"""Executable finite raw-process benchmarks with paired, prospective scorecards.

Publication manifests select the registered cells; smoke runs use a separate stream
and always retain their smoke label. Optional state/group scores are unavailable
unless an actual state/group method is selected, rather than invented from graphs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from effectome.connectivity import ConnectivityConfig
from effectome.connectivity.sparse_var import SparseVARModel, fit_sparse_var_model
from effectome.connectivity.switching_var import SwitchingVARModel, fit_switching_var_model
from effectome.connectivity.tvvar import TVVARModel, fit_tvvar_model
from effectome.data_module import PreprocessConfig, WindowConfig, make_windows, preprocess
from effectome.data_module.schema import NeuralRecording, WindowedSegments
from effectome.utils.synthetic import SyntheticConfig, make_synthetic_recording
from effectome.workflows.evaluation import (
    EvaluationConfig,
    paired_log_loss_delta,
    run_primary_evaluation,
    score_classification_records,
    score_regression_records,
)
from effectome.workflows.resume import content_fingerprint

from .validation import (
    ValidationAttempt,
    ValidationBudget,
    ValidationCell,
    _atomic_attempt,
    benchmark_cells,
    deterministic_seed,
    edge_recovery_metrics,
    run_validation_cells,
    synthetic_config_for_cell,
)

_FITTERS: dict[str, Callable[..., SparseVARModel | SwitchingVARModel | TVVARModel]] = {
    "constant_sparse_var": fit_sparse_var_model,
    "windowed_sparse_var": fit_sparse_var_model,
    "tvvar_smooth": fit_tvvar_model,
    "tvvar_fused": fit_tvvar_model,
    "switching_var": fit_switching_var_model,
}


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: str | Path
    profile: str = "smoke"
    master_seed: int = 260925
    suites: tuple[str, ...] = ("core", "stress", "nonlinear", "independent_calibration")
    include_grouping: bool = False
    max_cells: int | None = 1
    cell_offset: int = 0
    smoke_neurons: int = 4
    smoke_samples: int = 160
    smoke_history: int = 32
    smoke_target: int = 8
    smoke_stride: int = 16
    history_seconds: float = 120.0
    target_seconds: float = 5.0
    stride_seconds: float = 5.0
    methods: tuple[str, ...] = tuple(_FITTERS)
    max_origins: int | None = None
    selection_threshold: float = 0.01
    lambda1: float = 0.01
    lambda2: float = 0.001
    lambda_temporal: float = 0.01
    solver_max_iterations: int = 2000
    solver_tolerance: float = 1e-5
    switching_states: int = 2
    observed_projection_replicates: int = 0
    observed_projection_samples: int = 0
    observed_projection_tolerance: float = 0.1
    calibration_draws: int = 19
    calibration_alpha: float = 0.05
    calibration_tolerance: float | None = None
    resume: bool = True
    budget: ValidationBudget = field(default_factory=ValidationBudget)

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "publication"}:
            raise ValueError("benchmark profile must be smoke or publication")
        if self.profile == "smoke" and self.master_seed != 260925:
            raise ValueError("smoke must use the separate development stream 260925")
        if self.profile == "publication" and self.master_seed != 260926:
            raise ValueError("publication benchmark must use evaluation stream 260926")
        if self.max_cells is not None and self.max_cells < 1:
            raise ValueError("max_cells must be positive or null")
        if self.cell_offset < 0 or (self.max_origins is not None and self.max_origins < 1):
            raise ValueError("cell offset and origin limits must be nonnegative/positive")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("benchmark methods must be nonempty and unique")
        if "constant_sparse_var" not in self.methods:
            raise ValueError("paired neural benchmark requires constant_sparse_var")
        if not 0 < self.calibration_alpha < 1 or self.calibration_draws < 1:
            raise ValueError("invalid calibration alpha or draw count")
        if self.calibration_draws + 1 < 1 / self.calibration_alpha:
            raise ValueError("calibration Monte Carlo resolution cannot attain alpha")
        if self.calibration_tolerance is not None and not 0 < self.calibration_tolerance < 1:
            raise ValueError("calibration tolerance must lie in (0, 1)")
        if self.selection_threshold <= 0 or self.solver_max_iterations < 1:
            raise ValueError("selection threshold and solver iterations must be positive")


def selected_benchmark_cells(config: BenchmarkConfig) -> tuple[ValidationCell, ...]:
    available = benchmark_cells(include_grouping=config.include_grouping)
    known = {cell.suite for cell in available}
    if set(config.suites) - known:
        raise ValueError(f"unknown/unselected benchmark suites: {sorted(set(config.suites) - known)}")
    cells = tuple(cell for cell in available if cell.suite in config.suites)
    stop = None if config.max_cells is None else config.cell_offset + config.max_cells
    selected = cells[config.cell_offset : stop]
    if not selected:
        raise ValueError("benchmark selection contains no cells")
    return selected


def benchmark_preflight(config: BenchmarkConfig) -> dict[str, Any]:
    """Resolve every selected config without generating data or fitting models."""
    cells = selected_benchmark_cells(config)
    resolved = [_resolved_config(cell, config) for cell in cells]
    unresolved: list[str] = []
    if config.profile == "publication" and config.calibration_tolerance is None:
        unresolved.append("freeze calibration_tolerance before held-out calibration")
    if config.profile == "publication" and (
        not config.observed_projection_replicates or not config.observed_projection_samples
    ):
        unresolved.append("observed-process edge recovery unavailable without independent projection truth")
    return {
        "profile": config.profile,
        "selected_cells": len(cells),
        "suite_counts": {suite: sum(cell.suite == suite for cell in cells) for suite in config.suites},
        "full_registered_manifest": len(cells)
        == len(benchmark_cells(include_grouping=config.include_grouping)),
        "full_registered_grid": (
            len(cells) == len(benchmark_cells(include_grouping=config.include_grouping))
            and config.profile == "publication"
            and config.max_origins is None
        ),
        "methods": list(config.methods),
        "unsupported_methods": sorted(set(config.methods) - set(_FITTERS)),
        "native_fps": sorted({item.fps for item in resolved}),
        "node_counts": sorted({item.n_neurons for item in resolved}),
        "manifest_hash": content_fingerprint([asdict(cell) for cell in cells]),
        "config_hash": content_fingerprint(asdict(config)),
        "clock_manifest_hash": content_fingerprint(
            (Path(__file__).resolve().parents[3] / "conf/validation/worm_0_clock.yaml").read_text()
        ),
        "unresolved_scientific_gates": unresolved,
        "cells": [asdict(cell) | {"cell_id": cell.cell_id} for cell in cells],
    }


def _resolved_config(cell: ValidationCell, config: BenchmarkConfig) -> SyntheticConfig:
    synthetic = synthetic_config_for_cell(cell, master_seed=config.master_seed, profile=config.profile)
    replacements: dict[str, Any] = {
        "observed_projection_replicates": config.observed_projection_replicates,
        "observed_projection_samples": config.observed_projection_samples,
        "observed_projection_tolerance": config.observed_projection_tolerance,
    }
    if config.profile == "smoke":
        replacements.update(
            n_neurons=config.smoke_neurons,
            n_timepoints=config.smoke_samples,
            behavior_drivers=min(4, config.smoke_neurons),
        )
    return replace(synthetic, **replacements)


def _window_config(config: BenchmarkConfig) -> WindowConfig:
    values: dict[str, Any] = dict(
        mode="temporal", standardize_per_window=False, categorical_summary="last_valid"
    )
    if config.profile == "smoke":
        values.update(
            length=config.smoke_history,
            history_length=config.smoke_history,
            target_length=config.smoke_target,
            stride=config.smoke_stride,
        )
    else:
        values.update(
            history_seconds=config.history_seconds,
            target_seconds=config.target_seconds,
            stride_seconds=config.stride_seconds,
        )
    return WindowConfig(**values)


def _conditional_endpoint(windows: WindowedSegments, seed: int) -> WindowedSegments:
    """Generate a null from variables actually present in the issued M0 history.

    The per-sample generator endpoint has a different conditioning set. This
    benchmark target is explicitly constructed from the evaluator's history means
    and variances, then assigned to the next window's strictly future target.
    """
    noise = np.random.default_rng(np.random.SeedSequence([seed, 923_771])).normal(size=windows.n_windows)
    labels = np.zeros(windows.n_windows)
    for origin in range(windows.n_windows - 1):
        history = np.asarray(windows.segments[origin])
        labels[origin + 1] = float(history.mean() + np.mean(history.var(axis=0))) + noise[origin]
    return replace(windows, behavior_per_window={**windows.behavior_per_window, "conditional_null": labels})


def _evaluation_config(config: BenchmarkConfig, *, conditional_null: bool) -> EvaluationConfig:
    return EvaluationConfig(
        mode="development",
        endpoint_name="conditional_null" if conditional_null else "motif",
        endpoint_kind="synthetic_M0_conditional_null" if conditional_null else "synthetic_regime_label",
        endpoint_type="continuous" if conditional_null else "categorical",
        class_vocabulary=() if conditional_null else (0, 1),
        class_vocabulary_source="not_applicable" if conditional_null else "dataset_manifest",
        categorical_summary="not_applicable" if conditional_null else "last_valid",
        neural_max_lag=2,
        n_outer_folds=2 if config.profile == "smoke" else 5,
        neural_lambda1_grid=(config.lambda1,),
        neural_lambda2=config.lambda2,
        decoder_c_grid=(1.0,),
        decoder_alpha_grid=(1.0,),
        selection_seed=config.master_seed,
        solver_max_iterations=config.solver_max_iterations,
        solver_tolerance=config.solver_tolerance,
    )


def _method_config(method: str, config: BenchmarkConfig, seed: int) -> ConnectivityConfig:
    name = "sparse_var" if method in {"constant_sparse_var", "windowed_sparse_var"} else method
    extra: dict[str, Any] = {
        "max_iter": config.solver_max_iterations,
        "tol": config.solver_tolerance,
        "lambda2": config.lambda2,
    }
    if method != "switching_var":
        extra.update(lambda1=config.lambda1, standardize=True)
    if method.startswith("tvvar"):
        extra.update(lambda_temporal=config.lambda_temporal, knot_samples=16)
    if method == "switching_var":
        extra.update(n_states=config.switching_states, seed=seed)
    return ConnectivityConfig(name=name, max_lag=2, ridge=config.lambda2, extra=extra)


def _recovery_scores(
    recording: NeuralRecording, start: int, stop: int, estimate: np.ndarray, threshold: float
) -> dict[str, Any]:
    weights = np.asarray(recording.metadata["temporal_weights"])
    latent = np.einsum("c,clij->lij", weights[stop - 1], recording.metadata["true_lag_graphs"])
    result: dict[str, Any] = {
        "latent_endpoint_discrepancy": edge_recovery_metrics(latent, estimate, selection_threshold=threshold),
        "latent_interpretation": "descriptive_discrepancy_only_not_observed_process_recovery",
        "observed_projection": {
            "status": "unavailable",
            "reason": "independent_projection_truth_not_estimated",
        },
    }
    projection = recording.metadata.get("observed_projection_truth")
    homogeneous = np.allclose(weights[start:stop], weights[stop - 1])
    if projection is not None:
        reason = None
        stress = recording.metadata["stress_config"]
        if stress["variance_ratio"] != 1 or stress["measurement_change_ratio"] != 1:
            reason = "nonstationary_observation_or_innovation_projection_not_estimated"
        elif not projection["converged"]:
            reason = "observed_projection_MC_precision_gate_failed"
        elif not homogeneous or not np.isclose(weights[stop - 1].max(), 1):
            reason = "history_crosses_or_mixes_regimes"
        if reason:
            result["observed_projection"] = {"status": "unavailable", "reason": reason}
        else:
            truth = projection["coefficients"][int(np.argmax(weights[stop - 1]))]
            result["observed_projection"] = {
                "status": "available",
                "truth_id": projection["truth_id"],
                "metrics": edge_recovery_metrics(truth, estimate, selection_threshold=threshold),
            }
    return result


def _neural_comparisons(
    recording: NeuralRecording, windows: WindowedSegments, config: BenchmarkConfig, seed: int
) -> dict[str, Any]:
    if windows.n_windows < 2:
        return {"status": "unavailable", "reason": "insufficient_windows", "methods": {}, "paired": {}}
    indices = list(range(1, windows.n_windows))
    if config.max_origins is not None:
        indices = indices[: config.max_origins]
    raw = recording.traces.T.astype(float)
    prefix_stop = windows.anchors[0].context_stop
    scale = np.nanstd(raw[:prefix_stop], axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    method_scores: dict[str, Any] = {}
    for method in config.methods:
        method_started = time.perf_counter()
        if method not in _FITTERS:
            method_scores[method] = {
                "status": "unsupported",
                "reason": "no_audited_predictive_adapter",
                "origins": [],
            }
            continue
        rows: list[dict[str, Any]] = []
        model: SparseVARModel | SwitchingVARModel | TVVARModel | None = None
        state = None
        observed_stop = 0
        startup_failure: str | None = None
        fit_config = _method_config(method, config, seed)
        if method != "windowed_sparse_var":
            try:
                model = _FITTERS[method](raw[:prefix_stop], cfg=fit_config)
                state = model.initial_state(recording_identity=recording.identity.recording_id)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                startup_failure = f"{type(exc).__name__}:{exc}"
        for index in indices:
            anchor = windows.anchors[index]
            issue = anchor.context_stop
            row: dict[str, Any] = {
                "origin": index,
                "issue_sample": issue,
                "issue_seconds": float(recording.time[issue - 1]),
                "target_seconds": float(recording.time[issue]) if issue < len(raw) else None,
                "sample_convention": "issue_sample_is_exclusive_observation_stop",
                "status": "failed",
                "mse": None,
                "standardized_mse": None,
            }
            try:
                if startup_failure:
                    raise ValueError(startup_failure)
                if issue >= len(raw):
                    raise ValueError("no_future_neural_sample")
                if method == "windowed_sparse_var":
                    model = _FITTERS[method](windows.segments[index], cfg=fit_config)
                    state = model.initial_state(recording_identity=recording.identity.recording_id)
                    observed_stop = anchor.context_start
                assert model is not None and state is not None
                observed = raw[observed_stop:issue]
                state = model.observe(
                    state,
                    observed,
                    sample_indices=np.arange(observed_stop, issue),
                    masks=np.isfinite(observed),
                )
                observed_stop = issue
                forecast = model.forecast(state, horizon_samples=1)
                if forecast.failure_reason:
                    raise ValueError(forecast.failure_reason)
                if not np.all(np.isfinite(forecast.mean)):
                    raise ValueError("nonfinite_neural_forecast")
                if not np.all(np.isfinite(raw[issue])):
                    raise ValueError("missing_future_neural_sample")
                error = forecast.mean[0] - raw[issue]
                graph = model.graph_at(state)
                row.update(
                    status="issued",
                    mse=float(np.mean(error**2)),
                    standardized_mse=float(np.mean((error / scale) ** 2)),
                    model_hash=model.model_hash,
                    recovery=_recovery_scores(
                        recording,
                        anchor.context_start,
                        issue,
                        graph.lagged_coefficients,
                        config.selection_threshold,
                    ),
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                row["failure_reason"] = f"{type(exc).__name__}:{exc}"
            rows.append(row)
        issued = [row for row in rows if row["status"] == "issued"]
        method_scores[method] = {
            "status": "available" if issued else "unavailable",
            "origins": rows,
            "fit_scope": "independent_trailing_history"
            if method == "windowed_sparse_var"
            else "fixed_prefix_then_causal_replay",
            "training_stop": prefix_stop,
            "runtime_seconds": time.perf_counter() - method_started,
            "issued": len(issued),
            "failed": len(rows) - len(issued),
            "standardized_mse": float(np.mean([row["standardized_mse"] for row in issued]))
            if issued
            else None,
        }
    baseline = {
        row["origin"]: row
        for row in method_scores["constant_sparse_var"]["origins"]
        if row["status"] == "issued"
    }
    paired: dict[str, Any] = {}
    for method, scores in method_scores.items():
        if method == "constant_sparse_var":
            continue
        joined = [
            (baseline[row["origin"]], row)
            for row in scores["origins"]
            if row["status"] == "issued" and row["origin"] in baseline
        ]
        paired[method] = {
            "status": "available" if joined else "unavailable",
            "matched_origins": len(joined),
            "standardized_mse_gain": float(
                np.mean(
                    [base["standardized_mse"] - candidate["standardized_mse"] for base, candidate in joined]
                )
            )
            if joined
            else None,
            "origin_ids": [candidate["origin"] for _, candidate in joined],
            "inference": "descriptive_paired_origins_not_independent_animals",
        }
    return {
        "status": "available",
        "methods": method_scores,
        "paired": paired,
        "scale_policy": "fixed_initial_training_prefix",
        "scale_fit_stop": prefix_stop,
    }


def _score_recording(
    cell: ValidationCell, synthetic: SyntheticConfig, config: BenchmarkConfig
) -> dict[str, Any]:
    recording = make_synthetic_recording(synthetic)
    processed = preprocess(recording, PreprocessConfig())
    windows = make_windows(processed, _window_config(config))
    conditional = cell.suite == "independent_calibration" and cell.scenario == "endpoint"
    if conditional:
        windows = _conditional_endpoint(windows, int(synthetic.endpoint_seed or synthetic.seed))
    evaluation_cfg = _evaluation_config(config, conditional_null=conditional)
    evaluation = run_primary_evaluation(windows, evaluation_cfg)
    scorer = score_regression_records if conditional else score_classification_records
    score = [
        scorer(evaluation.predictions, model_name=name, endpoint_name=evaluation_cfg.endpoint_name)
        for name in ("M0_activity", "M1_effectome")
    ]
    delta = paired_log_loss_delta(
        evaluation.predictions,
        baseline_model="M0_activity",
        candidate_model="M1_effectome",
        endpoint_name=evaluation_cfg.endpoint_name,
    )
    neural = _neural_comparisons(processed, windows, config, synthetic.seed)
    return {
        "pipeline": "raw_identity_preprocess_training_owned_primary_plus_predictive_comparators",
        "graph_id": recording.metadata["graph_id"],
        "trajectory_id": recording.metadata["trajectory_id"],
        "input_hash": content_fingerprint(
            {
                "traces": recording.traces,
                "time": recording.time,
                "mask": recording.neural_validity,
                "behavior": recording.behavior,
            }
        ),
        "clock": {
            "fps": recording.fps,
            "n_samples": recording.n_timepoints,
            "source": "conf/validation/worm_0_clock.yaml",
        },
        "random_streams": {
            "graph": synthetic.graph_seed,
            "trajectory": synthetic.seed,
            "endpoint": synthetic.endpoint_seed,
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
        },
        "n_windows": windows.n_windows,
        "n_predictions": len(evaluation.predictions),
        "predictions": [asdict(record) for record in evaluation.predictions],
        "prediction_scores": [asdict(item) for item in score],
        "paired_prediction": asdict(delta),
        "evaluation_diagnostics": evaluation.diagnostics,
        "neural_comparison": neural,
        "optional_metrics": {
            "state_recovery": {
                "status": "unavailable",
                "reason": "no_selected_inductive_state_recovery_adapter",
            },
            "group_recovery": {"status": "unavailable", "reason": "optional_grouping_not_selected"},
        },
        "conditional_endpoint_contract": (
            {
                "conditioning_variables": ["history_activity_mean", "mean_history_per_neuron_variance"],
                "noise": "independent_fresh_gaussian",
                "future_window_offset": 1,
                "matches_M0_feature_span": True,
            }
            if conditional
            else None
        ),
    }


def _require_complete_primary(result: Mapping[str, Any]) -> None:
    delta = result.get("paired_prediction", {})
    if not delta.get("n_matched_predictions") or any(
        delta.get(key, 1)
        for key in ("n_baseline_only", "n_candidate_only", "n_failed_baseline", "n_failed_candidate")
    ):
        raise ValueError("complete registered M0/M1 prediction family is required")
    diagnostics = result.get("evaluation_diagnostics", {})
    if diagnostics.get("unavailable_folds", 1):
        raise ValueError("complete registered evaluator folds are required")
    folds = diagnostics.get("folds", {})
    expected = sum(fold["test_examples"] for fold in folds.values())
    if (
        not folds
        or len(folds) != diagnostics.get("available_folds")
        or delta["n_matched_predictions"] != expected
    ):
        raise ValueError("every expected registered M0/M1 origin pair is required")


def _null_statistic(result: Mapping[str, Any], family: str) -> float:
    if family == "endpoint":
        delta = result["paired_prediction"]
        statistic = float(delta["equal_animal_delta_log_loss"])
        if not np.isfinite(statistic):
            raise ValueError("conditional null statistic must be finite")
        _require_complete_primary(result)
        return statistic
    comparison = result["neural_comparison"]
    paired = comparison["paired"]
    methods = comparison["methods"]
    if any(value["status"] != "available" or value.get("failed", 0) for value in methods.values()):
        raise ValueError("fixed-interaction null requires every registered predictive method")
    baseline_origins = [row["origin"] for row in methods["constant_sparse_var"]["origins"]]
    if any(
        value["status"] != "available" or value["origin_ids"] != baseline_origins for value in paired.values()
    ):
        raise ValueError("fixed-interaction null requires matched origins for the complete model family")
    gains = [value["standardized_mse_gain"] for value in paired.values()]
    if not gains:
        raise ValueError("fixed-interaction null has no paired dynamic comparator")
    if not np.all(np.isfinite(gains)):
        raise ValueError("every registered model gain must be finite")
    _require_complete_primary(result)
    return max(float(value) for value in gains)


def _calibrate_cell(
    cell: ValidationCell, synthetic: SyntheticConfig, config: BenchmarkConfig, observed: Mapping[str, Any]
) -> dict[str, Any]:
    """Known-DGP full-pipeline Monte Carlo null, conditional on the generated graph.

    The maximum across the frozen dynamic method family is recomputed in every
    independent raw null draw. It is a simulator calibration test, never a claim
    that an observational animal satisfies this known generative null.
    """
    observed_statistic = _null_statistic(observed, cell.scenario)
    values: list[float] = []
    failures: list[dict[str, Any]] = []
    seed_mapping: list[dict[str, int | None]] = []
    for draw in range(config.calibration_draws):
        seed = deterministic_seed(
            config.master_seed,
            suite="calibration_reference",
            scenario=cell.scenario,
            graph=cell.graph,
            replicate=draw,
            purpose="independent_raw_null",
        )
        reference = replace(
            synthetic,
            seed=seed,
            endpoint_seed=seed,
            observed_projection_replicates=0,
            observed_projection_samples=0,
        )
        seed_mapping.append(
            {
                "draw": draw,
                "graph_seed": reference.graph_seed,
                "trajectory_seed": reference.seed,
                "endpoint_seed": reference.endpoint_seed,
            }
        )
        try:
            values.append(_null_statistic(_score_recording(cell, reference, config), cell.scenario))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failures.append({"draw": draw, "seed": seed, "reason": f"{type(exc).__name__}:{exc}"})
    if failures:
        return {
            "status": "unavailable",
            "reason": "registered_null_draw_failure",
            "p_value": None,
            "reject": False,
            "failed_draws": failures,
            "successful_statistics": values,
            "seed_mapping": seed_mapping,
        }
    p_value = (1 + sum(value >= observed_statistic for value in values)) / (config.calibration_draws + 1)
    return {
        "status": "available",
        "null_kind": "known_DGP_full_raw_pipeline_conditional_on_graph",
        "statistic": observed_statistic,
        "null_statistics": values,
        "p_value": p_value,
        "alpha": config.calibration_alpha,
        "reject": p_value <= config.calibration_alpha,
        "null_draws": config.calibration_draws,
        "seed_mapping": seed_mapping,
        "graph_seed": synthetic.graph_seed,
        "selection_policy": "maximum_over_registered_dynamic_comparators"
        if cell.scenario != "endpoint"
        else "fixed_M0_M1_probability_loss_gain",
    }


def _binomial_report(rejections: int, available: int, attempted: int) -> dict[str, Any]:
    low = 0.0 if rejections == 0 else float(beta.ppf(0.025, rejections, available - rejections + 1))
    high = 1.0 if rejections == available else float(beta.ppf(0.975, rejections + 1, available - rejections))
    return {
        "attempted": attempted,
        "available": available,
        "unavailable": attempted - available,
        "rejections": rejections,
        "rate": rejections / available if available else None,
        "binomial_95_interval": [low, high] if available else None,
        "independent_units": "graph_trajectory_pairs",
    }


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run/resume registered cells, then atomically save a complete scorecard."""
    preflight = benchmark_preflight(config)
    if config.profile == "publication" and "independent_calibration" in config.suites:
        if config.calibration_tolerance is None:
            raise ValueError("freeze calibration_tolerance before held-out calibration")
    cells = selected_benchmark_cells(config)

    def factory(cell: ValidationCell, _seed: int) -> Mapping[str, Any]:
        return {
            "synthetic": asdict(_resolved_config(cell, config)),
            "benchmark": asdict(config),
            "cell": asdict(cell),
            "manifest_hash": preflight["manifest_hash"],
            "clock_manifest_hash": preflight["clock_manifest_hash"],
        }

    def evaluate(cell: ValidationCell, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if cell.suite == "grouping":
            return {
                "status": "unsupported",
                "reason": "optional_grouping_requires_registered_family_generator",
            }
        synthetic = SyntheticConfig(**payload["synthetic"])
        result = _score_recording(cell, synthetic, config)
        result["profile"] = config.profile
        if cell.suite == "independent_calibration":
            result["calibration"] = _calibrate_cell(cell, synthetic, config, result)
        return result

    attempts = run_validation_cells(
        cells,
        output_dir=config.output_dir,
        master_seed=config.master_seed,
        config_factory=factory,
        evaluator=evaluate,
        budget=config.budget,
        resume=config.resume,
    )
    report = _benchmark_report(config, preflight, cells, attempts)
    _atomic_attempt(Path(config.output_dir) / "report.json", report)
    return report


def _benchmark_report(
    config: BenchmarkConfig,
    preflight: Mapping[str, Any],
    cells: Sequence[ValidationCell],
    attempts: Sequence[ValidationAttempt],
) -> dict[str, Any]:
    calibration: dict[str, Any] = {}
    for family in ("variance", "measurement", "endpoint"):
        selected = [
            attempt
            for cell, attempt in zip(cells, attempts, strict=True)
            if cell.suite == "independent_calibration" and cell.scenario == family
        ]
        records = [
            attempt.result["calibration"]
            for attempt in selected
            if attempt.status == "completed" and "calibration" in attempt.result
        ]
        available = [record for record in records if record["status"] == "available"]
        calibration[family] = _binomial_report(
            sum(bool(record["reject"]) for record in available), len(available), len(selected)
        )
        sufficient = len(selected) == 200 and len(available) == 200
        rate = calibration[family]["rate"]
        calibration[family]["pass"] = (
            rate <= config.calibration_alpha + config.calibration_tolerance
            if sufficient and config.calibration_tolerance is not None and rate is not None
            else None
        )
        calibration[family]["status"] = "complete" if sufficient else "incomplete_registered_family"

    def complete(attempt: ValidationAttempt) -> bool:
        if attempt.status != "completed" or attempt.result.get("status") == "unsupported":
            return False
        try:
            _require_complete_primary(attempt.result)
        except ValueError:
            return False
        comparison = attempt.result.get("neural_comparison", {})
        if any(
            method["status"] != "available" or method.get("failed", 0)
            for method in comparison.get("methods", {}).values()
        ):
            return False
        calibration_result = attempt.result.get("calibration")
        return calibration_result is None or calibration_result["status"] == "available"

    all_completed = bool(attempts) and all(complete(attempt) for attempt in attempts)
    full_grid_completed = (
        bool(preflight["full_registered_grid"]) and all_completed and config.max_origins is None
    )
    return {
        "profile": config.profile,
        "publication_grid_attempted": config.profile == "publication",
        "publication_grid_executed": config.profile == "publication" and full_grid_completed,
        "full_registered_grid_attempted": bool(preflight["full_registered_grid"]),
        "full_registered_grid_executed": full_grid_completed,
        "preflight": dict(preflight),
        "calibration": calibration,
        "counts": {
            status: sum(attempt.status == status for attempt in attempts)
            for status in ("completed", "failed", "timeout", "memory_exceeded")
        },
        "attempts": [asdict(attempt) for attempt in attempts],
        "scientific_status": "smoke_only" if config.profile == "smoke" else "registered_benchmark_outputs",
        "scope": (
            "Appendix-C finite suite; correlated_innovations stress retained; "
            "variance-only separate calibration"
        ),
        "state_and_group_inference": "optional_unavailable_outputs_are_not_successes",
    }


def load_benchmark_config(path: str | Path, **overrides: Any) -> BenchmarkConfig:
    """Load the checked-in runtime section without silently ignoring unknown keys."""
    from omegaconf import OmegaConf

    manifest = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("runner"), dict):
        raise ValueError("validation manifest requires a runner mapping")
    values = {**manifest["runner"], **overrides}
    for key in ("methods", "suites"):
        if key in values:
            values[key] = tuple(values[key])
    if isinstance(values.get("budget"), dict):
        values["budget"] = ValidationBudget(**values["budget"])
    return BenchmarkConfig(**values)


def write_preflight(path: str | Path, config: BenchmarkConfig) -> None:
    _atomic_attempt(Path(path), benchmark_preflight(config))


def print_benchmark_summary(report: Mapping[str, Any]) -> str:
    """Compact JSON for a CLI; full origin records live in the persisted report."""
    return json.dumps({key: report[key] for key in ("profile", "counts", "scientific_status")}, indent=2)
