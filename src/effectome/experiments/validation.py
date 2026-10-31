"""Validation manifest helpers for revised synthetic benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.metrics import average_precision_score

from effectome.data_module import PreprocessConfig, WindowConfig, make_windows, preprocess
from effectome.utils.synthetic import SyntheticConfig, make_synthetic_recording
from effectome.workflows import (
    EvaluationConfig,
    content_fingerprint,
    implementation_fingerprint,
    run_primary_evaluation,
)

SuiteName = Literal["core", "stress", "nonlinear", "grouping", "independent_calibration"]


@dataclass(frozen=True)
class ValidationSuiteSpec:
    name: SuiteName
    cells: int
    replicates_per_cell: int
    recordings: int
    optional: bool = False


@dataclass(frozen=True)
class ValidationCell:
    """One persisted dataset attempt in the finite validation manifest."""

    suite: SuiteName
    scenario: str
    graph: int
    replicate: int
    condition: str = "baseline"

    @property
    def cell_id(self) -> str:
        return (
            f"{self.suite}:{self.scenario}:{self.condition}:"
            f"g{self.graph:03d}:r{self.replicate:03d}"
        )


@dataclass(frozen=True)
class ValidationBudget:
    """Per-cell resource policy recorded before benchmark execution."""

    timeout_seconds: float = 3_600.0
    memory_limit_mb: float = 16_384.0
    failure_policy: str = "record_and_continue"

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.memory_limit_mb <= 0:
            raise ValueError("validation resource limits must be positive")
        if self.failure_policy not in {"record_and_continue", "fail_fast"}:
            raise ValueError("unsupported validation failure policy")


@dataclass(frozen=True)
class ValidationAttempt:
    """Machine-readable outcome for one finite registered benchmark cell."""

    cell_id: str
    status: str
    seed: int
    config_hash: str
    elapsed_seconds: float
    peak_memory_mb: float
    budget: ValidationBudget
    result: dict[str, Any]
    actual_config: dict[str, Any]
    graph_id: str
    trajectory_id: str
    input_hash: str
    implementation_hash: str
    evaluator_hash: str
    runtime_method: str = "time.perf_counter"
    memory_method: str = "tracemalloc_python_peak"
    timeout_method: str = "posthoc_wall_clock_budget"
    failure_reason: str | None = None
    resumed: bool = False


@dataclass(frozen=True)
class DevelopmentSmokeConfig:
    """Notebook-friendly bounded WP9 smoke configuration (never a publication grid)."""

    output_dir: str | Path
    master_seed: int = 260925
    scenarios: tuple[str, ...] = ("recurrent_geometric",)
    n_neurons: int = 8
    n_timepoints: int = 320
    history_length: int = 64
    target_length: int = 8
    stride: int = 8
    max_cells: int = 1
    resume: bool = True
    observed_projection_replicates: int = 0
    observed_projection_samples: int = 0

    def __post_init__(self) -> None:
        if not self.scenarios or self.max_cells < 1:
            raise ValueError("smoke scenarios must be non-empty and max_cells positive")
        if self.n_neurons < 2 or self.n_timepoints <= self.history_length:
            raise ValueError("smoke dimensions must support at least one complete window")
        if not 1 <= self.target_length <= self.history_length or self.stride < 1:
            raise ValueError("invalid smoke window geometry")


@dataclass(frozen=True)
class DevelopmentSmokeResult:
    label: str
    attempts: tuple[ValidationAttempt, ...]
    summary: dict[str, Any]
    publication_grid_executed: bool = False


def _stable_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "sha256": hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest(),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    if isinstance(value, np.generic):
        return _stable_json(value.item())
    if isinstance(value, Mapping):
        return {str(key): _stable_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _atomic_attempt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(_stable_json(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _callable_fingerprint(callback: Callable[..., Any]) -> str:
    """Fingerprint callback code without depending on mutable closure state."""
    code = getattr(callback, "__code__", None)
    payload = {
        "module": getattr(callback, "__module__", type(callback).__module__),
        "qualname": getattr(callback, "__qualname__", type(callback).__qualname__),
        "bytecode": None if code is None else code.co_code.hex(),
        "constants": None if code is None else repr(code.co_consts),
        "names": None if code is None else code.co_names,
    }
    encoded = json.dumps(_stable_json(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def run_validation_cells(
    cells: Sequence[ValidationCell],
    *,
    output_dir: str | Path,
    master_seed: int,
    config_factory: Callable[[ValidationCell, int], Mapping[str, Any]],
    evaluator: Callable[[ValidationCell, Mapping[str, Any]], Mapping[str, Any]],
    budget: ValidationBudget | None = None,
    resume: bool = True,
    progress_callback: Callable[[int, int, ValidationAttempt], None] | None = None,
) -> tuple[ValidationAttempt, ...]:
    """Execute and atomically persist registered cells with resumable attempt records.

    The caller supplies the raw-data/evaluation adapter so benchmark cells use the same
    evaluator as real recordings. This function deliberately does not select thresholds or
    silently retry failures; every registered attempt remains represented in the output.
    """
    destination = Path(output_dir)
    budget = ValidationBudget() if budget is None else budget
    implementation_hash = implementation_fingerprint()
    evaluator_hash = _callable_fingerprint(evaluator)
    budget_payload = _stable_json(asdict(budget))
    attempts: list[ValidationAttempt] = []
    total = len(cells)
    for ordinal, cell in enumerate(cells, start=1):
        path = destination / cell.suite / f"{cell.cell_id.replace(':', '__')}.json"
        seed = deterministic_seed(
            master_seed,
            suite=cell.suite,
            scenario=cell.scenario,
            graph=cell.graph,
            replicate=cell.replicate,
            purpose="recording",
        )
        config = dict(config_factory(cell, seed))
        actual_config = dict(_stable_json(config))
        encoded = json.dumps(actual_config, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(encoded.encode()).hexdigest()
        graph_id = str(
            config.get(
                "graph_id",
                f"{cell.suite}:{cell.scenario}:g{cell.graph:03d}:seed-{config.get('graph_seed', seed)}",
            )
        )
        trajectory_id = str(
            config.get(
                "trajectory_id",
                f"{cell.cell_id}:seed-{config.get('seed', seed)}",
            )
        )
        stored: dict[str, Any] | None = None
        if resume and path.is_file():
            with path.open(encoding="utf-8") as handle:
                stored = json.load(handle)
        reusable = bool(
            stored is not None
            and int(stored.get("seed", -1)) == seed
            and stored.get("config_hash") == config_hash
            and stored.get("budget") == budget_payload
            and stored.get("implementation_hash") == implementation_hash
            and stored.get("evaluator_hash") == evaluator_hash
        )
        if reusable:
            assert stored is not None
            attempt = ValidationAttempt(
                cell_id=str(stored["cell_id"]),
                status=str(stored["status"]),
                seed=int(stored["seed"]),
                config_hash=str(stored["config_hash"]),
                elapsed_seconds=float(stored["elapsed_seconds"]),
                peak_memory_mb=float(stored["peak_memory_mb"]),
                budget=ValidationBudget(**stored["budget"]),
                result=dict(stored.get("result", {})),
                actual_config=dict(stored.get("actual_config", {})),
                graph_id=str(stored.get("graph_id", f"{cell.cell_id}:graph")),
                trajectory_id=str(stored.get("trajectory_id", f"{cell.cell_id}:trajectory")),
                input_hash=str(stored.get("input_hash", stored["config_hash"])),
                implementation_hash=str(stored["implementation_hash"]),
                evaluator_hash=str(stored["evaluator_hash"]),
                runtime_method=str(stored.get("runtime_method", "time.perf_counter")),
                memory_method=str(stored.get("memory_method", "tracemalloc_python_peak")),
                timeout_method=str(
                    stored.get("timeout_method", "posthoc_wall_clock_budget")
                ),
                failure_reason=stored.get("failure_reason"),
                resumed=True,
            )
        else:
            started = time.perf_counter()
            tracemalloc.start()
            status = "completed"
            failure_reason = None
            result: dict[str, Any] = {}
            try:
                result = dict(_stable_json(dict(evaluator(cell, config))))
            except Exception as exc:  # every failed registered attempt must remain visible
                status = "failed"
                failure_reason = f"{type(exc).__name__}:{exc}"
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed = time.perf_counter() - started
            peak_mb = float(peak / (1024**2))
            input_hash = str(result.get("input_hash", config_hash))
            graph_id = str(result.get("graph_id", graph_id))
            trajectory_id = str(result.get("trajectory_id", trajectory_id))
            if status == "completed" and elapsed > budget.timeout_seconds:
                status = "timeout"
                failure_reason = "elapsed time exceeded the registered cell budget"
            if status == "completed" and peak_mb > budget.memory_limit_mb:
                status = "memory_exceeded"
                failure_reason = "traced Python memory exceeded the registered cell budget"
            attempt = ValidationAttempt(
                cell_id=cell.cell_id,
                status=status,
                seed=seed,
                config_hash=config_hash,
                elapsed_seconds=float(elapsed),
                peak_memory_mb=peak_mb,
                budget=budget,
                result=result,
                actual_config=actual_config,
                graph_id=graph_id,
                trajectory_id=trajectory_id,
                input_hash=input_hash,
                implementation_hash=implementation_hash,
                evaluator_hash=evaluator_hash,
                failure_reason=failure_reason,
            )
            _atomic_attempt(path, _stable_json(asdict(attempt)))
            if status != "completed" and budget.failure_policy == "fail_fast":
                attempts.append(attempt)
                if progress_callback is not None:
                    progress_callback(ordinal, total, attempt)
                raise RuntimeError(f"validation cell {cell.cell_id} {status}: {failure_reason}")
        attempts.append(attempt)
        if progress_callback is not None:
            progress_callback(ordinal, total, attempt)
    return tuple(attempts)


REVISED_VALIDATION_SUITES: tuple[ValidationSuiteSpec, ...] = (
    ValidationSuiteSpec("core", cells=100, replicates_per_cell=5, recordings=500),
    ValidationSuiteSpec("stress", cells=90, replicates_per_cell=3, recordings=270),
    ValidationSuiteSpec("nonlinear", cells=10, replicates_per_cell=3, recordings=30),
    ValidationSuiteSpec("grouping", cells=40, replicates_per_cell=3, recordings=120, optional=True),
    ValidationSuiteSpec(
        "independent_calibration",
        cells=600,
        replicates_per_cell=1,
        recordings=600,
    ),
)


def benchmark_cells(*, include_grouping: bool = False) -> tuple[ValidationCell, ...]:
    """Materialize the finite Appendix-C benchmark without running any cell."""
    cells: list[ValidationCell] = []
    temporal = (
        "constant",
        "smooth_drift",
        "nonrecurring_epochs",
        "recurrent_geometric",
        "recurrent_non_geometric",
    )
    for scenario in temporal:
        for size in (20, 109):
            for graph in range(10):
                for replicate in range(5):
                    cells.append(
                        ValidationCell(
                            "core",
                            scenario,
                            graph,
                            replicate,
                            condition=f"n{size}",
                        )
                    )

    stress_conditions = (
        "snr_low",
        "snr_high",
        "dwell_short",
        "dwell_long",
        "kernel_heterogeneity",
        "burst_missingness",
        "latent_input",
        "instantaneous",
        "correlated_innovations",
    )
    for condition in stress_conditions:
        for graph in range(10):
            for replicate in range(3):
                cells.append(ValidationCell("stress", "recurrent_geometric", graph, replicate, condition))

    for graph in range(10):
        for replicate in range(3):
            cells.append(ValidationCell("nonlinear", "nonlinear_saturation", graph, replicate))

    if include_grouping:
        for family in ("assortative", "feedforward", "core_periphery", "no_group"):
            for graph in range(10):
                for replicate in range(3):
                    cells.append(ValidationCell("grouping", family, graph, replicate))

    for family in ("variance", "measurement", "endpoint"):
        for graph in range(200):
            cells.append(ValidationCell("independent_calibration", family, graph, 0))

    ids = [cell.cell_id for cell in cells]
    if len(ids) != len(set(ids)):  # defensive guard for future manifest edits
        raise RuntimeError("validation cell IDs must be unique")
    return tuple(cells)


def validation_totals(*, include_grouping: bool = False) -> dict[str, int]:
    selected = [
        suite for suite in REVISED_VALIDATION_SUITES if include_grouping or not suite.optional
    ]
    core_like = [
        suite
        for suite in selected
        if suite.name in {"core", "stress", "nonlinear", "grouping"}
    ]
    return {
        "core_stress_nonlinear": sum(
            suite.recordings
            for suite in REVISED_VALIDATION_SUITES
            if suite.name in {"core", "stress", "nonlinear"}
        ),
        "with_grouping": sum(suite.recordings for suite in core_like),
        "independent_calibration": next(
            suite.recordings
            for suite in REVISED_VALIDATION_SUITES
            if suite.name == "independent_calibration"
        ),
        "selected_total": sum(suite.recordings for suite in selected),
    }


def deterministic_seed(
    master_seed: int,
    *,
    suite: str,
    scenario: str,
    graph: int,
    replicate: int,
    purpose: str,
) -> int:
    """Derive stable integer streams without Python's randomized hash()."""
    tokens = [suite, scenario, str(graph), str(replicate), purpose]
    entropy = [int(master_seed)]
    for token in tokens:
        encoded = token.encode("utf-8")
        entropy.extend(encoded)
        entropy.append(0)
    sequence = np.random.SeedSequence(entropy)
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def synthetic_config_for_cell(
    cell: ValidationCell,
    *,
    master_seed: int,
    profile: str = "publication",
    fps: float | None = None,
) -> SyntheticConfig:
    """Resolve one registered cell to an explicit executable generator config."""
    if profile not in {"smoke", "publication"}:
        raise ValueError("validation profile must be smoke or publication")
    seed = deterministic_seed(
        master_seed,
        suite=cell.suite,
        scenario=cell.scenario,
        graph=cell.graph,
        replicate=cell.replicate,
        purpose="trajectory",
    )
    graph_seed = deterministic_seed(
        master_seed,
        suite=cell.suite,
        scenario=cell.scenario,
        graph=cell.graph,
        replicate=0,
        purpose="graph",
    )
    endpoint_seed = deterministic_seed(
        master_seed,
        suite=cell.suite,
        scenario=cell.scenario,
        graph=cell.graph,
        replicate=cell.replicate,
        purpose="endpoint",
    )
    n_neurons = 20 if cell.suite in {"core", "grouping"} else 109
    if cell.condition.startswith("n") and cell.condition[1:].isdigit():
        n_neurons = int(cell.condition[1:])
    if profile == "smoke":
        n_neurons = min(n_neurons, 8)
    if fps is None:
        from omegaconf import OmegaConf

        manifest = Path(__file__).resolve().parents[3] / "conf/validation/worm_0_clock.yaml"
        fps = float(OmegaConf.load(manifest).fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("validation requires a positive native manifest clock")

    registered_scenarios = {
        "constant",
        "smooth_drift",
        "nonrecurring_epochs",
        "recurrent_geometric",
        "recurrent_non_geometric",
    }
    scenario = cell.scenario if cell.scenario in registered_scenarios else "recurrent_geometric"
    values: dict[str, Any] = {
        "n_neurons": n_neurons,
        "n_timepoints": 320 if profile == "smoke" else 3137,
        "density": 0.05,
        "max_lag": 2,
        "coupling": 0.25,
        "noise_std": 0.7,
        "fps": fps,
        "cross_coefficient_range": (0.05, 0.20),
        "self_lag_coefficients": (0.5, 0.1),
        "calcium_kernel_seconds": 1.0,
        "measurement_snr": 3.0,
        "n_states": 1 if scenario == "constant" else 3,
        "regime_dwell": max(1, round(60.0 * fps)),
        "behavior_drivers": min(4, n_neurons),
        "regime_sign_flip_fraction": 0.1,
        "regime_sparsity_jitter": 0.1,
        "scenario": scenario,
        "seed": seed,
        "graph_seed": graph_seed,
        "endpoint_seed": endpoint_seed,
    }
    condition = cell.condition
    if condition == "snr_low":
        values["measurement_snr"] = 1.0
    elif condition == "snr_high":
        values["measurement_snr"] = 10.0
    elif condition == "dwell_short":
        values["regime_dwell"] = max(1, round(15.0 * fps))
    elif condition == "dwell_long":
        values["regime_dwell"] = max(1, round(180.0 * fps))
    elif condition == "kernel_heterogeneity":
        values["calcium_kernel_decay_range_seconds"] = (0.3, 3.0)
    elif condition == "burst_missingness":
        values["missing_fraction"] = 0.05
        values["missingness_mode"] = "burst"
        values["missing_block_length"] = 20
    elif condition == "latent_input":
        values["n_latent_confounders"] = 2
        values["latent_strength"] = 0.25
    elif condition == "instantaneous":
        values["instantaneous_density"] = 0.03
        values["instantaneous_strength"] = 0.15
    elif condition == "correlated_innovations":
        values["innovation_correlation"] = 0.25
    if cell.suite == "nonlinear" or cell.scenario == "nonlinear_saturation":
        values["nonlinear_saturation"] = True
    if cell.suite == "independent_calibration":
        values["scenario"] = "constant"
        values["n_states"] = 1
        if cell.scenario == "variance":
            values["variance_ratio"] = 4.0
        elif cell.scenario == "measurement":
            values["measurement_change_ratio"] = 4.0
        elif cell.scenario == "endpoint":
            values["endpoint_mode"] = "conditional_null"
            values["conditional_null_m0_variables"] = (
                "activity_mean",
                "activity_variance",
            )
    return SyntheticConfig(**values)


def _edge_metrics_values(
    truth_values: np.ndarray,
    estimate_values: np.ndarray,
    *,
    selection_threshold: float,
) -> dict[str, float | int]:
    truth_selected = ~np.isclose(truth_values, 0.0)
    estimate_selected = np.abs(estimate_values) >= selection_threshold
    selected_count = int(estimate_selected.sum())
    true_count = int(truth_selected.sum())
    selected_null = int(np.sum(estimate_selected & ~truth_selected))
    wrong_sign = int(
        np.sum(
            estimate_selected
            & truth_selected
            & (np.sign(estimate_values) != np.sign(truth_values))
        )
    )
    return {
        "n_edges": int(truth_values.size),
        "n_true": true_count,
        "n_selected": selected_count,
        "prevalence_baseline": float(true_count / max(1, truth_values.size)),
        "auprc": (
            float(average_precision_score(truth_selected.astype(int), np.abs(estimate_values)))
            if np.any(truth_selected)
            else float("nan")
        ),
        "fdp": float(selected_null / max(1, selected_count)),
        "signed_fdp": float((selected_null + wrong_sign) / max(1, selected_count)),
        "sensitivity": float(
            np.sum(estimate_selected & truth_selected) / max(1, true_count)
        ),
        "coefficient_rmse": float(np.sqrt(np.mean((estimate_values - truth_values) ** 2))),
        "wrong_sign_selected_true": wrong_sign,
    }


def edge_recovery_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    *,
    selection_threshold: float,
) -> dict[str, Any]:
    """Report lag-resolved recovery, prevalence, FDP, sensitivity, and signed FDP."""
    truth_array = np.asarray(truth, dtype=float)
    estimate_array = np.asarray(estimate, dtype=float)
    if truth_array.shape != estimate_array.shape or truth_array.ndim != 3:
        raise ValueError("truth and estimate must share shape [lag, source, target]")
    if truth_array.shape[1] != truth_array.shape[2]:
        raise ValueError("edge matrices must be square")
    if selection_threshold < 0 or not np.isfinite(selection_threshold):
        raise ValueError("selection_threshold must be finite and nonnegative")
    off_diagonal = ~np.eye(truth_array.shape[1], dtype=bool)
    by_lag = tuple(
        _edge_metrics_values(
            truth_array[lag][off_diagonal],
            estimate_array[lag][off_diagonal],
            selection_threshold=selection_threshold,
        )
        for lag in range(truth_array.shape[0])
    )
    aggregate_truth = np.concatenate(
        [truth_array[lag][off_diagonal] for lag in range(truth_array.shape[0])]
    )
    aggregate_estimate = np.concatenate(
        [estimate_array[lag][off_diagonal] for lag in range(estimate_array.shape[0])]
    )
    overall = _edge_metrics_values(
        aggregate_truth,
        aggregate_estimate,
        selection_threshold=selection_threshold,
    )
    return {"by_lag": by_lag, "overall": overall}


def _smoke_evaluator(
    cell: ValidationCell,
    config: Mapping[str, Any],
    *,
    smoke: DevelopmentSmokeConfig,
) -> Mapping[str, Any]:
    synthetic_fields = {item.name for item in fields(SyntheticConfig)}
    synthetic_config = {
        key: value for key, value in config.items() if key in synthetic_fields
    }
    recording = make_synthetic_recording(SyntheticConfig(**synthetic_config))
    processed = preprocess(recording, PreprocessConfig())
    windows = make_windows(
        processed,
        WindowConfig(
            length=smoke.history_length,
            history_length=smoke.history_length,
            target_length=smoke.target_length,
            stride=smoke.stride,
            mode="temporal",
            categorical_summary="last_valid",
            standardize_per_window=False,
        ),
    )
    evaluation = run_primary_evaluation(
        windows,
        EvaluationConfig(
            mode="development",
            endpoint_name="motif",
            endpoint_kind="synthetic_regime_label",
            endpoint_type="categorical",
            class_vocabulary=(0, 1),
            n_outer_folds=2,
            min_train_examples=2,
            min_test_examples=1,
            neural_lambda1_grid=(0.0,),
            decoder_c_grid=(1.0,),
            selection_seed=smoke.master_seed,
        ),
    )
    input_hash = content_fingerprint(
        {
            "cell_id": cell.cell_id,
            "traces": recording.traces,
            "time": recording.time,
            "behavior": recording.behavior,
            "observation_mask": recording.metadata["observation_mask"],
        }
    )
    return {
        "pipeline": "preprocess_window_primary_evaluation",
        "profile": "smoke",
        "publication_grid_executed": False,
        "input_hash": input_hash,
        "graph_id": recording.metadata["graph_id"],
        "trajectory_id": recording.metadata["trajectory_id"],
        "n_windows": windows.n_windows,
        "n_predictions": len(evaluation.predictions),
        "evaluation_status": evaluation.diagnostics["status"],
    }


def run_development_smoke(config: DevelopmentSmokeConfig) -> DevelopmentSmokeResult:
    """Run a bounded, resumable raw-to-score smoke path without notebook execution."""
    cells = tuple(
        ValidationCell("core", scenario, graph=index, replicate=0, condition="smoke")
        for index, scenario in enumerate(config.scenarios)
    )[: config.max_cells]

    def make_config(cell: ValidationCell, _seed: int) -> Mapping[str, Any]:
        resolved = synthetic_config_for_cell(
            cell,
            master_seed=config.master_seed,
            profile="smoke",
        )
        resolved = replace(
            resolved,
            n_neurons=config.n_neurons,
            n_timepoints=config.n_timepoints,
            behavior_drivers=min(resolved.behavior_drivers, config.n_neurons),
            observed_projection_replicates=config.observed_projection_replicates,
            observed_projection_samples=config.observed_projection_samples,
        )
        return {
            **asdict(resolved),
            "_smoke_contract": {
                "history_length": config.history_length,
                "target_length": config.target_length,
                "stride": config.stride,
                "evaluation_profile": "development_primary",
            },
        }

    attempts = run_validation_cells(
        cells,
        output_dir=config.output_dir,
        master_seed=config.master_seed,
        config_factory=make_config,
        evaluator=lambda cell, payload: _smoke_evaluator(cell, payload, smoke=config),
        resume=config.resume,
    )
    counts = {
        status: sum(attempt.status == status for attempt in attempts)
        for status in ("completed", "failed", "timeout", "memory_exceeded")
    }
    return DevelopmentSmokeResult(
        label="smoke",
        attempts=attempts,
        summary={**counts, "total": len(attempts)},
        publication_grid_executed=False,
    )


__all__ = [
    "DevelopmentSmokeConfig",
    "DevelopmentSmokeResult",
    "REVISED_VALIDATION_SUITES",
    "ValidationCell",
    "ValidationAttempt",
    "ValidationBudget",
    "ValidationSuiteSpec",
    "benchmark_cells",
    "deterministic_seed",
    "edge_recovery_metrics",
    "run_development_smoke",
    "run_validation_cells",
    "synthetic_config_for_cell",
    "validation_totals",
]
