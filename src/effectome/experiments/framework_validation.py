"""Reproducible raw-process validation of the shared-transition framework."""

from __future__ import annotations

import csv
import itertools
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.transition_examples import (
    TRANSITION_EXAMPLE_SCENARIOS,
    TransitionExampleConfig,
    make_transition_example,
    make_transition_example_windows,
)
from effectome.workflows.cross_method import (
    _json_safe,
    load_cross_method_profile,
    run_cross_method_comparison,
)
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint


@dataclass(frozen=True)
class FrameworkValidationConfig:
    profile: str = "smoke"
    seeds: tuple[int, ...] = (0, 1, 2)
    scenarios: tuple[str, ...] = ("one_to_one", "matched_marginals", "fixed_circuit", "stationary")
    history_lengths: tuple[int, ...] = (256,)
    dwell_lengths: tuple[int, ...] = (512,)
    neuron_counts: tuple[int, ...] = (4,)
    positive_lags: tuple[int, ...] = (1,)
    observation_noise: tuple[float, ...] = (0.0,)
    calcium_decays: tuple[float, ...] = (0.0,)
    n_samples: int = 4096
    stride: int = 16
    target_length: int = 8
    estimator_profile: str = "csl_sanity"

    def __post_init__(self) -> None:
        integer_axes = (
            self.seeds,
            self.history_lengths,
            self.dwell_lengths,
            self.neuron_counts,
            self.positive_lags,
        )
        integer_values = (self.n_samples, self.stride, self.target_length)
        if any(type(x) is not int for axis in integer_axes for x in axis) or any(
            type(x) is not int for x in integer_values
        ):
            raise ValueError("sample counts, horizons and seeds must be integers")
        if self.profile not in {"smoke", "development", "manuscript"}:
            raise ValueError("unknown validation profile")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("independent seeds must be nonempty unique nonnegative integers")
        if (
            not self.scenarios
            or len(set(self.scenarios)) != len(self.scenarios)
            or set(self.scenarios) - set(TRANSITION_EXAMPLE_SCENARIOS)
        ):
            raise ValueError("validation scenarios must be known and unique")
        for name in ("history_lengths", "dwell_lengths", "neuron_counts", "positive_lags"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or min(values) < 1:
                raise ValueError(f"{name} must contain unique positive values")
        if min(self.neuron_counts) < 2 or self.n_samples <= max(self.history_lengths):
            raise ValueError("insufficient neurons or recording samples")
        if not 0 < self.target_length <= min(self.history_lengths) or self.stride < self.target_length:
            raise ValueError("target/stride must permit strictly future targets")
        if not self.observation_noise or any(not np.isfinite(x) or x < 0 for x in self.observation_noise):
            raise ValueError("observation noise must be finite nonnegative")
        if not self.calcium_decays or any(not np.isfinite(x) or not 0 <= x < 1 for x in self.calcium_decays):
            raise ValueError("calcium decays must lie in [0,1)")
        if len(set(self.observation_noise)) != len(self.observation_noise) or len(
            set(self.calcium_decays)
        ) != len(self.calcium_decays):
            raise ValueError("observation stress settings must be unique")
        if self.estimator_profile not in {"smoke", "development", "csl_sanity"}:
            raise ValueError("unknown estimator profile")


def load_framework_validation_profile(profile: str) -> FrameworkValidationConfig:
    from omegaconf import OmegaConf

    if profile not in {"smoke", "development", "manuscript"}:
        raise ValueError("unknown validation profile")
    path = Path(__file__).resolve().parents[3] / "conf" / "framework_validation" / f"{profile}.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("validation profile must be a mapping")
    values: dict[str, Any] = {str(key): value for key, value in raw.items()}
    for key in (
        "seeds",
        "scenarios",
        "history_lengths",
        "dwell_lengths",
        "neuron_counts",
        "positive_lags",
        "observation_noise",
        "calcium_decays",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return FrameworkValidationConfig(**values)


def framework_validation_cells(config: FrameworkValidationConfig) -> list[dict[str, Any]]:
    keys = (
        "scenario",
        "history_length",
        "regime_dwell",
        "n_neurons",
        "positive_lag",
        "observation_noise_std",
        "calcium_decay",
    )
    specs = itertools.product(
        config.scenarios,
        config.history_lengths,
        config.dwell_lengths,
        config.neuron_counts,
        config.positive_lags,
        config.observation_noise,
        config.calcium_decays,
    )
    cells = []
    for values in specs:
        spec = dict(zip(keys, values, strict=True))
        spec["spec_id"] = content_fingerprint(spec)[:16]
        for seed in config.seeds:
            cell = {**spec, "seed": seed}
            cell["cell_id"] = content_fingerprint(cell)[:20]
            cells.append(cell)
    return cells


def _recovery(
    row: dict[str, Any],
    result: dict[str, Any],
    comparison: dict[str, Any],
    recording: Any,
    windows: Any,
    cell: dict[str, Any],
) -> None:
    evaluation = np.asarray(comparison["events"]["evaluation_rows"], dtype=int)
    origins = np.asarray(comparison["events"]["origins"], dtype=int)[evaluation]
    truth = np.asarray(recording.metadata["true_regime"])
    pure = np.asarray([len(np.unique(truth[a.context_start : a.context_stop])) == 1 for a in windows.anchors])
    selected = origins[pure[origins]]
    row["n_pure_history_evaluation_origins"] = len(selected)
    native = result["native_graph_states"]
    if native["status"] == "available" and len(selected) > 1:
        codes = np.asarray(native["labels"])
        true_codes = np.asarray([truth[windows.anchors[int(i)].anchor_sample] for i in selected])
        row["native_state_ari"] = float(adjusted_rand_score(true_codes, codes[selected]))
        row["native_n_states"] = native["n_states"]
        corr = native["correspondence"]
        for key in (
            "heldout_entropy_effectome_given_manifold_nats",
            "heldout_entropy_manifold_given_effectome_nats",
        ):
            row[key] = corr[key]
    if not result["directed"]:
        row["graph_recovery_scope"] = "undirected_association_not_structural_coefficient_recovery"
        return
    if cell["calcium_decay"] > 0 or cell["observation_noise_std"] > 0:
        row["graph_recovery_scope"] = "latent_circuit_coefficients_not_observed_process_estimand"
        return
    coefficients = np.asarray(recording.metadata["true_coefficients"])
    scale = np.asarray(comparison["reference"]["scaler_scale"])
    lagged = result.get("lagged_graph_matrices")
    if lagged is None:
        row["graph_recovery_scope"] = "lagged_dynamical_coefficients_unavailable"
        return
    estimates = np.asarray(lagged)
    reference = np.asarray([coefficients[truth[windows.anchors[int(i)].anchor_sample]] for i in selected])
    if not len(reference):
        row["graph_recovery_scope"] = "no_pure_history_origins"
        return
    reference = reference * scale[None, :, None] / scale[None, None, :]
    # The generating process has one lag. Compare complete estimated lag stacks
    # against W at lag one and zero at other lags; do not cancel lag errors by summation.
    lag_reference = np.zeros_like(estimates[selected])
    lag_reference[:, 0] = reference
    offdiag = ~np.eye(windows.n_neurons, dtype=bool)
    a, b = lag_reference[:, :, offdiag], estimates[selected][:, :, offdiag]
    present, planted = np.abs(b) > 1e-8, np.abs(a) > 1e-8
    tp = int(np.sum(present & planted))
    row["support_precision"] = tp / int(present.sum()) if present.any() else None
    row["support_recall"] = tp / int(planted.sum()) if planted.any() else None
    if result.get("weight_semantics") == "binary_upstream_causal_support":
        row["graph_recovery_scope"] = "pure_history_structural_lag_support"
    else:
        row["graph_relative_frobenius_error"] = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12))
        row["graph_recovery_scope"] = "pure_history_structural_cross_coefficients_common_scaled_units"


def run_framework_validation(
    config: FrameworkValidationConfig,
    output_dir: str | Path,
    *,
    methods: dict[str, ConnectivityConfig] | None = None,
    max_cells: int | None = None,
) -> dict[str, Any]:
    """Execute declared cells and reuse only matching input/config/code fingerprints."""
    if max_cells is not None and (not isinstance(max_cells, int) or max_cells < 1):
        raise ValueError("max_cells must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base, defaults = load_cross_method_profile(config.estimator_profile)
    methods = defaults if methods is None else methods
    if not methods:
        raise ValueError("at least one validation method is required")
    cells = framework_validation_cells(config)
    resolved_protocol = {
        "comparison_defaults": asdict(base),
        "method_configs": {k: asdict(v) for k, v in methods.items()},
        "generator_defaults": asdict(TransitionExampleConfig()),
        "implementation_fingerprint": implementation_fingerprint(),
    }
    signature = content_fingerprint((asdict(config), resolved_protocol))
    rows, attempts, attribution_rows = [], [], []
    for cell in cells[:max_cells]:
        path = destination / "cells" / f"{cell['cell_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            previous = json.loads(path.read_text())
            if previous.get("signature") == signature:
                rows.extend(previous["rows"])
                attribution_rows.extend(previous.get("attribution_rows", []))
                attempts.append({**cell, "status": previous["status"], "reused": True})
                continue
        started = time.perf_counter()
        comparison = None
        current, current_attribution = [], []
        try:
            recording = make_transition_example(
                TransitionExampleConfig(
                    scenario=cell["scenario"],
                    seed=cell["seed"],
                    n_samples=config.n_samples,
                    n_neurons=cell["n_neurons"],
                    regime_dwell=cell["regime_dwell"],
                    trial_length=max(256, 2 * cell["history_length"]),
                )
            )
            traces = recording.traces.copy()
            for start, stop in recording.metadata["valid_ranges"]:
                for sample in range(start + 1, stop):
                    traces[:, sample] = (1 - cell["calcium_decay"]) * traces[:, sample] + (
                        cell["calcium_decay"] * traces[:, sample - 1]
                    )
            rng = np.random.default_rng(np.random.SeedSequence([cell["seed"], 9781]))
            traces += rng.normal(scale=cell["observation_noise_std"], size=traces.shape)
            observed = replace(recording, traces=traces)
            windows = make_transition_example_windows(
                observed,
                history_length=cell["history_length"],
                stride=config.stride,
                target_length=config.target_length,
            )
            comparison = run_cross_method_comparison(
                windows,
                replace(base, seed=cell["seed"], positive_lag=cell["positive_lag"]),
                method_configs=methods,
                neuron_ids=[str(i) for i in recording.neuron_ids],
                groups={"rotating_pair": ["0", "1"]},
                retain_graphs=True,
            )
            for method, result in comparison["methods"].items():
                row = {**cell, "method": method, "status": result["status"], "reason": result.get("reason")}
                if result["status"] == "available":
                    prediction = result["prediction"]
                    row.update(
                        log_loss_gain=prediction["log_loss_gain"],
                        baseline_log_loss=prediction["baseline_scores"]["log_loss"],
                        augmented_log_loss=prediction["augmented_scores"]["log_loss"],
                        brier_gain=prediction["baseline_scores"]["brier"]
                        - prediction["augmented_scores"]["brier"],
                        exit_log_loss_gain=prediction["baseline_scores"]["exit_log_loss"]
                        - prediction["augmented_scores"]["exit_log_loss"],
                        displacement_mse_gain=prediction["displacement"]["mse_gain"],
                        n_evaluation_origins=len(comparison["events"]["evaluation_rows"]),
                        horizon_seconds=comparison["reference"]["metadata"]["physical_horizon_seconds"],
                    )
                    _recovery(row, result, comparison, recording, windows, cell)
                    for contribution in prediction["omission_scores"]:
                        current_attribution.append({**cell, "method": method, **contribution})
                    controls = result.get("group_control_comparisons", [])
                    matched = [
                        item["gain_minus_matched_controls"]
                        for item in controls
                        if item["gain_minus_matched_controls"] is not None
                    ]
                    row["mean_planted_group_gain_minus_controls"] = (
                        float(np.mean(matched)) if matched else None
                    )
                current.append(row)
            status = "completed"
        except (ValueError, RuntimeError, ImportError, KeyError, np.linalg.LinAlgError) as error:
            current_attribution = []
            current = [
                {**cell, "method": method, "status": "unavailable", "reason": str(error)}
                for method in methods
            ]
            status = "failed"
        attempt = {
            **cell,
            "status": status,
            "reused": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        payload = {
            "signature": signature,
            "status": status,
            "cell": attempt,
            "rows": current,
            "attribution_rows": current_attribution,
            "comparison": comparison,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        rows.extend(current)
        attribution_rows.extend(current_attribution)
        attempts.append(attempt)
    available = sum(row["status"] == "available" for row in rows)
    complete = len(attempts) == len(cells)
    report = {
        "schema_version": "framework_validation_v1",
        "profile": config.profile,
        "config": asdict(config),
        "resolved_protocol": resolved_protocol,
        "manifest_fingerprint": signature,
        "cells": attempts,
        "planned_cells": cells,
        "rows": rows,
        "attribution_rows": attribution_rows,
        "planned_methods": list(methods),
        "attempted_cells": len(attempts),
        "completed_cells": sum(cell["status"] == "completed" for cell in attempts),
        "attempted_methods": len(rows),
        "available_methods": available,
        "execution_complete": complete,
        "all_requested_methods_available": available == len(cells) * len(methods),
        "publication_suite_complete": complete
        and config.profile == "manuscript"
        and len(config.seeds) >= 20
        and config == load_framework_validation_profile("manuscript")
        and methods == defaults,
        "scientific_validation_passed": False,
        "theoretical_controls": {
            "matched_marginals": "W=rR, Q=(1-r^2)I gives Sigma=I for both circuits",
            "fixed_circuit": "different initial conditions, no circuit switching",
            "stationary": "one fixed Gaussian VAR without trial resets",
        },
        "unit_of_replication": "independent_raw_process_seed",
        "driver_recovery_scope": "predictive_feature_omission_not_intervention_ground_truth",
    }
    (destination / "validation.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    fields = sorted({key for row in rows for key in row})
    with (destination / "run_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "transition_attribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in attribution_rows for key in row})
            or ["cell_id", "method", "unit", "source_state", "destination_state"],
        )
        writer.writeheader()
        writer.writerows(attribution_rows)
    return report
