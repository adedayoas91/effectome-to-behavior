"""Raw-recording checks for optional predictive graph-state summaries.

Every draw regenerates activity, estimates graphs and reselects states. A selected
multi-state representation is a compression/prediction result, not identification
of biological regimes. These checks do not use permutations of final labels.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module import WindowConfig, make_windows
from effectome.data_module.schema import NeuralRecording, WindowedSegments
from effectome.utils.synthetic import SyntheticConfig, make_synthetic_recording
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint, lock_fingerprint

SCENARIOS = (
    "constant", "variance_change", "kinetic_change", "smooth_drift",
    "recurrent", "nonrecurring", "nonlinear",
)
_NO_DISCRETE_REGIMES = frozenset({"constant", "variance_change", "kinetic_change", "smooth_drift"})


@dataclass(frozen=True)
class StateCalibrationConfig:
    """Finite development grid; counts are not publication-scale calibration."""

    profile: str = "smoke"
    scenarios: tuple[str, ...] = SCENARIOS
    history_lengths: tuple[int, ...] = (32,)
    dwell_lengths: tuple[int, ...] = (64,)
    replicates: int = 1
    n_neurons: int = 3
    n_samples: int = 960
    stride: int = 16
    fps: float = 1.0
    seed: int = 261031
    max_draws: int | None = None
    connectivity: ConnectivityConfig = field(default_factory=lambda: ConnectivityConfig(
        name="cgc", max_lag=1, alpha=0.05,
        extra={"support_test": "fastcgc", "beta": 0.05, "support_ridge": 1e-6},
    ))
    state_evaluation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "development"}:
            raise ValueError("state calibration profile must be smoke or development")
        if not self.scenarios or not set(self.scenarios) <= set(SCENARIOS):
            raise ValueError("unknown or empty state calibration scenarios")
        if not self.history_lengths or min(self.history_lengths) <= self.connectivity.max_lag:
            raise ValueError("history lengths must exceed the connectivity lag order")
        if not self.dwell_lengths or min(self.dwell_lengths) < 1:
            raise ValueError("dwell lengths must be positive")
        if min(self.replicates, self.n_neurons, self.n_samples, self.stride) < 1 or self.fps <= 0:
            raise ValueError("grid dimensions, stride and fps must be positive")
        if max(self.history_lengths) >= self.n_samples:
            raise ValueError("recording must be longer than every history")
        if self.max_draws is not None and self.max_draws < 1:
            raise ValueError("max_draws must be positive or None")


def state_calibration_profile(profile: str = "smoke", **overrides: Any) -> StateCalibrationConfig:
    if profile == "smoke":
        return StateCalibrationConfig(**{"profile": profile, **overrides})
    if profile == "development":
        return StateCalibrationConfig(**{
            "profile": profile, "history_lengths": (32, 64, 128),
            "dwell_lengths": (32, 64, 128), "replicates": 3,
            "n_neurons": 5, "n_samples": 1920, **overrides,
        })
    raise ValueError("state calibration profile must be smoke or development")


def _draw_seed(master: int, scenario: str, history: int, dwell: int, replicate: int) -> int:
    del history  # histories share the same raw draw for paired resolution comparisons
    words = [master, SCENARIOS.index(scenario), dwell, replicate]
    return int(np.random.SeedSequence(words).generate_state(1)[0])


def generate_state_calibration_recording(
    config: StateCalibrationConfig, *, scenario: str, dwell: int, seed: int
) -> NeuralRecording:
    """Reuse certified graph families and place changes in observed raw time.

    The existing generator certifies a common companion contraction. We reuse
    those coefficients, then explicitly schedule their observed-time mixture,
    avoiding a nonrecurring schedule accidentally consumed by burn-in.
    """
    if scenario not in SCENARIOS:
        raise ValueError("unknown state calibration scenario")
    n_states = 1 if scenario in {"constant", "variance_change", "kinetic_change"} else 2
    if scenario == "nonrecurring":
        n_states = 3
    template = make_synthetic_recording(SyntheticConfig(
        n_neurons=config.n_neurons, n_timepoints=config.n_samples,
        n_states=n_states, max_lag=1, behavior_drivers=1, fps=config.fps,
        density=0.6, coupling=0.35, scenario="constant" if n_states == 1 else "recurrent_geometric",
        regime_dwell=dwell, regime_sign_flip_fraction=0.5, seed=seed,
    ))
    graphs = np.asarray(template.metadata["true_lag_graphs"], dtype=float)[:, 0]
    timepoints, neurons = config.n_samples, config.n_neurons
    if scenario in {"recurrent", "nonlinear"}:
        truth = (np.arange(timepoints) // dwell) % n_states
    elif scenario == "nonrecurring":
        truth = np.minimum(np.arange(timepoints) * n_states // timepoints, n_states - 1)
    else:
        truth = np.zeros(timepoints, dtype=int)
    weights = np.eye(n_states)[truth]
    if scenario == "smooth_drift":
        fraction = np.linspace(0.0, 1.0, timepoints)
        weights = np.column_stack([1.0 - fraction, fraction])
    rng = np.random.default_rng(seed + 1)
    previous = np.zeros(neurons)
    for _ in range(1000):
        previous = graphs[0].T @ previous + rng.normal(scale=0.4, size=neurons)
    latent = np.empty((timepoints, neurons))
    for sample in range(timepoints):
        graph = np.tensordot(weights[sample], graphs, axes=(0, 0))
        noise_scale = 0.8 if scenario == "variance_change" and sample >= timepoints // 2 else 0.4
        value = graph.T @ previous + rng.normal(scale=noise_scale, size=neurons)
        latent[sample] = np.tanh(value) if scenario == "nonlinear" else value
        previous = latent[sample]
    # A changing measurement kernel is not a change in the latent interaction law.
    observed = latent.copy()
    if scenario == "kinetic_change":
        for sample in range(1, timepoints):
            decay = 0.1 if sample < timepoints // 2 else 0.8
            observed[sample] = decay * observed[sample - 1] + (1.0 - decay) * latent[sample]
    transitions = np.flatnonzero(np.diff(truth) != 0) + 1
    runs = np.diff(np.r_[0, transitions, timepoints])
    metadata = {
        **template.metadata,
        "true_states": truth,
        "temporal_weights": weights,
        "latent_neural_activity": latent,
        "state_calibration": {
            "scenario": scenario, "seed": seed, "requested_dwell_samples": dwell,
            "realized_median_dwell_samples": float(np.median(runs)) if len(runs) > 1 else None,
            "true_transition_samples": transitions.tolist(),
            "latent_law_constant": n_states == 1,
            "discrete_regimes_present": scenario not in _NO_DISCRETE_REGIMES,
            "observation_model": "piecewise_one_pole" if scenario == "kinetic_change" else "identity",
            "observation_decay_before_after": [0.1, 0.8] if scenario == "kinetic_change" else None,
            "observed_projection_truth": (
                "unavailable; latent coefficients are not fluorescence projection truth"
            ),
            "schedule_axis": "observed_raw_samples_after_burn_in",
        },
    }
    result = replace(
        template, traces=observed.T, behavior={}, endpoint_metadata={}, outcome_validity={},
        metadata=metadata, neural_validity=np.ones((neurons, timepoints), dtype=bool),
    )
    result.validate()
    return result


def eligible_window_truth(
    windows: WindowedSegments, raw_truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Only wholly unmixed histories can score instantaneous regime recovery."""
    labels = np.empty(windows.n_windows, dtype=int)
    eligible = np.zeros(windows.n_windows, dtype=bool)
    for index, anchor in enumerate(windows.anchors):
        history = raw_truth[anchor.context_start:anchor.context_stop]
        labels[index] = int(history[-1])
        eligible[index] = bool(np.all(history == labels[index]))
    return labels, eligible


def _recovery_metrics(recording: NeuralRecording, windows: WindowedSegments, evaluation: Any) -> dict:
    contract = recording.metadata["state_calibration"]
    result: dict[str, Any] = {
        "state_recovery_ari": None, "state_recovery_unavailable_reason": None,
        "mean_detection_delay_seconds": None, "missed_transition_count": None,
        "false_transition_count": None,
    }
    if evaluation.status != "available" or evaluation.selected_model is None:
        result["state_recovery_unavailable_reason"] = "state evaluation unavailable"
        return result
    labels, eligible = eligible_window_truth(windows, np.asarray(recording.metadata["true_states"]))
    selected = np.asarray(evaluation.evaluation_indices, dtype=int)
    eligible_indices = selected[eligible[selected]]
    result["eligible_unmixed_history_count"] = int(eligible_indices.size)
    result["mixed_history_count"] = int(np.count_nonzero(~eligible[selected]))
    if not contract["discrete_regimes_present"]:
        result["state_recovery_unavailable_reason"] = "DGP has no discrete latent regimes"
        return result
    if eligible_indices.size < 2 or np.unique(labels[eligible_indices]).size < 2:
        result["state_recovery_unavailable_reason"] = (
            "insufficient held-out unmixed histories spanning regimes"
        )
    else:
        result["state_recovery_ari"] = float(adjusted_rand_score(
            labels[eligible_indices], evaluation.selected_model.labels[eligible_indices]
        ))
    origins = np.asarray([anchor.context_stop - 1 for anchor in windows.anchors])
    predicted = evaluation.selected_model.labels[selected]
    changes = origins[selected[1:][np.diff(predicted) != 0]]
    truth_changes = np.asarray(contract["true_transition_samples"], dtype=int)
    first, last = origins[selected[0]], origins[selected[-1]]
    truth_changes = truth_changes[(truth_changes >= first) & (truth_changes <= last)]
    delays: list[float] = []
    used: set[int] = set()
    for i, change in enumerate(truth_changes):
        next_change = truth_changes[i + 1] if i + 1 < len(truth_changes) else last + 1
        matches = [int(value) for value in changes if change <= value < next_change and value not in used]
        if matches:
            match = min(matches)
            used.add(match)
            delays.append((match - int(change)) / recording.fps)
    result.update({
        "mean_detection_delay_seconds": float(np.mean(delays)) if delays else None,
        "missed_transition_count": int(len(truth_changes) - len(delays)),
        "false_transition_count": int(len(changes) - len(used)),
        "detection_delay_definition": (
            "first held-out label switch after a planted change, before next change"
        ),
    })
    return result


def run_state_calibration(config: StateCalibrationConfig) -> dict[str, Any]:
    """Run the complete raw-data-to-selected-state pipeline independently per draw."""
    from effectome.dynamics.state_evaluation import StateEvaluationConfig, evaluate_state_models

    rows: list[dict[str, Any]] = []
    for scenario in config.scenarios:
        for history in config.history_lengths:
            for dwell in config.dwell_lengths:
                for replicate in range(config.replicates):
                    if config.max_draws is not None and len(rows) >= config.max_draws:
                        break
                    seed = _draw_seed(config.seed, scenario, history, dwell, replicate)
                    row: dict[str, Any] = {
                        "scenario": scenario, "history_samples": history, "requested_dwell_samples": dwell,
                        "history_to_requested_dwell_ratio": history / dwell,
                        "history_seconds": history / config.fps, "stride_samples": config.stride,
                        "replicate": replicate, "seed": seed, "status": "unavailable",
                        "false_multistate_selection": None,
                    }
                    try:
                        recording = generate_state_calibration_recording(
                            config, scenario=scenario, dwell=dwell, seed=seed,
                        )
                        windows = make_windows(recording, WindowConfig(
                            mode="temporal", history_length=history, target_length=1,
                            stride=config.stride, standardize_per_window=False,
                        ))
                        series = ConnectivityFactory(config.connectivity).run(windows)
                        evaluation = evaluate_state_models(
                            series, StateEvaluationConfig(**config.state_evaluation)
                        )
                        row.update({
                            "status": evaluation.status,
                            "selected_algorithm": evaluation.selected_algorithm,
                            "selected_n_states": evaluation.selected_n_states,
                            "multistate_selected": evaluation.state_selected,
                            "false_multistate_selection": (
                                bool(evaluation.state_selected) if scenario in _NO_DISCRETE_REGIMES else None
                            ),
                            "raw_recording_hash": content_fingerprint(recording.traces),
                            "graph_hash": content_fingerprint(series.matrices),
                            "n_graph_windows": series.n_windows,
                            "realized_history_samples": windows.anchors[0].context_stop
                            - windows.anchors[0].context_start,
                            "realized_median_dwell_samples": recording.metadata["state_calibration"][
                                "realized_median_dwell_samples"
                            ],
                            "truth_contract": recording.metadata["state_calibration"],
                            "fit_indices": np.asarray(evaluation.fit_indices).tolist(),
                            "evaluation_indices": np.asarray(evaluation.evaluation_indices).tolist(),
                            "metrics": evaluation.metrics, "diagnostics": evaluation.diagnostics,
                            **evaluation.metrics,
                            "regime_claim_eligible": bool(
                                evaluation.diagnostics.get("regime_claim_eligible", False)
                            ),
                            "outer_predictive_validation_passed": bool(
                                evaluation.diagnostics.get("predictive_validation_passed", False)
                            ),
                            "backend": series.diagnostics.get("backend"),
                            "backend_source": (
                                series.diagnostics.get("lag_resolved", [{}])[0].get("backend_source")
                            ),
                            "backend_version": (
                                series.diagnostics.get("lag_resolved", [{}])[0].get("backend_version")
                            ),
                            **_recovery_metrics(recording, windows, evaluation),
                        })
                        if evaluation.status != "available":
                            row["unavailable_reason"] = evaluation.diagnostics.get(
                                "reason", "state evaluation unavailable"
                            )
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                        row["unavailable_reason"] = f"{type(error).__name__}: {error}"
                    rows.append(row)
    null_rows = [
        row for row in rows
        if row["scenario"] in _NO_DISCRETE_REGIMES and row["status"] == "available"
    ]
    return {
        "profile": config.profile, "claim_eligibility": "development_calibration_only",
        "configuration": asdict(config), "configuration_hash": content_fingerprint(asdict(config)),
        "implementation_hash": implementation_fingerprint(), "dependency_lock_hash": lock_fingerprint(),
        "raw_pipeline_rerun_per_draw": True, "label_permutation_used": False,
        "draws": rows,
        "summary": {
            "attempted_draws": len(rows),
            "available_draws": sum(row["status"] == "available" for row in rows),
            "eligible_no_discrete_regime_draws": len(null_rows),
            "unique_no_discrete_regime_raw_recordings": len({
                row["raw_recording_hash"] for row in null_rows
            }),
            "repeated_history_designs_are_independent_replicates": False,
            "biological_regime_claims_validated": False,
            "false_multistate_selection_frequency": (
                float(np.mean([row["false_multistate_selection"] for row in null_rows]))
                if null_rows else None
            ),
            "frequency_interpretation": "finite design-conditional diagnostic, not validated population FPR",
        },
    }


def write_state_calibration_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write a complete manifest plus a compact per-recording scorecard."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "state-calibration.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    csv_path = destination / "state-calibration.csv"
    fields = [
        "scenario", "history_samples", "requested_dwell_samples", "replicate", "seed", "status",
        "selected_algorithm", "selected_n_states", "multistate_selected", "false_multistate_selection",
        "state_recovery_ari", "mean_detection_delay_seconds", "missed_transition_count",
        "false_transition_count", "heldout_mse", "constant_mse", "continuous_mse", "relative_gain",
        "mean_membership_entropy", "regime_claim_eligible", "outer_predictive_validation_passed",
        "unavailable_reason",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["draws"])
    return json_path, csv_path
