"""Independent-seed driver recovery against actual simulator interventions."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from effectome.attribution.transition_prediction import _binary_log_loss
from effectome.connectivity import ConnectivityConfig
from effectome.experiments.driver_acceptance import (
    evaluate_driver_acceptance,
    load_driver_acceptance_criteria,
)
from effectome.experiments.driver_ground_truth import (
    DriverBenchmarkConfig,
    driver_intervention_effects,
    make_driver_benchmark,
)
from effectome.experiments.transition_examples import make_transition_example_windows
from effectome.utils.seed import set_seed
from effectome.workflows.cross_method import (
    _json_safe,
    load_cross_method_profile,
    run_cross_method_comparison,
)
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

DRIVER_SCENARIOS = ("single_driver", "redundant_group", "correlated_bystander", "null", "hidden_confounder")


@dataclass(frozen=True)
class DriverValidationConfig:
    profile: str = "smoke"
    scenarios: tuple[str, ...] = DRIVER_SCENARIOS
    positive_seeds: tuple[int, ...] = (0, 1, 2)
    null_seeds: tuple[int, ...] = (0, 1, 2)
    generator: DriverBenchmarkConfig = field(
        default_factory=lambda: DriverBenchmarkConfig(trial_length=512, cue_start=288, cue_stop=480)
    )
    history_length: int = 256
    target_length: int = 8
    stride: int = 16
    positive_lag: int = 1
    intervention_rollouts: int = 32
    estimator_profile: str = "csl_sanity"
    evaluation_fraction: float = 0.25
    manifold_dims: int = 2
    trial_origin_offset: int | None = None

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "development", "confirmatory"}:
            raise ValueError("unknown driver-validation profile")
        if (
            not self.scenarios
            or len(set(self.scenarios)) != len(self.scenarios)
            or set(self.scenarios) - set(DRIVER_SCENARIOS)
        ):
            raise ValueError("driver scenarios must be known and unique")
        for seeds in (self.positive_seeds, self.null_seeds):
            if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
                raise ValueError("seeds must be unique nonnegative integers")
        sizes = (
            self.history_length,
            self.target_length,
            self.stride,
            self.positive_lag,
            self.intervention_rollouts,
        )
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError("window, horizon and rollout counts must be positive integers")
        if not self.target_length <= self.stride <= self.history_length < self.generator.trial_length:
            raise ValueError("require target <= stride <= history < trial length")
        if self.history_length > self.generator.cue_start:
            raise ValueError("the planted cue must start after the first complete graph history")
        if self.estimator_profile not in {"smoke", "development", "csl_sanity"} or not (
            0 < self.evaluation_fraction < 1
        ):
            raise ValueError("invalid estimator or evaluation profile")
        if type(self.manifold_dims) is not int or self.manifold_dims < 1:
            raise ValueError("manifold_dims must be a positive integer")
        if self.trial_origin_offset is not None:
            origin = self.trial_origin_offset
            if type(origin) is not int or not (
                self.history_length - 1 <= origin < self.generator.trial_length - self.stride
            ):
                raise ValueError("scheduled origin and future target must lie inside a trial")
            if self.positive_lag != 1:
                raise ValueError("scheduled trial pairs require positive_lag=1")


def driver_validation_windows(recording: Any, config: DriverValidationConfig) -> Any:
    """A fixed phase per trial is selected without observing realized outcomes."""
    if config.trial_origin_offset is not None:
        origin = config.trial_origin_offset
        start, stop = origin + 1 - config.history_length, origin + config.stride + 1
        length = config.generator.trial_length
        ranges = [(trial * length + start, trial * length + stop)
                  for trial in range(config.generator.n_trials)]
        recording = replace(recording, metadata={**recording.metadata, "valid_ranges": ranges})
    return make_transition_example_windows(
        recording, history_length=config.history_length,
        target_length=config.target_length, stride=config.stride,
    )


def load_driver_validation_profile(profile: str = "smoke") -> DriverValidationConfig:
    from omegaconf import OmegaConf

    if profile not in {"smoke", "development", "confirmatory"}:
        raise ValueError("unknown driver-validation profile")
    path = Path(__file__).resolve().parents[3] / "conf" / "driver_validation" / f"{profile}.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("driver-validation profile must be a mapping")
    values: dict[str, Any] = {str(k): v for k, v in raw.items()}
    for key in ("scenarios", "positive_seeds", "null_seeds"):
        if key in values:
            values[key] = tuple(values[key])
    values["generator"] = DriverBenchmarkConfig(**values.get("generator", {}))
    return DriverValidationConfig(**values)


def driver_validation_cells(config: DriverValidationConfig) -> list[dict[str, Any]]:
    seeds = {s: config.null_seeds if s == "null" else config.positive_seeds for s in config.scenarios}
    cells = []
    # Round-robin ordering gives a bounded prefix coverage of every scenario.
    for index in range(max(map(len, seeds.values()))):
        for scenario, values in seeds.items():
            if index < len(values):
                cell = {"scenario": scenario, "seed": values[index]}
                cells.append({**cell, "cell_id": content_fingerprint(cell)[:20]})
    return cells


def _latent_wells(recording: Any, windows: Any) -> np.ndarray:
    coordinate = recording.metadata["population_coordinate"]
    return np.asarray(
        [int(np.mean(coordinate[a.target_start : a.target_stop]) >= 0) for a in windows.anchors], dtype=int
    )


def _minimum_class_agreement(predicted: np.ndarray, actual: np.ndarray) -> float:
    agreement = []
    for labels in (predicted, actual):
        for well in (0, 1):
            mask = labels == well
            agreement.append(float(np.mean(predicted[mask] == actual[mask])) if mask.any() else 0.0)
    return min(agreement)


def _alignment(recording: Any, windows: Any, comparison: dict) -> tuple[dict[int, int], float, dict]:
    truth = _latent_wells(recording, windows)
    reference = comparison["reference"]
    fit, evaluation = np.asarray(reference["fit_indices"]), np.asarray(reference["evaluation_indices"])
    states = np.asarray(reference["manifold_states"], dtype=int)
    counts = np.zeros((2, 2), dtype=int)
    np.add.at(counts, (states[fit], truth[fit]), 1)
    if np.any(counts.sum(0) == 0) or np.any(counts.sum(1) == 0):
        raise ValueError("training reference does not support both semantic wells and regions")
    row, column = linear_sum_assignment(-counts)
    mapping = dict(zip(row.tolist(), column.tolist(), strict=True))
    predictions = np.asarray([mapping[int(value)] for value in states])
    accuracy = float(np.mean(predictions[evaluation] == truth[evaluation]))
    return (
        mapping,
        accuracy,
        {
            "region_to_well": mapping,
            "training_contingency": counts,
            "training_accuracy": float(np.mean(predictions[fit] == truth[fit])),
            "evaluation_accuracy": accuracy,
            "mapping_uses_evaluation": False,
            "scope": "natural_observation_alignment_not_interventional_manifold_invariance",
        },
    )


def _cell_evidence(
    recording: Any, windows: Any, comparison: dict, cell: dict, config: DriverValidationConfig
) -> list:
    mapping, accuracy, alignment = _alignment(recording, windows, comparison)
    latent_wells = _latent_wells(recording, windows)
    inverse = {well: region for region, well in mapping.items()}
    events = comparison["events"]
    evaluation = np.asarray(events["evaluation_rows"], dtype=int)
    origins, targets = np.asarray(events["origins"])[evaluation], np.asarray(events["targets"])[evaluation]
    source_codes, target_codes = (
        np.asarray(events["source_states"])[evaluation],
        np.asarray(events["target_states"])[evaluation],
    )
    source_samples = np.asarray([windows.anchors[int(i)].anchor_sample for i in origins])
    target_samples = np.asarray([windows.anchors[int(i)].anchor_sample for i in targets])
    interventions = {f"neuron:{i}": [str(i)] for i in range(recording.n_neurons)}
    interventions.update(
        {"group:driver_pair": ["0", "1"], "group:control_pair": ["2", str(recording.n_neurons - 1)]}
    )
    oracle = driver_intervention_effects(
        recording,
        source_samples.tolist(),
        target_samples.tolist(),
        interventions,
        target_length=config.target_length,
        n_rollouts=config.intervention_rollouts,
        seed=cell["seed"] + 9571,
    )
    results = []
    for method, result in comparison["methods"].items():
        for source in (0, 1):
            destination, region = 1 - source, inverse[source]
            selected = source_codes == region
            source_purity = (
                float(np.mean(latent_wells[origins[selected]] == source)) if selected.any() else 0.0
            )
            predicted_wells = np.asarray([mapping[int(code)] for code in target_codes[selected]])
            actual_wells = latent_wells[targets[selected]]
            target_agreement = _minimum_class_agreement(predicted_wells, actual_wells)
            direction_alignment = min(source_purity, target_agreement)
            common = {
                **cell,
                "method": method,
                "source_state": source,
                "destination_state": destination,
                "status": result["status"],
                "reason": result.get("reason"),
                "alignment_accuracy": direction_alignment,
                "alignment": alignment,
                "global_alignment_accuracy": accuracy,
                "source_well_purity": source_purity,
                "minimum_target_class_agreement": target_agreement,
                "n_origins": int(selected.sum()),
                "n_events": int(np.sum(target_codes[selected] == inverse[destination])),
                "n_independent_trials": len(
                    np.unique(np.asarray(recording.metadata["true_trial_ids"])[source_samples[selected]])
                ),
            }
            if result["status"] != "available":
                results.append(common)
                continue
            scores = {
                r["unit"]: r["omission_delta_log_loss"]
                for r in result["prediction"]["omission_scores"]
                if r["source_state"] == region
                and r["destination_state"] == inverse[destination]
                and r["unit"] in interventions
            }
            sample_set = set(source_samples[selected].tolist())
            effects = {}
            for unit in interventions:
                # Keep exactly the same learned source risk set. A misaligned
                # origin's opposite-well contrast must be complemented, not dropped.
                values = [
                    r["probability_reduction"] * (1 if r["destination_state"] == destination else -1)
                    for r in oracle
                    if r["unit"] == unit and r["origin_sample"] in sample_set
                ]
                if values:
                    effects[unit] = float(np.mean(values))
            base = np.asarray(result["prediction"]["baseline_probabilities"])[selected, inverse[destination]]
            augmented = np.asarray(result["prediction"]["augmented_probabilities"])[
                selected, inverse[destination]
            ]
            observed = (target_codes[selected] == inverse[destination]).astype(float)

            common.update(
                driver_scores=scores,
                intervention_effects=effects,
                log_loss_gain=_binary_log_loss(base, observed) - _binary_log_loss(augmented, observed)
                if len(base)
                else None,
                oracle_scope="paired_SCM_outgoing_disconnection_same_origin_and_target_support",
                geometric_outcome_vs_latent_truth="training_aligned_approximation_audited_on_holdout",
            )
            results.append(common)
    return results


def run_driver_validation(
    config: DriverValidationConfig,
    output_dir: str | Path,
    *,
    methods: dict[str, ConnectivityConfig] | None = None,
    max_cells: int | None = None,
    cell_ids: set[str] | None = None,
    execute: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    """Run a declared prefix and retain all failures without relaxing the protocol."""
    if max_cells is not None and (type(max_cells) is not int or max_cells < 1):
        raise ValueError("max_cells must be a positive integer")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base, defaults = load_cross_method_profile(config.estimator_profile)
    requested = defaults if methods is None else methods
    if not requested:
        raise ValueError("at least one method is required")
    base = replace(
        base,
        n_dims=config.manifold_dims,
        n_manifold_states=2,
        evaluation_fraction=config.evaluation_fraction,
        positive_lag=config.positive_lag,
    )
    criteria = load_driver_acceptance_criteria()
    protocol = {
        "comparison": asdict(base),
        "methods": {k: asdict(v) for k, v in requested.items()},
        "acceptance": asdict(criteria),
        "implementation_fingerprint": implementation_fingerprint(),
    }
    signature = content_fingerprint((asdict(config), protocol))
    cells, attempts, rows = driver_validation_cells(config), [], []
    for cell in cells[:max_cells]:
        if cell_ids is not None and cell["cell_id"] not in cell_ids:
            continue
        path = destination / "cells" / f"{cell['cell_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            previous = json.loads(path.read_text())
            if previous["signature"] == signature:
                rows.extend(previous["seed_rows"])
                attempts.append({**previous["attempt"], "reused": True})
                continue
        if not execute:
            continue
        started = time.perf_counter()
        set_seed(cell["seed"])
        try:
            recording = make_driver_benchmark(
                replace(config.generator, scenario=cell["scenario"], seed=cell["seed"])
            )
            windows = driver_validation_windows(recording, config)
            comparison = run_cross_method_comparison(
                windows,
                replace(base, seed=cell["seed"]),
                method_configs=requested,
                neuron_ids=[str(i) for i in recording.neuron_ids],
                groups={"driver_pair": ["0", "1"], "control_pair": ["2", str(recording.n_neurons - 1)]},
                graph_origin_only=config.trial_origin_offset is not None,
                retain_graphs=True,
            )
            current = _cell_evidence(recording, windows, comparison, cell, config)
            status = "completed"
        except (ValueError, RuntimeError, ImportError, KeyError, np.linalg.LinAlgError) as error:
            current = [
                {
                    **cell,
                    "method": method,
                    "source_state": source,
                    "destination_state": 1 - source,
                    "status": "unavailable",
                    "reason": str(error),
                    "n_origins": 0,
                    "n_events": 0,
                }
                for method in requested
                for source in (0, 1)
            ]
            status = "failed"
        attempt = {
            **cell,
            "status": status,
            "elapsed_seconds": time.perf_counter() - started,
            "reused": False,
        }
        reference_hash = content_fingerprint(
            (comparison["reference"], comparison["events"])
        ) if status == "completed" else None
        payload = _json_safe({"signature": signature, "attempt": attempt, "seed_rows": current,
                              "shared_reference_fingerprint": reference_hash})
        if status == "completed":
            artifacts = {}
            for method, result in comparison["methods"].items():
                if result["status"] != "available":
                    continue
                arrays: dict[str, Any] = {
                    "matrices": np.asarray(result["graph_matrices"]),
                    "anchor_indices": np.asarray(result["graph_anchor_indices"]),
                }
                evidence = [window["native_evidence"]
                            for window in result["diagnostics"].get("lag_resolved", [])]
                if evidence:
                    for key in set.intersection(*(set(item) for item in evidence)):
                        arrays["evidence_" + key] = np.stack([item[key] for item in evidence])
                artifact_path = path.with_name(f"{cell['cell_id']}-{method}.npz")
                np.savez_compressed(artifact_path, **arrays)
                artifacts[method] = {
                    "path": artifact_path.name,
                    "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                    "upstream_provenance": result["diagnostics"].get("upstream_provenance"),
                    "graph_scope": result["graph_scope"],
                    "weight_semantics": result["weight_semantics"],
                }
            payload["graph_artifacts"] = artifacts
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        attempts.append(attempt)
        rows.extend(current)
    planned_seeds = {
        s: list(config.null_seeds if s == "null" else config.positive_seeds) for s in config.scenarios
    }
    acceptance = evaluate_driver_acceptance(rows, criteria, planned_seeds, list(requested))
    report = _json_safe(
        {
            "schema_version": "driver_validation_v1",
            "profile": config.profile,
            "config": asdict(config),
            "resolved_protocol": protocol,
            "manifest_fingerprint": signature,
            "planned_cells": cells,
            "planned_methods": list(requested),
            "cells": attempts,
            "seed_rows": rows,
            "acceptance": acceptance,
            "execution_complete": len(attempts) == len(cells),
            "attempted_method_runs": len(attempts) * len(requested),
            "available_method_runs": sum(r["status"] == "available" and r["source_state"] == 0 for r in rows),
            "scientific_validation_passed": False,
            "biological_necessity_validated": False,
            "synthetic_driver_recovery_passed": len(attempts) == len(cells)
            and acceptance["overall_status"] == "pass",
            "protocol_status": "frozen_development_protocol_not_externally_preregistered"
            if config.profile == "confirmatory" else "development_only",
            "independent_unit": "raw_process_seed; directions_and_origins_are_not_replicates",
        }
    )
    if write_report:
        path = destination / "driver-validation.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    return report
