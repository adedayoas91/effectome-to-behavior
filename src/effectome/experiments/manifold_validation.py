"""Cross frozen learned measurements with true and fitted neural dynamics.

Behavior is specified by the generator, independently of endpoint learning. This
development bridge never treats agreement within a chart as behavioral validity.
See docs/learned_manifold_validation_plan.md and its verified bibliography.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from effectome.connectivity import ConnectivityConfig
from effectome.connectivity.native_gc import conventional_gc_runtime
from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.learned_endpoints import endpoint_from_dict, fit_endpoint_arms
from effectome.experiments.manifold_endpoint_rollouts import paired_endpoint_probabilities, semantics_errors
from effectome.experiments.manifold_validation_config import ManifoldValidationConfig
from effectome.experiments.mechanism_controls import (
    MECHANISM_SCENARIOS,
    make_mechanism_control,
    mechanism_control_trials,
    oracle_mechanism,
)
from effectome.experiments.mechanism_replication import RNG_SCOPE, _inner_threads, _start_worker
from effectome.experiments.mechanism_validation import ORACLE_LANES, _discover_support, _write_json
from effectome.experiments.structural_mechanism import fit_additive_mechanism
from effectome.workflows.cross_method import _json_safe, load_cross_method_profile
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

logger = logging.getLogger(__name__)
ENDPOINT_ARMS = ("behavior_reference", "pca_geometry", "pca_behavior", "full_neural_behavior",
                 "constant", "proxy", "nuisance")
METRICS = ("end_to_end_mae", "within_chart_mae", "semantic_mae", "mean_triangle_bound",
           "bound_violation_count", "mean_factual_event_disagreement", "mean_intervened_event_disagreement")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_artifacts(folder: Path, records: dict[str, str]) -> None:
    for name, expected in records.items():
        if Path(name).name != name or not (folder / name).is_file() or _digest(folder / name) != expected:
            raise ValueError(f"cached learned-endpoint evidence failed integrity verification: {name}")


def _natural_diagnostics(endpoints: dict, evaluation: np.ndarray, target_length: int) -> dict:
    """Evaluate every chart on identical held-out observations, without alignment."""
    reference = endpoints["behavior_reference"]
    n_trials, length, n_nodes = evaluation.shape
    stop = length - length % target_length
    targets = evaluation[:, :stop].reshape(n_trials, -1, target_length, n_nodes).transpose(2, 0, 1, 3)
    source_gold = reference.source_states(evaluation[:, 0])
    target_gold = reference.target_states(targets)
    result = {}
    for name, endpoint in endpoints.items():
        source = endpoint.source_states(evaluation[:, 0])
        destination = endpoint.target_states(targets)
        counts = np.bincount(destination.ravel(), minlength=2)
        result[name] = {
            "source_accuracy": float(np.mean(source == source_gold)),
            "target_accuracy": float(np.mean(destination == target_gold)),
            "source_counts": np.bincount(source, minlength=2).tolist(),
            "target_counts": counts.tolist(), "collapsed": bool(np.any(counts == 0)),
            "ownership": "evaluation_only_no_refit_or_label_realignment",
        }
    return result


def _save_events(folder: Path, name: str, rollout: dict) -> dict[str, str]:
    path = folder / f"{name}-events.npz"
    events = {f"{arm}|{branch}": values for arm, branches in rollout["events"].items()
              for branch, values in branches.items()}
    np.savez_compressed(path, **events)
    return {path.name: _digest(path)}


def _run_cell(job: tuple) -> dict[str, Any]:
    config, scenario, seed, methods, root, signature = job
    mechanism = config.mechanism
    cell_id = f"{scenario}-{seed}"
    folder = Path(root) / "cells" / cell_id
    path = folder / "cell.json"
    folder.mkdir(parents=True, exist_ok=True)
    saved = json.loads(path.read_text()) if path.is_file() else {}
    if saved:
        if saved["signature"] != signature:
            raise ValueError("cached learned-endpoint cell belongs to another protocol")
        _verify_artifacts(folder, saved["artifacts"])
        for result in saved["methods"].values():
            _verify_artifacts(folder, result.get("artifacts", {}))
            _verify_artifacts(folder, result.get("support_metadata", {}).get("artifacts", {}))
    recording = make_mechanism_control(replace(mechanism.generator, scenario=scenario, seed=seed))
    trials = mechanism_control_trials(recording)
    split = int(len(trials) * (1 - mechanism.evaluation_fraction))
    training, evaluation = trials[:split], trials[split:]
    input_fingerprints = {
        "training_fingerprint": content_fingerprint(training),
        "evaluation_initial_fingerprint": content_fingerprint(evaluation[:, 0]),
        "evaluation_fingerprint": content_fingerprint(evaluation),
    }
    if saved and any(saved.get(key) != value for key, value in input_fingerprints.items()):
        raise ValueError("cached learned-endpoint inputs changed; preserve evidence and use a new directory")
    endpoint_index = int(recording.metadata["endpoint_index"])
    endpoint_path = folder / "endpoints.json"
    if saved:
        endpoints = {key: endpoint_from_dict(value)
                     for key, value in json.loads(endpoint_path.read_text()).items()}
    else:
        endpoints = fit_endpoint_arms(
            training, endpoint_index=endpoint_index, target_length=mechanism.rollout.target_length,
            n_components=config.n_components, validation_fraction=mechanism.fit.validation_fraction,
            decoder_cs=config.decoder_cs, seed=config.endpoint_seed,
        )
        _write_json(endpoint_path, {name: endpoint.to_dict() for name, endpoint in endpoints.items()})
    if set(endpoints) != set(ENDPOINT_ARMS):
        raise ValueError("the complete frozen endpoint panel is required")
    interventions = {f"neuron:{i}": [i] for i in range(recording.n_neurons)}
    interventions.update({"group:driver_pair": [0, 1], "group:control_pair": [2, 3]})
    rollout_seed = int(np.random.SeedSequence([mechanism.rollout.seed, seed]).generate_state(1)[0])
    arguments = {
        "endpoints": endpoints, "config": replace(mechanism.rollout, seed=rollout_seed),
        "origin_ids": np.arange(split, len(trials)),
        "training_bounds": (training.min(axis=(0, 1)), training.max(axis=(0, 1))),
    }
    truth = paired_endpoint_probabilities(oracle_mechanism(recording), evaluation[:, 0],
                                          interventions, **arguments)
    cell = saved or {
        "cell_id": cell_id, "signature": signature, "scenario": scenario, "seed": seed,
        "training_trial_ids": list(range(split)), "evaluation_trial_ids": list(range(split, len(trials))),
        **input_fingerprints,
        "rollout_seed": rollout_seed, "interventions": interventions,
        "artifacts": {endpoint_path.name: _digest(endpoint_path)},
        "natural_evaluation": _natural_diagnostics(endpoints, evaluation, mechanism.rollout.target_length),
        "methods": {},
    }
    for name in ("oracle", *ORACLE_LANES, *methods):
        if name in cell["methods"]:
            continue
        logger.info("Learned endpoint: scenario=%s seed=%s neural dynamics=%s", scenario, seed, name)
        try:
            if name == "oracle":
                model = oracle_mechanism(recording)
                estimated = truth
                support_metadata = {"support_source": "generating_parameters_evaluation_only",
                                    "native_window_fits": 0}
            else:
                if name in ORACLE_LANES:
                    support = np.asarray(recording.metadata["true_support"]).copy()
                    if name == "empty_support":
                        support[:] = False
                    support_metadata = {"support_source": name, "native_window_fits": 0,
                                        "uses_ground_truth_support": name == "known_support"}
                else:
                    support, support_metadata = _discover_support(
                        training, mechanism, methods[name], folder / f"{name}-native"
                    )
                model = fit_additive_mechanism(training, support, mechanism.fit)
                estimated = paired_endpoint_probabilities(model, evaluation[:, 0], interventions, **arguments)
            metrics = semantics_errors(truth, estimated)
            cell["methods"][name] = {
                "status": "available", "support_metadata": support_metadata,
                "model": {"intercept": model.intercept, "self_weights": model.self_weights,
                          "cross_weights": model.cross_weights, "noise_scale": model.noise_scale,
                          "support": model.support, "metadata": model.metadata},
                "endpoint_rollouts": estimated["endpoints"], "metrics": metrics,
                "support_diagnostics": estimated.get("support_diagnostics"),
                "artifacts": _save_events(folder, name, estimated),
            }
        except (ValueError, RuntimeError, ImportError, OSError, np.linalg.LinAlgError) as error:
            cell["methods"][name] = {"status": "unavailable", "reason": str(error)}
        _write_json(path, cell)
    return cell


def manifold_validation_inventory(config: ManifoldValidationConfig, methods: dict) -> dict:
    n_cells = 2 * len(config.seeds)
    mechanism = config.mechanism
    n_train = int(mechanism.generator.n_trials * (1 - mechanism.evaluation_fraction))
    n_discovery = n_train - max(1, int(np.ceil(mechanism.fit.validation_fraction * n_train)))
    return {
        "profile": config.profile, "development_seeds": list(config.seeds),
        "planned_cells": n_cells, "neural_dynamics": ["oracle", *ORACLE_LANES, *methods],
        "endpoint_arms": list(ENDPOINT_ARMS), "native_window_fits": n_cells * len(methods) * n_discovery,
        "fit_scope": "per_recording_outer_training_only", "fitting_enabled": False,
        "confirmatory_executed": False, "nomination_calibrated": False,
    }


def run_manifold_validation(
    config: ManifoldValidationConfig,
    output_dir: str | Path,
    *,
    workers: int = 4,
    methods: dict[str, ConnectivityConfig] | None = None,
) -> dict[str, Any]:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if methods is None:
        _, methods = load_cross_method_profile("csl_sanity")
    if set(methods) - set(CSL_METHODS):
        raise ValueError("only the four imported native methods may supply learned support")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    inventory = manifold_validation_inventory(config, methods)
    entry = Path(__file__).resolve().parents[3] / "synthetic"
    protocol = {
        "schema_version": "learned_endpoint_development_v1", "config": asdict(config),
        "methods": {key: asdict(value) for key, value in methods.items()},
        "implementation_fingerprint": implementation_fingerprint(),
        "entrypoints": {name: _digest(entry / name) for name in
                        ("manifold_validation.py", "_build_manifold_notebook.py")},
        "independent_behavior": "predeclared_last_neural_coordinate_sign_and_future_window_mean",
        "conditioning": "identical_origins_no_endpoint_specific_selection",
        "conventional_randomness": RNG_SCOPE,
        "references": "docs/learned_manifold_validation_plan.md",
    }
    frozen = root / "study-protocol.json"
    if frozen.is_file() and json.loads(frozen.read_text()) != _json_safe(protocol):
        raise ValueError("protocol/source changed; use a new development output directory")
    _write_json(frozen, protocol)
    signature = content_fingerprint(protocol)
    jobs = [(config, scenario, seed, methods, str(root), signature)
            for seed in config.seeds for scenario in MECHANISM_SCENARIOS]
    cells: list[dict] = []

    def report() -> dict:
        ordered = sorted(cells, key=lambda cell: (cell["seed"], cell["scenario"]))
        lanes = inventory["neural_dynamics"]
        return {
            **inventory, "fitting_enabled": True, "schema_version": "learned_endpoint_development_v1",
            "protocol": protocol, "signature": signature, "cells": ordered,
            "completed_cells": len(cells), "execution_complete": len(cells) == len(jobs),
            "attempted_method_runs": sum(len(cell["methods"]) for cell in cells),
            "available_method_runs": sum(result["status"] == "available" for cell in cells
                                         for result in cell["methods"].values()),
            "all_requested_lanes_available": len(cells) == len(jobs) and all(
                len(cell["methods"]) == len(lanes) and all(
                    result["status"] == "available" for result in cell["methods"].values()
                ) for cell in cells
            ),
            "scientific_validation_passed": False, "nonlinear_manifold_validated": False,
            "claim_scope": "linear_learned_endpoint_development_bridge_fully_observed_neural_SCM",
            "independent_unit": "raw_process_seed; endpoint_arms_and_trials_are_not_replications",
        }

    _write_json(root / "manifold-validation.json", report())
    if workers == 1:
        with conventional_gc_runtime():
            for job in jobs:
                cells.append(_run_cell(job))
                _write_json(root / "manifold-validation.json", report())
    else:
        with _inner_threads(), ProcessPoolExecutor(max_workers=workers, initializer=_start_worker) as pool:
            futures = [pool.submit(_run_cell, job) for job in jobs]
            for future in as_completed(futures):
                cells.append(future.result())
                _write_json(root / "manifold-validation.json", report())
                logger.info("Learned-endpoint progress: %s/%s cells", len(cells), len(jobs))
    result = report()
    render_manifold_validation_report(result, root / "analysis")
    return result


def render_manifold_validation_report(report: dict, output_dir: str | Path) -> Path:
    """Export seed-level diagnostic rows; no confirmation gates or pseudoreplication."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in report["cells"]:
        for name, result in cell["methods"].items():
            if result["status"] != "available":
                continue
            for arm, metrics in result["metrics"].items():
                rows.append({"seed": cell["seed"], "scenario": cell["scenario"], "method": name,
                             "endpoint": arm, **{key: metrics[key] for key in METRICS},
                             **cell["natural_evaluation"][arm]})
    if rows:
        with (root / "seed-metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Learned behavioral endpoint development", "",
             f"Completed cells: {report['completed_cells']}/{report['planned_cells']}. "
             f"Available neural-dynamics lanes: {report['available_method_runs']}/"
             f"{report['attempted_method_runs']}.", "",
             "This is a PCA bridge using development seeds, not a nonlinear manifold or confirmatory result.",
             "Behavioral error compares fitted chart effects with independently defined simulator behavior. "
             "Within-chart error compares fitted and simulator effects through the same frozen chart.", "",
             "| Scenario | Dynamics | Endpoint | Seeds | Behavioral MAE | Within-chart MAE | Semantic MAE |",
             "|---|---|---|---:|---:|---:|---:|"]
    groups = sorted({(row["scenario"], row["method"], row["endpoint"]) for row in rows})
    for scenario, method, arm in groups:
        selected = [row for row in rows if (row["scenario"], row["method"], row["endpoint"])
                    == (scenario, method, arm)]
        means = [np.mean([row[key] for row in selected])
                 for key in ("end_to_end_mae", "within_chart_mae", "semantic_mae")]
        lines.append(f"| {scenario} | {method} | {arm} | {len(selected)} | "
                     + " | ".join(f"{value:.6f}" for value in means) + " |")
    lines.extend(["", "No calibration/test pools were used. Negative charts remain in every comparison. "
                  "A collapsed chart cannot pass behavioral validity because its within-chart MAE is zero.",
                  "", "Method rationale and verified reference scopes: "
                  "`docs/learned_manifold_validation_plan.md` and its citation audit.", ""])
    path = root / "analysis-report.md"
    path.write_text("\n".join(lines))
    return path
