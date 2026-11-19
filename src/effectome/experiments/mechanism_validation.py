"""Development ladder: true mechanism, known support, then imported CSL support.

Support discovery and all mechanism fitting see training trials only. Simulator
parameters enter only the truth/oracle lanes and support-recovery evaluation.
The endpoint is directly observed; no latent-manifold validation is implied.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.connectivity.native_gc import conventional_gc_runtime
from effectome.data_module.schema import Window, WindowedSegments
from effectome.experiments.csl_sanity import CSL_METHODS, graph_recovery_metrics
from effectome.experiments.mechanism_controls import (
    MECHANISM_SCENARIOS,
    make_mechanism_control,
    mechanism_control_trials,
    oracle_mechanism,
)
from effectome.experiments.mechanism_metrics import mechanism_recovery_metrics, summarize_mechanism_cells
from effectome.experiments.mechanism_validation_config import MechanismValidationConfig
from effectome.experiments.structural_mechanism import (
    fit_additive_mechanism,
    paired_intervention_probabilities,
)
from effectome.utils.seed import set_seed
from effectome.workflows.cross_method import _json_safe, load_cross_method_profile
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

if TYPE_CHECKING:
    from effectome.experiments.behavior_supervised_config import BehaviorSupervisedConfig

logger = logging.getLogger(__name__)
ORACLE_LANES = ("known_support", "empty_support")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _discover_support(
    training_trials: np.ndarray,
    config: MechanismValidationConfig | BehaviorSupervisedConfig,
    method: ConnectivityConfig,
    artifact_prefix: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = config.discovery_burn_in
    stop = start + config.discovery_history_length
    validation_trials = max(1, int(np.ceil(config.fit.validation_fraction * len(training_trials))))
    discovery_trials = training_trials[:-validation_trials]
    segments = discovery_trials[:, start:stop].copy()
    windows = WindowedSegments(
        segments=segments,
        windows=[Window(start, stop) for _ in segments],
        behavior_per_window={},
        n_neurons=config.generator.n_neurons,
        fps=config.generator.fps,
        metadata={"independent_training_trial_ids": list(range(len(segments))),
                  "discovery_scope": "one_within_trial_history_per_outer_training_trial"},
    )
    series = ConnectivityFactory(method).run(windows)
    graphs = np.asarray(series.matrices)
    if graphs.shape != (len(segments), windows.n_neurons, windows.n_neurons):
        raise ValueError("native graphs do not match training-trial/neuron axes")
    if not np.isin(graphs, (0, 1)).all():
        raise ValueError("the mechanism ladder requires native binary CSL support")
    frequency = graphs.mean(axis=0)
    support = frequency >= config.support_selection_fraction
    np.fill_diagonal(support, False)
    archive = artifact_prefix.with_suffix(".npz")
    np.savez_compressed(archive, trial_supports=graphs, selection_frequency=frequency, support=support)
    diagnostics = artifact_prefix.with_suffix(".json")
    _write_json(diagnostics, series.diagnostics)
    return support, {
        "support_source": "native_CSL_training_trial_support_frequency",
        "selection_fraction": config.support_selection_fraction,
        "n_discovery_training_trials": len(discovery_trials),
        "discovery_trial_ids": list(range(len(discovery_trials))),
        "validation_trial_ids_excluded_from_discovery": list(
            range(len(discovery_trials), len(training_trials))
        ),
        "native_window_fits": len(segments),
        "uses_ground_truth_support": False,
        "upstream_provenance": series.diagnostics.get("upstream_provenance"),
        "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (archive, diagnostics)},
        "rng_caveat": "Conventional upstream CPU permutation randomness is not fully controlled.",
    }


def _run_cell(
    config: MechanismValidationConfig,
    scenario: str,
    seed: int,
    methods: dict[str, ConnectivityConfig],
    root: Path,
    signature: str,
) -> dict[str, Any]:
    cell_id = content_fingerprint((scenario, seed))[:20]
    path = root / "cells" / f"{cell_id}.json"
    saved: dict[str, Any] = {}
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["signature"] != signature:
            raise ValueError("cached mechanism cell belongs to another protocol")
        for result in saved["methods"].values():
            for name, digest in result.get("support_metadata", {}).get("artifacts", {}).items():
                artifact = path.parent / name
                if Path(name).name != name or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
                    raise ValueError("cached native mechanism evidence failed integrity verification")
    path.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    recording = make_mechanism_control(replace(config.generator, scenario=scenario, seed=seed))
    trials = mechanism_control_trials(recording)
    split = int(len(trials) * (1 - config.evaluation_fraction))
    training, evaluation = trials[:split], trials[split:]
    endpoint = int(recording.metadata["endpoint_index"])
    threshold = float(recording.metadata["endpoint_threshold"])
    interventions = {f"neuron:{i}": [i] for i in range(recording.n_neurons)}
    interventions.update({"group:driver_pair": [0, 1], "group:control_pair": [2, 3]})
    rollout_seed = int(np.random.SeedSequence([config.rollout.seed, seed]).generate_state(1)[0])
    rollout = replace(config.rollout, seed=rollout_seed)
    arguments: dict[str, Any] = {
        "endpoint_index": endpoint, "threshold": threshold, "config": rollout,
        "origin_ids": np.arange(split, len(trials)),
    }
    truth = paired_intervention_probabilities(
        oracle_mechanism(recording), evaluation[:, 0], interventions, **arguments
    )
    future = evaluation[:, rollout.horizon - rollout.target_length + 1:rollout.horizon + 1, endpoint]
    observed = (future.mean(axis=1) >= threshold) != truth["source_states"]
    cell: dict[str, Any] = {
        "cell_id": cell_id, "signature": signature, "scenario": scenario, "seed": seed,
        "training_trial_ids": list(range(split)), "evaluation_trial_ids": list(range(split, len(trials))),
        "training_input_fingerprint": content_fingerprint(training),
        "evaluation_initial_state_fingerprint": content_fingerprint(evaluation[:, 0]),
        "rollout_seed_ownership": {"configured_seed": config.rollout.seed, "raw_process_seed": seed,
                                   "effective_seed": rollout_seed},
        "endpoint": {"index": endpoint, "threshold": threshold, "fit_scope": "predeclared_design",
                     "scope": "directly_observed_coordinate_not_learned_manifold"},
        "oracle": truth,
        "methods": saved.get("methods", {}),
    }
    for name in (*ORACLE_LANES, *methods):
        if name in cell["methods"]:
            continue
        logger.info("Mechanism development: scenario=%s seed=%s lane=%s", scenario, seed, name)
        try:
            if name in ORACLE_LANES:
                support = np.asarray(recording.metadata["true_support"]).copy()
                if name == "empty_support":
                    support[:] = False
                support_metadata = {
                    "support_source": name, "uses_ground_truth_support": name == "known_support",
                    "native_window_fits": 0,
                }
            else:
                support, support_metadata = _discover_support(
                    training, config, methods[name], path.parent / f"{cell_id}-{name}-native"
                )
            model = fit_additive_mechanism(training, support, config.fit)
            estimated = paired_intervention_probabilities(model, evaluation[:, 0], interventions, **arguments)
            one_step = model.mean(evaluation[:, :-1])
            cell["methods"][name] = {
                "status": "available", "support": support, "support_metadata": support_metadata,
                "support_recovery": graph_recovery_metrics(support, recording.metadata["true_support"]),
                "model": {"intercept": model.intercept, "self_weights": model.self_weights,
                          "cross_weights": model.cross_weights, "noise_scale": model.noise_scale,
                          "metadata": model.metadata},
                "one_step_heldout_mse": float(np.mean((one_step - evaluation[:, 1:]) ** 2)),
                "rollout": estimated,
                "metrics": mechanism_recovery_metrics(
                    truth, estimated, observed, effect_threshold=config.effect_threshold,
                    nomination_threshold=config.nomination_threshold,
                ),
            }
        except (ValueError, RuntimeError, ImportError, OSError, np.linalg.LinAlgError) as error:
            cell["methods"][name] = {"status": "unavailable", "reason": str(error)}
        _write_json(path, cell)
    return cell


def run_mechanism_validation(
    config: MechanismValidationConfig,
    output_dir: str | Path,
    *,
    methods: dict[str, ConnectivityConfig] | None = None,
    oracle_only: bool = False,
) -> dict[str, Any]:
    """Run all declared cells, preserving unavailable lanes and per-lane caches."""
    if methods is None:
        _, methods = load_cross_method_profile("csl_sanity")
    if set(methods) - set(CSL_METHODS):
        raise ValueError("only the four imported native CSL methods may supply estimated support")
    requested = {} if oracle_only else methods
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "config": asdict(config), "methods": {name: asdict(value) for name, value in requested.items()},
        "implementation_fingerprint": implementation_fingerprint(), "oracle_only": oracle_only,
    }
    signature = content_fingerprint(protocol)
    frozen = root / "development-protocol.json"
    if frozen.is_file() and json.loads(frozen.read_text()) != _json_safe(protocol):
        raise ValueError("output directory belongs to another development protocol; use a new directory")
    _write_json(frozen, protocol)
    cells = []
    with conventional_gc_runtime():
        for seed in config.seeds:
            for scenario in MECHANISM_SCENARIOS:
                cells.append(_run_cell(config, scenario, seed, requested, root, signature))
                planned = [*ORACLE_LANES, *requested]
                attempted = sum(len(cell["methods"]) for cell in cells)
                report = {
                    "schema_version": "mechanism_driver_development_v1",
                    "profile": config.profile,
                    "protocol_status": "development_not_preregistered_not_confirmation",
                    "protocol": protocol,
                    "manifest_fingerprint": signature,
                    "planned_methods": planned,
                    "planned_cells": len(config.seeds) * len(MECHANISM_SCENARIOS),
                    "attempted_method_runs": attempted,
                    "available_method_runs": sum(result["status"] == "available" for cell in cells
                                                 for result in cell["methods"].values()),
                    "execution_complete": attempted == len(planned) * len(config.seeds) * 2,
                    "cells": cells,
                    "summary": summarize_mechanism_cells(_json_safe(cells), planned, config.seeds),
                    "independent_unit": "raw_process_seed; trials/directions/methods_are_not_seed_replicates",
                    "claim_scope": "model_matched_fully_observed_development_diagnostic",
                    "nomination_calibrated": False,
                    "scientific_validation_passed": False,
                    "confirmatory_executed": False,
                    "biological_necessity_validated": False,
                    "references": "docs/driver_methodology_revision.md",
                }
                _write_json(root / "mechanism-validation.json", report)
    return report
