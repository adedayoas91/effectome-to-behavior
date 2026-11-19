"""Train frozen behavioral readouts and evaluate them on paired neural futures."""

from __future__ import annotations

import csv
import json
import logging
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from effectome.connectivity import ConnectivityConfig
from effectome.connectivity.native_gc import conventional_gc_runtime
from effectome.manifold import ManifoldConfig, ManifoldFactory
from effectome.manifold.behavior_dynamics import BehaviorDynamicsManifold, behavior_runtime_info
from effectome.workflows.cross_method import _json_safe, load_cross_method_profile
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

from .behavior_readout import FrozenBehaviorMap, fit_linear_behavior_readouts, fixed_behavior_readouts
from .behavior_supervised_config import BehaviorSupervisedConfig
from .behavior_supervised_metrics import (
    assess_confirmation,
    calibrate_behavior_scores,
    decision_metrics,
    nominations,
)
from .continuous_behavior_rollouts import continuous_semantics_errors, paired_behavior_effects
from .csl_sanity import CSL_METHODS, graph_recovery_metrics
from .distributed_behavior_controls import (
    DISTRIBUTED_SCENARIOS,
    distributed_interventions,
    make_distributed_behavior_control,
)
from .manifold_validation import _digest, _verify_artifacts
from .mechanism_replication import RNG_SCOPE, _inner_threads, _start_worker
from .mechanism_validation import ORACLE_LANES, _discover_support, _write_json
from .structural_mechanism import fit_additive_mechanism

logger = logging.getLogger(__name__)
LINEAR_ARMS = ("pca_2d", "pca_4d", "pls_2d", "full_linear")
NETWORK_ARMS = (
    ("behavior_only_2d", 2, False),
    ("full_nonlinear", 10, False),
    ("bunddle_behavior_2d", 2, True),
    ("bunddle_behavior_4d", 4, True),
)
ARMS = (
    "behavior_reference",
    *LINEAR_ARMS,
    *(row[0] for row in NETWORK_ARMS),
    "constant",
    "driver_proxy",
    "nuisance_proxy",
)
REPORT = "behavior-supervised-validation.json"
METRICS = (
    "end_to_end_mae",
    "within_chart_mae",
    "semantic_mae",
    "positive_unit_mae",
    "positive_unit_semantic_mae",
    "mean_triangle_bound",
    "bound_violation_count",
    "mean_factual_behavior_disagreement",
    "mean_intervened_behavior_disagreement",
)


def behavior_supervised_inventory(config: BehaviorSupervisedConfig, methods: dict) -> dict:
    n_train = int(config.generator.n_trials * (1 - config.evaluation_fraction))
    n_discovery = n_train - max(1, int(np.ceil(config.fit.validation_fraction * n_train)))
    return {
        "profile": config.profile,
        "planned_cells": config.planned_cells,
        "neural_dynamics": ["oracle", *ORACLE_LANES, *methods],
        "readout_arms": list(ARMS),
        "planned_native_window_fits": len(methods) * n_discovery * config.planned_cells,
        "planned_nonlinear_readouts": len(NETWORK_ARMS) * config.planned_cells,
        "development_seeds": list(config.development_seeds),
        "independent_unit": "raw_process_seed",
        "behavior_dimensions": 2,
        "primary_readout": config.decisions.primary_readout,
    }


def _natural_diagnostics(
    readouts: dict[str, FrozenBehaviorMap],
    neural: np.ndarray,
    behavior: np.ndarray,
    config: BehaviorSupervisedConfig,
) -> dict:
    result = {}
    start = config.rollout.horizon + 1 - config.rollout.target_length
    stop = config.rollout.horizon + 1
    early = min(neural.shape[1], config.learning.early_samples)
    for arm, readout in readouts.items():
        pred = readout.predict(neural)
        codes = readout.encode(neural).reshape(-1, readout.encode(neural[0, 0]).size)
        covariance_eigenvalues = np.linalg.eigvalsh(np.atleast_2d(np.cov(codes, rowvar=False)))
        std = pred.reshape(-1, 2).std(axis=0)
        result[arm] = {
            "natural_mae": float(np.mean(np.abs(pred - behavior))),
            "origin_mae": float(np.mean(np.abs(pred[:, 0] - behavior[:, 0]))),
            "early_mae": float(np.mean(np.abs(pred[:, :early] - behavior[:, :early]))),
            "factual_window_mae": float(
                np.mean(np.abs(pred[:, start:stop].mean(axis=1) - behavior[:, start:stop].mean(axis=1)))
            ),
            "prediction_std_by_output": std.tolist(),
            "latent_covariance_eigenvalues": covariance_eigenvalues.tolist(),
            "collapsed": bool(np.any(std <= 1e-12)),
        }
    return result


def _fit_readouts(
    config: BehaviorSupervisedConfig,
    training: np.ndarray,
    behavior: np.ndarray,
    folder: Path,
    signature: str,
    training_digest: str,
) -> dict[str, FrozenBehaviorMap]:
    learning = config.learning
    readouts = fixed_behavior_readouts()
    linear_path = folder / "linear-readouts.json"
    if linear_path.is_file():
        payload = json.loads(linear_path.read_text())
        if (payload["signature"], payload["training_digest"]) != (signature, training_digest):
            raise ValueError("saved linear readouts belong to different training inputs")
        readouts.update(
            {arm: FrozenBehaviorMap.from_dict(value) for arm, value in payload["readouts"].items()}
        )
    else:
        linear = fit_linear_behavior_readouts(
            training,
            behavior,
            ridge_alphas=learning.ridge_alphas,
            seed=learning.seed,
            validation_fraction=learning.validation_fraction,
            max_pairs_per_trial=learning.max_pairs_per_trial,
            early_samples=learning.early_samples,
        )
        _write_json(
            linear_path,
            {
                "signature": signature,
                "training_digest": training_digest,
                "readouts": {arm: value.to_dict() for arm, value in linear.items()},
            },
        )
        readouts.update(linear)
    extra = asdict(learning)
    extra["trial_lengths"] = [training.shape[1]] * len(training)
    for name, dimension, uses_dynamics in NETWORK_ARMS:
        path = folder / f"{name}.json"
        if path.is_file():
            payload = json.loads(path.read_text())
            if (payload["signature"], payload["training_digest"]) != (signature, training_digest):
                raise ValueError("saved encoder belongs to different training inputs")
            readouts[name] = FrozenBehaviorMap.from_dict(payload["readout"])
            continue
        logger.info("%s fitting %s", folder.name, name)
        manifold = ManifoldFactory(
            ManifoldConfig(
                name="bunddle_behavior",
                behavior_key="continuous",
                target_length=1,
                n_dims=dimension,
                seed=learning.seed,
                extra={**extra, "dynamics_weights": learning.dynamics_weights if uses_dynamics else (0.0,)},
            )
        )
        if not isinstance(manifold, BehaviorDynamicsManifold):
            raise TypeError("unexpected continuous behavior encoder registry entry")
        manifold.fit(training.reshape(-1, training.shape[-1]), {"continuous": behavior.reshape(-1, 2)})
        readout = FrozenBehaviorMap.from_export(manifold.export_artifact(), name=name)
        _write_json(
            path, {"signature": signature, "training_digest": training_digest, "readout": readout.to_dict()}
        )
        readouts[name] = readout
    if set(readouts) != set(ARMS):
        raise ValueError("the declared readout panel is incomplete")
    return {name: readouts[name] for name in ARMS}


def _native_support(
    training: np.ndarray,
    config: BehaviorSupervisedConfig,
    method: ConnectivityConfig,
    folder: Path,
    name: str,
    signature: str,
) -> tuple[np.ndarray, dict]:
    path = folder / f"{name}-support.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved["signature"] != signature:
            raise ValueError("saved support belongs to a different protocol")
        _verify_artifacts(folder, saved["metadata"]["artifacts"])
        return np.asarray(saved["support"], dtype=bool), saved["metadata"]
    support, metadata = _discover_support(training, config, method, folder / f"{name}-native")
    _write_json(path, {"signature": signature, "support": support, "metadata": metadata})
    return support, metadata


def _run_cell(job: tuple) -> dict:
    config, phase, scenario, seed, methods, directory, signature = job
    cell_id = f"{phase}-{scenario}-{seed}"
    folder = Path(directory) / "cells" / cell_id
    path = folder / "cell.json"
    saved = json.loads(path.read_text()) if path.is_file() else {}
    generated = make_distributed_behavior_control(replace(config.generator, scenario=scenario, seed=seed))
    trials, labels = generated.neural_trials, generated.behavior_trials
    split = int(len(trials) * (1 - config.evaluation_fraction))
    training, evaluation = trials[:split], trials[split:]
    input_identity = {
        "training_fingerprint": content_fingerprint((training, labels[:split])),
        "evaluation_fingerprint": content_fingerprint((evaluation, labels[split:])),
    }
    if saved:
        if saved["signature"] != signature or any(
            saved.get(key) != value for key, value in input_identity.items()
        ):
            raise ValueError("saved study cell belongs to a different source, configuration or input")
        _verify_artifacts(folder, saved["readout_artifacts"])
        for method in saved["methods"].values():
            _verify_artifacts(folder, method.get("artifacts", {}))
            _verify_artifacts(folder, method.get("support_metadata", {}).get("artifacts", {}))
        if set(saved["methods"]) == {"oracle", *ORACLE_LANES, *methods}:
            return saved
    readouts = _fit_readouts(
        config, training, labels[:split], folder, signature, input_identity["training_fingerprint"]
    )
    interventions = distributed_interventions()
    rollout_seed = int(np.random.SeedSequence([config.rollout.seed, seed]).generate_state(1)[0])
    args = {
        "readouts": readouts,
        "config": replace(config.rollout, seed=rollout_seed),
        "origin_ids": np.arange(split, len(trials)),
        "training_bounds": (training.min(axis=(0, 1)), training.max(axis=(0, 1))),
        "retain_samples": config.retain_rollout_samples,
    }
    truth = paired_behavior_effects(generated.mechanism, evaluation[:, 0], interventions, **args)
    true_vectors = truth["readouts"]["behavior_reference"]["effect_vectors"]
    if scenario == "null" and any(np.any(vector != 0) for vector in true_vectors.values()):
        raise ValueError("declared behavioral null has a nonzero oracle intervention effect")
    if scenario == "driver" and not np.any(true_vectors["group:driver_pair"] != 0):
        raise ValueError("driver control has no oracle group effect")
    cell = saved or {
        "signature": signature,
        "phase": phase,
        "scenario": scenario,
        "seed": seed,
        "cell_id": cell_id,
        **input_identity,
        "training_trial_ids": list(range(split)),
        "evaluation_trial_ids": list(range(split, len(trials))),
        "rollout_seed": rollout_seed,
        "generator": generated.metadata,
        "interventions": interventions,
        "natural_evaluation": _natural_diagnostics(readouts, evaluation, labels[split:], config),
        "readout_artifacts": {
            name: _digest(folder / name)
            for name in ("linear-readouts.json", *(f"{arm[0]}.json" for arm in NETWORK_ARMS))
        },
        "methods": {},
    }
    for name in ("oracle", *ORACLE_LANES, *methods):
        if name in cell["methods"]:
            continue
        logger.info("%s evaluating dynamics %s", cell_id, name)
        try:
            if name == "oracle":
                model, estimated = generated.mechanism, truth
                support_metadata = {"native_window_fits": 0, "support_source": "generating_parameters"}
            else:
                if name in ORACLE_LANES:
                    support = (
                        generated.mechanism.support
                        if name == "known_support"
                        else np.zeros_like(generated.mechanism.support)
                    )
                    support_metadata = {"native_window_fits": 0, "support_source": name}
                else:
                    support, support_metadata = _native_support(
                        training, config, methods[name], folder, name, signature
                    )
                model = fit_additive_mechanism(training, support, config.fit)
                estimated = paired_behavior_effects(model, evaluation[:, 0], interventions, **args)
            metrics = continuous_semantics_errors(truth, estimated)
            for arm, values in metrics.items():
                values.update(decision_metrics(values, config.decisions))
                values["rollout_collapsed"] = estimated["readout_diagnostics"][arm]["collapsed"]
                if phase == "development":
                    values["provisional_nominations"] = nominations(values, config.decisions.nomination_floor)
            artifacts = {}
            if config.retain_rollout_samples:
                evidence = folder / f"{name}-behavior-samples.npz"
                samples = {
                    f"{arm}|{branch}": array
                    for arm, branches in estimated["target_behavior_means"].items()
                    for branch, array in branches.items()
                }
                np.savez_compressed(evidence, **samples)
                artifacts[evidence.name] = _digest(evidence)
            cell["methods"][name] = {
                "status": "available",
                "support_metadata": support_metadata,
                "support_recovery": graph_recovery_metrics(model.support, generated.mechanism.support),
                "model": asdict(model),
                "metrics": metrics,
                "artifacts": artifacts,
                "support_diagnostics": estimated["support_diagnostics"],
                "readout_diagnostics": estimated["readout_diagnostics"],
                "rollout_contract": estimated["metadata"],
            }
        except (ValueError, RuntimeError, ImportError, OSError, np.linalg.LinAlgError) as error:
            logger.error("%s %s unavailable: %s", cell_id, name, error)
            cell["methods"][name] = {"status": "unavailable", "reason": str(error)}
        _write_json(path, cell)
    return _json_safe(cell)


def _study_identity(config: BehaviorSupervisedConfig, methods: dict) -> dict:
    root = Path(__file__).resolve().parents[3]
    settings = asdict(config)
    for key in ("profile", "development_seeds", "calibration_seeds", "positive_seeds", "null_seeds"):
        settings.pop(key)
    surfaces = (
        "tools/behavior-encoder/worker.py",
        "tools/behavior-encoder/uv.lock",
        "tools/behavior-encoder/pyproject.toml",
        "tools/conventional-cgc/uv.lock",
        "synthetic/behavior_supervised.py",
        "synthetic/_build_behavior_supervised_notebook.py",
    )
    return {
        "settings": settings,
        "methods": {name: asdict(value) for name, value in methods.items()},
        "implementation_fingerprint": implementation_fingerprint(root),
        "external_source_sha256": {name: _digest(root / name) for name in surfaces},
        "encoder_runtime": behavior_runtime_info(config.learning.runtime_python),
        "main_runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "scikit-learn", "scipy", "fastcgc")},
        },
        "native_rng_scope": RNG_SCOPE,
    }


@contextmanager
def _run_lock(folder: Path):
    # Kernel releases this advisory lock if the controlling process exits.
    import fcntl

    with (folder / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"a behavior-supervised study is already using {folder}") from error
        yield


def run_behavior_supervised(
    config: BehaviorSupervisedConfig,
    output_dir: str | Path,
    *,
    workers: int = 4,
    methods: dict[str, ConnectivityConfig] | None = None,
    development_report: Path | None = None,
) -> dict:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if methods is None:
        _, methods = load_cross_method_profile("csl_sanity")
    if set(methods) - set(CSL_METHODS):
        raise ValueError("only the four imported native support estimators are supported")
    root = Path(output_dir)
    identity = _study_identity(config, methods)
    if config.profile == "confirmation":
        if set(methods) != set(CSL_METHODS) or development_report is None:
            raise ValueError(
                "confirmation requires all four native methods and the completed development report"
            )
        prior = json.loads(development_report.read_text())
        if (
            prior.get("profile") != "development"
            or not prior.get("execution_complete")
            or not prior.get("all_requested_lanes_available")
            or prior.get("procedure_fingerprint") != content_fingerprint(identity)
        ):
            raise ValueError(
                "confirmation requires completed development with identical code, runtime and settings"
            )
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        return _run_study(config, root, workers, methods, identity)


def _run_study(config, root, workers, methods, identity) -> dict:
    protocol = _json_safe(
        {
            "schema": "behavior_supervised_v1",
            "configuration": asdict(config),
            "procedure": identity,
            "candidate_interventions": distributed_interventions(),
        }
    )
    signature = content_fingerprint(protocol)
    frozen = root / "study-protocol.json"
    if frozen.is_file() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("run namespace belongs to a different source, runtime or protocol; use a new label")
    _write_json(frozen, protocol)
    inventory = behavior_supervised_inventory(config, methods)
    lanes = inventory["neural_dynamics"]
    cells: list[dict] = []
    calibration: dict | None = None
    stage = "initializing"

    def report() -> dict:
        rows = [
            {
                "cell_id": c["cell_id"],
                "seed": c["seed"],
                "scenario": c["scenario"],
                "phase": c["phase"],
                "path": f"cells/{c['cell_id']}/cell.json",
                "methods": {name: row["status"] for name, row in c["methods"].items()},
            }
            for c in cells
        ]
        available = sum(v == "available" for c in rows for v in c["methods"].values())
        attempted = sum(len(c["methods"]) for c in rows)
        complete = len(cells) == config.planned_cells
        result = {
            **inventory,
            "stage": stage,
            "signature": signature,
            "procedure_fingerprint": content_fingerprint(identity),
            "completed_cells": len(cells),
            "attempted_method_runs": attempted,
            "available_method_runs": available,
            "execution_complete": complete,
            "all_requested_lanes_available": complete and available == config.planned_cells * len(lanes),
            "native_window_fits": sum(
                r.get("support_metadata", {}).get("native_window_fits", 0)
                for c in cells
                for r in c["methods"].values()
            ),
            "cells": sorted(rows, key=lambda row: row["cell_id"]),
            "nomination_calibrated": calibration is not None,
            "confirmatory_executed": complete and config.profile == "confirmation",
            "scientific_validation_passed": False,
            "claim_scope": "fully_observed_model_matched_distributed_continuous_behavior",
        }
        if calibration is not None and complete:
            decision = assess_confirmation(
                cells, calibration, config.decisions, lanes, config.positive_seeds, config.null_seeds
            )
            result["confirmation"] = decision
            result["scientific_validation_passed"] = result["all_requested_lanes_available"] and all(
                decision["dynamics"][name]["passed"] for name in ("oracle", "known_support", *CSL_METHODS)
            )
        _write_json(root / REPORT, result)
        return result

    def run_phase(phase: str, pairs: list[tuple[str, int]]) -> None:
        nonlocal stage
        stage = phase
        report()
        jobs = [(config, phase, scenario, seed, methods, str(root), signature) for scenario, seed in pairs]
        if workers == 1:
            with conventional_gc_runtime():
                for job in jobs:
                    cells.append(_run_cell(job))
                    report()
        else:
            with (
                _inner_threads(),
                ProcessPoolExecutor(
                    max_workers=workers, mp_context=get_context("spawn"), initializer=_start_worker
                ) as pool,
            ):
                futures = [pool.submit(_run_cell, job) for job in jobs]
                for completed in as_completed(futures):
                    cells.append(completed.result())
                    report()
                    logger.info("Behavior study progress %d/%d cases", len(cells), config.planned_cells)

    if config.profile == "confirmation":
        run_phase("calibration", [("null", seed) for seed in config.calibration_seeds])
        calibration = calibrate_behavior_scores(
            cells, lanes, list(ARMS), config.calibration_seeds, config.decisions
        )
        calibration["study_signature"] = signature
        cutoff_path = root / "calibration.json"
        if cutoff_path.is_file() and json.loads(cutoff_path.read_text()) != calibration:
            raise ValueError("frozen calibration evidence changed; preserve the original run")
        _write_json(cutoff_path, calibration)
        run_phase("positive", [("driver", seed) for seed in config.positive_seeds])
        run_phase("null-test", [("null", seed) for seed in config.null_seeds])
    else:
        run_phase(
            "development",
            [(scenario, seed) for seed in config.development_seeds for scenario in DISTRIBUTED_SCENARIOS],
        )
    render_behavior_supervised_results(cells, root / "analysis")
    stage = "complete"
    return report()


def render_behavior_supervised_results(cells: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in sorted(cells, key=lambda item: item["cell_id"]):
        for method, outcome in cell["methods"].items():
            if outcome["status"] != "available":
                continue
            for arm, metrics in outcome["metrics"].items():
                rows.append(
                    {
                        "phase": cell["phase"],
                        "scenario": cell["scenario"],
                        "seed": cell["seed"],
                        "dynamics": method,
                        "readout": arm,
                        **{key: metrics[key] for key in METRICS},
                        **{
                            key: cell["natural_evaluation"][arm][key]
                            for key in ("natural_mae", "origin_mae", "early_mae", "factual_window_mae")
                        },
                    }
                )
    if not rows:
        return
    with (directory / "seed-metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Continuous behavioral intervention outcomes",
        "",
        "MAEs use the two declared continuous behavior coordinates. Each row averages seed summaries.",
        "",
        "| Phase | Scenario | Dynamics | Readout | Seeds | Behavioral MAE | Active MAE | Semantic MAE |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["phase"], row["scenario"], row["dynamics"], row["readout"])
        grouped.setdefault(key, []).append(row)
    for key, selected in sorted(grouped.items()):
        means = []
        for metric in ("end_to_end_mae", "positive_unit_mae", "semantic_mae"):
            values = [row[metric] for row in selected if row[metric] is not None]
            means.append(f"{np.mean(values):.6f}" if values else "—")
        lines.append("| " + " | ".join((*key, str(len(selected)), *means)) + " |")
    (directory / "analysis-report.md").write_text("\n".join(lines) + "\n")
