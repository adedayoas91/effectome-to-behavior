"""Freeze null calibration before fresh mechanism evaluation; retain first native fits."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from effectome.connectivity import ConnectivityConfig
from effectome.connectivity.native_gc import conventional_gc_runtime
from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.mechanism_calibration import calibrate_seed_maxima
from effectome.experiments.mechanism_replication_analysis import (
    render_mechanism_replication_report,
    summarize_replication,
)
from effectome.experiments.mechanism_replication_config import MechanismReplicationConfig
from effectome.experiments.mechanism_validation import ORACLE_LANES, _run_cell, _write_json
from effectome.workflows.cross_method import _json_safe, load_cross_method_profile
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

__all__ = ["run_mechanism_replication", "render_mechanism_replication_report"]
logger = logging.getLogger(__name__)
_WORKER_CONTEXT: Any = None
RNG_SCOPE = (
    "Conventional CPU permutation randomness is uncontrolled by the public upstream seed API. "
    "The fixed procedure retains first completed native evidence and never selects favorable reruns. "
    "Exchangeability includes estimator randomization; byte-identical rediscovery is not promised."
)


def _start_worker() -> None:
    global _WORKER_CONTEXT
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(message)s")
    _WORKER_CONTEXT = conventional_gc_runtime()
    _WORKER_CONTEXT.__enter__()
    atexit.register(_WORKER_CONTEXT.__exit__, None, None, None)


@contextmanager
def _inner_threads():
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
    saved = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _execute_cell(job: tuple) -> dict[str, Any]:
    config, phase, history, scenario, seed, methods, root, signature = job
    mechanism = replace(config.mechanism, seeds=(seed,), discovery_history_length=history)
    output = Path(root) / f"history-{history}" / phase
    cell = _run_cell(mechanism, scenario, seed, methods, output, signature)
    return {**cell, "phase": phase, "history_length": history}


def _jobs(config: MechanismReplicationConfig, methods: dict, root: Path, signature: str) -> tuple:
    def pool(phase: str, history: int, scenario: str, seeds: tuple[int, ...]) -> list[tuple]:
        return [(config, phase, history, scenario, seed, methods, str(root), signature) for seed in seeds]

    calibration = pool("calibration", config.primary_history, "null", config.calibration_seeds)
    evaluation = [
        *pool("positive", config.primary_history, "driver", config.positive_seeds),
        *pool("null-test", config.primary_history, "null", config.null_seeds),
        *pool("comparison", config.comparison_history, "driver", config.positive_seeds),
    ]
    return calibration, evaluation


def run_mechanism_replication(
    config: MechanismReplicationConfig,
    output_dir: str | Path,
    *,
    workers: int = 4,
    methods: dict[str, ConnectivityConfig] | None = None,
) -> dict[str, Any]:
    """Run all frozen cells, locking calibration before scheduling untouched test cells."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if methods is None:
        _, methods = load_cross_method_profile("csl_sanity")
    if set(methods) - set(CSL_METHODS):
        raise ValueError("only the four imported native CSL methods may be replicated")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    planned = [*ORACLE_LANES, *methods]
    protocol = {
        "schema_version": "mechanism_replication_v1",
        "config": asdict(config),
        "methods": {name: asdict(value) for name, value in methods.items()},
        "implementation_fingerprint": implementation_fingerprint(),
        "entrypoint_fingerprints": {
            name: hashlib.sha256((Path(__file__).resolve().parents[3] / name).read_bytes()).hexdigest()
            for name in ("synthetic/mechanism_replication.py", "synthetic/_build_replication_notebook.py")
        },
        "primary_nomination": "max(existing_effect_size_floor, calibration_seed_maximum_order_statistic)",
        "comparison_scope": "paired_positive_only_no_short_history_calibration_or_null_claim",
        "conventional_randomness": RNG_SCOPE,
    }
    frozen_path = root / "study-protocol.json"
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != _json_safe(protocol):
        raise ValueError("study protocol/source changed; use a new output directory and fresh test seeds")
    _write_json(frozen_path, protocol)
    signature = content_fingerprint(protocol)
    calibration_jobs, evaluation_jobs = _jobs(config, methods, root, signature)
    cells: list[dict[str, Any]] = []
    thresholds: dict[str, Any] = {}
    report: dict[str, Any] = {}

    def checkpoint(stage: str) -> dict[str, Any]:
        nonlocal report
        ordered = sorted(cells, key=lambda c: (c["phase"], c["history_length"], c["seed"]))
        report = {
            "schema_version": "mechanism_replication_v1",
            "profile": config.profile,
            "stage": stage,
            "protocol": protocol,
            "manifest_fingerprint": signature,
            "planned_methods": planned,
            "planned_cells": config.planned_cells,
            "completed_cells": len(cells),
            "planned_method_runs": config.planned_cells * len(planned),
            "attempted_method_runs": sum(len(c["methods"]) for c in cells),
            "available_method_runs": sum(
                r["status"] == "available" for c in cells for r in c["methods"].values()
            ),
            "execution_complete": len(cells) == config.planned_cells,
            "calibration": thresholds,
            "cells": ordered,
            "workers": workers,
            "independent_unit": "raw_process_seed; directions/trials/windows/methods are not replicates",
            "claim_scope": "fresh_seed_replication_of_fully_observed_model_matched_control_only",
            "scientific_validation_passed": False,
            "biological_necessity_validated": False,
            "graph_similarity_included": False,
            "conventional_randomness": RNG_SCOPE,
        }
        report.update(summarize_replication(ordered, planned, config, thresholds))
        _write_json(root / "replication-report.json", report)
        logger.info(
            "Replication %s: cells=%s/%s available=%s/%s",
            stage,
            len(cells),
            config.planned_cells,
            report["available_method_runs"],
            report["planned_method_runs"],
        )
        return report

    def collect(jobs: list[tuple], stage: str, executor: ProcessPoolExecutor | None) -> None:
        if executor is None:
            for job in jobs:
                cells.append(_execute_cell(job))
                checkpoint(stage)
        else:
            futures = [executor.submit(_execute_cell, job) for job in jobs]
            for future in as_completed(futures):
                cells.append(future.result())
                checkpoint(stage)

    def execute(executor: ProcessPoolExecutor | None) -> None:
        collect(calibration_jobs, "calibration", executor)
        for method in planned:
            calibrated = calibrate_seed_maxima(
                cells, method, config.calibration_seeds, alpha=config.calibration_alpha
            )
            calibrated["nomination_floor"] = config.nomination_floor
            calibrated["operational_threshold"] = (
                max(config.nomination_floor, calibrated["threshold"]) if calibrated["complete"] else None
            )
            thresholds[method] = calibrated
        calibration_path = root / "calibration.json"
        frozen = {
            "study_fingerprint": signature,
            "methods": thresholds,
            "uses_test_results": False,
            "selection_scope": "primary_history_global_null_only",
        }
        if calibration_path.exists() and json.loads(calibration_path.read_text()) != _json_safe(frozen):
            raise ValueError(
                "locked calibration evidence changed; test results cannot select its replacement"
            )
        _write_json(calibration_path, frozen)
        if not all(value["complete"] for value in thresholds.values()):
            checkpoint("blocked_calibration")
            return
        checkpoint("evaluation")
        # No positive/comparison/null-test jobs are submitted before the cutoff is locked.
        collect(evaluation_jobs, "evaluation", executor)
        checkpoint("complete")

    checkpoint("calibration")
    with _inner_threads():
        if workers == 1:
            with conventional_gc_runtime():
                execute(None)
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_start_worker,
            ) as executor:
                execute(executor)
    render_mechanism_replication_report(report, root / "analysis")
    return report
