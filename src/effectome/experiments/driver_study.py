"""Resumable independent method/seed execution of the frozen four-CSL study."""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import Any

from effectome.connectivity import ConnectivityConfig
from effectome.connectivity.native_gc import conventional_gc_runtime
from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.driver_acceptance import (
    evaluate_driver_acceptance,
    load_driver_acceptance_criteria,
)
from effectome.experiments.driver_validation import (
    DriverValidationConfig,
    driver_validation_cells,
    run_driver_validation,
)
from effectome.workflows.cross_method import _json_safe, load_cross_method_profile
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint


def _shard(config: DriverValidationConfig, root: Path, name: str, method: ConnectivityConfig,
           ids: set[str]) -> int:
    with conventional_gc_runtime():
        report = run_driver_validation(config, root / "methods" / name, methods={name: method},
                                       cell_ids=ids, write_report=False)
    return len(report["cells"])


def collect_driver_study(config: DriverValidationConfig, root: Path,
                         methods: dict[str, ConnectivityConfig]) -> dict[str, Any]:
    """Merge only same-protocol caches; require identical shared targets per cell."""
    parts = [run_driver_validation(config, root / "methods" / name, methods={name: cfg},
                                   execute=False, write_report=False) for name, cfg in methods.items()]
    reference_hashes: dict[str, str] = {}
    for part, name in zip(parts, methods, strict=True):
        for cell in part["cells"]:
            payload = json.loads((root / "methods" / name / "cells" / f"{cell['cell_id']}.json").read_text())
            value = payload.get("shared_reference_fingerprint")
            if value is not None:
                previous = reference_hashes.setdefault(cell["cell_id"], value)
                if previous != value:
                    raise ValueError("methods did not use identical shared references, events and baselines")
    rows = [row for part in parts for row in part["seed_rows"]]
    criteria = load_driver_acceptance_criteria()
    seeds = {s: list(config.null_seeds if s == "null" else config.positive_seeds) for s in config.scenarios}
    acceptance = evaluate_driver_acceptance(rows, criteria, seeds, list(methods))
    complete = all(part["execution_complete"] for part in parts)
    report = {
        **parts[0],
        "schema_version": "four_csl_driver_study_v2",
        "planned_methods": list(methods),
        "resolved_protocol": {**parts[0]["resolved_protocol"],
                              "methods": {name: asdict(cfg) for name, cfg in methods.items()}},
        "manifest_fingerprint": content_fingerprint([part["manifest_fingerprint"] for part in parts]),
        "method_manifest_fingerprints": {name: part["manifest_fingerprint"]
                                         for name, part in zip(methods, parts, strict=True)},
        "shared_reference_fingerprints": reference_hashes,
        "cells": [{**cell, "completed_methods": sum(any(c["cell_id"] == cell["cell_id"]
                    for c in part["cells"]) for part in parts)} for cell in driver_validation_cells(config)],
        "seed_rows": rows,
        "acceptance": acceptance,
        "execution_complete": complete,
        "attempted_method_runs": sum(part["attempted_method_runs"] for part in parts),
        "available_method_runs": sum(part["available_method_runs"] for part in parts),
        "synthetic_driver_recovery_passed": complete and set(methods) == set(CSL_METHODS)
        and acceptance["overall_status"] == "pass",
        "scientific_validation_passed": False,
    }
    path = root / "driver-validation.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return report


def run_driver_study(config: DriverValidationConfig, output_dir: str | Path, *, workers: int = 4,
                     max_cells: int | None = None) -> dict[str, Any]:
    """Freeze before launching; persist each native method/recording realization."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    _, methods = load_cross_method_profile(config.estimator_profile)
    if set(methods) != set(CSL_METHODS):
        raise ValueError("parallel study requires exactly the four imported CSL methods")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frozen = _json_safe({
        "config": asdict(config), "methods": {name: asdict(cfg) for name, cfg in methods.items()},
        "criteria": asdict(load_driver_acceptance_criteria()),
        "implementation_fingerprint": implementation_fingerprint(),
        "claim_scope": "synthetic_driver_recovery; no independent_methodological_confirmation",
    })
    protocol_path = root / "frozen-protocol.json"
    if protocol_path.is_file() and json.loads(protocol_path.read_text()) != frozen:
        raise ValueError("study directory belongs to a different frozen protocol; use a new namespace")
    protocol_path.write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    cells = driver_validation_cells(config)[:max_cells]
    shards = [{cell["cell_id"] for cell in cells[index::workers]} for index in range(workers)]
    # Fast-method jobs issue complete independent-seed results early; the full
    # four-method report remains incomplete until every conventional job finishes.
    names = sorted(methods, key=lambda name: (not name.startswith("fast_"), name))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(_shard, config, root, name, methods[name], ids)
                   for name in names for ids in shards if ids}
        completed_count = -1
        while pending:
            done, pending = wait(pending, timeout=20)
            for future in done:
                future.result()
            if implementation_fingerprint() != frozen["implementation_fingerprint"]:
                raise RuntimeError("scientific code changed during the frozen study")
            report = collect_driver_study(config, root, methods)
            if report["attempted_method_runs"] != completed_count:
                completed_count = report["attempted_method_runs"]
                print(f"Completed native method/recording runs: {completed_count}/"
                      f"{4 * len(driver_validation_cells(config))}", flush=True)
    return collect_driver_study(config, root, methods)
