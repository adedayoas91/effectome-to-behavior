"""Run prospective behavioral forecasts and intervention checks in recurrent circuits."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from effectome.connectivity import ConnectivityConfig
from effectome.connectivity.native_gc import conventional_gc_runtime
from effectome.experiments.behavior_supervised import _run_lock
from effectome.experiments.behavior_supervised_config import BehaviorLearningConfig
from effectome.experiments.behavior_transition import TransitionExperimentConfig, run_transition_case
from effectome.experiments.behavior_transition_controls import TRANSITION_SCENARIOS, TransitionControlConfig
from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.manifold_validation import _digest
from effectome.experiments.mechanism_replication import RNG_SCOPE, _inner_threads, _start_worker
from effectome.experiments.mechanism_validation import _write_json
from effectome.manifold.behavior_dynamics import behavior_runtime_info
from effectome.workflows.cross_method import load_cross_method_profile
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

REPORT = "transition-report.json"
SEEDS = {
    "smoke": (81000,),
    "development": tuple(range(81001, 81006)),
    "confirmation": tuple(range(82000, 82020)),
}
logger = logging.getLogger(__name__)


def _config(profile: str, runtime: str | None) -> TransitionExperimentConfig:
    if profile == "smoke":
        return TransitionExperimentConfig(
            generator=TransitionControlConfig(n_trials=16, history_lengths=(256,)),
            learning=BehaviorLearningConfig(max_epochs=24, min_epochs=8, runtime_python=runtime),
            split_sizes=(6, 4, 2, 4),
            n_rollouts=16,
        )
    return TransitionExperimentConfig(learning=BehaviorLearningConfig(runtime_python=runtime))


def _identity(config: TransitionExperimentConfig, methods: dict[str, ConnectivityConfig]) -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "tools/behavior-encoder/worker.py",
        root / "tools/behavior-encoder/uv.lock",
        root / "tools/conventional-cgc/uv.lock",
    )
    return {
        "settings": asdict(config),
        "methods": {method: asdict(value) for method, value in methods.items()},
        "implementation_fingerprint": implementation_fingerprint(root),
        "source_sha256": {path.relative_to(root).as_posix(): _digest(path) for path in paths},
        "encoder_runtime": behavior_runtime_info(config.learning.runtime_python),
        "main_runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{package: version(package) for package in ("numpy", "scikit-learn", "scipy")},
            **(
                {"fastcgc": version("fastcgc")}
                if any(method.startswith("fast_") for method in methods)
                else {}
            ),
        },
        "native_rng_scope": RNG_SCOPE,
    }


def _seed_rows(cases: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        for length, block in case["history_lengths"].items():
            for method, value in {"oracle": {}, **block["methods"]}.items():
                rows.append(
                    {
                        "seed": case["seed"],
                        "scenario": case["scenario"],
                        "history_samples": int(length),
                        "method": method,
                        "status": value.get("status", "available"),
                        "history_mae": block["history_only_mae"],
                        "oracle_mae": block["oracle_dynamic_mae"],
                        "oracle_rollout_mae": block["oracle_rollout_mae"],
                        "static_mae": value.get("static_mae"),
                        "dynamic_mae": value.get("dynamic_mae"),
                        "static_rollout_mae": value.get("static_rollout_mae"),
                        "dynamic_rollout_mae": value.get("dynamic_rollout_mae"),
                        "gain_over_history": value.get(
                            "gain_over_history", block["oracle_gain"] if not value else None
                        ),
                        "gain_over_static": value.get("gain_over_static"),
                        "test_support_jaccard": value.get("test_support_jaccard"),
                        "encoder_origin_mae": case["encoder_origin_mae"],
                        "reason": value.get("reason"),
                    }
                )
    return rows


def _decision(rows: list[dict], config: TransitionExperimentConfig, profile: str) -> dict | None:
    if profile != "confirmation":
        return None
    primary_length = max(config.generator.history_lengths)
    threshold = config.practical_forecast_gain
    tests: dict[str, Any] = {}
    for scenario in TRANSITION_SCENARIOS:
        selected = [
            row
            for row in rows
            if row["scenario"] == scenario
            and row["history_samples"] == primary_length
            and row["method"] == config.primary_method
            and row["status"] == "available"
        ]
        complete = sorted(row["seed"] for row in selected) == list(SEEDS[profile])
        for comparator in ("history", "static"):
            gains = np.asarray([row[f"gain_over_{comparator}"] for row in selected])
            hits = int(
                np.count_nonzero(gains > threshold if scenario == "driver_switch" else gains < threshold)
            )
            p = float(binomtest(hits, len(gains), 0.5, alternative="greater").pvalue) if complete else None
            tests[f"{scenario}_vs_{comparator}"] = {
                "n_seeds": len(gains),
                "n_gains_on_expected_side": hits,
                "mean_gain": float(gains.mean()) if gains.size else None,
                "p_value": p,
                "passed": p is not None and p <= 0.05,
            }
    return {
        "primary_method": config.primary_method,
        "primary_history_samples": primary_length,
        "practical_gain": threshold,
        "tests": tests,
        "prediction_passed": all(row["passed"] for row in tests.values()),
        "scope": "intersection_of_prespecified_forecast_gates; perturbation_metrics_descriptive",
    }


def run_study(
    config: TransitionExperimentConfig,
    profile: str,
    methods: dict[str, ConnectivityConfig],
    root: Path,
    workers: int,
    development_report: Path | None = None,
) -> dict[str, Any]:
    identity = content_fingerprint(_identity(config, methods))
    if profile == "confirmation":
        if development_report is None or config.primary_method not in methods:
            raise ValueError("confirmation requires the frozen primary method and a development report")
        prior = json.loads(development_report.read_text())
        if (prior["profile"], prior["procedure_fingerprint"], prior["all_lanes_available"]) != (
            "development",
            identity,
            True,
        ):
            raise ValueError("confirmation requires complete development evidence with the same procedure")
    seeds = SEEDS[profile]
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        return _run_locked(config, profile, methods, root, workers, identity, seeds)


def _run_locked(config, profile, methods, root, workers, identity, seeds) -> dict:
    signature = content_fingerprint({"procedure": identity, "profile": profile, "seeds": seeds})
    protocol = root / "transition-protocol.json"
    if protocol.is_file() and json.loads(protocol.read_text())["signature"] != signature:
        raise ValueError("this study label belongs to another protocol or runtime; choose another label")
    _write_json(
        protocol,
        {
            "signature": signature,
            "procedure_fingerprint": identity,
            "configuration": asdict(config),
            "profile": profile,
            "seeds": seeds,
            "scenarios": TRANSITION_SCENARIOS,
            "methods": list(methods),
        },
    )
    jobs = [
        (config, scenario, seed, methods, root / "cases" / f"{scenario}-{seed}", signature)
        for seed in seeds
        for scenario in TRANSITION_SCENARIOS
    ]
    cases: list[dict] = []

    def save(case: dict | None = None) -> dict:
        if case is not None:
            cases.append(case)
            logger.info("Transition study %d/%d cases saved", len(cases), len(jobs))
        complete = len(cases) == len(jobs)
        rows = _seed_rows(sorted(cases, key=lambda row: (row["seed"], row["scenario"])))
        result = {
            "profile": profile,
            "procedure_fingerprint": identity,
            "signature": signature,
            "completed_cases": len(cases),
            "planned_cases": len(jobs),
            "execution_complete": complete,
            "all_lanes_available": complete and all(row["status"] == "available" for row in rows),
            "native_window_fits": sum(
                method.get("window_fits", 0)
                for c in cases
                for block in c["history_lengths"].values()
                for method in block["methods"].values()
            ),
            "cases": [f"cases/{c['scenario']}-{c['seed']}/case.json" for c in cases],
            "independent_unit": "simulation_process_seed",
            "confirmation": _decision(rows, config, profile) if complete else None,
        }
        if rows:
            with (root / "seed-metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        _write_json(root / REPORT, result)
        return result

    result = save()
    if workers == 1:
        with _inner_threads(), conventional_gc_runtime():
            for job in jobs:
                result = save(run_transition_case(*job))
    else:
        with (
            _inner_threads(),
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=get_context("spawn"),
                initializer=_start_worker,
            ) as pool,
        ):
            for future in as_completed([pool.submit(run_transition_case, *job) for job in jobs]):
                result = save(future.result())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true")
    action.add_argument("--status", action="store_true")
    parser.add_argument("--profile", choices=tuple(SEEDS), default="development")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--method", choices=CSL_METHODS, action="append")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--development-report", type=Path)
    parser.add_argument("--encoder-python")
    args = parser.parse_args()
    if args.workers < 1 or (args.oracle_only and args.method):
        parser.error("workers must be positive; choose --oracle-only or --method")
    config = _config(args.profile, args.encoder_python)
    root = args.output_root or Path("outputs/behavior-transition") / args.profile / "recurrent-coupling-01"
    if args.status:
        path = root / REPORT
        print(
            path.read_text() if path.is_file() else json.dumps({"status": "not_started", "report": str(path)})
        )
        return
    _, inventory = load_cross_method_profile("csl_sanity")
    methods = (
        {}
        if args.oracle_only
        else {name: inventory[name] for name in (args.method or [config.primary_method])}
    )
    print(
        json.dumps(
            {
                "profile": args.profile,
                "fit": args.run,
                "output": str(root),
                "methods": list(methods),
                "history_samples": config.generator.history_lengths,
                "process_seeds": SEEDS[args.profile],
                "cases": len(SEEDS[args.profile]) * len(TRANSITION_SCENARIOS),
                "origins_per_case": 3 * config.generator.n_trials,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.run:
        from notebooks._shared import require_conventional_runtime

        require_conventional_runtime(methods)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(process)d %(message)s")
        report = run_study(config, args.profile, methods, root, args.workers, args.development_report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
