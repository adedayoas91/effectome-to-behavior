"""Validate the isolated effectome-to-behavior rollout against paired controls."""

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
from effectome.experiments.behavior_readout import fixed_behavior_readouts
from effectome.experiments.behavior_supervised import _run_lock
from effectome.experiments.behavior_supervised_config import BehaviorLearningConfig
from effectome.experiments.behavior_transition import _fit_encoder
from effectome.experiments.behavior_transition_controls import (
    TRANSITION_SCENARIOS,
    TransitionControlConfig,
    make_transition_control,
)
from effectome.experiments.behavior_transition_graphs import (
    discover_transition_supports,
    fit_transition_mechanism,
    trailing_histories,
)
from effectome.experiments.distributed_behavior_controls import distributed_interventions
from effectome.experiments.manifold_validation import _digest, _verify_artifacts
from effectome.experiments.mechanism_replication import _inner_threads
from effectome.experiments.mechanism_validation import _write_json
from effectome.experiments.structural_mechanism import MechanismRolloutConfig
from effectome.manifold.behavior_dynamics import behavior_runtime_info
from effectome.workflows.cross_method import load_cross_method_profile
from effectome.workflows.effectome_rollout import forecast_mechanisms, run_effectome_rollout
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

logger = logging.getLogger(__name__)
REPORT = "rollout-report.json"
SEEDS = {"smoke": (83000,), "confirmation": tuple(range(84000, 84020))}
READOUT_TRIALS = 10
POOLED_STOP = 24
GENERATOR = TransitionControlConfig(history_lengths=(1024,))
N_ROLLOUTS = 128
RIDGE_ALPHA = 1.0
CONSENSUS = 0.25
ACTIVE_FLOOR = 0.05
GAIN_MARGIN = 0.005
LEVEL = 0.05


def _case(
    scenario: str,
    seed: int,
    folder: Path,
    signature: str,
    method: ConnectivityConfig,
    learner: BehaviorLearningConfig,
) -> dict[str, Any]:
    path = folder / "case.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if (saved["signature"], saved["scenario"], saved["seed"]) != (signature, scenario, seed):
            raise ValueError("saved case belongs to another study")
        _verify_artifacts(folder, saved["artifacts"])
        return saved

    trials = make_transition_control(GENERATOR, scenario=scenario, seed=seed)
    encoder = folder / "encoder.json"
    case_id = content_fingerprint((signature, scenario, seed))
    logger.info("Fitting encoder and causal histories for %s", folder.name)
    readout = _fit_encoder(
        learner, trials.neural[:READOUT_TRIALS], trials.behavior[:READOUT_TRIALS], encoder, case_id
    )
    length = GENERATOR.history_lengths[0]
    histories, windows = trailing_histories(trials.neural, trials.origins, length)
    trial_ids = np.repeat(np.arange(GENERATOR.n_trials), trials.origins.shape[1])
    pool = np.flatnonzero((trial_ids >= READOUT_TRIALS) & (trial_ids < POOLED_STOP))
    test = np.flatnonzero(trial_ids >= POOLED_STOP)
    pooled_path, dynamic_path = folder / "native-pooled.json", folder / "native-dynamic.json"
    pool_windows = [windows[i] for i in pool]
    pool_identity = content_fingerprint((case_id, histories[pool], pool_windows, method, GENERATOR.fps))
    pooled_support = (
        discover_transition_supports(
            histories[pool], pool_windows, method, GENERATOR.fps, pooled_path, pool_identity
        ).mean(axis=0)
        >= CONSENSUS
    )
    pooled = fit_transition_mechanism(histories[pool], pooled_support, RIDGE_ALPHA)
    fixed_local = tuple(
        fit_transition_mechanism(history[None], pooled_support, RIDGE_ALPHA) for history in histories[test]
    )
    config = MechanismRolloutConfig(GENERATOR.horizon, GENERATOR.target_length, N_ROLLOUTS, seed)
    units = distributed_interventions()
    states = histories[test, -1]
    regimes = trials.regimes.ravel()[test]
    oracle = forecast_mechanisms(
        [trials.mechanisms[i] for i in regimes],
        states,
        test,
        fixed_behavior_readouts()["behavior_reference"],
        units,
        config,
    )
    dynamic, mechanisms = run_effectome_rollout(
        histories[test],
        [windows[i] for i in test],
        test,
        method=method,
        fps=GENERATOR.fps,
        evidence=dynamic_path,
        signature=case_id,
        readout=readout,
        interventions=units,
        config=config,
        ridge_alpha=RIDGE_ALPHA,
    )
    arms = {
        "pooled": forecast_mechanisms([pooled] * len(test), states, test, readout, units, config),
        "local_fixed_support": forecast_mechanisms(fixed_local, states, test, readout, units, config),
        "local_dynamic_support": dynamic,
        "oracle": oracle,
    }
    trial, origin = trial_ids[test], trials.origins.ravel()[test]
    offsets = np.arange(config.horizon - config.target_length + 1, config.horizon + 1)
    target = trials.behavior[trial[:, None], origin[:, None] + offsets].mean(axis=1)
    current = trials.behavior[trial, origin]
    true_effects = np.stack([oracle.effects[unit] for unit in units], axis=1)
    active = np.linalg.norm(true_effects, axis=-1) / np.sqrt(2) >= ACTIVE_FLOOR
    metrics: dict[str, Any] = {}
    for name, result in arms.items():
        error = np.abs(np.stack([result.effects[unit] for unit in units], axis=1) - true_effects).mean(axis=2)
        metrics[name] = {
            "forecast_mae": float(np.abs(result.factual - target).mean()),
            "active_effect_mae": float(error[active].mean()) if active.any() else None,
            "all_effect_mae": float(error.mean()),
        }
    record = {
        "signature": signature,
        "seed": seed,
        "scenario": scenario,
        "artifacts": {p.name: _digest(p) for p in (encoder, pooled_path, dynamic_path)},
        "trial_counts": {
            "encoder": READOUT_TRIALS,
            "pooled": POOLED_STOP - READOUT_TRIALS,
            "evaluation": GENERATOR.n_trials - POOLED_STOP,
        },
        "native_window_fits": len(pool) + len(test),
        "origins": {
            "event": test,
            "trial": trial,
            "sample": origin,
            "oracle_regime": regimes,
            "target_delta": target - current,
            "raw_histories": [[windows[i].start, windows[i].stop] for i in test],
        },
        "candidate_members": units,
        "active_origin_unit_pairs": int(active.sum()),
        "metrics": metrics,
        "forecast_deltas": {name: result.factual - current for name, result in arms.items()},
        "effect_vectors": {name: result.effects for name, result in arms.items()},
        "mechanisms": {
            "pooled": asdict(pooled),
            "local_fixed_support": [asdict(m) for m in fixed_local],
            "local_dynamic_support": [asdict(m) for m in mechanisms],
        },
    }
    _write_json(path, record)
    return json.loads(path.read_text())


def _seed_row(case: dict[str, Any]) -> dict[str, Any]:
    arms = case["metrics"]
    pooled, fixed, dynamic = (
        arms[key]["forecast_mae"]
        for key in (
            "pooled",
            "local_fixed_support",
            "local_dynamic_support",
        )
    )
    return {
        "seed": case["seed"],
        "scenario": case["scenario"],
        "active_origin_unit_pairs": case["active_origin_unit_pairs"],
        **{f"{arm}_{key}": value for arm, metrics in arms.items() for key, value in metrics.items()},
        "local_adaptation_gain": pooled - fixed,
        "support_rediscovery_gain": fixed - dynamic,
        "dynamic_vs_pooled_gain": pooled - dynamic,
    }


def decision(rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, Any]:
    """Require a meaningful driver gain and its absence in each paired control."""
    gates = {}
    for scenario in TRANSITION_SCENARIOS:
        selected = [row for row in rows if row["scenario"] == scenario]
        gains = np.asarray([row["dynamic_vs_pooled_gain"] for row in selected])
        complete = sorted(row["seed"] for row in selected) == list(seeds) and np.isfinite(gains).all()
        count = int(
            np.count_nonzero(gains > GAIN_MARGIN if scenario == "driver_switch" else gains < GAIN_MARGIN)
        )
        p = float(binomtest(count, len(gains), 0.5, alternative="greater").pvalue) if complete else None
        gates[scenario] = {
            "seeds": len(gains),
            "gains_on_expected_side": count,
            "p_value": p,
            "passed": p is not None and p <= LEVEL,
        }
    return {
        "margin_mae": GAIN_MARGIN,
        "level": LEVEL,
        "tests": gates,
        "forecast_confirmation_passed": all(gate["passed"] for gate in gates.values()),
    }


def _run(
    root: Path, profile: str, workers: int, learner: BehaviorLearningConfig, method: ConnectivityConfig
) -> dict:
    package = Path(__file__).resolve().parents[1]
    seeds = SEEDS[profile]
    protocol = {
        "profile": profile,
        "seeds": seeds,
        "scenarios": TRANSITION_SCENARIOS,
        "generator": asdict(GENERATOR),
        "learning": asdict(learner),
        "method": asdict(method),
        "pooling_start": READOUT_TRIALS,
        "evaluation_start": POOLED_STOP,
        "rollouts": N_ROLLOUTS,
        "ridge_alpha": RIDGE_ALPHA,
        "consensus": CONSENSUS,
        "active_floor": ACTIVE_FLOOR,
        "decision_margin": GAIN_MARGIN,
        "test_level": LEVEL,
        "implementation": implementation_fingerprint(package),
        "entrypoint_sha256": _digest(Path(__file__)),
        "encoder_worker_sha256": _digest(package / "tools/behavior-encoder/worker.py"),
        "encoder_lock_sha256": _digest(package / "tools/behavior-encoder/uv.lock"),
        "encoder_runtime": behavior_runtime_info(learner.runtime_python),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "scikit-learn", "scipy", "fastcgc")},
        },
    }
    signature = content_fingerprint(protocol)
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        path = root / "rollout-protocol.json"
        if path.is_file() and json.loads(path.read_text())["signature"] != signature:
            raise ValueError("saved output belongs to another protocol or runtime; choose a new output root")
        _write_json(path, {"signature": signature, **protocol})
        jobs = [
            (scenario, seed, root / "cases" / f"{scenario}-{seed}", signature, method, learner)
            for seed in seeds
            for scenario in TRANSITION_SCENARIOS
        ]
        cases: list[dict] = []

        def save() -> dict:
            ordered = sorted(cases, key=lambda case: (case["seed"], case["scenario"]))
            rows = [_seed_row(case) for case in ordered]
            if rows:
                temp = root / "seed-metrics.csv.tmp"
                with temp.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                temp.replace(root / "seed-metrics.csv")
            complete = len(cases) == len(jobs)
            report = {
                "profile": profile,
                "signature": signature,
                "completed_cases": len(cases),
                "planned_cases": len(jobs),
                "execution_complete": complete,
                "native_window_fits": sum(case["native_window_fits"] for case in cases),
                "cases": [f"cases/{case['scenario']}-{case['seed']}/case.json" for case in ordered],
                "confirmation": decision(rows, seeds) if complete and profile == "confirmation" else None,
            }
            _write_json(root / REPORT, report)
            return report

        report = save()
        with (
            _inner_threads(),
            ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool,
        ):
            for future in as_completed([pool.submit(_case, *job) for job in jobs]):
                cases.append(future.result())
                report = save()
                logger.info("Rollout study %s/%s cases saved", len(cases), len(jobs))
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--profile", choices=tuple(SEEDS), default="smoke")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--encoder-python")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = args.output_root or Path("outputs/effectome-rollout") / args.profile / "mechanism-rollout-01"
    if args.status:
        path = root / REPORT
        print(
            path.read_text() if path.is_file() else json.dumps({"status": "not_started", "report": str(path)})
        )
        return
    print(
        json.dumps(
            {
                "profile": args.profile,
                "seeds": SEEDS[args.profile],
                "output": str(root),
                "method": "fast_cgc",
                "history_samples": GENERATOR.history_lengths[0],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.run:
        _, inventory = load_cross_method_profile("csl_sanity")
        report = _run(
            root,
            args.profile,
            args.workers,
            BehaviorLearningConfig(runtime_python=args.encoder_python),
            inventory["fast_cgc"],
        )
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
