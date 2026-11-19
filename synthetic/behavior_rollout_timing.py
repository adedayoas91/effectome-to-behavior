"""Test when fixed-support local refitting starts improving behavior forecasts."""

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

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.behavior_readout import fixed_behavior_readouts
from effectome.experiments.behavior_rollout_timing_decisions import timing_decision
from effectome.experiments.behavior_rollout_timing_kernels import fit_timing_mechanisms, timing_forecasts
from effectome.experiments.behavior_supervised import _run_lock
from effectome.experiments.behavior_supervised_config import BehaviorLearningConfig
from effectome.experiments.behavior_transition import _fit_encoder
from effectome.experiments.behavior_transition_controls import (
    TRANSITION_SCENARIOS,
    TransitionControlConfig,
    TransitionControlData,
    make_transition_control,
)
from effectome.experiments.behavior_transition_graphs import (
    discover_transition_supports,
    fit_transition_mechanism,
    trailing_histories,
)
from effectome.experiments.manifold_validation import _digest, _verify_artifacts
from effectome.experiments.mechanism_replication import _inner_threads
from effectome.experiments.mechanism_validation import _write_json
from effectome.experiments.structural_mechanism import MechanismRolloutConfig
from effectome.manifold.behavior_dynamics import behavior_runtime_info
from effectome.workflows.cross_method import load_cross_method_profile
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint
from synthetic.effectome_rollout import (
    CONSENSUS,
    GAIN_MARGIN,
    LEVEL,
    N_ROLLOUTS,
    POOLED_STOP,
    READOUT_TRIALS,
    RIDGE_ALPHA,
)

logger = logging.getLogger(__name__)
REPORT = "timing-report.json"
SEEDS = {"smoke": (85000,), "confirmation": tuple(range(86000, 86020))}
GENERATOR = TransitionControlConfig(n_trials=72, history_lengths=(256, 1024))
DELAYS = (0, 8, 16, 32, 64, 128, 192, 256, 512, 768, 1023)


def _evaluation_origins(
    data: TransitionControlData, delay: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use generating switch times to index scoring events on untouched trials."""
    switches = data.regime_starts[POOLED_STOP:, 1:]
    origins = switches + delay
    if np.any(origins[:, 0] + horizon >= switches[:, 1]) or np.any(origins + horizon >= data.neural.shape[1]):
        raise ValueError("a scored future crosses a synthetic regime or trial boundary")
    trial = np.repeat(np.arange(POOLED_STOP, len(data.neural)), switches.shape[1])
    return origins, trial, switches.ravel()


def _case(
    scenario: str,
    seed: int,
    path: Path,
    signature: str,
    method: ConnectivityConfig,
    learner: BehaviorLearningConfig,
) -> dict[str, Any]:
    record_path = path / "case.json"
    if record_path.is_file():
        saved = json.loads(record_path.read_text())
        if (saved["signature"], saved["scenario"], saved["seed"]) != (signature, scenario, seed):
            raise ValueError("saved timing case belongs to another study")
        _verify_artifacts(path, saved["artifacts"])
        return saved

    logger.info("Fitting training readout and support for %s", path.name)
    data = make_transition_control(GENERATOR, scenario=scenario, seed=seed)
    encoder_path, evidence = path / "encoder.json", path / "native-training.json"
    case_id = content_fingerprint((signature, scenario, seed))
    readout = _fit_encoder(
        learner, data.neural[:READOUT_TRIALS], data.behavior[:READOUT_TRIALS], encoder_path, case_id
    )
    train = slice(READOUT_TRIALS, POOLED_STOP)
    stable_history, train_windows = trailing_histories(
        data.neural[train], data.origins[train], max(GENERATOR.history_lengths)
    )
    native_id = content_fingerprint((case_id, stable_history, train_windows, method, GENERATOR.fps))
    votes = discover_transition_supports(
        stable_history, train_windows, method, GENERATOR.fps, evidence, native_id
    )
    support = votes.mean(axis=0) >= CONSENSUS
    pooled = fit_transition_mechanism(stable_history, support, RIDGE_ALPHA)
    rollout = MechanismRolloutConfig(GENERATOR.horizon, GENERATOR.target_length, N_ROLLOUTS, seed)
    gold_readout = fixed_behavior_readouts()["behavior_reference"]
    epochs = np.arange(rollout.horizon - rollout.target_length + 1, rollout.horizon + 1)
    regime = data.regimes[POOLED_STOP:, 1:].ravel()
    truth_mechanisms = [data.mechanisms[item] for item in regime]
    blocks, rows = {}, []
    for delay in DELAYS:
        origins, trial, switches = _evaluation_origins(data, delay, rollout.horizon)
        sample = origins.ravel()
        identifiers = trial * data.neural.shape[1] + sample
        states = data.neural[trial, sample]
        target = data.behavior[trial[:, None], sample[:, None] + epochs].mean(axis=1)
        pred_pooled = timing_forecasts([pooled] * len(trial), states, identifiers, readout, rollout)
        pred_oracle = timing_forecasts(truth_mechanisms, states, identifiers, gold_readout, rollout)
        pooled_error = float(np.abs(pred_pooled - target).mean())
        oracle_error = float(np.abs(pred_oracle - target).mean())
        pooled_conditional_error = float(np.abs(pred_pooled - pred_oracle).mean())
        block = {
            "trial": trial,
            "switch_sample": switches,
            "forecast_origin": sample,
            "innovation_id": identifiers,
            "generating_regime": regime,
            "target": target,
            "pooled": pred_pooled,
            "oracle": pred_oracle,
            "local": {},
        }
        for length in GENERATOR.history_lengths:
            history, windows = trailing_histories(data.neural[POOLED_STOP:], origins, length)
            local = fit_timing_mechanisms(history, support, RIDGE_ALPHA)
            pred_local = timing_forecasts(local, states, identifiers, readout, rollout)
            local_error = float(np.abs(pred_local - target).mean())
            local_conditional_error = float(np.abs(pred_local - pred_oracle).mean())
            block["local"][str(length)] = {
                "forecast": pred_local,
                "history_start": sample + 1 - length,
                "history_stop": sample + 1,
            }
            if np.any([window.length != length for window in windows]):
                raise ValueError("unexpected trailing window length")
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "history_samples": length,
                    "delay_samples": delay,
                    "delay_seconds": delay / GENERATOR.fps,
                    "origins": len(trial),
                    "post_switch_pair_fraction": min(delay + 1, length - 1) / (length - 1),
                    "pooled_mae": pooled_error,
                    "local_mae": local_error,
                    "oracle_mae": oracle_error,
                    "forecast_gain": pooled_error - local_error,
                    "pooled_conditional_mae": pooled_conditional_error,
                    "local_conditional_mae": local_conditional_error,
                    "conditional_gain": pooled_conditional_error - local_conditional_error,
                }
            )
        blocks[str(delay)] = block

    saved = {
        "seed": seed,
        "scenario": scenario,
        "signature": signature,
        "artifacts": {file.name: _digest(file) for file in (encoder_path, evidence)},
        "trial_counts": {
            "encoder": READOUT_TRIALS,
            "pooled": POOLED_STOP - READOUT_TRIALS,
            "evaluation": len(data.neural) - POOLED_STOP,
        },
        "pooled_mechanism": asdict(pooled),
        "native_training_fits": len(stable_history),
        "forecasts": blocks,
        "seed_metrics": rows,
    }
    _write_json(record_path, saved)
    return json.loads(record_path.read_text())


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
        "delays": DELAYS,
        "encoder_trial_stop": READOUT_TRIALS,
        "evaluation_trial_start": POOLED_STOP,
        "rollouts": N_ROLLOUTS,
        "ridge_alpha": RIDGE_ALPHA,
        "consensus": CONSENSUS,
        "margin": GAIN_MARGIN,
        "level": LEVEL,
        "joint_test": "intersection_union_then_holm",
        "implementation": implementation_fingerprint(package),
        "entrypoint_sha256": _digest(Path(__file__)),
        "isolated_rollout_config_sha256": _digest(package / "synthetic/effectome_rollout.py"),
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
        path = root / "timing-protocol.json"
        if path.is_file() and json.loads(path.read_text())["signature"] != signature:
            raise ValueError("saved timing output belongs to another protocol or runtime")
        _write_json(path, {"signature": signature, **protocol})
        jobs = [
            (scenario, seed, root / "cases" / f"{scenario}-{seed}", signature, method, learner)
            for seed in seeds
            for scenario in TRANSITION_SCENARIOS
        ]
        cases: list[dict] = []

        def save() -> dict:
            ordered = sorted(cases, key=lambda case: (case["seed"], case["scenario"]))
            rows = [row for case in ordered for row in case["seed_metrics"]]
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
                "native_training_fits": sum(case["native_training_fits"] for case in cases),
                "cases": [f"cases/{case['scenario']}-{case['seed']}/case.json" for case in ordered],
                "confirmation": timing_decision(
                    rows, seeds, GENERATOR.history_lengths, DELAYS, margin=GAIN_MARGIN, level=LEVEL
                )
                if complete and profile == "confirmation"
                else None,
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
                logger.info("Timing study %s/%s cases saved", len(cases), len(jobs))
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
    root = args.output_root or Path("outputs/effectome-rollout-timing") / args.profile / "post-switch-01"
    if args.status:
        path = root / REPORT
        if path.is_file():
            print(path.read_text())
        else:
            print(json.dumps({"status": "not_started", "report": str(path)}))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(
        json.dumps(
            {
                "profile": args.profile,
                "seeds": SEEDS[args.profile],
                "output": str(root),
                "histories": GENERATOR.history_lengths,
                "delays": DELAYS,
                "evaluation_trials": GENERATOR.n_trials - POOLED_STOP,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.run:
        _, methods = load_cross_method_profile("csl_sanity")
        report = _run(
            root,
            args.profile,
            args.workers,
            BehaviorLearningConfig(runtime_python=args.encoder_python),
            methods["fast_cgc"],
        )
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
