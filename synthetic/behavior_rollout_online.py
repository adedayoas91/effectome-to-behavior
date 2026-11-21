"""Evaluate an online choice between pooled and fixed-support local forecasts."""

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
from effectome.experiments.behavior_rollout_online_decisions import online_decision
from effectome.experiments.behavior_rollout_online_policy import refit_decisions
from effectome.experiments.behavior_rollout_timing_kernels import fit_timing_mechanisms, timing_forecasts
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
REPORT = "online-report.json"
CALIBRATION = tuple(range(87000, 87005))
CONFIRMATION = tuple(range(88000, 88020))
SMOKE = (86999,)
GENERATOR = TransitionControlConfig(n_trials=40)
HISTORY = 256
STRIDE = 16
EARLY = 192
BATCH = 96
LOOKBACKS = (4, 8, 16)
MARGINS = (0.0, 0.005, 0.01, 0.02, 0.04)


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
        case = json.loads(path.read_text())
        if (case["scenario"], case["seed"], case["signature"]) != (scenario, seed, signature):
            raise ValueError("cached case belongs to another protocol")
        _verify_artifacts(folder, case["artifacts"])
        return case

    logger.info("Fitting %s", folder.name)
    data = make_transition_control(GENERATOR, scenario=scenario, seed=seed)
    case_id = content_fingerprint((signature, scenario, seed))
    encoder_path, evidence = folder / "encoder.json", folder / "native-training.json"
    readout = _fit_encoder(
        learner,
        data.neural[:READOUT_TRIALS],
        data.behavior[:READOUT_TRIALS],
        encoder_path,
        case_id,
    )
    train = slice(READOUT_TRIALS, POOLED_STOP)
    training, windows = trailing_histories(
        data.neural[train], data.origins[train], max(GENERATOR.history_lengths)
    )
    votes = discover_transition_supports(
        training,
        windows,
        method,
        GENERATOR.fps,
        evidence,
        content_fingerprint((case_id, training, windows, method, GENERATOR.fps)),
    )
    support = votes.mean(axis=0) >= CONSENSUS
    pooled = fit_transition_mechanism(training, support, RIDGE_ALPHA)
    cfg = MechanismRolloutConfig(GENERATOR.horizon, GENERATOR.target_length, N_ROLLOUTS, seed)

    neural = data.neural[POOLED_STOP:]
    points = np.arange(HISTORY - 1, neural.shape[1] - cfg.horizon, STRIDE)
    origins = np.broadcast_to(points, (len(neural), len(points)))
    trial_ids = np.repeat(np.arange(len(neural)), len(points))
    locations = origins.ravel()
    step = np.arange(cfg.horizon - cfg.target_length + 1, cfg.horizon + 1)
    base, local = np.empty((len(locations), 2)), np.empty((len(locations), 2))
    for start in range(0, len(locations), BATCH):
        group = slice(start, start + BATCH)
        trial, origin = trial_ids[group], locations[group]
        ids = (trial + POOLED_STOP) * neural.shape[1] + origin
        history = neural[trial[:, None], origin[:, None] - np.arange(HISTORY - 1, -1, -1)]
        states = history[:, -1]
        base[group] = timing_forecasts([pooled] * len(trial), states, ids, readout, cfg)
        local[group] = timing_forecasts(
            fit_timing_mechanisms(history, support, RIDGE_ALPHA), states, ids, readout, cfg
        )

    # The online scorer can observe only these decoder targets once their interval ends.
    history_targets = readout.predict(neural)[:, points[:, None] + step].mean(axis=2)
    # Generator behavior and switch locations are used only to evaluate a completed policy.
    scored_targets = data.behavior[POOLED_STOP:, points[:, None] + step].mean(axis=2)
    case = {
        "scenario": scenario,
        "seed": seed,
        "signature": signature,
        "artifacts": {file.name: _digest(file) for file in (encoder_path, evidence)},
        "origin_samples": points,
        "training_native_fits": len(training),
        "pooled_mechanism": asdict(pooled),
        "pooled": base.reshape(*origins.shape, 2),
        "local": local.reshape(*origins.shape, 2),
        "decoded_observations": history_targets,
        "behavior_targets": scored_targets,
        "scoring_switch_samples": data.regime_starts[POOLED_STOP:, 1:],
    }
    _write_json(path, case)
    return json.loads(path.read_text())


def _score(case: dict, lookback: int | None, margin: float) -> dict[str, Any]:
    pooled, local = np.asarray(case["pooled"]), np.asarray(case["local"])
    origins = np.broadcast_to(case["origin_samples"], pooled.shape[:2])
    choose = (
        np.zeros_like(origins, dtype=bool)
        if lookback is None
        else refit_decisions(
            origins,
            pooled,
            local,
            np.asarray(case["decoded_observations"]),
            horizon=GENERATOR.horizon,
            lookback=lookback,
            margin=margin,
        )
    )
    return score_choices(case, choose)


def score_choices(case: dict, choose: np.ndarray) -> dict[str, Any]:
    pooled, local = np.asarray(case["pooled"]), np.asarray(case["local"])
    origins = np.broadcast_to(case["origin_samples"], pooled.shape[:2])
    if choose.dtype != np.bool_ or choose.shape != origins.shape:
        raise ValueError("forecast choices must follow the trial-by-origin grid")
    observed = np.asarray(case["behavior_targets"])
    pe, le = np.abs(pooled - observed).mean(axis=-1), np.abs(local - observed).mean(axis=-1)
    selected = np.where(choose, le, pe)
    age = origins[..., None] - np.asarray(case["scoring_switch_samples"])[:, None, :]
    early = np.any((age >= 0) & (age < EARLY), axis=-1)
    if not early.any():
        raise ValueError("the scoring schedule has no origins in the early post-switch band")
    return {
        "seed": case["seed"],
        "scenario": case["scenario"],
        "origins": int(origins.size),
        "early_origins": int(early.sum()),
        "pooled_mae": float(pe.mean()),
        "local_mae": float(le.mean()),
        "gated_mae": float(selected.mean()),
        "pooled_gain": float((pe - selected).mean()),
        "local_gain": float((le - selected).mean()),
        "early_local_gain": float((le[early] - selected[early]).mean()),
        "local_fraction": float(choose.mean()),
        "early_local_fraction": float(choose[early].mean()),
    }


def _calibrate(cases: list[dict]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for lookback in LOOKBACKS:
        for margin in MARGINS:
            rows = [_score(case, lookback, margin) for case in cases]
            gains = {
                name: float(np.mean([r["pooled_gain"] for r in rows if r["scenario"] == name]))
                for name in TRANSITION_SCENARIOS
            }
            early = float(np.mean([r["early_local_gain"] for r in rows if r["scenario"] == "driver_switch"]))
            eligible = early > 0 and all(gains[name] > -GAIN_MARGIN for name in TRANSITION_SCENARIOS[1:])
            candidates.append(
                {
                    "lookback": lookback,
                    "margin": margin,
                    "gains": gains,
                    "early_local_gain": early,
                    "eligible": eligible,
                }
            )
    eligible_rules = [c for c in candidates if c["eligible"]]
    if not eligible_rules:
        return {"lookback": None, "margin": 0.0, "candidates": candidates}
    selected = max(eligible_rules, key=lambda candidate: candidate["gains"]["driver_switch"])
    return {"lookback": selected["lookback"], "margin": selected["margin"], "candidates": candidates}


def _run(
    root: Path, profile: str, workers: int, learner: BehaviorLearningConfig, method: ConnectivityConfig
) -> dict:
    repo = Path(__file__).resolve().parents[1]
    calibration, confirmation = (SMOKE, ()) if profile == "smoke" else (CALIBRATION, CONFIRMATION)
    protocol = {
        "profile": profile,
        "calibration_seeds": calibration,
        "confirmation_seeds": confirmation,
        "generator": asdict(GENERATOR),
        "learning": asdict(learner),
        "method": asdict(method),
        "lookbacks": LOOKBACKS,
        "margins": MARGINS,
        "history": HISTORY,
        "stride": STRIDE,
        "early": EARLY,
        "readout_trials": READOUT_TRIALS,
        "pooled_stop": POOLED_STOP,
        "rollouts": N_ROLLOUTS,
        "ridge_alpha": RIDGE_ALPHA,
        "consensus": CONSENSUS,
        "gain_margin": GAIN_MARGIN,
        "level": LEVEL,
        "implementation": implementation_fingerprint(repo),
        "entrypoint_sha256": _digest(Path(__file__)),
        "encoder_sha256": _digest(repo / "tools/behavior-encoder/worker.py"),
        "encoder_lock_sha256": _digest(repo / "tools/behavior-encoder/uv.lock"),
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
        frozen = root / "online-protocol.json"
        if frozen.is_file() and json.loads(frozen.read_text())["signature"] != signature:
            raise ValueError("output root belongs to another protocol")
        _write_json(frozen, {"signature": signature, **protocol})
        report: dict[str, Any] = {
            "profile": profile,
            "signature": signature,
            "completed_cases": 0,
            "planned_cases": (len(calibration) + len(confirmation)) * len(TRANSITION_SCENARIOS),
        }
        _write_json(root / REPORT, report)

        def collect(seeds: tuple[int, ...], phase: str) -> list[dict]:
            jobs = [
                (scenario, seed, root / "cases" / f"{scenario}-{seed}", signature, method, learner)
                for seed in seeds
                for scenario in TRANSITION_SCENARIOS
            ]
            cases = []
            with (
                _inner_threads(),
                ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool,
            ):
                for future in as_completed([pool.submit(_case, *args) for args in jobs]):
                    cases.append(future.result())
                    report["completed_cases"] += 1
                    _write_json(root / REPORT, report)
                    logger.info("%s case %s/%s saved", phase, len(cases), len(jobs))
            return sorted(cases, key=lambda case: (case["seed"], case["scenario"]))

        dev = collect(calibration, "Calibration")
        # Freeze the rule before any confirmation case is collected or scored.
        policy = _calibrate(dev)
        policy_path = root / "frozen-policy.json"
        policy_record = {"signature": signature, **policy}
        if policy_path.is_file() and json.loads(policy_path.read_text()) != policy_record:
            raise ValueError("calibration does not reproduce the frozen policy")
        _write_json(policy_path, policy_record)
        report["policy"] = {"lookback": policy["lookback"], "margin": policy["margin"]}
        _write_json(root / REPORT, report)
        test = collect(confirmation, "Confirmation") if confirmation else dev
        rows = [_score(case, **report["policy"]) for case in test]
        with (root / "seed-metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        report["confirmation"] = (
            online_decision(rows, confirmation, margin=GAIN_MARGIN, level=LEVEL) if confirmation else None
        )
        report["execution_complete"] = True
        _write_json(root / REPORT, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--profile", choices=("smoke", "confirmation"), default="smoke")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--encoder-python")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = (
        args.output_root or Path("outputs/effectome-rollout-online") / args.profile / "forecast-selection-01"
    )
    if args.status:
        path = root / REPORT
        print(
            path.read_text() if path.is_file() else json.dumps({"status": "not_started", "report": str(path)})
        )
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(
        json.dumps({"profile": args.profile, "output": str(root), "history": HISTORY, "stride": STRIDE}),
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
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
