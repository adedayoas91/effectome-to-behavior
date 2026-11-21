"""Fit and evaluate the end-to-end fixed-support behavioral rollout."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib.metadata import version
from math import isfinite
from multiprocessing import get_context
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.behavior_readout import FrozenBehaviorMap, fixed_behavior_readouts
from effectome.experiments.behavior_rollout_online_decisions import online_decision
from effectome.experiments.behavior_rollout_online_policy import fast_refit_decisions, refit_decisions
from effectome.experiments.behavior_rollout_timing_kernels import fit_timing_mechanisms
from effectome.experiments.behavior_supervised import _run_lock
from effectome.experiments.behavior_supervised_config import BehaviorLearningConfig
from effectome.experiments.behavior_transition_controls import TRANSITION_SCENARIOS, make_transition_control
from effectome.experiments.distributed_behavior_controls import distributed_interventions
from effectome.experiments.manifold_validation import _digest, _verify_artifacts
from effectome.experiments.mechanism_replication import _inner_threads
from effectome.experiments.mechanism_validation import _write_json
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig
from effectome.manifold.behavior_dynamics import behavior_runtime_info
from effectome.workflows.cross_method import load_cross_method_profile
from effectome.workflows.effectome_rollout import forecast_mechanisms
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint
from synthetic.behavior_rollout_online import GENERATOR, HISTORY, STRIDE, _case, score_choices
from synthetic.effectome_rollout import ACTIVE_FLOOR, GAIN_MARGIN, LEVEL, N_ROLLOUTS, POOLED_STOP, RIDGE_ALPHA

REPO = Path(__file__).resolve().parents[1]
REPORT = "pipeline-report.json"
SEEDS = {"smoke": (88999,), "confirmation": tuple(range(89000, 89020))}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Policy:
    lookback: int
    margin: float
    consecutive_losses: int
    recovery_forecasts: int
    effect_stride_origins: int

    def __post_init__(self) -> None:
        for field in ("lookback", "consecutive_losses", "recovery_forecasts", "effect_stride_origins"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if isinstance(self.margin, bool) or not isinstance(self.margin, (int, float)):
            raise ValueError("margin must be a finite nonnegative number")
        if not isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be a finite nonnegative number")


def _effect_metrics(effects: np.ndarray, truth: np.ndarray, active: np.ndarray) -> dict[str, Any]:
    error = np.abs(effects - truth).mean(axis=-1)
    return {
        "all_effect_mae": float(error.mean()),
        "active_effect_mae": float(error[active].mean()) if active.any() else None,
    }


def _effects(case: dict, folder: Path, picks: dict[str, np.ndarray], policy: Policy) -> dict[str, Any]:
    data = make_transition_control(GENERATOR, scenario=case["scenario"], seed=case["seed"])
    readout = FrozenBehaviorMap.from_dict(json.loads((folder / "encoder.json").read_text())["readout"])
    pooled = AdditiveMechanism(**case["pooled_mechanism"])
    points = np.asarray(case["origin_samples"])
    sampled = np.arange(0, len(points), policy.effect_stride_origins)
    trials = np.arange(POOLED_STOP, len(data.neural))
    trial = np.repeat(trials, len(sampled))
    point_index = np.tile(sampled, len(trials))
    origin = points[point_index]
    config = MechanismRolloutConfig(GENERATOR.horizon, GENERATOR.target_length, N_ROLLOUTS, case["seed"])

    # Score oracle effects where the current mechanism remains in force for the full horizon.
    switches = data.regime_starts[trial, 1:]
    episode = (switches <= origin[:, None]).sum(axis=1)
    stable = episode == (switches <= (origin + config.horizon)[:, None]).sum(axis=1)
    omitted = int((~stable).sum())
    trial, point_index, origin, episode = (value[stable] for value in (trial, point_index, origin, episode))
    if not len(origin):
        raise ValueError("no intervention origins stay within one generating regime")
    ids = trial * data.neural.shape[1] + origin
    histories = data.neural[trial[:, None], origin[:, None] - np.arange(HISTORY - 1, -1, -1)]
    states = histories[:, -1]
    local = fit_timing_mechanisms(histories, pooled.support, RIDGE_ALPHA)
    units = distributed_interventions()
    forecasts = {
        "pooled": forecast_mechanisms([pooled] * len(ids), states, ids, readout, units, config),
        "local": forecast_mechanisms(local, states, ids, readout, units, config),
        "oracle": forecast_mechanisms(
            [data.mechanisms[i] for i in data.regimes[trial, episode]],
            states,
            ids,
            fixed_behavior_readouts()["behavior_reference"],
            units,
            config,
        ),
    }
    for name in ("pooled", "local"):
        np.testing.assert_allclose(
            forecasts[name].factual,
            np.asarray(case[name])[trial - POOLED_STOP, point_index],
            rtol=0,
            atol=1e-10,
        )
    effects = {
        name: np.stack([value.effects[key] for key in units], axis=1) for name, value in forecasts.items()
    }
    truth = effects["oracle"]
    active = np.linalg.norm(truth, axis=-1) / np.sqrt(2) >= ACTIVE_FLOOR
    selected = {name: mask[trial - POOLED_STOP, point_index] for name, mask in picks.items()}
    for name, mask in selected.items():
        effects[name] = np.where(mask[:, None, None], effects["local"], effects["pooled"])
    return {
        "trial": trial,
        "origin": origin,
        "history_start": origin + 1 - HISTORY,
        "history_stop": origin + 1,
        "innovation_id": ids,
        "trial_crossings_excluded": omitted,
        "selected_local": selected,
        "units": units,
        "active_origin_unit_pairs": int(active.sum()),
        "vectors": effects,
        "metrics": {
            name: _effect_metrics(values, truth, active)
            for name, values in effects.items()
            if name != "oracle"
        },
    }


def _pipeline_case(
    scenario: str,
    seed: int,
    folder: Path,
    signature: str,
    method: ConnectivityConfig,
    learner: BehaviorLearningConfig,
    policy: Policy,
) -> dict[str, Any]:
    path = folder / "pipeline-case.json"
    if path.is_file():
        result = json.loads(path.read_text())
        if (result["scenario"], result["seed"], result["signature"]) != (scenario, seed, signature):
            raise ValueError("cached pipeline case belongs to another protocol")
        _verify_artifacts(folder, result["artifacts"])
        return result
    case = _case(scenario, seed, folder, signature, method, learner)
    pooled, local, observed = (np.asarray(case[key]) for key in ("pooled", "local", "decoded_observations"))
    origins = np.broadcast_to(case["origin_samples"], pooled.shape[:2])
    picks = {
        "slow": refit_decisions(
            origins,
            pooled,
            local,
            observed,
            horizon=GENERATOR.horizon,
            lookback=policy.lookback,
            margin=policy.margin,
        ),
        "fast": fast_refit_decisions(
            origins,
            pooled,
            local,
            observed,
            horizon=GENERATOR.horizon,
            lookback=policy.lookback,
            margin=policy.margin,
            consecutive_losses=policy.consecutive_losses,
            recovery_forecasts=policy.recovery_forecasts,
        ),
    }
    effects = _effects(case, folder, picks, policy)
    out = {
        "scenario": scenario,
        "seed": seed,
        "signature": signature,
        "artifacts": {**case["artifacts"], "case.json": _digest(folder / "case.json")},
        "selection": picks,
        "effects": effects,
        "forecast_metrics": {name: score_choices(case, mask) for name, mask in picks.items()},
    }
    _write_json(path, out)
    return json.loads(path.read_text())


def _run(
    root: Path,
    profile: str,
    workers: int,
    learner: BehaviorLearningConfig,
    method: ConnectivityConfig,
    *,
    seeds: tuple[int, ...] | None = None,
) -> dict:
    if seeds is None:
        seeds = SEEDS[profile]
    config = OmegaConf.to_container(
        OmegaConf.load(REPO / "conf/behavior_rollout_pipeline/fixed.yaml"), resolve=True
    )
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise ValueError("fixed policy configuration must be a mapping")
    policy = Policy(**cast(dict[str, Any], config))
    protocol = {
        "profile": profile,
        "seeds": seeds,
        "scenarios": TRANSITION_SCENARIOS,
        "policy": asdict(policy),
        "generator": asdict(GENERATOR),
        "learner": asdict(learner),
        "method": asdict(method),
        "history": HISTORY,
        "forecast_stride": STRIDE,
        "rollouts": N_ROLLOUTS,
        "active_effect_floor": ACTIVE_FLOOR,
        "gain_margin": GAIN_MARGIN,
        "level": LEVEL,
        "implementation": implementation_fingerprint(REPO),
        "entrypoint_sha256": _digest(Path(__file__)),
        "forecast_entrypoint_sha256": _digest(REPO / "synthetic/behavior_rollout_online.py"),
        "rollout_settings_sha256": _digest(REPO / "synthetic/effectome_rollout.py"),
        "encoder_sha256": _digest(REPO / "tools/behavior-encoder/worker.py"),
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
        plan = root / "pipeline-protocol.json"
        if plan.is_file() and json.loads(plan.read_text())["signature"] != signature:
            raise ValueError("output root belongs to another protocol")
        _write_json(plan, {"signature": signature, **protocol})
        jobs = [
            (scenario, seed, root / "cases" / f"{scenario}-{seed}", signature, method, learner, policy)
            for seed in seeds
            for scenario in TRANSITION_SCENARIOS
        ]
        report: dict[str, Any] = {
            "profile": profile,
            "signature": signature,
            "policy": asdict(policy),
            "completed_cases": 0,
            "planned_cases": len(jobs),
            "execution_complete": False,
        }
        _write_json(root / REPORT, report)
        cases = []
        with (
            _inner_threads(),
            ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool,
        ):
            for future in as_completed([pool.submit(_pipeline_case, *args) for args in jobs]):
                cases.append(future.result())
                report["completed_cases"] = len(cases)
                _write_json(root / REPORT, report)
                logger.info("End-to-end case %s/%s saved", len(cases), len(jobs))
        cases.sort(key=lambda case: (case["seed"], case["scenario"]))
        rows = [
            {
                "policy": name,
                **case["forecast_metrics"][name],
                **case["effects"]["metrics"][name],
                "pooled_active_effect_mae": case["effects"]["metrics"]["pooled"]["active_effect_mae"],
                "local_active_effect_mae": case["effects"]["metrics"]["local"]["active_effect_mae"],
                "effect_origins": len(case["effects"]["origin"]),
            }
            for case in cases
            for name in ("slow", "fast")
        ]
        temporary = root / "seed-metrics.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / "seed-metrics.csv")
        report["confirmation"] = (
            online_decision(
                [row for row in rows if row["policy"] == "fast"],
                seeds,
                margin=GAIN_MARGIN,
                level=LEVEL,
            )
            if profile == "confirmation"
            else None
        )
        report["execution_complete"] = True
        _write_json(root / REPORT, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--profile", choices=tuple(SEEDS), default="smoke")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--encoder-python")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = args.output_root or Path("outputs/effectome-rollout-pipeline") / args.profile / "fast-fallback-01"
    if args.status:
        path = root / REPORT
        print(
            path.read_text() if path.is_file() else json.dumps({"status": "not_started", "report": str(path)})
        )
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _, methods = load_cross_method_profile("csl_sanity")
    print(
        json.dumps(
            _run(
                root,
                args.profile,
                args.workers,
                BehaviorLearningConfig(runtime_python=args.encoder_python),
                methods["fast_cgc"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
