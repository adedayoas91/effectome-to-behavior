"""Compare learned support with dense, shuffled, and direct history baselines."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib.metadata import version
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.behavior_readout import FrozenBehaviorMap, fixed_behavior_readouts
from effectome.experiments.behavior_rollout_history_baseline import fit_history_baseline
from effectome.experiments.behavior_rollout_online_decisions import online_decision
from effectome.experiments.behavior_rollout_online_policy import fast_refit_decisions
from effectome.experiments.behavior_rollout_support_decisions import support_decision
from effectome.experiments.behavior_rollout_timing_kernels import fit_timing_mechanisms, timing_forecasts
from effectome.experiments.behavior_supervised import _run_lock
from effectome.experiments.behavior_supervised_config import BehaviorLearningConfig
from effectome.experiments.behavior_transition import _fit_encoder
from effectome.experiments.behavior_transition_controls import (
    TRANSITION_SCENARIOS,
    TransitionControlData,
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
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig
from effectome.manifold.behavior_dynamics import behavior_runtime_info
from effectome.workflows.cross_method import load_cross_method_profile
from effectome.workflows.effectome_rollout import forecast_mechanisms
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint
from synthetic.behavior_rollout_online import BATCH, EARLY, GENERATOR, HISTORY, STRIDE, score_choices
from synthetic.behavior_rollout_pipeline import REPO, Policy, _effect_metrics
from synthetic.effectome_rollout import (
    ACTIVE_FLOOR,
    CONSENSUS,
    GAIN_MARGIN,
    POOLED_STOP,
    READOUT_TRIALS,
    RIDGE_ALPHA,
)

CONFIG = REPO / "conf/behavior_rollout_pipeline/support.yaml"
REPORT = "support-report.json"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Profile:
    seed_start: int
    seed_count: int
    shuffled_masks: int
    rollouts: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in asdict(self).values()):
            raise ValueError("profile seeds, shuffle count and rollout budget must be positive integers")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.seed_count))


def _support_masks(learned: np.ndarray, seed: int, n_shuffles: int) -> dict[str, np.ndarray]:
    if (
        learned.ndim != 2
        or not len(learned)
        or learned.shape[0] != learned.shape[1]
        or learned.dtype != np.bool_
        or np.any(np.diag(learned))
    ):
        raise ValueError("learned support must be a square boolean matrix without self edges")
    dense = ~np.eye(len(learned), dtype=bool)
    masks = {"learned": learned, "dense": dense}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    for index in range(n_shuffles):
        shuffled = np.zeros_like(learned)
        for target, count in enumerate(learned.sum(axis=0)):
            parents = rng.choice(np.flatnonzero(dense[:, target]), size=count, replace=False)
            shuffled[parents, target] = True
        masks[f"shuffle_{index + 1}"] = shuffled
    return masks


@dataclass(frozen=True)
class ForecastGrid:
    neural: np.ndarray
    points: np.ndarray
    origins: np.ndarray
    trial: np.ndarray
    origin: np.ndarray
    innovation_ids: np.ndarray
    decoded_targets: np.ndarray
    targets: np.ndarray
    switches: np.ndarray

    @classmethod
    def from_data(cls, data: TransitionControlData, readout: FrozenBehaviorMap) -> ForecastGrid:
        neural = data.neural[POOLED_STOP:]
        points = np.arange(HISTORY - 1, neural.shape[1] - GENERATOR.horizon, STRIDE)
        origins = np.broadcast_to(points, (len(neural), len(points)))
        trial = np.repeat(np.arange(len(neural)), len(points))
        origin = origins.ravel()
        offsets = np.arange(GENERATOR.horizon - GENERATOR.target_length + 1, GENERATOR.horizon + 1)
        return cls(
            neural,
            points,
            origins,
            trial,
            origin,
            (trial + POOLED_STOP) * neural.shape[1] + origin,
            readout.predict(neural)[:, points[:, None] + offsets].mean(axis=2),
            data.behavior[POOLED_STOP:, points[:, None] + offsets].mean(axis=2),
            data.regime_starts[POOLED_STOP:, 1:],
        )

    def histories(self, index: slice | np.ndarray) -> np.ndarray:
        samples = self.origin[index, None] - np.arange(HISTORY - 1, -1, -1)
        return self.neural[self.trial[index, None], samples]

    def errors(self, prediction: np.ndarray) -> dict[str, float]:
        error = np.abs(prediction - self.targets).mean(axis=-1)
        age = self.origins[..., None] - self.switches[:, None, :]
        early = ((age >= 0) & (age < EARLY)).any(axis=-1)
        return {"forecast_mae": float(error.mean()), "early_forecast_mae": float(error[early].mean())}


@dataclass(frozen=True)
class EffectGrid:
    index: np.ndarray
    histories: np.ndarray
    innovation_ids: np.ndarray
    truth: np.ndarray
    active: np.ndarray
    crossings_excluded: int

    @classmethod
    def from_data(
        cls, data: TransitionControlData, grid: ForecastGrid, policy: Policy, config: MechanismRolloutConfig
    ) -> EffectGrid:
        sampled = np.arange(grid.origins.size).reshape(grid.origins.shape)[:, :: policy.effect_stride_origins]
        index = sampled.ravel()
        trial, origin = grid.trial[index] + POOLED_STOP, grid.origin[index]
        switches = data.regime_starts[trial, 1:]
        episode = (switches <= origin[:, None]).sum(axis=1)
        stable = episode == (switches <= (origin + config.horizon)[:, None]).sum(axis=1)
        if not np.any(stable):
            raise ValueError("no sampled intervention origin has a single generating regime")
        index = index[stable]
        ids = grid.innovation_ids[index]
        histories = grid.histories(index)
        verified_regimes = data.regimes[trial[stable], episode[stable]]
        oracle = forecast_mechanisms(
            [data.mechanisms[regime] for regime in verified_regimes],
            histories[:, -1],
            ids,
            fixed_behavior_readouts()["behavior_reference"],
            distributed_interventions(),
            config,
        )
        truth = np.stack(list(oracle.effects.values()), axis=1)
        active = np.linalg.norm(truth, axis=-1) / np.sqrt(truth.shape[-1]) >= ACTIVE_FLOOR
        return cls(index, histories, ids, truth, active, int((~stable).sum()))

    def score(
        self,
        chosen: np.ndarray,
        prediction: np.ndarray,
        pooled: AdditiveMechanism,
        readout: FrozenBehaviorMap,
        config: MechanismRolloutConfig,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        local_indices = np.flatnonzero(chosen.ravel()[self.index])
        models = [pooled] * len(self.index)
        if local_indices.size:
            local = fit_timing_mechanisms(self.histories[local_indices], pooled.support, RIDGE_ALPHA)
            for index, model in zip(local_indices, local, strict=True):
                models[index] = model
        rollouts = forecast_mechanisms(
            models, self.histories[:, -1], self.innovation_ids, readout, distributed_interventions(), config
        )
        np.testing.assert_allclose(
            rollouts.factual,
            prediction.reshape(-1, 2)[self.index],
            rtol=0,
            atol=1e-10,
        )
        vectors = np.stack(list(rollouts.effects.values()), axis=1)
        return vectors, _effect_metrics(vectors, self.truth, self.active)


def _paired_forecasts(
    grid: ForecastGrid, pooled: AdditiveMechanism, readout: FrozenBehaviorMap, config: MechanismRolloutConfig
) -> tuple[np.ndarray, np.ndarray]:
    base = np.empty((grid.origins.size, 2))
    local = np.empty_like(base)
    for start in range(0, len(base), BATCH):
        batch = slice(start, start + BATCH)
        history = grid.histories(batch)
        ids = grid.innovation_ids[batch]
        states = history[:, -1]
        base[batch] = timing_forecasts([pooled] * len(states), states, ids, readout, config)
        local[batch] = timing_forecasts(
            fit_timing_mechanisms(history, pooled.support, RIDGE_ALPHA), states, ids, readout, config
        )
    return base.reshape(*grid.origins.shape, 2), local.reshape(*grid.origins.shape, 2)


def _replicate_mean(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Collapse within-case shuffle draws before the seed-level comparisons."""
    first = rows[0]
    result = {key: first[key] for key in ("scenario", "seed", "origins", "effect_origins")}
    for field in (field for field in first if field not in result and field != "arm"):
        values = [row[field] for row in rows]
        result[field] = mean(values) if all(value is not None for value in values) else None
    return {"arm": arm, **result}


def _case(
    scenario: str,
    seed: int,
    folder: Path,
    signature: str,
    method: ConnectivityConfig,
    learner: BehaviorLearningConfig,
    policy: Policy,
    profile: Profile,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "support-case.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if (saved["scenario"], saved["seed"], saved["signature"]) != (scenario, seed, signature):
            raise ValueError("cached case belongs to another support protocol")
        _verify_artifacts(folder, saved["artifacts"])
        return saved["rows"], saved["learned_selector_metrics"]

    identity = content_fingerprint((signature, scenario, seed))
    data = make_transition_control(GENERATOR, scenario=scenario, seed=seed)
    encoder_path, evidence = folder / "encoder.json", folder / "native-training.json"
    readout = _fit_encoder(
        learner, data.neural[:READOUT_TRIALS], data.behavior[:READOUT_TRIALS], encoder_path, identity
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
        content_fingerprint((identity, training, windows, method, GENERATOR.fps)),
    )
    learned = votes.mean(axis=0) >= CONSENSUS
    config = MechanismRolloutConfig(GENERATOR.horizon, GENERATOR.target_length, profile.rollouts, seed)
    grid = ForecastGrid.from_data(data, readout)
    effects = EffectGrid.from_data(data, grid, policy, config)
    base_row = {"scenario": scenario, "seed": seed, "origins": grid.origins.size}
    rows = []
    variants = {}
    learned_selector = {}
    for name, mask in _support_masks(learned, seed, profile.shuffled_masks).items():
        pooled = fit_transition_mechanism(training, mask, RIDGE_ALPHA)
        base, local = _paired_forecasts(grid, pooled, readout, config)
        chosen = fast_refit_decisions(
            grid.origins,
            base,
            local,
            grid.decoded_targets,
            horizon=config.horizon,
            lookback=policy.lookback,
            margin=policy.margin,
            consecutive_losses=policy.consecutive_losses,
            recovery_forecasts=policy.recovery_forecasts,
        )
        prediction = np.where(chosen[..., None], local, base)
        vectors, effect_scores = effects.score(chosen, prediction, pooled, readout, config)
        if name == "learned":
            learned_selector = score_choices(
                {
                    **base_row,
                    "origin_samples": grid.points,
                    "pooled": base,
                    "local": local,
                    "behavior_targets": grid.targets,
                    "scoring_switch_samples": grid.switches,
                },
                chosen,
            )
        row = {
            "arm": name,
            **base_row,
            **grid.errors(prediction),
            **effect_scores,
            "effect_origins": len(effects.index),
            "support_edges": int(mask.sum()),
            "learned_overlap_edges": int((mask & learned).sum()),
            "local_fraction": float(chosen.mean()),
        }
        rows.append(row)
        variants[name] = {
            "metrics": row,
            "pooled_fit": asdict(pooled),
            "pooled_forecast": base,
            "local_forecast": local,
            "selected_local": chosen,
            "effects": vectors,
        }
        logger.info("%s saved scores for %s", folder.name, name)

    groups = np.repeat(np.arange(READOUT_TRIALS, POOLED_STOP), data.origins.shape[1])
    direct, direct_fit = fit_history_baseline(
        training,
        groups,
        grid.neural,
        grid.points,
        readout,
        history=HISTORY,
        horizon=config.horizon,
        target_length=config.target_length,
        stride=STRIDE,
    )
    if direct.shape != grid.targets.shape or not np.isfinite(direct).all():
        raise ValueError("direct forecasts must match the held-out behavior grid")
    rows = [
        rows[0],
        rows[1],
        _replicate_mean(rows[2:], "shuffled"),
        {
            "arm": "direct_history",
            **base_row,
            **grid.errors(direct),
            "all_effect_mae": None,
            "active_effect_mae": None,
            "effect_origins": None,
            "support_edges": None,
            "learned_overlap_edges": None,
            "local_fraction": None,
        },
    ]
    payload = {
        "signature": signature,
        "scenario": scenario,
        "seed": seed,
        "rows": rows,
        "learned_selector_metrics": learned_selector,
        "artifacts": {artifact.name: _digest(artifact) for artifact in (encoder_path, evidence)},
        "origin_samples": grid.points,
        "decoded_targets": grid.decoded_targets,
        "behavior_targets": grid.targets,
        "scoring_switch_samples": grid.switches,
        "effect_index": effects.index,
        "effect_innovation_ids": effects.innovation_ids,
        "effect_crossings_excluded": effects.crossings_excluded,
        "effect_units": distributed_interventions(),
        "oracle_effects": effects.truth,
        "oracle_active": effects.active,
        "generative": variants,
        "direct_predictions": direct,
        "direct_fit": direct_fit,
    }
    _write_json(path, payload)
    return rows, learned_selector


def _run(
    root: Path,
    name: str,
    workers: int,
    learner: BehaviorLearningConfig,
    method: ConnectivityConfig,
) -> dict[str, Any]:
    settings = OmegaConf.load(CONFIG)
    profile = Profile(**settings.profiles[name])
    policy = Policy(**settings.policy)
    test = {
        "forecast_margin": float(settings.forecast_margin),
        "effect_margin": float(settings.effect_margin),
        "level": float(settings.level),
    }
    protocol = {
        "profile": name,
        "settings": asdict(profile),
        "policy": asdict(policy),
        "scenarios": TRANSITION_SCENARIOS,
        "generator": asdict(GENERATOR),
        "learner": asdict(learner),
        "method": asdict(method),
        "history": HISTORY,
        "stride": STRIDE,
        "early": EARLY,
        "support_consensus": CONSENSUS,
        "ridge_alpha": RIDGE_ALPHA,
        "effect_floor": ACTIVE_FLOOR,
        "decision": test,
        "partitions": {"readout_stop": READOUT_TRIALS, "training_stop": POOLED_STOP},
        "implementation": implementation_fingerprint(REPO),
        "source": {
            relative: _digest(REPO / relative)
            for relative in (
                "conf/behavior_rollout_pipeline/support.yaml",
                "synthetic/behavior_rollout_support.py",
                "synthetic/behavior_rollout_online.py",
                "synthetic/behavior_rollout_pipeline.py",
                "synthetic/effectome_rollout.py",
                "tools/behavior-encoder/worker.py",
            )
        },
        "encoder_runtime": behavior_runtime_info(learner.runtime_python),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{package: version(package) for package in ("numpy", "scikit-learn", "scipy", "fastcgc")},
        },
    }
    signature = content_fingerprint(protocol)
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        plan = root / "support-protocol.json"
        if plan.is_file() and json.loads(plan.read_text())["signature"] != signature:
            raise ValueError("output root belongs to another support protocol")
        _write_json(plan, {"signature": signature, **protocol})
        cases = {
            (scenario, seed): root / "cases" / f"{scenario}-{seed}"
            for seed, scenario in product(profile.seeds, TRANSITION_SCENARIOS)
        }
        report: dict[str, Any] = {
            "profile": name,
            "signature": signature,
            "planned_cases": len(cases),
            "completed_cases": 0,
            "execution_complete": False,
            "support": None,
            "learned_selector": None,
        }
        _write_json(root / REPORT, report)
        rows = []
        selector_rows = []
        with (
            _inner_threads(),
            ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool,
        ):
            jobs = [
                pool.submit(_case, scenario, seed, folder, signature, method, learner, policy, profile)
                for (scenario, seed), folder in cases.items()
            ]
            for future in as_completed(jobs):
                case_rows, selector = future.result()
                rows.extend(case_rows)
                selector_rows.append(selector)
                report["completed_cases"] += 1
                _write_json(root / REPORT, report)
                logger.info(
                    "Support comparisons complete for %s/%s cases",
                    report["completed_cases"],
                    len(cases),
                )
        rows.sort(key=lambda row: (row["seed"], row["scenario"], row["arm"]))
        temporary = root / "seed-metrics.csv.tmp"
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / "seed-metrics.csv")
        if name == "confirmation":
            report["support"] = support_decision(rows, profile.seeds, **test)
            report["learned_selector"] = online_decision(
                sorted(selector_rows, key=lambda row: (row["seed"], row["scenario"])),
                profile.seeds,
                margin=GAIN_MARGIN,
                level=test["level"],
            )
        report["execution_complete"] = True
        _write_json(root / REPORT, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--profile", choices=("smoke", "confirmation"), default="smoke")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--encoder-python")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = args.output_root or Path("outputs/effectome-rollout-support") / args.profile / "support-01"
    if args.status:
        path = root / REPORT
        if path.is_file():
            print(path.read_text())
        else:
            print(json.dumps({"status": "not_started", "report": str(path)}))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _, methods = load_cross_method_profile("csl_sanity")
    report = _run(
        root,
        args.profile,
        args.workers,
        BehaviorLearningConfig(runtime_python=args.encoder_python),
        methods["fast_cgc"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
