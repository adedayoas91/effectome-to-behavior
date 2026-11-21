"""Tune the online policy by replay, then test one frozen choice on fresh seeds."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any, cast

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from omegaconf import OmegaConf

from effectome.experiments.behavior_rollout_online_decisions import online_decision
from effectome.experiments.behavior_rollout_online_policy import (
    _matured_gains,
    _rolling_choice,
    fast_refit_decisions,
)
from effectome.experiments.behavior_supervised import _run_lock
from effectome.experiments.behavior_supervised_config import BehaviorLearningConfig
from effectome.experiments.behavior_transition_controls import TRANSITION_SCENARIOS
from effectome.experiments.manifold_validation import _digest, _verify_artifacts
from effectome.experiments.mechanism_validation import _write_json
from effectome.workflows.cross_method import load_cross_method_profile
from effectome.workflows.resume import content_fingerprint
from synthetic.behavior_rollout_online import GENERATOR, score_choices
from synthetic.behavior_rollout_pipeline import REPO, Policy, _effect_metrics, _run
from synthetic.effectome_rollout import ACTIVE_FLOOR, GAIN_MARGIN, LEVEL, POOLED_STOP

CONFIG = REPO / "conf/behavior_rollout_pipeline/tuning.yaml"
FIXED = REPO / "conf/behavior_rollout_pipeline/fixed.yaml"
DEFAULT_ROOT = Path("outputs/effectome-rollout-tuning/tuning-01")
logger = logging.getLogger(__name__)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _seeds(config: Any) -> tuple[int, ...]:
    return tuple(range(config.seed_start, config.seed_start + config.seed_count))


def _load_case(folder: Path, scenario: str, seed: int, signature: str) -> dict[str, Any]:
    case = _read(folder / "case.json")
    if (case["scenario"], case["seed"], case["signature"]) != (scenario, seed, signature):
        raise ValueError(f"forecast case has the wrong identity: {folder}")
    _verify_artifacts(folder, case["artifacts"])
    for field in ("pooled", "local", "decoded_observations", "behavior_targets", "scoring_switch_samples"):
        case[field] = np.asarray(case[field])
    return case


def _load_effects(folder: Path, case: dict[str, Any], signature: str) -> dict[str, Any]:
    result = _read(folder / "pipeline-case.json")
    identity = (case["scenario"], case["seed"], signature)
    if (result["scenario"], result["seed"], result["signature"]) != identity:
        raise ValueError(f"intervention case has the wrong identity: {folder}")
    _verify_artifacts(folder, result["artifacts"])
    return result


@dataclass(frozen=True)
class EffectReplay:
    trial: np.ndarray
    point: np.ndarray
    pooled: np.ndarray
    local: np.ndarray
    truth: np.ndarray
    active: np.ndarray

    @classmethod
    def from_case(cls, case: dict[str, Any], result: dict[str, Any]) -> EffectReplay:
        effects = result["effects"]
        origin = np.asarray(effects["origin"])
        points = np.asarray(case["origin_samples"])
        point = np.searchsorted(points, origin)
        if np.any(point == len(points)) or not np.array_equal(points[point], origin):
            raise ValueError("intervention origins do not match the forecast grid")
        trial = np.asarray(effects["trial"]) - POOLED_STOP
        if np.any((trial < 0) | (trial >= len(case["pooled"]))):
            raise ValueError("intervention trials do not match the forecast grid")
        values = effects["vectors"]
        pooled, local, truth = (np.asarray(values[name]) for name in ("pooled", "local", "oracle"))
        active = np.linalg.norm(truth, axis=-1) / np.sqrt(truth.shape[-1]) >= ACTIVE_FLOOR
        return cls(trial, point, pooled, local, truth, active)

    def score(self, choice: np.ndarray) -> dict[str, Any]:
        selected = np.where(choice[self.trial, self.point, None, None], self.local, self.pooled)
        return _effect_metrics(selected, self.truth, self.active)


def _inputs(case: dict[str, Any]) -> tuple[np.ndarray, ...]:
    return (
        np.broadcast_to(case["origin_samples"], case["pooled"].shape[:2]),
        case["pooled"],
        case["local"],
        case["decoded_observations"],
    )


def _replay(case: dict[str, Any], policy: Policy) -> np.ndarray:
    return fast_refit_decisions(
        *_inputs(case),
        horizon=GENERATOR.horizon,
        lookback=policy.lookback,
        margin=policy.margin,
        consecutive_losses=policy.consecutive_losses,
        recovery_forecasts=policy.recovery_forecasts,
    )


def _grid_masks(case: dict[str, Any], policies: list[Policy]) -> Iterator[np.ndarray]:
    gains, matured = _matured_gains(*_inputs(case), horizon=GENERATOR.horizon)
    event = np.arange(gains.shape[1])

    def latest(condition: np.ndarray, width: int) -> np.ndarray:
        # Window-triggered events first exist at index width - 1.
        padded = np.pad(condition, ((0, 0), (width - 1, 0)), constant_values=False)
        history = np.maximum.accumulate(np.where(padded, event, -1), axis=1)
        return np.take_along_axis(history, np.maximum(matured - 1, 0), axis=1)

    lookbacks = {(policy.lookback, policy.margin) for policy in policies}
    fallback_counts = {policy.consecutive_losses for policy in policies}
    recovery_keys = {(policy.recovery_forecasts, policy.margin) for policy in policies}
    unavailable = np.full_like(matured, -1)
    rolling = {key: _rolling_choice(gains, matured, *key) for key in lookbacks}
    fallbacks = {
        width: (
            latest(sliding_window_view(gains < 0, width, axis=1).all(axis=-1), width)
            if width <= len(event) else unavailable
        )
        for width in fallback_counts
    }
    recoveries = {
        (width, margin): (
            latest(sliding_window_view(gains, width, axis=1).mean(axis=-1) > margin, width)
            if width <= len(event) else unavailable
        )
        for width, margin in recovery_keys
    }
    for policy in policies:
        last_loss = fallbacks[policy.consecutive_losses]
        recovery = recoveries[policy.recovery_forecasts, policy.margin]
        # A qualifying recovery window must begin strictly after the most recent fallback.
        unblocked = (last_loss < 0) | (recovery >= last_loss + policy.recovery_forecasts)
        yield rolling[policy.lookback, policy.margin] & unblocked


def _rank(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    def priority(candidate: dict[str, Any]) -> tuple[float, ...]:
        return (
            candidate["both_panels_pass"],
            0 if candidate["both_panels_pass"] else candidate["worst_seed_count"],
            -candidate["driver_mae"],
            candidate["early_gain"],
            -candidate["active_effect_mae"],
        )

    return max((row for row in candidates if row["effect_eligible"]), key=priority)


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _select(root: Path, config: Any, fixed: Policy) -> dict[str, Any]:
    development_seeds = [_seeds(panel) for panel in config.development]
    confirmation_seeds = _seeds(config.confirmation)
    protocol = {
        "grid_sha256": _digest(CONFIG),
        "fixed_sha256": _digest(FIXED),
        "source_sha256": _digest(Path(__file__)),
        "development_protocol_sha256": {
            panel.path: _digest(REPO / panel.path / panel.protocol) for panel in config.development
        },
        "development_seeds": development_seeds,
        "confirmation_seeds": confirmation_seeds,
        "margin_mae": GAIN_MARGIN,
        "level": LEVEL,
    }
    used = {seed for panel in development_seeds for seed in panel}
    if used.intersection(confirmation_seeds):
        raise ValueError("development and confirmation seeds overlap")
    signature = content_fingerprint(protocol)
    path = root / "frozen-policy.json"
    if path.is_file():
        frozen = _read(path)
        if frozen["selection_signature"] != signature:
            raise ValueError("output root contains a different frozen selection")
        return frozen
    _write_json(root / "tuning-protocol.json", {"signature": signature, **protocol})

    policies = [
        Policy(window, margin, loss, recovery, fixed.effect_stride_origins)
        for window, margin, loss, recovery in product(
            config.lookbacks, config.margins, config.consecutive_losses, config.recovery_forecasts
        )
    ]
    if fixed not in policies:
        raise ValueError("the grid must contain the fixed baseline")
    fixed_index = policies.index(fixed)
    panel_scores = []
    effect_scores: list[list[float]] = [[] for _ in policies]
    for panel in config.development:
        panel_path = REPO / panel.path
        panel_signature = _read(panel_path / panel.protocol)["signature"]
        scores: list[list[dict[str, Any]]] = [[] for _ in policies]
        for seed, scenario in product(_seeds(panel), TRANSITION_SCENARIOS):
            folder = panel_path / "cases" / f"{scenario}-{seed}"
            case = _load_case(folder, scenario, seed, panel_signature)
            effects = (
                EffectReplay.from_case(case, _load_effects(folder, case, panel_signature))
                if panel.protocol == "pipeline-protocol.json" and scenario == "driver_switch"
                else None
            )
            for index, mask in enumerate(_grid_masks(case, policies)):
                if index == fixed_index:
                    np.testing.assert_array_equal(mask, _replay(case, fixed))
                scores[index].append(score_choices(case, mask))
                if effects is not None:
                    effect_scores[index].append(effects.score(mask)["active_effect_mae"])
        panel_scores.append(scores)
        logger.info("Replayed %s policies on %s", len(policies), panel.path)

    baseline_effect = mean(effect_scores[fixed_index])
    candidates = []
    for index, policy in enumerate(policies):
        results = [scores[index] for scores in panel_scores]
        decisions = [
            online_decision(rows, _seeds(panel), margin=GAIN_MARGIN, level=LEVEL)
            for rows, panel in zip(results, config.development, strict=True)
        ]
        driver = [row for rows in results for row in rows if row["scenario"] == "driver_switch"]
        effect_mae = mean(effect_scores[index])
        candidates.append({
            **asdict(policy),
            "both_panels_pass": all(result["passed"] for result in decisions),
            "worst_seed_count": min(
                test["gains_on_expected_side"] for result in decisions for test in result["tests"].values()
            ),
            "driver_mae": mean(row["gated_mae"] for row in driver),
            "early_gain": mean(row["early_local_gain"] for row in driver),
            "active_effect_mae": effect_mae,
            "effect_eligible": effect_mae <= baseline_effect,
        })
    winner = _rank(candidates)
    parameters = {field: winner[field] for field in asdict(fixed)}
    selected = Policy(**parameters)
    for panel in config.development:
        panel_path = REPO / panel.path
        panel_signature = _read(panel_path / panel.protocol)["signature"]
        for seed, scenario in product(_seeds(panel), TRANSITION_SCENARIOS):
            folder = panel_path / "cases" / f"{scenario}-{seed}"
            case = _load_case(folder, scenario, seed, panel_signature)
            np.testing.assert_array_equal(next(_grid_masks(case, [selected])), _replay(case, selected))
    _write_table(root / "grid-metrics.csv", candidates)
    frozen = {
        "selection_signature": signature,
        "policy": parameters,
        "development": winner,
        "fixed_development": candidates[fixed_index],
        "policies_evaluated": len(candidates),
        "eligible_policies": sum(row["effect_eligible"] for row in candidates),
        "eligible_passing_both": sum(
            row["effect_eligible"] and row["both_panels_pass"] for row in candidates
        ),
    }
    _write_json(path, frozen)
    return frozen


def _confirm(
    root: Path, config: Any, frozen: dict[str, Any], workers: int, encoder_python: str | None
) -> dict:
    panel = root / "confirmation"
    seeds = _seeds(config.confirmation)
    baseline = panel / "pipeline-report.json"
    if not baseline.is_file() or not _read(baseline)["execution_complete"]:
        _, methods = load_cross_method_profile("csl_sanity")
        _run(
            panel, "confirmation", workers, BehaviorLearningConfig(runtime_python=encoder_python),
            methods["fast_cgc"], seeds=seeds,
        )
    protocol = _read(panel / "pipeline-protocol.json")
    fixed_config = OmegaConf.to_container(OmegaConf.load(FIXED))
    if tuple(protocol["seeds"]) != seeds or protocol["policy"] != fixed_config:
        raise ValueError("fresh confirmation bundle does not match the declared seeds and fixed baseline")
    policy = Policy(**frozen["policy"])
    rows = []
    for seed, scenario in product(seeds, TRANSITION_SCENARIOS):
        folder = panel / "cases" / f"{scenario}-{seed}"
        case = _load_case(folder, scenario, seed, protocol["signature"])
        result = _load_effects(folder, case, protocol["signature"])
        effects = EffectReplay.from_case(case, result)
        shape = case["pooled"].shape[:2]
        for name, mask in {
            "tuned": _replay(case, policy),
            "fixed_fallback": np.asarray(result["selection"]["fast"]),
            "previous_online": np.asarray(result["selection"]["slow"]),
            "always_pooled": np.zeros(shape, dtype=bool),
            "always_local": np.ones(shape, dtype=bool),
        }.items():
            rows.append({
                "policy": name,
                **score_choices(case, mask),
                **effects.score(mask),
                "effect_origins": len(effects.trial),
            })
    tests = {
        name: online_decision(
            [row for row in rows if row["policy"] == name], seeds, margin=GAIN_MARGIN, level=LEVEL
        )
        for name in ("tuned", "fixed_fallback", "previous_online")
    }
    if tests["fixed_fallback"] != _read(baseline)["confirmation"]:
        raise ValueError("fixed baseline could not be reproduced from the confirmation artifacts")
    _write_table(root / "confirmation-metrics.csv", rows)
    report = {
        "selection_signature": frozen["selection_signature"],
        "confirmation_signature": protocol["signature"],
        "policy": frozen["policy"],
        "cases": len(seeds) * len(TRANSITION_SCENARIOS),
        "seeds": seeds,
        "tests": tests,
    }
    _write_json(root / "tuning-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--select", action="store_true")
    actions.add_argument("--confirm", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--encoder-python")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    root = args.output_root
    if args.status:
        for path in ("frozen-policy.json", "confirmation/pipeline-report.json", "tuning-report.json"):
            file = root / path
            if file.is_file():
                print(file.read_text())
        return
    config = OmegaConf.load(CONFIG)
    fixed_config = OmegaConf.to_container(OmegaConf.load(FIXED))
    if not isinstance(fixed_config, dict) or any(not isinstance(key, str) for key in fixed_config):
        raise ValueError("fixed policy configuration must be a mapping")
    fixed = Policy(**cast(dict[str, Any], fixed_config))
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        frozen = _select(root, config, fixed)
        logger.info("Frozen policy %s", frozen["policy"])
        if args.confirm:
            report = _confirm(root, config, frozen, args.workers, args.encoder_python)
            logger.info("Tuned joint decision %s", report["tests"]["tuned"])


if __name__ == "__main__":
    main()
