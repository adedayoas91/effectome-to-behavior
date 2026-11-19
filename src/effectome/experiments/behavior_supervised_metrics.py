"""Seed-level continuous effects and prospective, separate-null decisions."""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal
from typing import Any

import numpy as np
from scipy.stats import beta

from .behavior_supervised_config import BehaviorDecisionConfig


def decision_metrics(metrics: dict, config: BehaviorDecisionConfig) -> dict:
    """Aggregate magnitudes after taking the norm at each factual origin."""
    units = metrics["per_unit"]
    scores = {unit: float(np.mean(row["estimated_effect_magnitudes"])) for unit, row in units.items()}
    gold = {unit: float(np.mean(row["gold_effect_magnitudes"])) for unit, row in units.items()}
    active = [unit for unit, value in gold.items() if value >= config.gold_effect_floor]
    return {
        "candidate_scores": scores,
        "gold_scores": gold,
        "eligible_units": active,
        "positive_unit_mae": float(np.mean([units[unit]["end_to_end_absolute_error"] for unit in active]))
        if active
        else None,
        "positive_unit_semantic_mae": float(
            np.mean([units[unit]["semantic_absolute_error"] for unit in active])
        )
        if active
        else None,
    }


def nominations(summary: dict, threshold: float) -> dict:
    chosen = {unit for unit, score in summary["candidate_scores"].items() if score > threshold}
    gold = set(summary["eligible_units"])
    output: dict[str, Any] = {
        "threshold": threshold,
        "nominated_units": sorted(chosen),
        "false_nominated_units": sorted(chosen - gold),
    }
    for kind in ("neuron", "group"):
        positive = {unit for unit in gold if unit.startswith(f"{kind}:")}
        selected = {unit for unit in chosen if unit.startswith(f"{kind}:")}
        correct = len(positive & selected)
        output[kind] = {
            "gold_count": len(positive),
            "nominee_count": len(selected),
            "recall": correct / len(positive) if positive else None,
            "precision": correct / len(selected) if selected else (0.0 if positive else None),
        }
    return output


def calibrate_behavior_scores(
    cells: list[dict],
    methods: list[str],
    arms: list[str],
    seeds: tuple[int, ...],
    config: BehaviorDecisionConfig,
) -> dict:
    """Freeze strict exceedance cutoffs from one full null maximum per seed."""
    observed = {cell["seed"]: cell for cell in cells}
    if (
        len(cells) != len(observed)
        or set(observed) != set(seeds)
        or any(cell["phase"] != "calibration" or cell["scenario"] != "null" for cell in cells)
    ):
        raise ValueError("calibration requires exactly the declared separate null seeds")
    rank = int(
        (Decimal(len(seeds) + 1) * (1 - Decimal(str(config.calibration_alpha)))).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    if rank > len(seeds):
        raise ValueError("calibration seed count does not resolve the requested rank")
    thresholds: dict[str, dict[str, Any]] = {}
    for method in methods:
        thresholds[method] = {}
        for arm in arms:
            maxima = {}
            candidates = None
            for seed in sorted(seeds):
                result = observed[seed]["methods"].get(method, {})
                if result.get("status") != "available" or arm not in result["metrics"]:
                    raise ValueError(f"calibration evidence unavailable for seed {seed}, {method}/{arm}")
                scores = result["metrics"][arm]["candidate_scores"]
                if not scores or not all(np.isfinite(value) and value >= 0 for value in scores.values()):
                    raise ValueError("calibration magnitudes must be finite and nonnegative")
                if candidates is not None and set(scores) != candidates:
                    raise ValueError("calibration candidates must match across seeds")
                candidates = set(scores)
                maxima[str(seed)] = max(scores.values())
            empirical = sorted(maxima.values())[rank - 1]
            thresholds[method][arm] = {
                "cutoff": max(config.nomination_floor, empirical),
                "empirical_cutoff": empirical,
                "seed_maxima": maxima,
            }
    return {
        "rank": rank,
        "alpha": config.calibration_alpha,
        "calibration_seeds": list(seeds),
        "nomination_floor": config.nomination_floor,
        "comparison": "strictly_greater_than",
        "global_null_marginal_tail_bound": (len(seeds) + 1 - rank) / (len(seeds) + 1),
        "scope": "whole_seed_global_null_per_dynamics_and_readout",
        "thresholds": thresholds,
    }


def assess_confirmation(
    cells: list[dict],
    calibration: dict,
    config: BehaviorDecisionConfig,
    methods: list[str],
    positive_seeds: tuple[int, ...],
    null_seeds: tuple[int, ...],
) -> dict:
    """Keep group and single-neuron eligibility visible at their own denominators."""
    positive = [cell for cell in cells if cell["phase"] == "positive"]
    null = [cell for cell in cells if cell["phase"] == "null-test"]
    arm = config.primary_readout
    table = {}
    for method in methods:
        cutoff = calibration["thresholds"][method][arm]["cutoff"]
        available = [c for c in positive if c["methods"].get(method, {}).get("status") == "available"]
        null_available = [c for c in null if c["methods"].get(method, {}).get("status") == "available"]
        rows = [c["methods"][method]["metrics"][arm] for c in available]
        positive_nom = [nominations(row, cutoff) for row in rows]
        null_nom = [nominations(c["methods"][method]["metrics"][arm], cutoff) for c in null_available]
        complete_positive = len(available) == len(positive_seeds) >= config.min_positive_seeds and {
            c["seed"] for c in available
        } == set(positive_seeds)
        complete_null = len(null_available) == len(null_seeds) >= config.min_null_seeds and {
            c["seed"] for c in null_available
        } == set(null_seeds)
        false_count, n = sum(bool(row["nominated_units"]) for row in null_nom), len(null_nom)
        upper = (
            (
                1.0
                if false_count == n
                else float(beta.ppf(config.confidence_level, false_count + 1, n - false_count))
            )
            if complete_null
            else None
        )

        def mean(values: list) -> float | None:
            present = [value for value in values if value is not None]
            return float(np.mean(present)) if present else None

        behavioral = mean([row["end_to_end_mae"] for row in rows])
        semantic = mean([row["semantic_mae"] for row in rows])
        active = mean([row["positive_unit_mae"] for row in rows])
        origin = mean([c["natural_evaluation"][arm]["origin_mae"] for c in available])
        gates = {
            "complete_positive": complete_positive,
            "complete_null": complete_null,
            "behavioral_mae": behavioral is not None and behavioral <= config.max_behavioral_mae,
            "positive_unit_mae": active is not None and active <= config.max_positive_unit_mae,
            "semantic_mae": semantic is not None and semantic <= config.max_semantic_mae,
            "origin_decoding": origin is not None and origin <= config.max_origin_decoding_mae,
            "noncollapsed_behavior": bool(rows)
            and all(
                not row["rollout_collapsed"] and not c["natural_evaluation"][arm]["collapsed"]
                for row, c in zip(rows, available, strict=True)
            ),
            "null_specificity": upper is not None and upper <= config.max_null_probability,
        }
        recovery = {}
        for kind in ("neuron", "group"):
            recoveries = [row[kind] for row in positive_nom if row[kind]["gold_count"] > 0]
            recall = mean([row["recall"] for row in recoveries])
            precision = mean([row["precision"] for row in recoveries])
            recovery[kind] = {
                "eligible_seeds": len(recoveries),
                "mean_recall": recall,
                "mean_precision": precision,
            }
            gates[f"{kind}_power"] = (
                len(recoveries) == len(positive_seeds)
                and recall is not None
                and recall >= config.min_nomination_recall
                and precision is not None
                and precision >= config.min_nomination_precision
            )
        table[method] = {
            "passed": all(gates.values()),
            "gates": gates,
            "cutoff": cutoff,
            "behavioral_mae": behavioral,
            "semantic_mae": semantic,
            "origin_mae": origin,
            "positive_unit_mae": active,
            "available_positive_seeds": len(available),
            "available_null_seeds": n,
            "null_false_seeds": false_count,
            "null_exact_upper": upper,
            "recovery": recovery,
            "nominations": {str(c["seed"]): row for c, row in zip(available, positive_nom, strict=True)},
            "null_nominations": {
                str(c["seed"]): row for c, row in zip(null_available, null_nom, strict=True)
            },
        }
    return {"primary_readout": arm, "dynamics": table, "null_coverage": "per_dynamics_readout"}
