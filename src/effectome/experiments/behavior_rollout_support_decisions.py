"""Paired seed-level decisions for the fixed-policy graph support comparison."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from math import isfinite
from numbers import Integral, Real
from typing import Any

from .behavior_rollout_timing_decisions import _axis, _component
from .behavior_transition_controls import TRANSITION_SCENARIOS
from .framework_validation_analysis import _holm

_COMPARATORS = ("dense", "shuffled")
_ARMS = ("learned", *_COMPARATORS, "direct_history")


def _mae(value: Any) -> float | None:
    if not isinstance(value, Real) or isinstance(value, bool):
        return None
    mae = float(value)
    return mae if isfinite(mae) and mae >= 0 else None


def _paired_gains(
    cells: dict[tuple[str, str, int], list[dict[str, Any]]],
    seeds: tuple[int, ...],
    comparator: str,
    scenario: str,
    metric: str,
) -> list[dict[str, Any]]:
    gains = []
    for seed in seeds:
        learned = cells.get(("learned", scenario, seed), [])
        reference = cells.get((comparator, scenario, seed), [])
        if len(learned) == len(reference) == 1:
            left, right = _mae(learned[0].get(metric)), _mae(reference[0].get(metric))
            if left is not None and right is not None:
                gains.append({"seed": seed, "forecast_gain": right - left})
    return gains


def support_decision(
    rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    *,
    forecast_margin: float = 0.005,
    effect_margin: float = 0.005,
    level: float = 0.05,
) -> dict[str, Any]:
    """Evaluate learned support against dense and shuffled support on matched seeds.

    Positive gains indicate lower MAE with learned support. The forecast claim
    requires practical gains under driver switching and noninferiority in each
    control. The effect claim requires practical gains for oracle-active
    effects under driver switching. Each claim uses an intersection of exact
    sign tests with Holm adjustment across the two claims.
    """
    seeds = _axis(seeds, "seeds", 0)
    expected = set(seeds)
    if any(_mae(value) is None for value in (forecast_margin, effect_margin)):
        raise ValueError("forecast and effect margins must be finite and nonnegative")
    if not 0 < level < 1:
        raise ValueError("level must lie in (0, 1)")

    cells: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        arm, scenario, seed = row.get("arm"), row.get("scenario"), row.get("seed")
        if arm not in _ARMS or scenario not in TRANSITION_SCENARIOS:
            raise ValueError("row lies outside the declared arm and scenario grid")
        if not isinstance(seed, Integral) or isinstance(seed, bool) or seed not in expected:
            raise ValueError("row lies outside the declared seed grid")
        cells[arm, scenario, int(seed)].append(row)

    endpoints: dict[str, dict[str, Any]] = {}
    families = (
        ("forecast", "forecast_mae", TRANSITION_SCENARIOS, forecast_margin),
        ("active_effect", "active_effect_mae", ("driver_switch",), effect_margin),
    )
    for endpoint, metric, scenarios, margin in families:
        tests = {}
        for comparator, scenario in product(_COMPARATORS, scenarios):
            threshold = margin if scenario == "driver_switch" else -margin
            gains = _paired_gains(cells, seeds, comparator, scenario, metric)
            tests[f"{scenario}_vs_{comparator}"] = {
                "comparator": comparator,
                "scenario": scenario,
                "metric": metric,
                "threshold_mae": threshold,
                **_component(gains, expected, threshold, driver_switch=True),
            }
        complete = all(test["complete"] for test in tests.values())
        endpoints[endpoint] = {
            "complete": complete,
            "joint_p_value": max(test["p_value"] for test in tests.values()) if complete else None,
            "tests": tests,
        }

    # Keep incomplete endpoints in the prespecified family without testing them.
    adjusted = _holm([end["joint_p_value"] if end["complete"] else 1.0 for end in endpoints.values()])
    for end, p_value in zip(endpoints.values(), adjusted, strict=True):
        end["holm_p_value"] = p_value
        end["passed"] = end["complete"] and p_value <= level

    descriptive = {}
    for scenario, comparator in product(TRANSITION_SCENARIOS, _ARMS[1:]):
        metrics = (
            ("forecast_mae", "early_forecast_mae")
            if comparator == "direct_history"
            else ("early_forecast_mae",)
        )
        for metric in metrics:
            gains = _paired_gains(cells, seeds, comparator, scenario, metric)
            descriptive[f"{scenario}_vs_{comparator}_{metric}"] = {
                "scenario": scenario,
                "comparator": comparator,
                "metric": metric,
                "seed_count": len(expected),
                "available_seed_count": len(gains),
                "mean_gain": sum(pair["forecast_gain"] for pair in gains) / len(gains) if gains else None,
            }
    return {
        "forecast_margin": forecast_margin,
        "effect_margin": effect_margin,
        "level": level,
        "family_size": len(endpoints),
        "complete": all(end["complete"] for end in endpoints.values()),
        "passed": any(end["passed"] for end in endpoints.values()),
        "endpoints": endpoints,
        "descriptive": descriptive,
    }
