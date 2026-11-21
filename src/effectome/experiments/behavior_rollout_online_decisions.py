"""Prespecified seed-level tests for a causal mechanism-selection rule."""

from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Any

from .behavior_rollout_timing_decisions import _axis, _component
from .behavior_transition_controls import TRANSITION_SCENARIOS


def online_decision(
    rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    *,
    margin: float = 0.005,
    level: float = 0.05,
) -> dict[str, Any]:
    """Require both driver gains and noninferiority in both controls.

    Every gate counts independent seeds strictly above its margin. The
    intersection-union test uses the largest component p-value. Missing
    evidence prevents a positive joint decision.
    """
    expected = set(_axis(seeds, "seeds", 0))
    if not isfinite(margin) or margin < 0 or not 0 < level < 1:
        raise ValueError("margin must be finite and nonnegative and level must lie in (0, 1)")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scenario = row.get("scenario")
        if scenario not in TRANSITION_SCENARIOS:
            raise ValueError("row lies outside the declared scenario grid")
        grouped[scenario].append(row)

    gates = (
        ("driver_vs_pooled", "driver_switch", "pooled_gain", margin),
        ("driver_early_vs_local", "driver_switch", "early_local_gain", margin),
        ("constant_noninferiority", "constant", "pooled_gain", -margin),
        ("nuisance_switch_noninferiority", "nuisance_switch", "pooled_gain", -margin),
    )
    tests = {}
    for name, scenario, metric, threshold in gates:
        gains = [{"seed": row.get("seed"), "forecast_gain": row.get(metric)} for row in grouped[scenario]]
        gate = _component(gains, expected, threshold, driver_switch=True)
        tests[name] = {
            "scenario": scenario,
            "metric": metric,
            "threshold_mae": threshold,
            **gate,
            "passed": gate["complete"] and gate["p_value"] <= level,
        }

    complete = all(test["complete"] for test in tests.values())
    joint_p = max(test["p_value"] if test["complete"] else 1.0 for test in tests.values())
    return {
        "margin_mae": margin,
        "level": level,
        "complete": complete,
        "joint_p_value": joint_p,
        "passed": complete and joint_p <= level,
        "tests": tests,
    }
