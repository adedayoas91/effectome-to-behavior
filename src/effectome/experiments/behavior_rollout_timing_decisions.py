"""Seed-level practical gain tests across a fixed rollout timing grid."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
from math import isfinite
from numbers import Integral, Real
from typing import Any

from scipy.stats import binomtest

from .behavior_transition_controls import TRANSITION_SCENARIOS
from .framework_validation_analysis import _holm


def _axis(values: tuple[int, ...], name: str, minimum: int) -> tuple[int, ...]:
    if (
        not values
        or any(isinstance(x, bool) or not isinstance(x, Integral) or x < minimum for x in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain distinct integers of at least {minimum}")
    return tuple(sorted(map(int, values)))


def _component(
    rows: list[dict[str, Any]], expected: set[int], margin: float, driver_switch: bool
) -> dict[str, Any]:
    counts = Counter(row["seed"] for row in rows)
    gains = []
    for row in rows:
        seed, gain = row["seed"], row["forecast_gain"]
        if (
            seed in expected
            and not isinstance(seed, bool)
            and isinstance(seed, Integral)
            and counts[seed] == 1
            and isinstance(gain, Real)
            and not isinstance(gain, bool)
            and isfinite(gain)
        ):
            gains.append(float(gain))
    complete = len(gains) == len(rows) == len(expected)
    hits = sum(gain > margin if driver_switch else gain < margin for gain in gains)
    return {
        "complete": complete,
        "seed_count": len(expected),
        "available_seed_count": len(gains),
        "observed_row_count": len(rows),
        "mean_gain": sum(gains) / len(gains) if gains else None,
        "gains_on_expected_side": hits,
        "p_value": float(binomtest(hits, len(gains), 0.5, alternative="greater").pvalue)
        if complete
        else None,
    }


def timing_decision(
    rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    histories: tuple[int, ...],
    delays: tuple[int, ...],
    *,
    margin: float = 0.005,
    level: float = 0.05,
) -> dict[str, Any]:
    """Test when fixed-support local refitting beats pooling after a switch.

    Each row holds one paired forecast gain for an independent process seed.
    The driver and two control sign tests form an intersection-union test at
    each history-delay cell. Holm correction covers the entire declared grid,
    with p=1 for incomplete cells. Timing claims require complete evidence.
    """
    expected = set(_axis(seeds, "seeds", 0))
    histories = _axis(histories, "histories", 1)
    delays = _axis(delays, "delays", 0)
    if not isfinite(margin) or margin < 0 or not 0 < level < 1:
        raise ValueError("margin must be finite and nonnegative and level must lie in (0, 1)")

    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        h, d, scenario = row["history_samples"], row["delay_samples"], row["scenario"]
        if h not in histories or d not in delays or scenario not in TRANSITION_SCENARIOS:
            raise ValueError("row lies outside the declared history, delay and scenario grid")
        grouped[h, d, scenario].append(row)

    tests: list[dict[str, Any]] = []
    for h, d in product(histories, delays):
        scenarios = {
            scenario: _component(grouped[h, d, scenario], expected, margin, scenario == "driver_switch")
            for scenario in TRANSITION_SCENARIOS
        }
        complete = all(gate["complete"] for gate in scenarios.values())
        joint_p = max(gate["p_value"] if gate["complete"] else 1.0 for gate in scenarios.values())
        tests.append(
            {
                "history_samples": h,
                "delay_samples": d,
                "complete": complete,
                "joint_p_value": joint_p,
                "scenarios": scenarios,
            }
        )

    complete = all(test["complete"] for test in tests)
    for test, adjusted in zip(tests, _holm([test["joint_p_value"] for test in tests]), strict=True):
        test["holm_p_value"] = adjusted
        test["passed"] = complete and adjusted <= level

    earliest = {}
    sustained = {}
    for h in histories:
        profile = [test["passed"] for test in tests if test["history_samples"] == h]
        earliest[str(h)] = next(
            (delay for delay, passed in zip(delays, profile, strict=True) if passed), None
        )
        sustained[str(h)] = next((delay for i, delay in enumerate(delays) if all(profile[i:])), None)
    return {
        "margin_mae": margin,
        "level": level,
        "family_size": len(tests),
        "complete": complete,
        "passed": any(test["passed"] for test in tests),
        "earliest_supported_delay_samples": earliest,
        "earliest_sustained_delay_samples": sustained,
        "tests": tests,
    }
