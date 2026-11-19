"""Verify forecast gain timing decisions on hand-checkable independent seeds."""

from itertools import product

import pytest

from effectome.experiments.behavior_rollout_timing_decisions import timing_decision
from effectome.experiments.behavior_transition_controls import TRANSITION_SCENARIOS


def _rows(seeds, histories, delays):
    return [
        {
            "seed": seed,
            "scenario": scenario,
            "history_samples": history,
            "delay_samples": delay,
            "forecast_gain": 0.02 if scenario == "driver_switch" else 0.0,
        }
        for seed, history, delay, scenario in product(seeds, histories, delays, TRANSITION_SCENARIOS)
    ]


def test_ties_counts_and_joint_p_value_are_exact():
    seeds, histories, delays = tuple(range(10)), (256,), (0, 8)
    rows = _rows(seeds, histories, delays)
    for row in rows:
        if row["scenario"] == "constant" and row["seed"] == 0 and row["delay_samples"] == 0:
            row["forecast_gain"] = 0.005
        if row["scenario"] == "nuisance_switch" and row["delay_samples"] == 8:
            row["forecast_gain"] = 0.006

    result = timing_decision(rows, seeds, histories, delays)
    first, second = result["tests"]
    assert first["scenarios"]["constant"]["gains_on_expected_side"] == 9
    assert first["scenarios"]["driver_switch"]["p_value"] == 1 / 1024
    assert first["joint_p_value"] == 11 / 1024
    assert first["holm_p_value"] == 22 / 1024
    assert first["passed"]
    assert second["joint_p_value"] == 1
    assert not second["passed"]
    assert result["earliest_supported_delay_samples"] == {"256": 0}
    assert result["earliest_sustained_delay_samples"] == {"256": None}


def test_holm_adjusts_over_both_history_lengths():
    seeds, histories, delays = tuple(range(5)), (256, 1024), (0,)
    result = timing_decision(_rows(seeds, histories, delays), seeds, histories, delays)
    assert result["complete"]
    assert result["family_size"] == 2
    assert [test["joint_p_value"] for test in result["tests"]] == [1 / 32, 1 / 32]
    assert [test["holm_p_value"] for test in result["tests"]] == [2 / 32, 2 / 32]
    assert not result["passed"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nonfinite", "unexpected seed"])
def test_incomplete_families_keep_all_tests_and_block_timing_claims(defect):
    seeds, histories, delays = tuple(range(10)), (256, 1024), (0,)
    rows = _rows(seeds, histories, delays)
    if defect == "missing":
        rows.pop(0)
    elif defect == "duplicate":
        rows.append(rows[0].copy())
    elif defect == "nonfinite":
        rows[0]["forecast_gain"] = float("nan")
    else:
        rows[0]["seed"] = 200

    result = timing_decision(rows, seeds, histories, delays)
    incomplete, available = result["tests"]
    assert result["family_size"] == 2
    assert not result["complete"]
    assert incomplete["joint_p_value"] == 1
    assert incomplete["scenarios"]["driver_switch"]["available_seed_count"] == 9
    assert incomplete["scenarios"]["driver_switch"]["p_value"] is None
    assert available["complete"]
    assert available["holm_p_value"] == 2 / 1024
    assert not result["passed"]
    assert result["earliest_supported_delay_samples"] == {"256": None, "1024": None}


def test_earliest_sustained_delay_can_follow_an_isolated_pass():
    seeds, histories, delays = tuple(range(10)), (1024,), (32, 0, 16, 8)
    rows = _rows(seeds, histories, delays)
    for row in rows:
        if row["scenario"] == "driver_switch" and row["delay_samples"] == 8:
            row["forecast_gain"] = 0.005
    result = timing_decision(rows, seeds, histories, delays)
    assert [test["delay_samples"] for test in result["tests"]] == [0, 8, 16, 32]
    assert [test["passed"] for test in result["tests"]] == [True, False, True, True]
    assert result["earliest_supported_delay_samples"] == {"1024": 0}
    assert result["earliest_sustained_delay_samples"] == {"1024": 16}
