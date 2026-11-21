"""Check support claims with hand-calculable paired seed panels."""

from copy import deepcopy
from itertools import product

import pytest

from effectome.experiments.behavior_rollout_support_decisions import support_decision
from effectome.experiments.behavior_transition_controls import TRANSITION_SCENARIOS


def _rows(seeds: tuple[int, ...]) -> list[dict]:
    return [
        {
            "arm": arm,
            "scenario": scenario,
            "seed": seed,
            "forecast_mae": 0.125 if arm == "learned" or scenario != "driver_switch" else 0.140,
            "early_forecast_mae": 0.16 if arm == "learned" else 0.20,
            "active_effect_mae": 0.06 if arm == "learned" else 0.08,
        }
        for arm, scenario, seed in product(("learned", "dense", "shuffled"), TRANSITION_SCENARIOS, seeds)
    ]


def test_complete_panel_passes_both_claims_without_the_direct_baseline():
    seeds = tuple(range(8))
    rows = _rows(seeds)
    original = deepcopy(rows)
    result = support_decision(rows, seeds)

    assert result["complete"] and result["passed"]
    assert rows == original
    endpoints = result["endpoints"]
    assert len(endpoints["forecast"]["tests"]) == 6
    assert len(endpoints["active_effect"]["tests"]) == 2
    for endpoint in endpoints.values():
        assert endpoint["joint_p_value"] == 1 / 256
        assert endpoint["holm_p_value"] == 2 / 256
        assert endpoint["passed"]
    tests = endpoints["forecast"]["tests"]
    assert tests["driver_switch_vs_dense"]["mean_gain"] == pytest.approx(0.015)
    assert tests["constant_vs_shuffled"]["threshold_mae"] == -0.005
    assert tests["constant_vs_shuffled"]["gains_on_expected_side"] == 8


def test_holm_prevents_positive_claims_from_a_small_perfect_panel():
    seeds = tuple(range(5))
    result = support_decision(_rows(seeds), seeds)

    assert result["complete"] and not result["passed"]
    for endpoint in result["endpoints"].values():
        assert endpoint["joint_p_value"] == 1 / 32
        assert endpoint["holm_p_value"] == 1 / 16
        assert not endpoint["passed"]


@pytest.mark.parametrize(("scenario", "comparator"), [("constant", "dense"), ("nuisance_switch", "shuffled")])
def test_comparator_control_can_block_the_forecast_claim(scenario, comparator):
    seeds = tuple(range(8))
    rows = _rows(seeds)
    for row in rows:
        if row["scenario"] == scenario and row["arm"] == comparator:
            row["forecast_mae"] = 0.10

    result = support_decision(rows, seeds)
    endpoints = result["endpoints"]
    assert endpoints["forecast"]["complete"]
    assert endpoints["forecast"]["joint_p_value"] == 1
    assert not endpoints["forecast"]["passed"]
    assert endpoints["active_effect"]["passed"]


def test_an_effect_gain_exactly_at_the_margin_does_not_count():
    seeds = tuple(range(8))
    rows = _rows(seeds)
    for row in rows:
        if row["scenario"] == "driver_switch" and row["arm"] in {"learned", "dense"}:
            row["active_effect_mae"] = 0.005 if row["arm"] == "dense" else 0.0

    result = support_decision(rows, seeds)
    tests = result["endpoints"]["active_effect"]["tests"]
    assert tests["driver_switch_vs_dense"]["gains_on_expected_side"] == 0
    assert tests["driver_switch_vs_shuffled"]["gains_on_expected_side"] == len(seeds)
    assert not result["endpoints"]["active_effect"]["passed"]
    assert result["endpoints"]["forecast"]["passed"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "unavailable", "nonfinite"])
def test_incomplete_effect_pairs_receive_no_p_value(defect):
    seeds = tuple(range(8))
    rows = _rows(seeds)
    index = next(
        i
        for i, row in enumerate(rows)
        if (row["arm"], row["scenario"], row["seed"]) == ("dense", "driver_switch", 0)
    )
    if defect == "missing":
        del rows[index]
    elif defect == "duplicate":
        rows.append(rows[index].copy())
    else:
        rows[index]["active_effect_mae"] = None if defect == "unavailable" else float("nan")

    result = support_decision(rows, seeds)
    effect = result["endpoints"]["active_effect"]
    assert not result["complete"]
    assert not effect["complete"] and not effect["passed"]
    assert effect["joint_p_value"] is None
    assert effect["holm_p_value"] == 1
    incomplete = effect["tests"]["driver_switch_vs_dense"]
    assert incomplete["available_seed_count"] == 7
    assert incomplete["p_value"] is None
    # Effect-only missingness does not block a complete forecast endpoint.
    assert result["endpoints"]["forecast"]["passed"] == (defect in {"unavailable", "nonfinite"})


def test_direct_history_comparison_is_descriptive():
    seeds = tuple(range(8))
    rows = _rows(seeds)
    rows.append(
        {
            "arm": "direct_history",
            "scenario": "driver_switch",
            "seed": 1,
            "forecast_mae": 0.120,
            "early_forecast_mae": 0.17,
            "active_effect_mae": None,
        }
    )

    result = support_decision(rows, seeds)
    comparison = result["descriptive"]["driver_switch_vs_direct_history_forecast_mae"]
    assert result["complete"] and result["passed"]
    assert comparison["available_seed_count"] == 1
    assert comparison["mean_gain"] == pytest.approx(-0.005)
    assert "p_value" not in comparison


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("arm", "oracle", "arm and scenario"),
        ("scenario", "new", "arm and scenario"),
        ("seed", 8, "seed grid"),
        ("seed", True, "seed grid"),
    ],
)
def test_rows_outside_the_declared_grid_raise(key, value, message):
    seeds = tuple(range(8))
    rows = _rows(seeds)
    rows[0][key] = value

    with pytest.raises(ValueError, match=message):
        support_decision(rows, seeds)
