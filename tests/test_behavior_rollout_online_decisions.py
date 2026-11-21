"""Check the online policy decision against hand-calculable seed panels."""

from copy import deepcopy
from itertools import product

import pytest

from effectome.experiments.behavior_rollout_online_decisions import online_decision
from effectome.experiments.behavior_transition_controls import TRANSITION_SCENARIOS


def _rows(seeds):
    return [
        {
            "seed": seed,
            "scenario": scenario,
            "pooled_gain": 0.020 if scenario == "driver_switch" else -0.004,
            **({"early_local_gain": 0.030} if scenario == "driver_switch" else {}),
        }
        for scenario, seed in product(TRANSITION_SCENARIOS, seeds)
    ]


def test_joint_p_uses_all_four_gates_and_ties_do_not_count():
    seeds = tuple(range(10))
    rows = _rows(seeds)
    rows[0]["early_local_gain"] = 0.005
    rows[10]["pooled_gain"] = -0.005
    original = deepcopy(rows)

    result = online_decision(rows, seeds)
    assert rows == original
    assert result["complete"] and result["passed"]
    assert result["joint_p_value"] == 11 / 1024
    tests = result["tests"]
    assert tests["driver_vs_pooled"]["p_value"] == 1 / 1024
    assert tests["driver_early_vs_local"]["gains_on_expected_side"] == 9
    assert tests["constant_noninferiority"]["gains_on_expected_side"] == 9
    assert tests["constant_noninferiority"]["mean_gain"] == pytest.approx(-0.0041)


@pytest.mark.parametrize(
    ("scenario", "metric", "gain", "failed"),
    [
        ("driver_switch", "pooled_gain", 0.005, "driver_vs_pooled"),
        ("driver_switch", "early_local_gain", 0.005, "driver_early_vs_local"),
        ("constant", "pooled_gain", -0.006, "constant_noninferiority"),
        ("nuisance_switch", "pooled_gain", -0.005, "nuisance_switch_noninferiority"),
    ],
)
def test_every_gate_can_block_the_claim(scenario, metric, gain, failed):
    seeds = tuple(range(10))
    rows = _rows(seeds)
    for row in rows:
        if row["scenario"] == scenario:
            row[metric] = gain

    result = online_decision(rows, seeds)
    assert result["complete"] and not result["passed"]
    assert result["joint_p_value"] == 1
    assert [name for name, gate in result["tests"].items() if not gate["passed"]] == [failed]


@pytest.mark.parametrize(
    "defect", ["missing row", "duplicate", "nonfinite", "unexpected seed", "missing gain"]
)
def test_incomplete_input_cannot_pass(defect):
    seeds = tuple(range(10))
    rows = _rows(seeds)
    if defect == "missing row":
        rows.pop()
    elif defect == "duplicate":
        rows.append(rows[0].copy())
    elif defect == "nonfinite":
        rows[0]["early_local_gain"] = float("inf")
    elif defect == "unexpected seed":
        rows[0]["seed"] = 300
    else:
        del rows[0]["early_local_gain"]

    result = online_decision(rows, seeds)
    assert not result["complete"] and not result["passed"]
    assert result["joint_p_value"] == 1
    assert any(test["p_value"] is None for test in result["tests"].values())


def test_unknown_scenario_is_rejected():
    seeds = tuple(range(10))
    rows = _rows(seeds)
    rows[0]["scenario"] = "unexpected"
    with pytest.raises(ValueError, match="scenario grid"):
        online_decision(rows, seeds)
