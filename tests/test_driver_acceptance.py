"""Acceptance decisions retain seed denominators and simulator-grounded truth."""

from copy import deepcopy
from dataclasses import replace

import pytest

from effectome.experiments.driver_acceptance import (
    DriverAcceptanceCriteria,
    evaluate_driver_acceptance,
    load_driver_acceptance_criteria,
)


def _row(scenario, seed, source, method="method", gain=0.02):
    active = f"neuron:{source}"
    scores = {f"neuron:{i}": 0.0 for i in range(6)}
    effects = dict(scores)
    scores.update({"group:driver_pair": 0.0, "group:control_pair": 0.0})
    effects.update({"group:driver_pair": 0.0, "group:control_pair": 0.0})
    if scenario in {"single_driver", "correlated_bystander"}:
        scores[active], effects[active] = 0.2, 0.3
        scores["group:driver_pair"], effects["group:driver_pair"] = 0.3, 0.4
    elif scenario == "redundant_group":
        scores["group:driver_pair"], effects["group:driver_pair"] = 0.3, 0.5
        effects["neuron:0"] = effects["neuron:1"] = 0.02
    elif scenario == "hidden_confounder":
        scores[active] = 0.4
    return {
        "scenario": scenario,
        "seed": seed,
        "method": method,
        "source_state": source,
        "destination_state": 1 - source,
        "status": "available",
        "reason": None,
        "n_origins": 24,
        "n_events": 6,
        "alignment_accuracy": 1.0,
        "log_loss_gain": gain,
        "driver_scores": scores,
        "intervention_effects": effects,
    }


def _data(n_positive=20, n_null=60, hidden=True):
    planned = {
        scenario: list(range(n_positive))
        for scenario in ("single_driver", "redundant_group", "correlated_bystander")
    }
    planned["null"] = list(range(n_null))
    if hidden:
        planned["hidden_confounder"] = list(range(n_positive))
    rows = [
        _row(scenario, seed, source)
        for scenario, seeds in planned.items()
        for seed in seeds
        for source in (0, 1)
    ]
    return rows, planned


def test_default_configuration_is_explicit_proposed_and_validated():
    criteria = load_driver_acceptance_criteria()
    assert criteria == DriverAcceptanceCriteria()
    assert criteria.min_positive_seeds == 20
    assert criteria.min_null_seeds == 60
    with pytest.raises(ValueError, match="integer"):
        replace(criteria, min_positive_seeds=True)
    with pytest.raises(ValueError, match="strictly smaller"):
        replace(criteria, confidence_level=1)


def test_three_seed_pilot_is_inconclusive_even_with_perfect_recovery():
    rows, planned = _data(3, 3)
    result = evaluate_driver_acceptance(rows, DriverAcceptanceCriteria(), planned, ["method"])
    assert result["overall_status"] == "inconclusive"
    assert result["supported_methods"] == []
    single = result["methods"]["method"]["scenarios"]["single_driver"]
    assert single["summaries"]["topk_recall"]["mean"] == 1
    assert single["summaries"]["log_loss_gain"]["n_seeds"] == 3
    assert not result["directions_are_replicates"]
    assert not result["biological_necessity_validated"]


def test_full_dummy_recovery_and_zero_of_sixty_null_seeds_pass_proposed_gates():
    rows, planned = _data()
    result = evaluate_driver_acceptance(rows, DriverAcceptanceCriteria(), planned, ["method"])
    assert result["methods"]["method"]["status"] == "pass"
    scenarios = result["methods"]["method"]["scenarios"]
    null = scenarios["null"]["null_nomination"]
    assert null["n_independent_seeds"] == 60
    assert null["seeds_with_any_false_nominee"] == 0
    assert null["one_sided_exact_upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 60))
    assert null["one_sided_exact_upper_bound"] < 0.05
    assert scenarios["redundant_group"]["gates"]["joint_group_recovery"]
    assert scenarios["hidden_confounder"]["status"] == "diagnostic"
    assert scenarios["hidden_confounder"]["null_nomination"]["fraction"] == 1
    assert not result["confounding_identification_established"]


def test_missing_fits_support_and_methods_remain_visible_without_complete_seed_intervals():
    rows, planned = _data()
    failed = next(row for row in rows if row["scenario"] == "single_driver" and row["seed"] == 0)
    failed.update(status="unavailable", reason="nonconverged")
    result = evaluate_driver_acceptance(rows, DriverAcceptanceCriteria(), planned, ["method", "missing"])
    single = result["methods"]["method"]["scenarios"]["single_driver"]
    assert single["status"] == "inconclusive"
    assert single["n_planned_directions"] == single["n_attempted_directions"] == 40
    assert single["n_available_directions"] == 39
    assert single["n_complete_eligible_seeds"] == 19
    assert single["summaries"]["log_loss_gain"]["mean_interval"] is None
    assert single["ineligible_directions"][0]["reasons"] == ["nonconverged"]
    assert result["methods"]["missing"]["scenarios"]["single_driver"]["n_attempted_directions"] == 0
    assert result["methods"]["missing"]["status"] == "inconclusive"
    rows[1]["n_events"] = 1
    result = evaluate_driver_acceptance(rows, DriverAcceptanceCriteria(), planned, ["method"])
    reasons = result["methods"]["method"]["scenarios"]["single_driver"]["ineligible_directions"]
    assert any("insufficient_n_events" in row["reasons"] for row in reasons)


def test_negative_prediction_fails_only_after_sufficient_independent_repeats():
    for count in (3, 20):
        rows, planned = _data(count, 60)
        for row in rows:
            if row["scenario"] == "single_driver":
                row["log_loss_gain"] = -0.03
        result = evaluate_driver_acceptance(rows, DriverAcceptanceCriteria(), planned, ["method"])
        single = result["methods"]["method"]["scenarios"]["single_driver"]
        assert single["status"] == ("fail" if count == 20 else "inconclusive")


def test_oracle_interventions_define_gold_not_nominal_neuron_names():
    rows, planned = _data()
    for row in rows:
        if row["scenario"] in {"single_driver", "correlated_bystander"}:
            for unit in row["driver_scores"]:
                if unit.startswith("neuron:"):
                    row["intervention_effects"][unit] = 0.0
                    row["driver_scores"][unit] = 0.0
            row["intervention_effects"]["neuron:4"] = 0.4
            row["driver_scores"]["neuron:4"] = 0.3
    result = evaluate_driver_acceptance(rows, DriverAcceptanceCriteria(), planned, ["method"])
    assert result["methods"]["method"]["scenarios"]["single_driver"]["summaries"]["topk_recall"]["mean"] == 1
    empty = deepcopy(rows)
    for row in empty:
        if row["scenario"] == "single_driver":
            row["intervention_effects"]["neuron:4"] = 0.0
    result = evaluate_driver_acceptance(empty, DriverAcceptanceCriteria(), planned, ["method"])
    single = result["methods"]["method"]["scenarios"]["single_driver"]
    assert single["status"] == "inconclusive"
    assert single["summaries"]["topk_recall"]["mean"] is None


def test_null_false_nomination_is_any_direction_or_unit_once_per_seed():
    rows, planned = _data()
    for row in rows:
        if row["scenario"] == "null" and row["seed"] == 0:
            row["driver_scores"]["neuron:2"] = 0.4
            row["driver_scores"]["group:control_pair"] = 0.5
    result = evaluate_driver_acceptance(rows, DriverAcceptanceCriteria(), planned, ["method"])
    null = result["methods"]["method"]["scenarios"]["null"]
    assert null["null_nomination"]["seeds_with_any_false_nominee"] == 1
    assert null["null_nomination"]["fraction"] == pytest.approx(1 / 60)
    assert null["status"] == "fail"


def test_duplicate_or_undeclared_rows_fail_clearly():
    rows, planned = _data(3, 3)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_driver_acceptance([*rows, rows[0]], DriverAcceptanceCriteria(), planned, ["method"])
    rows[0]["source_state"] = 2
    with pytest.raises(ValueError, match="outside"):
        evaluate_driver_acceptance(rows, DriverAcceptanceCriteria(), planned, ["method"])
