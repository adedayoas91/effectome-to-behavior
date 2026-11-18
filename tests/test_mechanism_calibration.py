"""Separate-null calibration must not learn from missing, test or gold effects."""

from copy import deepcopy

import numpy as np
import pytest

from effectome.experiments.mechanism_calibration import (
    apply_nomination_threshold,
    calibrate_seed_maxima,
)

METHOD = "fast_cgc"
SEEDS = tuple(range(1000, 1019))


def _metrics(maximum: float = 0.2) -> dict:
    directions = []
    for source in (0, 1):
        directions.append({
            "source_state": source, "destination_state": 1 - source,
            "status": "available", "n_origins": 6,
            "estimated_probability_effects": {
                "neuron:0": maximum - 0.01 * source,
                "neuron:1": -0.01,
                "group:driver_pair": maximum - 0.02,
                "group:control_pair": 0.0,
            },
            "true_probability_effects": {
                "neuron:0": 0.4, "neuron:1": 0.0,
                "group:driver_pair": 0.6, "group:control_pair": 0.0,
            },
            "intervention_positive_neurons": ["neuron:0"],
            "topk_recall": 0.75, "driver_nomination_recall": 0.0,
            "nominated_units": [], "false_nominated_units": [],
        })
    return {
        "directions": directions, "factual_brier": 0.03,
        "factual_log_loss": 0.2, "mean_absolute_effect_error": 0.04,
        "any_false_nomination": False, "nomination_scope": "old_development_cutoff",
    }


def _cells(seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    return [{
        "scenario": "null", "seed": seed,
        "methods": {METHOD: {"status": "available", "metrics": _metrics(0.01 * i)}},
    } for i, seed in enumerate(seeds)]


def _direction(cells: list[dict], index: int = 0, source: int = 0) -> dict:
    return cells[index]["methods"][METHOD]["metrics"]["directions"][source]


def test_boundary_calibration_has_one_seed_maximum_and_a_strict_cutoff():
    cells = _cells()
    result = calibrate_seed_maxima(cells, METHOD, SEEDS)
    assert result["complete"] and result["status"] == "complete"
    assert result["rank"] == 19 and result["planned_seed_count"] == 19
    assert result["threshold"] == pytest.approx(0.18)
    assert result["finite_sample_tail_probability_bound"] == 0.05
    assert result["nomination_comparison"] == "strictly_greater_than"
    assert len(result["seed_maxima"]) == 19
    applied = apply_nomination_threshold(_metrics(result["threshold"]), result["threshold"])
    assert not applied["directions"][0]["nominated_units"]


def test_maximum_covers_groups_and_both_directions():
    cells = _cells()
    _direction(cells, 4, 1)["estimated_probability_effects"]["group:control_pair"] = 0.71
    result = calibrate_seed_maxima(cells, METHOD, SEEDS)
    assert result["threshold"] == 0.71
    assert next(r["maximum_score"] for r in result["seed_maxima"] if r["seed"] == SEEDS[4]) == 0.71


def test_finite_sample_order_statistic_uses_second_largest_of_39_not_interpolation():
    seeds = tuple(range(2000, 2039))
    result = calibrate_seed_maxima(_cells(seeds), METHOD, seeds)
    assert result["rank"] == 38
    assert result["threshold"] == pytest.approx(0.37)
    assert result["finite_sample_tail_probability_bound"] == 0.05


def test_order_statistic_tail_bound_by_exhaustive_exchangeable_test_assignment():
    # With 20 distinct exchangeable maxima, each can be held out as the test
    # seed. A strict maximum-of-19 threshold exceeds in exactly 1/20 assignments.
    hits = []
    for test_index in range(20):
        calibration_values = [i / 100 for i in range(20) if i != test_index]
        cells = _cells()
        for cell, value in zip(cells, calibration_values, strict=True):
            cell["methods"][METHOD]["metrics"] = _metrics(value)
        cutoff = calibrate_seed_maxima(cells, METHOD, SEEDS)["threshold"]
        hits.append(test_index / 100 > cutoff)
    assert sum(hits) / len(hits) == 0.05


def test_clamping_negative_order_statistic_to_zero_is_conservative():
    cells = _cells()
    for cell in cells:
        for row in cell["methods"][METHOD]["metrics"]["directions"]:
            row["estimated_probability_effects"] = dict.fromkeys(row["estimated_probability_effects"], -0.2)
    result = calibrate_seed_maxima(cells, METHOD, SEEDS)
    assert result["threshold"] == 0
    assert all(row["maximum_score"] == -0.2 for row in result["seed_maxima"])


@pytest.mark.parametrize("bad_alpha", [0, 1, -0.1, np.nan, np.inf, True, "0.05"])
def test_invalid_alpha_is_rejected(bad_alpha):
    with pytest.raises(ValueError, match="alpha"):
        calibrate_seed_maxima(_cells(), METHOD, SEEDS, bad_alpha)


def test_calibration_requires_finite_sample_alpha_resolution():
    with pytest.raises(ValueError, match="under-resolved"):
        calibrate_seed_maxima(_cells(SEEDS[:18]), METHOD, SEEDS[:18])


@pytest.mark.parametrize("bad_seeds", [(), (1, 1), (-1, 2), (True, 2), (1.0, 2)])
def test_invalid_or_reused_planned_seed_ids_are_rejected(bad_seeds):
    with pytest.raises(ValueError, match="distinct nonnegative integers"):
        calibrate_seed_maxima([], METHOD, bad_seeds)


@pytest.mark.parametrize("bad_seed", [-1, 999, True, 1000.0])
def test_bad_or_unplanned_null_seed_is_rejected(bad_seed):
    cells = _cells()
    cells[0]["seed"] = bad_seed
    with pytest.raises(ValueError, match="declared calibration pool"):
        calibrate_seed_maxima(cells, METHOD, SEEDS)


def test_duplicate_null_cells_are_not_independent_replication():
    cells = _cells()
    cells.append(deepcopy(cells[0]))
    with pytest.raises(ValueError, match="duplicate null"):
        calibrate_seed_maxima(cells, METHOD, SEEDS)


@pytest.mark.parametrize("missing_kind", ["cell", "method", "unavailable", "unsupported", "origins"])
def test_missing_evidence_blocks_calibration_without_changing_planned_denominator(missing_kind):
    cells = _cells()
    if missing_kind == "cell":
        cells.pop()
    elif missing_kind == "method":
        cells[-1]["methods"] = {}
    elif missing_kind == "unavailable":
        cells[-1]["methods"][METHOD] = {"status": "unavailable", "reason": "runtime missing"}
    elif missing_kind == "unsupported":
        _direction(cells, -1, 1).update(status="unsupported", n_origins=0)
    else:
        _direction(cells, -1, 1)["n_origins"] = 0
    result = calibrate_seed_maxima(cells, METHOD, SEEDS)
    assert not result["complete"] and result["threshold"] is None
    assert result["planned_seed_count"] == 19 and result["available_seed_count"] == 18
    assert result["finite_sample_tail_probability_bound"] is None
    assert SEEDS[-1] in result["missing_seeds"] + result["unavailable_seeds"]


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, True, None, "0.1"])
def test_nonfinite_or_nonnumeric_scores_are_rejected(score):
    cells = _cells()
    _direction(cells)["estimated_probability_effects"]["neuron:0"] = score
    with pytest.raises(ValueError, match="finite real"):
        calibrate_seed_maxima(cells, METHOD, SEEDS)


@pytest.mark.parametrize("change", ["source", "duplicate_source", "candidate_direction", "candidate_seed"])
def test_inconsistent_source_and_candidate_schemas_are_rejected(change):
    cells = _cells()
    if change == "source":
        _direction(cells)["source_state"] = 2
    elif change == "duplicate_source":
        _direction(cells, 0, 1)["source_state"] = 0
    elif change == "candidate_direction":
        _direction(cells)["estimated_probability_effects"].pop("neuron:1")
    else:
        for row in cells[0]["methods"][METHOD]["metrics"]["directions"]:
            row["estimated_probability_effects"].pop("neuron:1")
    with pytest.raises(ValueError, match="source_state|candidate set"):
        calibrate_seed_maxima(cells, METHOD, SEEDS)


def test_calibration_ignores_gold_nonnull_cells_and_input_order_without_mutation():
    cells = _cells()
    original = deepcopy(cells)
    before = calibrate_seed_maxima(cells, METHOD, SEEDS)
    assert cells == original
    for cell in cells:
        for row in cell["methods"][METHOD]["metrics"]["directions"]:
            row["true_probability_effects"] = "gold must not be read"
        cell["oracle"] = "oracle must not be read"
    cells.append({"scenario": "driver", "seed": -99, "methods": "must not be read"})
    after = calibrate_seed_maxima(cells[::-1], METHOD, SEEDS[::-1])
    assert after["threshold"] == before["threshold"]
    assert after["fingerprint"] == before["fingerprint"]
    assert after["ignored_non_null_cells"] == 1


def test_applying_cutoff_preserves_scores_rankings_factual_errors_and_original_metrics():
    original = _metrics(0.4)
    snapshot = deepcopy(original)
    applied = apply_nomination_threshold(original, 0.3)
    assert original == snapshot
    assert applied["nomination_complete"]
    assert applied["nomination_threshold"] == 0.3
    for before, after in zip(original["directions"], applied["directions"], strict=True):
        assert after["estimated_probability_effects"] == before["estimated_probability_effects"]
        assert after["true_probability_effects"] == before["true_probability_effects"]
        assert after["topk_recall"] == before["topk_recall"]
        assert after["nominated_units"] == ["group:driver_pair", "neuron:0"]
        assert after["driver_nomination_recall"] == 1
        assert after["false_nominated_units"] == []
    for key in ["mean_absolute_effect_error", "factual_brier", "factual_log_loss"]:
        assert applied[key] == original[key]


def test_gold_only_changes_nomination_evaluation_not_which_units_are_nominated():
    original = _metrics(0.4)
    changed = deepcopy(original)
    for row in changed["directions"]:
        row["true_probability_effects"] = dict.fromkeys(row["true_probability_effects"], 0.0)
    first = apply_nomination_threshold(original, 0.3)
    second = apply_nomination_threshold(changed, 0.3)
    assert not first["any_false_nomination"] and second["any_false_nomination"]
    for a, b in zip(first["directions"], second["directions"], strict=True):
        assert a["nominated_units"] == b["nominated_units"]
        assert b["driver_nomination_recall"] is None


def test_application_retains_unsupported_directions_as_incomplete():
    metrics = _metrics()
    metrics["directions"][1] = {"source_state": 1, "status": "unsupported", "n_origins": 0}
    applied = apply_nomination_threshold(metrics, 0.1)
    assert not applied["nomination_complete"]
    assert applied["directions"][1] == metrics["directions"][1]


@pytest.mark.parametrize("cutoff", [-0.1, np.nan, np.inf, True])
def test_application_rejects_invalid_cutoffs(cutoff):
    with pytest.raises(ValueError, match="threshold"):
        apply_nomination_threshold(_metrics(), cutoff)
