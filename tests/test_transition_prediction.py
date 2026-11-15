"""Shared transition comparisons preserve ownership and transition-specific evidence."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from effectome.attribution.transition_prediction import evaluate_transition_predictors


def _inputs(n_events=800):
    rng = np.random.default_rng(194)
    useful = rng.normal(size=n_events)
    return {
        "baseline": rng.normal(size=(n_events, 2)),
        "graph_features": np.column_stack((useful, rng.normal(size=n_events))),
        "source_states": np.zeros(n_events, dtype=int),
        "target_states": (useful + rng.normal(scale=0.2, size=n_events) > 0).astype(int),
        "train_rows": np.arange(n_events // 2),
        "evaluation_rows": np.arange(n_events // 2, n_events),
        "feature_blocks": {"useful": [0], "irrelevant": [1]},
        "n_states": 2,
    }


def test_planted_predictor_and_omission_scores_include_nontransition_origins():
    inputs = _inputs()
    result = evaluate_transition_predictors(**inputs)
    assert result["log_loss_gain"] > 0.3
    units = {row["unit"]: row for row in result["omission_scores"]}
    assert units["useful"]["omission_delta_log_loss"] > 0.3
    assert abs(units["irrelevant"]["omission_delta_log_loss"]) < 0.03
    assert units["useful"]["n_origins"] == len(inputs["evaluation_rows"])
    assert 0 < units["useful"]["n_events"] < units["useful"]["n_origins"]
    assert not units["useful"]["is_validated_driver"]
    assert np.asarray(result["augmented_probabilities"]).shape == (400, 2)
    assert result["transition_matrices"]["source_counts"] == [400, 0]
    assert result["transition_matrices"]["unsupported_rows"] == [1]
    assert result["transition_matrices"]["empirical"][1] is None
    assert result["transition_matrices"]["augmented_predicted"][1] is None
    json.dumps(result, allow_nan=False)


def test_joint_group_omission_detects_redundant_sources():
    inputs = _inputs()
    signal = inputs["graph_features"][:, 0]
    inputs["graph_features"] = np.column_stack((signal, signal))
    inputs["feature_blocks"] = {"neuron_1": [0], "neuron_2": [1], "group": [0, 1]}
    inputs["c"] = 1000.0
    result = evaluate_transition_predictors(**inputs)
    units = {row["unit"]: row for row in result["omission_scores"]}
    assert units["group"]["omission_delta_log_loss"] > 0.3
    assert abs(units["neuron_1"]["omission_delta_log_loss"]) < 0.01
    assert abs(units["neuron_2"]["omission_delta_log_loss"]) < 0.01


def test_heldout_outcomes_do_not_change_predictions_or_fitted_parameters():
    inputs = _inputs()
    first = evaluate_transition_predictors(**inputs)
    changed_targets = inputs["target_states"].copy()
    changed_targets[inputs["evaluation_rows"]] = 1 - changed_targets[inputs["evaluation_rows"]]
    second = evaluate_transition_predictors(**{**inputs, "target_states": changed_targets})
    assert first["training_parameters"] == second["training_parameters"]
    np.testing.assert_array_equal(first["augmented_probabilities"], second["augmented_probabilities"])
    changed_graph = inputs["graph_features"].copy()
    changed_graph[inputs["evaluation_rows"]] *= 100
    third = evaluate_transition_predictors(**{**inputs, "graph_features": changed_graph})
    assert first["training_parameters"] == third["training_parameters"]


def test_one_training_class_keeps_declared_unseen_class_probability_columns():
    inputs = _inputs(100)
    inputs["target_states"][:50] = 0
    inputs["target_states"][50:] = 2
    inputs["n_states"] = 3
    result = evaluate_transition_predictors(**inputs)
    probabilities = np.asarray(result["augmented_probabilities"])
    assert result["class_order"] == [0, 1, 2]
    assert probabilities.shape == (50, 3)
    assert np.all(probabilities > 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert result["training_parameters"]["augmented"]["kind"] == "training_occupancy"
    assert np.isfinite(result["augmented_scores"]["log_loss"])
    assert result["augmented_scores"]["log_loss"] > 20


def test_logistic_training_with_missing_class_keeps_vocabulary_order():
    inputs = _inputs()
    inputs["n_states"] = 3
    inputs["target_states"][700:] = 2
    result = evaluate_transition_predictors(**inputs)
    probabilities = np.asarray(result["augmented_probabilities"])
    assert probabilities.shape == (400, 3)
    np.testing.assert_allclose(probabilities[:, 2], 1e-12)
    assert result["training_parameters"]["augmented"]["observed_training_classes"] == [0, 1]


def test_nonconverged_decoder_is_unavailable_instead_of_publishing_predictions(monkeypatch):
    original_fit = LogisticRegression.fit

    def nonconverged_fit(model, *args, **kwargs):
        fitted = original_fit(model, *args, **kwargs)
        fitted.n_iter_ = np.asarray([fitted.max_iter])
        return fitted

    monkeypatch.setattr(LogisticRegression, "fit", nonconverged_fit)
    with pytest.raises(RuntimeError, match="did not converge"):
        evaluate_transition_predictors(**_inputs())


def test_no_observed_exit_reports_destination_score_as_unavailable():
    inputs = _inputs(100)
    inputs["target_states"][50:] = 0
    result = evaluate_transition_predictors(**inputs)
    assert result["augmented_scores"]["n_exits"] == 0
    assert result["augmented_scores"]["conditional_destination_log_loss"] is None
    assert result["omission_scores"][0]["n_origins"] == 50
    assert result["omission_scores"][0]["n_events"] == 0


def test_single_declared_state_and_empty_graph_are_available_without_fabricated_exits():
    inputs = _inputs(100)
    inputs.update(
        target_states=np.zeros(100, dtype=int),
        graph_features=np.empty((100, 0)),
        feature_blocks={},
        n_states=1,
    )
    result = evaluate_transition_predictors(**inputs)
    assert result["transition_matrices"]["empirical"] == [[1.0]]
    assert result["omission_scores"] == []
    assert result["augmented_scores"]["conditional_destination_log_loss"] is None
    assert result["log_loss_gain"] == 0


def test_displacement_prediction_is_training_owned_and_uses_future_vectors_only_as_targets():
    inputs = _inputs()
    signal = inputs["graph_features"][:, :1]
    displacement = np.column_stack((signal, -2 * signal))
    first = evaluate_transition_predictors(**inputs, future_displacement=displacement)
    assert first["displacement"]["mse_gain"] > 1
    assert np.asarray(first["displacement"]["augmented_predictions"]).shape == (400, 2)
    changed = displacement.copy()
    changed[inputs["evaluation_rows"]] *= -100
    second = evaluate_transition_predictors(**inputs, future_displacement=changed)
    assert first["displacement"]["training_parameters"] == second["displacement"]["training_parameters"]
    assert first["displacement"]["augmented_predictions"] == second["displacement"]["augmented_predictions"]


def test_multiclass_competing_outcomes_remain_in_transition_omission_score():
    inputs = _inputs(120)
    inputs["n_states"] = 3
    inputs["target_states"] = np.tile([0, 1, 2], 40)
    result = evaluate_transition_predictors(**inputs)
    unit = next(row for row in result["omission_scores"] if row["destination_state"] == 1)
    assert unit["n_origins"] == 60
    assert unit["n_events"] == 20
    assert result["augmented_scores"]["n_exits"] == 40


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"n_states": 0}, "n_states"),
        ({"c": 0}, "c must"),
        ({"c": np.nan}, "c must"),
        ({"train_rows": np.array([0.0, 1.0])}, "train_rows"),
        ({"train_rows": np.array([1, 1])}, "train_rows"),
        ({"train_rows": np.array([], dtype=int)}, "train_rows"),
        ({"train_rows": np.arange(55)}, "disjoint"),
        ({"evaluation_rows": np.array([100])}, "evaluation_rows"),
        ({"feature_blocks": {"invalid": [2]}}, "feature_blocks"),
        ({"feature_blocks": {"duplicate": [0, 0]}}, "feature_blocks"),
        ({"feature_blocks": {"empty": []}}, "feature_blocks"),
        ({"source_states": np.full(100, 2)}, "source_states"),
        ({"target_states": np.full(100, 0.5)}, "target_states"),
        ({"baseline": np.full((100, 2), np.nan)}, "finite aligned"),
        ({"graph_features": np.ones((99, 2))}, "finite aligned"),
        ({"future_displacement": np.ones((99, 2))}, "future_displacement"),
        ({"future_displacement": np.empty((100, 0))}, "future_displacement"),
    ],
)
def test_invalid_inputs_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        evaluate_transition_predictors(**{**_inputs(100), **overrides})
