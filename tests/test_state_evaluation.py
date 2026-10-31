"""Nested graph-summary evaluation must preserve raw support and frozen fitting."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor
from effectome.dynamics import GraphStateConfig, TransitionConfig
from effectome.dynamics.state_evaluation import (
    StateEvaluationConfig,
    evaluate_state_models,
    fit_frozen_state_transitions,
    future_graph_pairs,
)


def _series(n=85):
    data = np.random.default_rng(20).normal(size=(n, 3, 3))
    data[:, np.arange(3), np.arange(3)] = 0
    anchors = [
        TemporalAnchor(
            dataset_id="dataset",
            recording_id="recording",
            animal_id="animal",
            session_id=None,
            segment_id=None,
            context_start=i * 5,
            context_stop=i * 5 + 10,
            target_start=i * 5 + 8,
            target_stop=i * 5 + 10,
            anchor_sample=i * 5 + 9,
            anchor_time_seconds=float(i * 5 + 9),
            sampling_rate_hz=1.0,
        )
        for i in range(n)
    ]
    return ConnectivitySeries(
        matrices=data,
        lagged_matrices=data[:, None],
        window_starts=np.arange(n) * 5,
        method="fixture",
        directed=True,
        anchors=anchors,
    )


def _cfg():
    return StateEvaluationConfig(
        graph=GraphStateConfig(
            n_states=1,
            feature_representation="signed_lag_offdiag",
            assignment="kmeans",
            metric="frobenius",
            pca_components=3,
        ),
        candidate_counts=(1, 2),
        ridge_penalties=(1.0,),
    )


def test_future_graph_targets_and_nested_splits_are_raw_disjoint():
    series = _series()
    result = evaluate_state_models(series, _cfg())
    assert result.status == "available"
    for origin, target in future_graph_pairs(series):
        assert series.anchors[target].context_start >= series.anchors[origin].context_stop
    for train_key, test_key in [
        ("outer_train_pairs", "outer_test_pairs"),
        ("inner_train_pairs", "inner_validation_pairs"),
    ]:
        train = np.unique(result.diagnostics[train_key])
        test = np.unique(result.diagnostics[test_key])
        for a in train:
            for b in test:
                assert series.anchors[a].context_stop <= series.anchors[b].context_start
    assert not result.diagnostics["selection_uses_outer_test"]
    assert not result.diagnostics["regime_claim_eligible"]
    assert {"constant", "continuous"} <= {t["algorithm"] for t in result.diagnostics["inner_trials"]}


def test_outer_test_values_do_not_change_selected_model_or_parameters():
    series = _series()
    first = evaluate_state_models(series, _cfg())
    changed = deepcopy(series)
    first_test = min(np.ravel(first.diagnostics["outer_test_pairs"]))
    changed.matrices[first_test:] *= 300
    changed.lagged_matrices[first_test:] *= 300
    second = evaluate_state_models(changed, _cfg())
    assert (first.selected_algorithm, first.selected_n_states) == (
        second.selected_algorithm,
        second.selected_n_states,
    )
    for field in ("feature_mean", "feature_scale", "feature_centroids", "pca_components"):
        np.testing.assert_allclose(
            getattr(first.selected_model, field), getattr(second.selected_model, field)
        )
    assert first.selected_model.descriptive_centroids is None


def test_gap_flags_prevent_future_graph_pairs_crossing_bad_data():
    series = _series(12)
    series.anchors[5] = replace(series.anchors[5], bad_frame_before=True)
    assert all(not (a < 5 <= b) for a, b in future_graph_pairs(series))


def test_short_series_is_unavailable_without_shrinking_purge():
    result = evaluate_state_models(_series(8), _cfg())
    assert result.status == "unavailable"
    assert result.selected_model is None
    assert "reason" in result.diagnostics


def test_frozen_transition_parameters_ignore_outer_test_codes():
    series = _series()
    result = evaluate_state_models(series, _cfg())
    first = fit_frozen_state_transitions(series, result, TransitionConfig(n_null=0))
    changed = deepcopy(result)
    changed.selected_model.labels[result.evaluation_indices] = 0
    second = fit_frozen_state_transitions(series, changed, TransitionConfig(n_null=0))
    np.testing.assert_array_equal(first.transition_matrix, second.transition_matrix)
    assert np.isnan(first.p_value)
    assert not first.regime_claim_eligible
    assert first.null_scope == "not_run_frozen_training_transition_prediction"


def test_config_requires_one_state_and_registered_assignments():
    with pytest.raises(ValueError, match="one-state"):
        StateEvaluationConfig(candidate_counts=(2, 3))
    with pytest.raises(ValueError, match="supports"):
        StateEvaluationConfig(assignments=("causal_magic",))


def test_invalid_anchor_is_excluded_and_breaks_future_pairs_on_both_sides():
    series = _series(12)
    series.anchors[5] = replace(series.anchors[5], valid=False, invalid_reason="masked_history")
    pairs = future_graph_pairs(series)
    assert not np.any(pairs == 5)
    assert all(not (a < 5 < b) for a, b in pairs)
