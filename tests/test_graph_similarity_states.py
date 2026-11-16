"""Support geometry reaches the active state path and preserves holdout ownership."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor
from effectome.dynamics import GraphStateConfig, fit_graph_states
from effectome.dynamics.metrics import kmedoids
from effectome.dynamics.state_evaluation import StateEvaluationConfig, evaluate_state_models


def _series(graphs: np.ndarray) -> ConnectivitySeries:
    count, lags, nodes, _ = graphs.shape
    return ConnectivitySeries(
        matrices=graphs.max(axis=1), lagged_matrices=graphs.copy(),
        window_starts=np.arange(count), method="planted_support", directed=True,
        signed=False, weighted=False,
        diagnostics={"node_ids": [f"neuron:{i}" for i in range(nodes)],
                     "physical_lags": [0.1 * (i + 1) for i in range(lags)],
                     "representation_units": "binary_support"},
    )


def _graphs() -> np.ndarray:
    graphs = np.zeros((3, 2, 4, 4))
    graphs[0, 0, 0, 2] = graphs[0, 0, 2, 3] = 1
    graphs[1, 0, 1, 2] = graphs[1, 0, 2, 3] = 1
    graphs[2, 0, 0, 2] = graphs[2, 0, 0, 3] = 1
    return np.tile(graphs, (5, 1, 1, 1))


def _config(metric="edge_jaccard", count=3):
    return GraphStateConfig(
        n_states=count, metric=metric, feature_representation="binary_lag_offdiag",
        assignment="kmedoids", standardize=False, analysis_scope="prospective", seed=0,
    )


@pytest.mark.parametrize("metric,count", [("edge_jaccard", 3), ("directed_wl", 2)])
def test_similarity_states_have_frozen_train_medoids_and_consistent_assignments(metric, count):
    series = _series(_graphs())
    indices = np.arange(9)
    model = fit_graph_states(series, _config(metric, count), fit_indices=indices)
    assert model.metric == model.requested_metric == metric
    assert model.assignment == "kmedoids"
    assert model.effective_algorithm == "kmedoids_alternating"
    assert set(model.medoid_indices) <= set(indices)
    assert len(set(model.labels[indices])) == count
    np.testing.assert_array_equal(model.labels, model.transform(series))
    np.testing.assert_array_equal(model.membership_probabilities, model.transform_proba(series))
    altered = deepcopy(series)
    altered.lagged_matrices[9:] = 1 - altered.lagged_matrices[9:]
    altered.matrices[9:] = altered.lagged_matrices[9:].max(axis=1)
    changed = fit_graph_states(altered, _config(metric, count), fit_indices=indices)
    np.testing.assert_array_equal(model.medoid_indices, changed.medoid_indices)
    np.testing.assert_array_equal(model.lagged_medoid_matrices, changed.lagged_medoid_matrices)
    np.testing.assert_array_equal(model.lag_centroids, changed.lag_centroids)
    np.testing.assert_array_equal(model.labels[:9], changed.labels[:9])
    assert model.descriptive_centroids is None
    assert not model.lagged_medoid_matrices.flags.writeable
    with pytest.raises(ValueError, match="Euclidean"):
        model.transform_features(series)


def test_identity_and_topology_groups_remain_distinct():
    series = _series(_graphs())
    identity = fit_graph_states(series, _config())
    topology = fit_graph_states(series, _config("directed_wl", 2))
    assert identity.labels[0] != identity.labels[1]
    assert topology.labels[0] == topology.labels[1]
    assert topology.labels[0] != topology.labels[2]
    with pytest.raises(ValueError, match="distinguishable"):
        fit_graph_states(series, _config("directed_wl", 3))


@pytest.mark.parametrize("key,value", [
    ("node_ids", ["neuron:1", "neuron:0", "neuron:2", "neuron:3"]),
    ("physical_lags", [0.2, 0.4]), ("representation_units", "different_support_definition"),
])
def test_similarity_assignment_rejects_registered_schema_changes(key, value):
    series = _series(_graphs())
    model = fit_graph_states(series, _config())
    changed = deepcopy(series)
    changed.diagnostics[key] = value
    with pytest.raises(ValueError, match="schema"):
        model.transform(changed)
    with pytest.raises(ValueError, match="schema"):
        model.transform(series.lagged_matrices)


@pytest.mark.parametrize("options", [
    {"standardize": True}, {"pca_components": 1}, {"assignment": "gaussian_mixture"},
    {"feature_representation": "signed_lag_offdiag"},
])
def test_similarity_settings_are_not_silently_ignored(options):
    with pytest.raises(ValueError):
        fit_graph_states(_series(_graphs()), replace(_config(), **options))


def test_schema_binary_support_and_feature_weight_requirements_are_explicit():
    series = _series(_graphs())
    with pytest.raises(ValueError, match="feature weights"):
        fit_graph_states(series, _config(), feature_weights=np.ones(24))
    series.diagnostics = {}
    with pytest.raises(ValueError, match="schema"):
        fit_graph_states(series, _config())
    series = _series(_graphs())
    series.lagged_matrices[0, 0, 0, 2] = 0.4
    with pytest.raises(ValueError, match="binary"):
        fit_graph_states(series, _config())


def test_one_empty_state_is_assignable_without_positive_overlap_claim():
    series = _series(np.zeros((5, 1, 3, 3)))
    model = fit_graph_states(series, _config(count=1), fit_indices=np.arange(3))
    np.testing.assert_array_equal(model.transform(series), np.zeros(5))
    assert model.n_states == 1
    with pytest.raises(ValueError, match="distinguishable"):
        fit_graph_states(series, _config(count=2))


def test_predictive_state_count_selection_includes_constant_and_ignores_silhouette():
    series = _series(_graphs())
    cfg = replace(_config(), select_k=True)
    with pytest.raises(ValueError, match="predictive losses"):
        fit_graph_states(series, cfg)
    model = fit_graph_states(series, cfg, predictive_losses={1: 0., 2: 1., 3: 2., 4: 3., 6: 4.})
    assert model.n_states == 1
    assert model.selection_loss == 0


def test_optional_medoid_initialization_is_validated_and_legacy_default_remains_callable():
    distance = np.array([[0., 0., 1.], [0., 0., 1.], [1., 1., 0.]])
    labels, medoids = kmedoids(distance, 2, initial_medoids=np.array([0, 2]))
    assert labels[0] == labels[1] != labels[2]
    assert len(medoids) == 2
    kmedoids(distance, 1)
    for invalid in (np.array([0, 0]), np.array([0., 2.]), np.array([0, 3]), np.array([0])):
        with pytest.raises(ValueError, match="initial_medoids"):
            kmedoids(distance, 2, initial_medoids=invalid)


def _nonlinear_future_series() -> ConnectivitySeries:
    # Four binary edge patterns follow a nonlinear deterministic recurrence.
    # Labels are never inputs to the quantizer or forecast model.
    states = np.tile([0, 3, 1, 2], 40)
    graphs = np.zeros((len(states), 1, 3, 3))
    graphs[:, 0, 0, 1] = states // 2
    graphs[:, 0, 0, 2] = states % 2
    series = _series(graphs)
    series.anchors = [TemporalAnchor(
        dataset_id="planted", recording_id="recurrence", animal_id=None,
        session_id=None, segment_id=None, context_start=i, context_stop=i + 1,
        target_start=i, target_stop=i + 1, anchor_sample=i,
        anchor_time_seconds=float(i), sampling_rate_hz=1.0,
    ) for i in range(len(graphs))]
    return series


def test_medoid_selection_uses_common_frozen_response_and_outer_mutation_cannot_reselect():
    series = _nonlinear_future_series()
    cfg = StateEvaluationConfig(
        graph=GraphStateConfig(metric="frobenius", feature_representation="signed_lag_offdiag",
                               assignment="kmeans", pca_components=2, seed=0),
        assignments=("edge_jaccard",), candidate_counts=(1, 2, 3, 4),
        ridge_penalties=(0.1,), minimum_relative_gain=0.,
    )
    result = evaluate_state_models(series, cfg)
    assert result.status == "available"
    assert result.selected_algorithm == "edge_jaccard"
    assert result.selected_n_states == 4
    assert result.response_model.metric == "frobenius"
    assert result.selected_model.metric == "edge_jaccard"
    assert result.metrics["heldout_mse"] < result.metrics["continuous_mse"]
    assert result.diagnostics["response_coordinates_shared_across_candidates"]
    assert not result.diagnostics["selection_uses_outer_test"]
    assert not result.diagnostics["regime_claim_eligible"]
    changed = deepcopy(series)
    first_test = min(np.ravel(result.diagnostics["outer_test_pairs"]))
    changed.lagged_matrices[first_test:] = 1 - changed.lagged_matrices[first_test:]
    changed.matrices[first_test:] = changed.lagged_matrices[first_test:, 0]
    second = evaluate_state_models(changed, cfg)
    assert (second.selected_algorithm, second.selected_n_states) == ("edge_jaccard", 4)
    np.testing.assert_array_equal(result.selected_model.medoid_indices, second.selected_model.medoid_indices)
    np.testing.assert_array_equal(result.response_model.pca_components, second.response_model.pca_components)
