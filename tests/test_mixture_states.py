"""Frozen graph representations and probabilistic state assignment."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from effectome.data_module.schema import ConnectivitySeries
from effectome.dynamics import GraphStateConfig, fit_graph_states


def _series(features: np.ndarray) -> ConnectivitySeries:
    lagged = np.zeros((len(features), 1, 2, 2), dtype=np.float64)
    lagged[:, 0, 0, 1] = features[:, 0]
    lagged[:, 0, 1, 0] = features[:, 1]
    return ConnectivitySeries(
        matrices=lagged[:, 0],
        lagged_matrices=lagged,
        window_starts=np.arange(len(features)),
        method="synthetic_graphs",
        directed=True,
    )


def _config(**kwargs) -> GraphStateConfig:
    return GraphStateConfig(
        feature_representation="signed_lag_offdiag",
        assignment="gaussian_mixture",
        analysis_scope="prospective",
        **kwargs,
    )


def test_diagonal_mixture_recovers_unequal_spreads_and_soft_membership():
    rng = np.random.default_rng(41)
    features = np.concatenate([
        rng.normal(loc=(-2, 0), scale=(0.3, 1.4), size=(240, 2)),
        rng.normal(loc=(2, 0), scale=(1.0, 0.2), size=(240, 2)),
    ])
    series = _series(features)
    model = fit_graph_states(series, _config(n_states=2, standardize=False, seed=8))
    assert model.mixture is not None
    assert model.mixture.covariance_type == "diag"
    assert model.mixture.reg_covar == 1e-3
    assert adjusted_rand_score(np.repeat([0, 1], 240), model.labels) > 0.9
    assert np.max(model.mixture.covariances_[:, 1]) / np.min(model.mixture.covariances_[:, 1]) > 10
    probabilities = model.transform_proba(series)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(np.argmax(probabilities, axis=1), model.labels)
    np.testing.assert_allclose(probabilities, model.membership_probabilities)
    assert np.any(model.assignment_entropy > 0.01)


def test_mixture_fitting_pca_and_probabilities_ignore_heldout_changes():
    rng = np.random.default_rng(9)
    features = np.r_[rng.normal((-2, 0), 0.4, (60, 2)), rng.normal((2, 0), 0.8, (60, 2))]
    all_features = np.r_[features, rng.normal(size=(15, 2))]
    changed = all_features.copy()
    changed[120:] *= 1000
    fit_indices = np.arange(120)
    config = _config(n_states=2, pca_components=1, seed=2)
    first = fit_graph_states(_series(all_features), config, fit_indices=fit_indices)
    second = fit_graph_states(_series(changed), config, fit_indices=fit_indices)
    for field in ["feature_mean", "feature_scale", "pca_components", "pca_mean", "feature_centroids"]:
        np.testing.assert_allclose(getattr(first, field), getattr(second, field))
    np.testing.assert_allclose(first.mixture.covariances_, second.mixture.covariances_)
    np.testing.assert_allclose(
        first.membership_probabilities[:120], second.membership_probabilities[:120]
    )
    reference = _series(all_features[:10])
    enlarged = _series(np.r_[all_features[:10], changed[120:]])
    np.testing.assert_allclose(first.transform_proba(reference), first.transform_proba(enlarged)[:10])
    assert first.transform_features(reference).shape == (10, 1)
    np.testing.assert_allclose(first.centroids, second.centroids)
    assert first.descriptive_centroids is second.descriptive_centroids is None


@pytest.mark.parametrize("assignment", ["kmeans", "gaussian_mixture"])
def test_one_state_is_supported_without_spurious_assignment_uncertainty(assignment):
    series = _series(np.random.default_rng(4).normal(size=(20, 2)))
    model = fit_graph_states(series, GraphStateConfig(
        feature_representation="signed_lag_offdiag", assignment=assignment, n_states=1,
    ))
    assert model.n_states == 1
    assert np.isnan(model.silhouette)
    np.testing.assert_array_equal(model.transform(series), np.zeros(20, dtype=int))
    np.testing.assert_array_equal(model.transform_proba(series), np.ones((20, 1)))
    np.testing.assert_array_equal(model.assignment_entropy, np.zeros(20))


def test_mixture_pickle_roundtrip_preserves_frozen_transform_and_actual_provenance():
    series = _series(np.random.default_rng(5).normal(size=(40, 2)))
    model = fit_graph_states(series, _config(n_states=2, metric="causal_kernel", pca_components=1))
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(model.transform_proba(series), restored.transform_proba(series))
    assert restored.requested_metric == "causal_kernel"
    assert restored.metric == "frobenius"
    assert restored.effective_algorithm == "gaussian_mixture_diag"
    assert restored.effective_distance == "diagonal_gaussian_negative_log_density"
    assert restored.inductive_assignment_valid
    assert restored.assignment == "gaussian_mixture"


def test_supplied_reliability_weights_are_frozen_and_applied_before_pca():
    series = _series(np.random.default_rng(6).normal(size=(30, 2)))
    weights = np.array([4.0, 0.0])
    model = fit_graph_states(series, _config(n_states=1, standardize=False), feature_weights=weights)
    weights[:] = 99
    np.testing.assert_array_equal(model.feature_weights, [4.0, 0.0])
    transformed = model.transform_features(series)
    np.testing.assert_allclose(transformed[:, 0], 2 * series.matrices[:, 0, 1])
    np.testing.assert_array_equal(transformed[:, 1], np.zeros(30))
    for invalid in [np.array([0., 0.]), np.array([-1., 1.]), np.array([np.nan, 1.]), np.ones(3)]:
        with pytest.raises(ValueError, match="feature_weights"):
            fit_graph_states(series, _config(n_states=1), feature_weights=invalid)


@pytest.mark.parametrize("value", [0.0, -1., np.nan, np.inf])
def test_mixture_covariance_regularization_must_be_finite_positive(value):
    with pytest.raises(ValueError, match="mixture_reg_covar"):
        _config(mixture_reg_covar=value)


def test_mixture_requires_signed_lag_features_and_two_training_observations():
    series = _series(np.random.default_rng(7).normal(size=(20, 2)))
    with pytest.raises(ValueError, match="signed_lag_offdiag"):
        fit_graph_states(series, GraphStateConfig(assignment="gaussian_mixture"))
    with pytest.raises(ValueError, match="two training"):
        fit_graph_states(series, _config(n_states=1), fit_indices=np.array([0]))


def test_legacy_spectral_proxy_is_rejected_for_prospective_analysis():
    series = _series(np.random.default_rng(8).normal(size=(30, 2)))
    with pytest.raises(ValueError, match="inductive extension"):
        fit_graph_states(series, GraphStateConfig(metric="causal_kernel", analysis_scope="prospective"))
    descriptive = fit_graph_states(series, GraphStateConfig(metric="legacy_rbf", n_states=2))
    assert not descriptive.inductive_assignment_valid
    assert descriptive.effective_distance == "rbf_on_duplicated_signed_graph_features"
    assert descriptive.effective_algorithm == "spectral_legacy_original_feature_assignment_proxy"
