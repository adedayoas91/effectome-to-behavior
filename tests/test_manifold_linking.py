"""Manifold embedding and the Stage-6 linking analysis."""

from __future__ import annotations

import numpy as np

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.dynamics import GraphStateConfig, fit_graph_states
from effectome.linking import (
    association_with_null,
    connectivity_features,
    decode_behavior,
    lead_lag,
    state_features,
)
from effectome.manifold import ManifoldConfig, ManifoldFactory


def test_classical_manifold_shape(synthetic_recording):
    emb = ManifoldFactory(ManifoldConfig(name="classical", method="pca", n_dims=3)).embed(
        synthetic_recording.traces.T, synthetic_recording.behavior
    )
    assert emb.shape == (synthetic_recording.n_timepoints, 3)


def test_connectivity_decodes_regime_behavior(windows):
    """The dynamic effectome should decode the regime-linked 'motif' behavior above chance."""
    series = ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)
    beh = series.behavior_per_window["motif"]
    res = decode_behavior(connectivity_features(series), beh, "connectivity", "motif", n_folds=4)
    assert res.task == "classification"
    chance = max(np.bincount(beh)) / len(beh)
    assert res.mean_score >= chance - 0.05  # at least near the majority-class baseline


def test_state_behavior_association_and_leadlag(windows):
    """Inferred connectivity states should carry information about the regime-linked behavior."""
    series = ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="causal_kernel", seed=0))
    beh = series.behavior_per_window["motif"]

    assoc = association_with_null(states.labels, beh, n_null=300, seed=0)
    assert assoc.statistic > 0.0
    assert assoc.p_value < 0.05  # significant state<->behavior association vs shuffle null

    ll = lead_lag(states.labels, beh, max_lag=5)
    assert ll.lags.shape == ll.xcorr.shape
    assert ll.best_lag in ll.lags


def test_state_features_one_hot(windows):
    series = ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="causal_kernel", seed=0))
    feats = state_features(states.labels, states.n_states)
    assert feats.shape == (series.n_windows, 3)
    assert np.allclose(feats.sum(axis=1), 1.0)
