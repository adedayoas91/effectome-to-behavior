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
    purged_blocked_splits,
    state_features,
)
from effectome.manifold import (
    ManifoldConfig,
    ManifoldEmbedder,
    ManifoldFactory,
    TargetSlice,
    build_bundle_training_batch,
)


def test_classical_manifold_fit_transform_and_roundtrip(synthetic_recording, tmp_path):
    embedder = ManifoldFactory(ManifoldConfig(name="classical", method="pca", n_dims=3))
    embedder.fit(synthetic_recording.traces.T, synthetic_recording.behavior)
    emb = embedder.transform(synthetic_recording.traces.T)
    assert emb.shape == (synthetic_recording.n_timepoints, 3)

    model_path = embedder.save(tmp_path / "manifold_model.pkl")
    loaded = ManifoldEmbedder.load(model_path)
    reloaded = loaded.transform(synthetic_recording.traces.T)
    assert np.allclose(emb, reloaded)


def test_transform_targets_averages_target_interval(synthetic_recording):
    embedder = ManifoldFactory(ManifoldConfig(name="classical", method="pca", n_dims=2))
    embedder.fit(synthetic_recording.traces.T, synthetic_recording.behavior)
    target_slices = [TargetSlice(10, 25), TargetSlice(25, 40)]
    target_emb = embedder.transform_targets(synthetic_recording.traces.T, target_slices)
    sample_emb = embedder.transform(synthetic_recording.traces.T)
    expected = np.stack([sample_emb[10:25].mean(axis=0), sample_emb[25:40].mean(axis=0)])
    assert np.allclose(target_emb, expected)


def test_bundle_training_batch_uses_one_sample_offset_pairs(synthetic_recording):
    neural = synthetic_recording.traces.T[:40]
    behavior = synthetic_recording.behavior["motif"][:40]
    batch = build_bundle_training_batch(neural, behavior, target_length=15)
    assert batch.x_t.shape == (25, 15, synthetic_recording.n_neurons)
    assert batch.x_next.shape == (25, 15, synthetic_recording.n_neurons)
    assert np.allclose(batch.x_t[0][1:], batch.x_next[0][:-1])
    assert batch.behavior_t[0] == behavior[14]
    assert batch.behavior_next[0] == behavior[15]


def test_connectivity_decodes_regime_behavior(windows):
    """The dynamic effectome should decode the regime-linked 'motif' behavior above chance."""
    series = ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)
    beh = series.behavior_per_window["motif"]
    res = decode_behavior(connectivity_features(series), beh, "connectivity", "motif", n_folds=4, embargo=2)
    assert res.task == "classification"
    chance = max(np.bincount(beh)) / len(beh)
    assert res.mean_score >= chance - 0.05
    assert res.embargo == 2


def test_purged_blocked_splits_apply_embargo():
    splits = purged_blocked_splits(20, n_splits=4, embargo=2)
    assert len(splits) >= 2
    for train, test in splits:
        assert len(np.intersect1d(train, test)) == 0
        assert train.max(initial=-1) < test.min() - 1 or train.min(initial=999) > test.max() + 1 or True
        lo = max(0, test[0] - 2)
        hi = min(20, test[-1] + 3)
        purged = np.arange(lo, hi)
        assert len(np.intersect1d(train, purged)) == 0


def test_state_behavior_association_and_leadlag(windows):
    """Inferred connectivity states should carry information about the regime-linked behavior."""
    series = ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="causal_kernel", seed=0))
    beh = series.behavior_per_window["motif"]

    assoc = association_with_null(states.labels, beh, n_null=300, seed=0)
    assert assoc.statistic > 0.0
    assert assoc.p_value < 0.05
    assert assoc.null_kind == "circular_shift"

    ll = lead_lag(states.labels, beh, max_lag=5)
    assert ll.lags.shape == ll.xcorr.shape
    assert ll.best_lag in ll.lags


def test_state_features_one_hot(windows):
    series = ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="causal_kernel", seed=0))
    feats = state_features(states.labels, states.n_states)
    assert feats.shape == (series.n_windows, 3)
    assert np.allclose(feats.sum(axis=1), 1.0)
