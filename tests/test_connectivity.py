"""Connectivity estimators: interface + ground-truth recovery."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from effectome.connectivity import CONNECTIVITY_REGISTRY, ConnectivityConfig, ConnectivityFactory
from effectome.experiments import analyze_with_cgc, analyze_with_cgc_star


def test_registry_has_methods():
    assert {"correlation", "granger", "granger_star", "pcmci", "jpcmci"} <= set(CONNECTIVITY_REGISTRY)


def test_correlation_runs(windows):
    est = ConnectivityFactory(ConnectivityConfig(name="correlation", partial=True))
    series = est.run(windows)
    assert series.matrices.shape == (windows.n_windows, windows.n_neurons, windows.n_neurons)
    assert not series.directed


def test_granger_recovers_ground_truth(windows, true_graph):
    """Multivariate Granger should rank true edges above non-edges (AUROC > 0.6)."""
    est = ConnectivityFactory(ConnectivityConfig(name="granger", max_lag=1, ridge=1.0))
    series = est.run(windows)
    mean_influence = series.matrices.mean(axis=0)

    n = mean_influence.shape[0]
    off = ~np.eye(n, dtype=bool)
    scores = np.abs(mean_influence[off])
    labels = true_graph[off]
    auroc = roc_auc_score(labels, scores)
    assert auroc > 0.6, f"Granger AUROC too low: {auroc:.3f}"


def test_granger_preserves_sign(windows):
    est = ConnectivityFactory(ConnectivityConfig(name="granger", max_lag=1, ridge=1.0))
    series = est.run(windows)
    mats = series.matrices[:, ~np.eye(series.n_neurons, dtype=bool)]
    assert np.any(mats > 0.0)
    assert np.any(mats < 0.0)


def test_notebook_cgc_adapters_return_weighted_matrices(synthetic_recording):
    X = synthetic_recording.traces.T
    cgc = analyze_with_cgc(X, [1, 2])
    cgc_star = analyze_with_cgc_star(X, [1])
    assert set(cgc) == {1, 2}
    assert set(cgc_star) == {1}
    assert cgc[1].shape == (X.shape[1], X.shape[1])
    assert cgc_star[1].shape == (X.shape[1], X.shape[1])
