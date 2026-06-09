"""Connectivity estimators: interface + ground-truth recovery."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from effectome.connectivity import CONNECTIVITY_REGISTRY, ConnectivityConfig, ConnectivityFactory


def test_registry_has_methods():
    assert {"correlation", "granger", "pcmci"} <= set(CONNECTIVITY_REGISTRY)


def test_correlation_runs(windows):
    est = ConnectivityFactory(ConnectivityConfig(name="correlation", partial=True))
    series = est.run(windows)
    assert series.matrices.shape == (windows.n_windows, windows.n_neurons, windows.n_neurons)
    assert not series.directed


def test_granger_recovers_ground_truth(windows, true_graph):
    """Multivariate Granger should rank true edges above non-edges (AUROC > 0.6)."""
    est = ConnectivityFactory(ConnectivityConfig(name="granger", max_lag=1, ridge=1.0, absolute=True))
    series = est.run(windows)
    mean_influence = series.matrices.mean(axis=0)

    n = mean_influence.shape[0]
    off = ~np.eye(n, dtype=bool)
    scores = mean_influence[off]
    labels = true_graph[off]
    auroc = roc_auc_score(labels, scores)
    assert auroc > 0.6, f"Granger AUROC too low: {auroc:.3f}"
