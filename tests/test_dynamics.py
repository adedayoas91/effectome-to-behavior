"""Connectivity-state clustering and Markov transition modeling."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module.schema import ConnectivitySeries
from effectome.dynamics import (
    GraphStateConfig,
    TransitionConfig,
    affine_invariant_distance,
    fit_graph_states,
    fit_transitions,
    gromov_wasserstein_distance,
    kmedoids,
    log_euclidean_distance,
    pairwise_distances,
)


def _series(windows):
    return ConnectivityFactory(ConnectivityConfig(name="granger", absolute=True)).run(windows)


def test_graph_states_recover_regimes(windows):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, seed=0))
    assert states.labels.shape[0] == series.n_windows
    assert states.centroids.shape == (3, series.n_neurons, series.n_neurons)
    # The synthetic data has contiguous regimes -> states should be temporally blocky,
    # i.e. far fewer switches than windows.
    switches = int(np.sum(np.diff(states.labels) != 0))
    assert switches < series.n_windows // 2


def test_transitions_beat_shuffle_null(windows):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, seed=0))
    tm = fit_transitions(states.labels, states.n_states, TransitionConfig(n_null=200, seed=0))
    assert tm.transition_matrix.shape == (3, 3)
    assert np.allclose(tm.transition_matrix.sum(axis=1), 1.0)
    # Observed sequence is more structured (higher LL) than time-shuffled surrogates.
    assert tm.p_value < 0.05


# --------------------------------------------------------------------------- #
# Metric-aware (Riemannian / Gromov-Wasserstein) graph clustering
# --------------------------------------------------------------------------- #


def _spd(n: int, seed: int) -> np.ndarray:
    """A well-conditioned symmetric positive-definite test matrix."""
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def test_graph_metric_distances_identity_and_symmetry():
    a, b = _spd(8, 1), _spd(8, 2)
    # Riemannian distances vanish on self and are symmetric.
    assert log_euclidean_distance(a, a) < 1e-8
    assert affine_invariant_distance(a, a) < 1e-8
    assert abs(log_euclidean_distance(a, b) - log_euclidean_distance(b, a)) < 1e-8
    assert abs(affine_invariant_distance(a, b) - affine_invariant_distance(b, a)) < 1e-6
    # Entropic Gromov-Wasserstein is symmetric and non-negative on connectivity-scale
    # (O(1)) graphs. Its self-distance is a positive constant (entropic bias), so we do
    # not assert it is zero; clustering uses only relative distances.
    rng = np.random.default_rng(0)
    c, d = np.abs(rng.normal(size=(8, 8))), np.abs(rng.normal(size=(8, 8)))
    assert gromov_wasserstein_distance(c, d) >= 0.0
    assert abs(gromov_wasserstein_distance(c, d) - gromov_wasserstein_distance(d, c)) < 1e-4


def test_kmedoids_recovers_two_blobs():
    pts = np.array([0.0, 0.1, 0.2, 5.0, 5.1, 5.2])
    d = np.abs(pts[:, None] - pts[None, :])
    labels, medoids = kmedoids(d, 2, seed=0)
    assert len(np.unique(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert len(medoids) == 2


def test_pairwise_distances_shape_symmetry(windows):
    series = _series(windows)
    d = pairwise_distances(series.matrices[:10], "affine_invariant")
    assert d.shape == (10, 10)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0, atol=1e-6)


@pytest.mark.parametrize("metric", ["cosine", "log_euclidean", "affine_invariant"])
def test_graph_states_metric_paths_blocky(windows, metric):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric=metric, seed=0))
    assert states.labels.shape[0] == series.n_windows
    assert states.centroids.shape == (3, series.n_neurons, series.n_neurons)
    assert states.metric == metric
    # Contiguous regimes -> the label sequence is more temporally coherent (fewer state
    # switches) than a random shuffle of the *same* labels. This self-calibrates against
    # the realized state frequencies, so it is robust to the metric and clustering quality.
    switches = int(np.sum(np.diff(states.labels) != 0))
    shuffled = np.random.default_rng(0).permutation(states.labels)
    shuffled_switches = int(np.sum(np.diff(shuffled) != 0))
    assert switches < shuffled_switches


def test_log_euclidean_centroids_are_symmetric(windows):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="log_euclidean", seed=0))
    for c in states.centroids:
        assert np.allclose(c, c.T, atol=1e-6)  # log-Euclidean Fréchet mean is symmetric


def test_gromov_wasserstein_clustering_small(windows):
    series = _series(windows)
    # Subset windows to keep the O(K^2) GW pairwise cost cheap in tests.
    small = ConnectivitySeries(
        matrices=series.matrices[:14],
        window_starts=series.window_starts[:14],
        method=series.method,
        directed=series.directed,
    )
    states = fit_graph_states(
        small,
        GraphStateConfig(n_states=2, metric="gromov_wasserstein", gw_max_iter=50, seed=0),
    )
    assert states.labels.shape[0] == 14
    assert states.metric == "gromov_wasserstein"
    assert states.centroids.shape == (2, series.n_neurons, series.n_neurons)


def test_unknown_metric_raises(windows):
    series = _series(windows)
    with pytest.raises(ValueError, match="unknown metric"):
        fit_graph_states(series, GraphStateConfig(metric="not_a_metric"))
