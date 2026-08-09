"""Connectivity-state clustering and boundary-safe Markov transition modeling."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor
from effectome.dynamics import (
    GraphStateConfig,
    ProbabilisticStateConfig,
    TransitionConfig,
    affine_invariant_distance,
    fit_graph_states,
    fit_probabilistic_states,
    fit_transitions,
    gromov_wasserstein_distance,
    kmedoids,
    log_euclidean_distance,
    pairwise_distances,
)


def _series(windows):
    return ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)


def test_graph_states_recover_regimes(windows):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="causal_kernel", seed=0))
    assert states.labels.shape[0] == series.n_windows
    assert states.centroids.shape == (3, series.n_neurons, series.n_neurons)
    switches = int(np.sum(np.diff(states.labels) != 0))
    assert switches < series.n_windows // 2


def test_graph_state_transform_is_holdout_safe(windows):
    series = _series(windows)
    fit_idx = np.arange(0, series.n_windows // 2)
    model = fit_graph_states(
        series,
        GraphStateConfig(n_states=3, metric="causal_kernel", seed=0),
        fit_indices=fit_idx,
    )

    heldout = ConnectivitySeries(
        matrices=series.matrices[fit_idx[-1] + 1 : fit_idx[-1] + 5],
        window_starts=series.window_starts[fit_idx[-1] + 1 : fit_idx[-1] + 5],
        method=series.method,
        directed=series.directed,
    )
    prefix = model.transform(heldout)
    fitted_starts = np.array(model.window_starts, copy=True)

    altered = np.concatenate(
        [
            heldout.matrices,
            np.random.default_rng(0).normal(scale=3.0, size=(3,) + heldout.matrices.shape[1:]),
        ],
        axis=0,
    )
    altered_starts = np.concatenate(
        [heldout.window_starts, heldout.window_starts[-1] + 20 + np.arange(3) * 20]
    )
    altered_labels = model.transform(altered, window_starts=altered_starts)
    assert np.array_equal(prefix, altered_labels[: len(prefix)])
    assert np.array_equal(model.window_starts, fitted_starts)


def test_transitions_beat_block_null(windows):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="causal_kernel", seed=0))
    tm = fit_transitions(
        states.labels,
        states.n_states,
        TransitionConfig(n_null=200, seed=0, null_mode="block"),
        window_starts=states.window_starts,
        boundary_indices=states.boundary_indices,
    )
    assert tm.transition_matrix.shape == (3, 3)
    assert np.allclose(tm.transition_matrix.sum(axis=1), 1.0)
    assert tm.p_value < 0.05
    assert tm.heldout_delta_log_likelihood == pytest.approx(
        tm.heldout_transition_log_likelihood - tm.heldout_baseline_log_likelihood
    )


def test_transitions_do_not_cross_recording_boundaries():
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    starts = np.array([0, 20, 40, 60, 0, 20, 40, 60])
    tm = fit_transitions(
        labels,
        2,
        TransitionConfig(laplace=0.0, n_null=8, seed=0),
        window_starts=starts,
    )
    assert np.allclose(tm.transition_matrix[1], [0.0, 1.0])
    assert np.array_equal(tm.boundary_indices, np.array([0, 4]))
    assert np.array_equal(tm.segment_lengths, np.array([4, 4]))


def test_graph_states_use_typed_anchor_recording_boundaries():
    matrices = np.zeros((8, 3, 3), dtype=np.float32)
    matrices[:4, 0, 1] = 1.0
    matrices[4:, 1, 2] = 1.0
    anchors = [
        TemporalAnchor(
            dataset_id="ds",
            recording_id="rec-a" if idx < 4 else "rec-b",
            animal_id=None,
            session_id=None,
            segment_id=None,
            context_start=idx * 20,
            context_stop=idx * 20 + 20,
            target_start=idx * 20,
            target_stop=idx * 20 + 20,
            anchor_sample=idx * 20 + 19,
            anchor_time_seconds=float(idx),
            sampling_rate_hz=10.0,
        )
        for idx in range(8)
    ]
    series = ConnectivitySeries(
        matrices=matrices,
        window_starts=np.arange(8) * 20,
        method="manual",
        directed=True,
        anchors=anchors,
    )
    states = fit_graph_states(series, GraphStateConfig(n_states=2, metric="frobenius", seed=0))
    assert np.array_equal(states.boundary_indices, np.array([0, 4]))


# --------------------------------------------------------------------------- #
# Metric-aware (Riemannian / Gromov-Wasserstein) graph clustering
# --------------------------------------------------------------------------- #


def _spd(n: int, seed: int) -> np.ndarray:
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def test_graph_metric_distances_identity_and_symmetry():
    a, b = _spd(8, 1), _spd(8, 2)
    assert log_euclidean_distance(a, a) < 1e-8
    assert affine_invariant_distance(a, a) < 1e-8
    assert abs(log_euclidean_distance(a, b) - log_euclidean_distance(b, a)) < 1e-8
    assert abs(affine_invariant_distance(a, b) - affine_invariant_distance(b, a)) < 1e-6
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


@pytest.mark.parametrize("metric", ["causal_kernel", "cosine", "log_euclidean", "affine_invariant"])
def test_graph_states_metric_paths_blocky(windows, metric):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric=metric, seed=0))
    assert states.labels.shape[0] == series.n_windows
    assert states.centroids.shape == (3, series.n_neurons, series.n_neurons)
    assert states.metric == metric
    switches = int(np.sum(np.diff(states.labels) != 0))
    shuffled = np.random.default_rng(0).permutation(states.labels)
    shuffled_switches = int(np.sum(np.diff(shuffled) != 0))
    assert switches < shuffled_switches


def test_log_euclidean_centroids_are_symmetric(windows):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="log_euclidean", seed=0))
    for centroid in states.centroids:
        assert np.allclose(centroid, centroid.T, atol=1e-6)


def test_gromov_wasserstein_clustering_small(windows):
    series = _series(windows)
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


def test_probabilistic_states_filtered_prefix_is_future_safe(windows):
    series = _series(windows)
    model = fit_probabilistic_states(
        series,
        ProbabilisticStateConfig(n_states=3, n_components=4, n_iter=25, seed=0),
    )
    prefix = series.matrices[:8]
    future = np.concatenate(
        [prefix, np.ones((3,) + prefix.shape[1:], dtype=prefix.dtype) * 4.0],
        axis=0,
    )
    prefix_projection = model.transform(prefix)
    future_projection = model.transform(future)
    assert np.allclose(prefix_projection.filtered_probs, future_projection.filtered_probs[: len(prefix)])
    assert not np.allclose(prefix_projection.smoothed_probs, future_projection.smoothed_probs[: len(prefix)])


def test_probabilistic_states_report_holdout_scores(windows):
    series = _series(windows)
    fit_idx = np.arange(series.n_windows // 2, dtype=int)
    model = fit_probabilistic_states(
        series,
        ProbabilisticStateConfig(n_states=3, n_components=4, n_iter=25, seed=0),
        fit_indices=fit_idx,
    )
    assert model.heldout_log_likelihood is not None
    assert model.heldout_iid_log_likelihood is not None


def test_probabilistic_states_flag_non_geometric_dwell():
    labels = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=int)
    matrices = np.zeros((labels.size, 2, 2), dtype=np.float32)
    matrices[labels == 1, 0, 1] = 5.0
    series = ConnectivitySeries(
        matrices=matrices,
        window_starts=np.arange(labels.size, dtype=int),
        method="manual",
        directed=True,
    )
    model = fit_probabilistic_states(
        series,
        ProbabilisticStateConfig(n_states=2, n_components=2, n_iter=30, seed=0),
    )
    assert np.any(model.hsmm_candidate.diagnostics.total_variation > 0.1)
    assert model.hsmm_candidate.status == "duration_diagnostic_only"
