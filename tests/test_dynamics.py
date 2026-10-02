"""Connectivity-state clustering and boundary-safe Markov transition modeling."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.community import CommunityConfig, fit_temporal_role_model
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
    return ConnectivityFactory(ConnectivityConfig(name="cgc")).run(windows)


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


def _lagged_state_series(lagged: np.ndarray) -> ConnectivitySeries:
    matrices = lagged.sum(axis=1).astype(np.float32)
    return ConnectivitySeries(
        matrices=matrices,
        lagged_matrices=lagged.astype(np.float32),
        window_starts=np.arange(lagged.shape[0], dtype=int),
        method="manual",
        directed=True,
    )


def test_state_fit_transform_consistency():
    lagged = np.zeros((8, 2, 3, 3), dtype=np.float64)
    lagged[:4, 0, 0, 1] = 2.0
    lagged[4:, 0, 1, 2] = 2.0
    series = _lagged_state_series(lagged)

    model = fit_graph_states(
        series,
        GraphStateConfig(
            n_states=2,
            feature_representation="signed_lag_offdiag",
            assignment="kmeans",
            seed=0,
        ),
    )

    assert np.array_equal(model.transform(series), model.labels)
    assert model.edge_indices is not None
    assert model.lag_count == 2
    assert model.n_nodes == 3


def test_signed_state_centroids_depend_only_on_fit_indices():
    lagged = np.zeros((8, 1, 3, 3), dtype=np.float64)
    lagged[:4, 0, 0, 1] = 1.0
    lagged[4:, 0, 0, 1] = 100.0
    fit_idx = np.arange(4)
    series = _lagged_state_series(lagged)
    altered = _lagged_state_series(lagged.copy())
    altered.lagged_matrices[4:, 0, 0, 1] = -100.0
    altered.matrices[4:, 0, 1] = -100.0

    cfg = GraphStateConfig(
        n_states=1,
        feature_representation="signed_lag_offdiag",
        assignment="kmeans",
        seed=0,
    )
    model = fit_graph_states(series, cfg, fit_indices=fit_idx)
    changed = fit_graph_states(altered, cfg, fit_indices=fit_idx)

    assert np.allclose(model.centroids, changed.centroids)
    assert not np.allclose(model.descriptive_centroids, changed.descriptive_centroids)


def test_signed_lag_features_preserve_opposite_lag_effects():
    lagged = np.zeros((6, 2, 3, 3), dtype=np.float64)
    lagged[:3, 0, 0, 1] = 1.0
    lagged[:3, 1, 0, 1] = -1.0
    lagged[3:, 0, 0, 1] = -1.0
    lagged[3:, 1, 0, 1] = 1.0
    series = _lagged_state_series(lagged)
    assert np.allclose(series.matrices, 0.0)

    model = fit_graph_states(
        series,
        GraphStateConfig(
            n_states=2,
            feature_representation="signed_lag_offdiag",
            assignment="kmeans",
            seed=0,
        ),
    )

    assert len(np.unique(model.labels)) == 2
    assert model.labels[0] == model.labels[1] == model.labels[2]
    assert model.labels[3] == model.labels[4] == model.labels[5]
    assert model.labels[0] != model.labels[3]


def test_one_state_quantizer():
    lagged = np.random.default_rng(0).normal(size=(5, 2, 3, 3))
    series = _lagged_state_series(lagged)

    model = fit_graph_states(
        series,
        GraphStateConfig(
            n_states=1,
            feature_representation="signed_lag_offdiag",
            assignment="kmeans",
            seed=0,
        ),
    )

    assert model.n_states == 1
    assert np.all(model.labels == 0)
    assert np.isnan(model.silhouette)


def test_registered_schema_and_predictive_selection():
    series = _lagged_state_series(np.random.default_rng(7).normal(size=(8, 2, 3, 3)))
    series.diagnostics.update(node_ids=["a", "b", "c"], physical_lags=[0.1, 0.2],
                              representation_units="raw")
    cfg = GraphStateConfig(select_k=True, feature_representation="signed_lag_offdiag")
    with pytest.raises(ValueError, match="predictive"):
        fit_graph_states(series, cfg)
    model = fit_graph_states(series, cfg, predictive_losses={1: 0., 2: 1., 3: 2., 4: 3., 6: 4.})
    assert model.n_states == 1
    np.testing.assert_allclose(model.lag_centroids[0], series.lagged_matrices.mean(axis=0))
    series.diagnostics["node_ids"] = ["b", "a", "c"]
    with pytest.raises(ValueError, match="schema"):
        model.transform(series)
    series.diagnostics["node_ids"] = ["a", "b", "c"]
    series.diagnostics["physical_lags"] = [0.2, 0.4]
    with pytest.raises(ValueError, match="schema"):
        model.transform(series)


def test_hmm_explicit_test_and_resumed_filter():
    series = _lagged_state_series(np.random.default_rng(8).normal(size=(20, 1, 3, 3)))
    cfg = ProbabilisticStateConfig(n_states=2, n_iter=3)
    model = fit_probabilistic_states(series, cfg, fit_indices=np.arange(10),
                                     test_indices=np.array([15, 16, 19]))
    assert model.test_indices.tolist() == [15, 16, 19]
    first = model.filter(series.matrices[:7])
    second = model.filter(series.matrices[7:], state=first.final_state)
    whole = model.filter(series.matrices)
    np.testing.assert_allclose(np.concatenate([first.filtered_probs, second.filtered_probs]),
                               whole.filtered_probs)
    assert len(second.state_snapshots) == 13
    unscored = fit_probabilistic_states(series, cfg, fit_indices=np.arange(10))
    assert unscored.heldout_log_likelihood is None


def test_state_schema_rejects_reordered_nodes_or_lags():
    lagged = np.random.default_rng(0).normal(size=(6, 2, 3, 3))
    series = _lagged_state_series(lagged)
    model = fit_graph_states(
        series,
        GraphStateConfig(
            n_states=2,
            feature_representation="signed_lag_offdiag",
            assignment="kmeans",
            seed=0,
        ),
    )

    with pytest.raises(ValueError, match="lag schema mismatch"):
        model.transform(np.zeros((2, 1, 3, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="node schema mismatch"):
        model.transform(np.zeros((2, 2, 4, 4), dtype=np.float64))


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
    assert 1.0 / 201.0 <= tm.p_value <= 1.0
    assert tm.heldout_delta_log_likelihood == pytest.approx(
        tm.heldout_transition_log_likelihood - tm.heldout_baseline_log_likelihood
    )


def test_fitted_temporal_roles_are_prefix_invariant_and_resumable():
    matrices = np.random.default_rng(918).normal(size=(12, 3, 3))
    matrices[:, np.arange(3), np.arange(3)] = 0.0
    series = ConnectivitySeries(
        matrices=matrices.astype(np.float32),
        window_starts=np.arange(12),
        method="manual",
        directed=True,
        diagnostics={"node_ids": ["a", "b", "c"]},
    )
    cfg = CommunityConfig(
        name="temporal",
        seed=3,
        extra={"n_communities": 2, "online_learning_rate": 0.2},
    )
    model = fit_temporal_role_model(series, cfg, fit_indices=np.arange(5))
    first = ConnectivitySeries(
        matrices=series.matrices[:7],
        window_starts=series.window_starts[:7],
        method=series.method,
        directed=True,
        diagnostics=dict(series.diagnostics),
    )
    tail_changed = ConnectivitySeries(
        matrices=np.concatenate([series.matrices[:7], series.matrices[7:] * -100.0]),
        window_starts=series.window_starts,
        method=series.method,
        directed=True,
        diagnostics=dict(series.diagnostics),
    )
    prefix = model.filter(first)
    altered = model.filter(tail_changed)
    np.testing.assert_array_equal(prefix.labels, altered.labels[:7])
    resumed = model.filter(
        ConnectivitySeries(
            matrices=series.matrices[7:],
            window_starts=series.window_starts[7:],
            method=series.method,
            directed=True,
            diagnostics=dict(series.diagnostics),
        ),
        state=prefix.final_state,
    )
    whole = model.filter(series)
    np.testing.assert_array_equal(
        np.concatenate([prefix.labels, resumed.labels]),
        whole.labels,
    )
    reordered = ConnectivitySeries(
        matrices=series.matrices,
        window_starts=series.window_starts,
        method=series.method,
        directed=True,
        diagnostics={"node_ids": ["b", "a", "c"]},
    )
    with pytest.raises(ValueError, match="schema"):
        model.filter(reordered)


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


def test_transition_progress_callback_reports_all_null_draws(windows):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="causal_kernel", seed=0))
    progress: list[tuple[int, int]] = []
    fit_transitions(
        states.labels,
        states.n_states,
        TransitionConfig(n_null=6, seed=0),
        window_starts=states.window_starts,
        boundary_indices=states.boundary_indices,
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    assert progress == [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]


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
    assert switches <= shuffled_switches


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
    assert np.array_equal(
        prefix_projection.filtered_labels,
        future_projection.filtered_labels[: len(prefix)],
    )
    assert np.allclose(
        prefix_projection.filtered_entropy,
        future_projection.filtered_entropy[: len(prefix)],
    )
    assert not np.allclose(prefix_projection.smoothed_probs, future_projection.smoothed_probs[: len(prefix)])
    assert np.array_equal(prefix_projection.labels, prefix_projection.filtered_labels)
    assert np.allclose(prefix_projection.uncertainty, prefix_projection.filtered_entropy)
    assert np.array_equal(model.labels, model.filtered_labels)
    assert np.allclose(model.uncertainty, model.filtered_entropy)


def test_probabilistic_states_report_holdout_scores(windows):
    series = _series(windows)
    fit_idx = np.arange(series.n_windows // 2, dtype=int)
    model = fit_probabilistic_states(
        series,
        ProbabilisticStateConfig(n_states=3, n_components=4, n_iter=25, seed=0),
        fit_indices=fit_idx,
        test_indices=np.arange(len(fit_idx), series.n_windows),
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


def test_probabilistic_progress_callback_reports_em_iterations(windows):
    series = _series(windows)
    progress: list[tuple[int, int]] = []
    fit_probabilistic_states(
        series,
        ProbabilisticStateConfig(n_states=2, n_components=2, n_iter=8, seed=0),
        progress_callback=lambda done, total: progress.append((done, total)),
    )
    assert progress
    assert progress[0] == (1, 8)
    assert progress[-1][1] == 8
    assert [done for done, _ in progress] == sorted(done for done, _ in progress)
    assert progress[-1][0] <= 8
