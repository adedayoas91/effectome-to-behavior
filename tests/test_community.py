"""Community detection: registry, signed role features, and temporal regularization."""

from __future__ import annotations

import numpy as np

from effectome.community import COMMUNITY_REGISTRY, CommunityConfig, CommunityFactory
from effectome.community.temporal import _signed_role_features
from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module.schema import ConnectivitySeries


def _series(windows):
    return ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)


def _manual_series(matrices: np.ndarray) -> ConnectivitySeries:
    return ConnectivitySeries(
        matrices=matrices.astype(np.float32),
        window_starts=np.arange(len(matrices)) * 20,
        method="manual",
        directed=True,
    )


def test_registry_has_methods():
    assert {"leiden", "markov_stability", "temporal", "temporal_greedy"} <= set(COMMUNITY_REGISTRY)


def test_static_community_shapes(windows):
    series = _series(windows)
    det = CommunityFactory(CommunityConfig(name="leiden"))
    com = det.run(series)
    assert com.labels.shape == (series.n_windows, series.n_neurons)
    assert (com.n_communities_per_window >= 1).all()


def test_temporal_greedy_baseline_runs(windows):
    series = _series(windows)
    det = CommunityFactory(CommunityConfig(name="temporal_greedy"))
    com = det.run(series)
    assert com.labels.shape == (series.n_windows, series.n_neurons)
    assert np.issubdtype(com.labels.dtype, np.integer)


def test_signed_role_features_preserve_sign_and_direction():
    matrix = np.array(
        [
            [0.0, 2.0, -1.0],
            [-3.0, 0.0, 0.5],
            [1.0, -2.0, 0.0],
        ]
    )
    abs_sym = 0.5 * (np.abs(matrix) + np.abs(matrix).T)
    assert not np.allclose(_signed_role_features(matrix), _signed_role_features(abs_sym))


def test_temporal_regularized_communities_expose_consensus_metrics():
    base = np.array(
        [
            [0.0, 2.0, 2.0, -1.0],
            [2.0, 0.0, 1.5, -1.0],
            [2.0, 1.5, 0.0, -1.0],
            [-1.0, -1.0, -1.0, 0.0],
        ]
    )
    switched = base.copy()
    switched[3, :2] = 1.8
    switched[:2, 3] = 1.8
    switched[3, 2] = -1.5
    switched[2, 3] = -1.5
    series = _manual_series(np.stack([base, base, switched, switched, switched]))
    det = CommunityFactory(
        CommunityConfig(
            name="temporal",
            symmetrize=False,
            use_absolute=False,
            seed=0,
            extra={"n_communities": 2, "n_runs": 3, "max_iter": 10, "temporal_penalty": 1.0},
        )
    )
    com = det.run(series)
    assert com.labels.shape == (series.n_windows, series.n_neurons)
    assert "flexibility" in com.__dict__
    assert "coassignment" in com.__dict__
    assert com.__dict__["coassignment"].shape == (series.n_windows, series.n_neurons, series.n_neurons)
    assert com.__dict__["flexibility"].max() > 0.0
    assert com.__dict__["switching_rate"] > 0.0
    assert 0.0 <= com.__dict__["stability"] <= 1.0


def test_prospective_temporal_mode_is_future_invariant():
    early = np.array(
        [
            [0.0, 2.5, 2.5, -1.0],
            [2.5, 0.0, 2.0, -1.0],
            [2.5, 2.0, 0.0, -1.0],
            [-1.0, -1.0, -1.0, 0.0],
        ]
    )
    late = np.array(
        [
            [0.0, -1.0, -1.0, 2.5],
            [-1.0, 0.0, -1.0, 2.5],
            [-1.0, -1.0, 0.0, 2.0],
            [2.5, 2.5, 2.0, 0.0],
        ]
    )
    alt_late = -late
    series_a = _manual_series(np.stack([early, early, late, late, late, late]))
    series_b = _manual_series(np.stack([early, early, late, late, alt_late, alt_late]))
    cfg = CommunityConfig(
        name="temporal",
        symmetrize=False,
        use_absolute=False,
        seed=0,
        extra={
            "n_communities": 2,
            "n_runs": 1,
            "mode": "prospective",
            "temporal_penalty": 1.0,
            "init_window_count": 2,
        },
    )
    labels_a = CommunityFactory(cfg).run(series_a).labels
    labels_b = CommunityFactory(cfg).run(series_b).labels
    assert np.array_equal(labels_a[:4], labels_b[:4])
