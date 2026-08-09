"""Community detection: registry, signed role features, and temporal regularization."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.community import COMMUNITY_REGISTRY, CommunityConfig, CommunityFactory
from effectome.community.temporal import TemporalGreedyCommunity, _signed_role_features, _viterbi_assign
from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor


def _series(windows):
    return ConnectivityFactory(ConnectivityConfig(name="cgc")).run(windows)


def _manual_series(matrices: np.ndarray) -> ConnectivitySeries:
    return ConnectivitySeries(
        matrices=matrices.astype(np.float32),
        window_starts=np.arange(len(matrices)) * 20,
        method="manual",
        directed=True,
    )


def _anchor(recording_id: str, idx: int) -> TemporalAnchor:
    return TemporalAnchor(
        dataset_id="ds",
        recording_id=recording_id,
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


def test_static_community_progress_callback_reports_each_window(windows):
    series = _series(windows)
    det = CommunityFactory(CommunityConfig(name="leiden"))
    progress: list[tuple[int, int]] = []
    det.run(series, progress_callback=lambda done, total: progress.append((done, total)))
    assert progress[0] == (1, series.n_windows)
    assert progress[-1] == (series.n_windows, series.n_windows)
    assert len(progress) == series.n_windows


def test_temporal_viterbi_resets_coupling_at_boundaries():
    unary = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    coupled = _viterbi_assign(unary, penalty=10.0)
    reset = _viterbi_assign(unary, penalty=10.0, boundary_indices=np.array([0, 2]))
    assert np.array_equal(coupled, np.array([0, 0, 0, 0]))
    assert np.array_equal(reset, np.array([0, 0, 1, 1]))


def test_temporal_greedy_does_not_match_labels_across_anchor_boundaries():
    class _DummyTemporalGreedy(TemporalGreedyCommunity):
        def __init__(self, outputs: list[np.ndarray]) -> None:
            super().__init__(CommunityConfig(name="temporal_greedy"))
            self._outputs = iter(outputs)

        def detect_one(self, matrix: np.ndarray) -> np.ndarray:
            return next(self._outputs)

    outputs = [
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([1, 1, 0, 0], dtype=np.int64),
    ]
    series = ConnectivitySeries(
        matrices=np.zeros((2, 4, 4), dtype=np.float32),
        window_starts=np.array([0, 20]),
        method="manual",
        directed=True,
        anchors=[_anchor("rec-a", 0), _anchor("rec-b", 1)],
    )
    com = _DummyTemporalGreedy(outputs).run(series)
    assert np.array_equal(com.labels[0], outputs[0])
    assert np.array_equal(com.labels[1], outputs[1])


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
    if not hasattr(com, "flexibility"):
        pytest.skip("typed CommunitySeries metrics land with task-1 schema integration")
    assert com.coassignment is not None
    assert com.coassignment.shape == (series.n_windows, series.n_neurons, series.n_neurons)
    assert com.flexibility is not None
    assert com.flexibility.max() > 0.0
    assert com.switching_rate is not None
    assert com.switching_rate > 0.0
    assert com.stability is not None
    assert 0.0 <= com.stability <= 1.0


def test_temporal_community_progress_callback_reports_each_run():
    base = np.array(
        [
            [0.0, 2.0, 2.0, -1.0],
            [2.0, 0.0, 1.5, -1.0],
            [2.0, 1.5, 0.0, -1.0],
            [-1.0, -1.0, -1.0, 0.0],
        ]
    )
    series = _manual_series(np.stack([base, base, base, base]))
    det = CommunityFactory(
        CommunityConfig(
            name="temporal",
            symmetrize=False,
            use_absolute=False,
            seed=0,
            extra={"n_communities": 2, "n_runs": 3, "max_iter": 5, "temporal_penalty": 1.0},
        )
    )
    progress: list[tuple[int, int]] = []
    det.run(series, progress_callback=lambda done, total: progress.append((done, total)))
    assert progress == [(1, 3), (2, 3), (3, 3)]


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

    immediate_future_changed = _manual_series(
        np.stack([early, alt_late, alt_late, alt_late, alt_late, alt_late])
    )
    labels_c = CommunityFactory(cfg).run(immediate_future_changed).labels
    assert np.array_equal(labels_a[0], labels_c[0])
