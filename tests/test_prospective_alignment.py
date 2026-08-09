"""Regression tests for prospective manifold and community-transition features."""

from __future__ import annotations

import numpy as np

from effectome.connectivity import ConnectivityConfig
from effectome.data_module.schema import CommunitySeries, ConnectivitySeries, TemporalAnchor
from effectome.dynamics.graph_states import boundary_indices_for_series
from effectome.linking import (
    anchor_continuity_labels,
    community_reconfiguration_features,
    future_manifold_displacement,
    valid_positive_lag_origins,
)
from effectome.manifold import causal_forward_fill_nonfinite


def test_future_manifold_displacement_is_forward_and_chart_invariant() -> None:
    embedding = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]])
    displacement, magnitude = future_manifold_displacement(
        embedding,
        np.array([0, 1]),
        lag=1,
        groups=np.array(["fold-a", "fold-a", "fold-a"], dtype=object),
    )
    assert np.array_equal(displacement, [[3.0, 4.0], [0.0, 4.0]])
    assert np.array_equal(magnitude, [5.0, 4.0])

    rotated = embedding @ np.array([[0.0, -1.0], [1.0, 0.0]])
    _, rotated_magnitude = future_manifold_displacement(
        rotated,
        np.array([0, 1]),
        lag=1,
    )
    assert np.allclose(rotated_magnitude, magnitude)


def test_future_manifold_displacement_rejects_fold_crossing() -> None:
    embedding = np.array([[0.0], [1.0], [2.0]])
    groups = np.array(["fold-a", "fold-a", "fold-b"], dtype=object)
    with np.testing.assert_raises_regex(ValueError, "continuity group"):
        future_manifold_displacement(embedding, np.array([1]), lag=1, groups=groups)


def test_community_reconfiguration_features_keep_neuron_identity() -> None:
    labels = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]])
    series = CommunitySeries(
        labels=labels,
        method="temporal",
        n_communities_per_window=np.array([2, 2, 2]),
        switching=np.array([[False, True, False], [True, False, True]]),
    )
    features = community_reconfiguration_features(series)
    assert features.shape == (3, 4)
    assert np.allclose(features[:, 0], [0.0, 1 / 3, 2 / 3])
    assert np.array_equal(features[:, 1:], [[0, 0, 0], [0, 1, 0], [1, 0, 1]])


def test_positive_lag_origins_reset_at_sparse_bad_frames() -> None:
    def anchor(index: int, *, bad_after: bool = False, bad_before: bool = False) -> TemporalAnchor:
        return TemporalAnchor(
            dataset_id="dataset",
            recording_id="recording",
            animal_id=None,
            session_id=None,
            segment_id=None,
            context_start=index,
            context_stop=index + 2,
            target_start=index + 1,
            target_stop=index + 2,
            anchor_sample=index + 1,
            anchor_time_seconds=float(index + 1),
            sampling_rate_hz=1.0,
            bad_frame_after=bad_after,
            bad_frame_before=bad_before,
        )

    anchors = [anchor(0, bad_after=True), anchor(2, bad_before=True), anchor(4)]
    assert np.array_equal(valid_positive_lag_origins(3, 1, anchors), np.array([1]))
    continuity = anchor_continuity_labels(anchors)
    assert continuity[0] != continuity[1]
    assert continuity[1] == continuity[2]
    series = ConnectivitySeries(
        matrices=np.zeros((3, 1, 1), dtype=float),
        window_starts=np.array([0, 2, 4]),
        method="cgc",
        directed=True,
        anchors=anchors,
    )
    assert np.array_equal(boundary_indices_for_series(series), np.array([0, 1]))


def test_connectivity_lag_horizon_resolves_from_each_recording_frequency() -> None:
    cfg = ConnectivityConfig(name="cgc", max_lag=2, max_lag_seconds=0.7)
    worm = cfg.resolve(2.9046296296296297)
    fish = cfg.resolve(6.0268)
    assert worm.max_lag == 2
    assert fish.max_lag == 4
    assert worm.extra["n_lags"] == 2
    assert fish.extra["n_lags"] == 4
    assert fish.extra["lag_contract"]["request_mode"] == "duration_first"


def test_manifold_missing_values_are_filled_causally_without_time_compression() -> None:
    neural = np.array([[np.nan, 2.0], [1.0, np.nan], [np.nan, 4.0], [3.0, 5.0]])
    filled, contract = causal_forward_fill_nonfinite(neural)
    assert np.array_equal(filled, [[0.0, 2.0], [1.0, 2.0], [1.0, 4.0], [3.0, 5.0]])
    assert contract["nonfinite_value_count"] == 3
    assert contract["nonfinite_frame_count"] == 3
    assert contract["future_samples_used"] is False
    assert contract["clock_compressed"] is False
