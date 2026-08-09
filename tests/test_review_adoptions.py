"""Focused regression tests for reviewer-adopted leakage and sensitivity controls."""

from __future__ import annotations

import numpy as np

from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor
from effectome.linking import (
    DependencySupport,
    SignedEffectomeScaler,
    activity_magnitude_features,
    anchor_dependency_intervals,
    combined_continuity_groups,
    lead_lag,
    manifold_speed,
    purged_blocked_splits,
    signed_effectome_agreement,
)
from effectome.utils.synthetic import SyntheticConfig, make_synthetic_recording


def _anchor(start: int, stop: int, target_length: int = 2) -> TemporalAnchor:
    return TemporalAnchor(
        dataset_id="test",
        recording_id="recording",
        animal_id="animal",
        session_id=None,
        segment_id=None,
        context_start=start,
        context_stop=stop,
        target_start=stop - target_length,
        target_stop=stop,
        anchor_sample=stop - 1,
        anchor_time_seconds=float(stop - 1),
        sampling_rate_hz=1.0,
    )


def test_dependency_support_expands_raw_interval_and_purges_overlap() -> None:
    anchors = [_anchor(start, start + 4) for start in range(0, 40, 4)]
    support = DependencySupport(lag_extension=1, preprocessing_past=1, preprocessing_future=1)
    assert anchor_dependency_intervals(anchors[1], support) == ((2, 9),)

    splits = purged_blocked_splits(
        len(anchors),
        n_splits=5,
        anchors=anchors,
        dependency_support=support,
    )
    for train, test in splits:
        for train_idx in train:
            train_interval = anchor_dependency_intervals(anchors[int(train_idx)], support)[0]
            for test_idx in test:
                test_interval = anchor_dependency_intervals(anchors[int(test_idx)], support)[0]
                assert max(train_interval[0], test_interval[0]) >= min(
                    train_interval[1], test_interval[1]
                )


def test_future_outcome_target_is_part_of_dependency_set() -> None:
    anchors = [_anchor(start, start + 3, target_length=1) for start in range(0, 30, 3)]
    outcome_anchors = anchors[1:] + [anchors[-1]]
    intervals = anchor_dependency_intervals(anchors[0], outcome_anchor=outcome_anchors[0])
    assert intervals == ((0, 3), (5, 6))
    splits = purged_blocked_splits(
        len(anchors),
        n_splits=5,
        anchors=anchors,
        outcome_anchors=outcome_anchors,
    )
    assert len(splits) == 5


def test_signed_scaling_preserves_sign_and_agreement_is_scale_free() -> None:
    matrices = np.array(
        [
            [[0.0, 1.0, -2.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
            [[0.0, 2.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ]
    )
    reference = ConnectivitySeries(
        matrices=matrices,
        window_starts=np.array([0, 1]),
        method="cgc",
        directed=True,
    )
    candidate = ConnectivitySeries(
        matrices=10.0 * matrices,
        window_starts=np.array([0, 1]),
        method="cgc_star",
        directed=True,
    )
    scaler = SignedEffectomeScaler.fit(reference.matrices, fit_indices=[0])
    transformed = scaler.transform(reference.matrices)
    assert np.array_equal(np.sign(transformed), np.sign(reference.matrices))
    agreement = signed_effectome_agreement(reference, candidate)
    assert np.isclose(agreement.edge_rank_correlation, 1.0)
    assert agreement.sign_agreement == 1.0
    assert agreement.support_jaccard == 1.0


def test_activity_features_and_adjusted_lead_lag_are_finite() -> None:
    anchors = [_anchor(0, 4), _anchor(2, 6), _anchor(4, 8), _anchor(6, 10)]
    traces = np.arange(30, dtype=float).reshape(10, 3)
    activity = activity_magnitude_features(traces, anchors)
    assert activity.shape == (4, 3)
    result = lead_lag(
        np.array([0, 0, 1, 1]),
        np.array([0.0, 0.5, 1.0, 0.5]),
        max_lag=1,
        controls=activity,
    )
    assert np.all(np.isfinite(result.xcorr))


def test_manifold_derivative_breaks_at_cross_fit_fold_boundaries() -> None:
    embedding = np.array([[0.0, 0.0], [1.0, 0.0], [20.0, 0.0], [21.0, 0.0]])
    groups = combined_continuity_groups(
        np.array(["recording"] * 4, dtype=object),
        np.array([0, 0, 1, 1]),
        4,
    )
    assert np.array_equal(manifold_speed(embedding, groups=groups), [0.0, 1.0, 0.0, 1.0])


def test_synthetic_stress_dgp_records_lags_confounders_and_calcium() -> None:
    cfg = SyntheticConfig(
        n_neurons=8,
        n_timepoints=160,
        density=0.7,
        max_lag=2,
        n_states=2,
        regime_dwell=25,
        behavior_drivers=2,
        regime_sign_flip_fraction=0.5,
        regime_sparsity_jitter=0.25,
        instantaneous_density=0.4,
        instantaneous_strength=0.15,
        n_latent_confounders=2,
        latent_strength=0.25,
        calcium_decay_range=(0.55, 0.85),
        calcium_gain_heterogeneity=0.2,
        seed=9,
    )
    recording = make_synthetic_recording(cfg)
    recording.validate()
    metadata = recording.metadata
    assert metadata["true_lag_graphs"].shape == (2, 2, 8, 8)
    assert metadata["true_instantaneous_graphs"].shape == (2, 8, 8)
    assert metadata["latent_factors"].shape == (160, 2)
    assert metadata["latent_loadings"].shape == (8, 2)
    assert np.any(metadata["regime_sign_flip_masks"])
    assert np.all((metadata["calcium_decay"] >= 0.55) & (metadata["calcium_decay"] <= 0.85))
    assert not np.allclose(recording.traces.T, metadata["latent_neural_activity"])
    lag_graphs = metadata["true_lag_graphs"]
    for state_graphs in lag_graphs:
        n = state_graphs.shape[1]
        p = state_graphs.shape[0]
        companion = np.zeros((n * p, n * p))
        companion[:n] = np.concatenate([graph.T for graph in state_graphs], axis=1)
        if p > 1:
            companion[n:, :-n] = np.eye(n * (p - 1))
        assert np.max(np.abs(np.linalg.eigvals(companion))) < 1.0
