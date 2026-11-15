"""Shared targets and correspondence must preserve training and temporal ownership."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from effectome.data_module import PreprocessConfig, WindowConfig, make_windows, preprocess
from effectome.data_module.schema import NeuralRecording
from effectome.linking.shared_transitions import (
    SharedTransitionConfig,
    fit_shared_transition_reference,
    state_correspondence,
)
from effectome.workflows.feature_ownership import chronological_feature_split


def _windows():
    rng = np.random.default_rng(18)
    recording = NeuralRecording(
        traces=rng.normal(size=(3, 640)),
        time=np.arange(640, dtype=float),
        coords=None,
        neuron_ids=np.arange(3),
        behavior={},
        fps=1.0,
    )
    recording = preprocess(
        recording,
        PreprocessConfig(
            detrend=False,
            zscore=False,
            smooth_window=0,
            deconvolve=False,
            drop_low_variance=0.0,
        ),
    )
    windows = make_windows(
        recording,
        WindowConfig(
            mode="temporal",
            history_length=24,
            target_length=4,
            stride=12,
            standardize_per_window=False,
        ),
    )
    fit, evaluation = chronological_feature_split(windows)
    return windows, fit, evaluation


def _reference(windows, fit, evaluation, **overrides):
    return fit_shared_transition_reference(
        windows,
        fit_indices=fit,
        evaluation_indices=evaluation,
        config=SharedTransitionConfig(n_dims=2, n_manifold_states=2, **overrides),
    )


def test_shared_reference_fit_is_invariant_to_heldout_values():
    windows, fit, evaluation = _windows()
    first = _reference(windows, fit, evaluation)
    changed = deepcopy(windows)
    changed.segments[evaluation] = changed.segments[evaluation] * 20 + 100
    second = _reference(changed, fit, evaluation)
    for field in ("scaler_mean", "scaler_scale", "pca_components", "state_centers"):
        np.testing.assert_allclose(getattr(first, field), getattr(second, field))
    np.testing.assert_allclose(first.baseline_features[fit], second.baseline_features[fit])
    np.testing.assert_array_equal(first.events.origins, second.events.origins)
    assert not first.metadata["regime_claim_eligible"]


def test_baseline_at_origin_does_not_use_future_windows():
    windows, fit, evaluation = _windows()
    first = _reference(windows, fit, evaluation)
    origin = int(evaluation[2])
    changed = deepcopy(windows)
    changed.segments[origin + 1 :] *= 25
    second = _reference(changed, fit, evaluation)
    np.testing.assert_allclose(first.baseline_features[: origin + 1], second.baseline_features[: origin + 1])


def test_events_have_common_strict_future_support_and_disjoint_partitions():
    windows, fit, evaluation = _windows()
    result = _reference(windows, fit, evaluation)
    events = result.events
    assert events.origins.size == len(events.source_states) == len(events.horizon_seconds)
    assert np.all(events.horizon_seconds == 12.0)
    for origin, target in zip(events.origins, events.targets, strict=True):
        assert windows.anchors[target].target_start >= windows.anchors[origin].context_stop
    for rows, partition in ((events.train_rows, fit), (events.evaluation_rows, evaluation)):
        assert np.all(np.isin(events.origins[rows], partition))
        assert np.all(np.isin(events.targets[rows], partition))
    train_support = np.unique(np.r_[events.origins[events.train_rows], events.targets[events.train_rows]])
    test_support = np.unique(
        np.r_[events.origins[events.evaluation_rows], events.targets[events.evaluation_rows]]
    )
    for a in train_support:
        for b in test_support:
            assert windows.anchors[a].context_stop <= windows.anchors[b].context_start


@pytest.mark.parametrize("flag", ["gap_before", "bad_frame_before", "valid"])
def test_events_break_at_gaps_and_invalid_anchors(flag):
    windows, fit, evaluation = _windows()
    index = int(evaluation[4])
    windows.anchors[index] = replace(windows.anchors[index], **{flag: flag != "valid"})
    result = _reference(windows, fit, evaluation)
    assert all(
        not (a < index <= b) for a, b in zip(result.events.origins, result.events.targets, strict=True)
    )
    if flag == "valid":
        assert index not in result.events.origins


def test_shared_reference_rejects_leakage_and_unsupported_regions():
    windows, fit, evaluation = _windows()
    with pytest.raises(ValueError, match="raw sample"):
        _reference(windows, np.arange(evaluation[0]), evaluation)
    with pytest.raises(ValueError, match="distinct training"):
        fit_shared_transition_reference(
            windows,
            fit_indices=fit,
            evaluation_indices=evaluation,
            config=SharedTransitionConfig(n_dims=2, n_manifold_states=100),
        )
    windows.metadata["neural_standardization"] = {"enabled": True}
    with pytest.raises(ValueError, match="per-window"):
        _reference(windows, fit, evaluation)


def test_same_manifold_region_can_have_distinct_effectome_codes():
    graph = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    manifold = np.zeros(8, dtype=int)
    result = state_correspondence(
        graph, manifold, fit_indices=np.arange(4), evaluation_indices=np.arange(4, 8)
    )
    assert result["heldout_effectome_count_per_manifold"].tolist() == [2]
    assert result["heldout_entropy_effectome_given_manifold_nats"] == pytest.approx(np.log(2))
    assert result["heldout_entropy_manifold_given_effectome_nats"] == 0.0


def test_same_effectome_code_can_have_distinct_manifold_states_and_futures():
    graph = np.zeros(8, dtype=int)
    manifold = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    result = state_correspondence(
        graph,
        manifold,
        fit_indices=np.arange(4),
        evaluation_indices=np.arange(4, 8),
        future_targets=manifold,
    )
    assert result["heldout_manifold_count_per_effectome"].tolist() == [2]
    assert result["heldout_future_count_per_effectome"].tolist() == [2]
    assert result["heldout_entropy_future_given_effectome_nats"] == pytest.approx(np.log(2))
    assert not result["biological_equivalence_established"]


def test_correspondence_mapping_is_learned_only_from_training():
    graph = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    manifold = graph.copy()
    first = state_correspondence(
        graph, manifold, fit_indices=np.arange(4), evaluation_indices=np.arange(4, 8)
    )
    manifold[4:] = 1 - manifold[4:]
    second = state_correspondence(
        graph, manifold, fit_indices=np.arange(4), evaluation_indices=np.arange(4, 8)
    )
    np.testing.assert_array_equal(
        first["training_effectome_to_manifold"], second["training_effectome_to_manifold"]
    )
    np.testing.assert_array_equal(first["training_contingency"], second["training_contingency"])


def test_baseline_resets_dwell_and_velocity_at_ownership_boundaries():
    windows, fit, evaluation = _windows()
    result = _reference(windows, fit, evaluation)
    names = result.metadata["baseline_feature_names"]
    reset_columns = [
        i for i, name in enumerate(names) if name.startswith("past_") or name == "state_dwell_seconds"
    ]
    np.testing.assert_array_equal(result.baseline_features[evaluation[0], reset_columns], 0.0)


def test_correspondence_unknown_heldout_codes_do_not_expand_training_mapping():
    graph = np.asarray([0, 0, 1, 1, 9, 9, 9, 9])
    manifold = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    result = state_correspondence(
        graph, manifold, fit_indices=np.arange(4), evaluation_indices=np.arange(4, 8)
    )
    assert result["training_effectome_to_manifold"].tolist() == [0, 1]
    assert result["training_contingency"].shape == (2, 2)
    assert result["heldout_mapping_coverage"] == 0.0


def test_nonfinite_histories_require_explicit_policy():
    windows, fit, evaluation = _windows()
    windows.segments[evaluation[0], -1, 0] = np.nan
    with pytest.raises(ValueError, match="finite neural histories"):
        _reference(windows, fit, evaluation)


def test_baseline_neural_history_distinguishes_equal_moment_histories():
    windows, fit, evaluation = _windows()
    first = _reference(windows, fit, evaluation)
    origin = int(evaluation[2])
    changed = deepcopy(windows)
    changed.segments[origin, -2:, 0] = changed.segments[origin, -2:, 0][::-1]
    second = _reference(changed, fit, evaluation)
    names = first.metadata["baseline_feature_names"]
    moment_columns = [i for i, name in enumerate(names) if "activity_mean" in name or "activity_std" in name]
    np.testing.assert_allclose(
        first.baseline_features[origin, moment_columns], second.baseline_features[origin, moment_columns]
    )
    current_column, previous_column = (
        names.index("neural_history_lag_0_0"),
        names.index("neural_history_lag_1_0"),
    )
    assert first.baseline_features[origin, current_column] != second.baseline_features[origin, current_column]
    assert (
        first.baseline_features[origin, previous_column] != second.baseline_features[origin, previous_column]
    )
    assert first.metadata["baseline_neural_history_samples"] == 2
    assert len(names) == first.baseline_features.shape[1]


def test_reference_rejects_context_observations_after_target_anchor():
    windows, fit, evaluation = _windows()
    anchor = windows.anchors[0]
    windows.anchors[0] = replace(
        anchor,
        target_stop=anchor.target_stop - 1,
        anchor_sample=anchor.anchor_sample - 1,
    )
    with pytest.raises(ValueError, match="trailing targets"):
        _reference(windows, fit, evaluation)


def test_shared_reference_rejects_irregular_physical_horizons():
    windows, fit, evaluation = _windows()
    index = int(evaluation[3])
    windows.anchors[index] = replace(
        windows.anchors[index],
        anchor_time_seconds=windows.anchors[index].anchor_time_seconds + 1.0,
    )
    assert all(
        a.anchor_time_seconds < b.anchor_time_seconds
        for a, b in zip(windows.anchors[:-1], windows.anchors[1:], strict=True)
    )
    with pytest.raises(ValueError, match="physical horizon"):
        _reference(windows, fit, evaluation)


def test_gaps_do_not_allow_distinct_horizons_for_the_same_source():
    windows, fit, evaluation = _windows()
    start = int(evaluation[3])
    before = windows.anchors[start - 1].anchor_time_seconds
    for offset, index in enumerate(range(start, windows.n_windows)):
        windows.anchors[index] = replace(
            windows.anchors[index],
            anchor_time_seconds=before + (offset + 1) * 24.0,
            gap_before=index == start,
        )
    with pytest.raises(ValueError, match="physical horizon"):
        _reference(windows, fit, evaluation)


def test_baseline_includes_every_declared_raw_neural_history_sample():
    windows, fit, evaluation = _windows()
    result = _reference(windows, fit, evaluation, neural_history_samples=4)
    names = result.metadata["baseline_feature_names"]
    origin = int(evaluation[2])
    scaled = (windows.segments[origin] - result.scaler_mean) / result.scaler_scale
    for lag in range(4):
        for neuron in range(windows.n_neurons):
            column = names.index(f"neural_history_lag_{lag}_{neuron}")
            assert result.baseline_features[origin, column] == pytest.approx(scaled[-lag - 1, neuron])
    assert not any(name.startswith("neural_history_lag_4_") for name in names)
    assert result.metadata["baseline_neural_history_samples"] == 4
    assert len(names) == result.baseline_features.shape[1]


def test_baseline_rejects_insufficient_declared_neural_history():
    windows, fit, evaluation = _windows()
    with pytest.raises(ValueError, match="neural history samples exceed"):
        _reference(windows, fit, evaluation, neural_history_samples=windows.segments.shape[1] + 1)
    with pytest.raises(ValueError, match="neural_history_samples"):
        SharedTransitionConfig(neural_history_samples=0)
