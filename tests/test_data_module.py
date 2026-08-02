"""Tests for schema, loaders, preprocessing, and windowing."""

from __future__ import annotations

import numpy as np

from effectome.data_module import (
    CalciumSegmentationConfig,
    PreprocessConfig,
    WindowConfig,
    get_loader,
    make_overlapping_calcium_windows,
    make_taper,
    make_windows,
    preprocess,
    segment_calcium_traces,
)


def test_synthetic_loader_shapes():
    rec = get_loader("synthetic")({"name": "synthetic", "n_neurons": 10, "n_timepoints": 500})
    assert rec.traces.shape == (10, 500)
    assert rec.time.shape == (500,)
    assert "true_graphs" in rec.metadata


def test_preprocess_zscore(synthetic_recording):
    pre = preprocess(synthetic_recording, PreprocessConfig(detrend=True, zscore=True))
    # z-scored traces have ~unit variance per neuron.
    assert np.allclose(pre.traces.std(axis=1), 1.0, atol=0.2)
    # input is not mutated.
    assert pre.traces is not synthetic_recording.traces


def test_sliding_windows_shapes(synthetic_recording):
    w = make_windows(synthetic_recording, WindowConfig(length=100, stride=50, mode="sliding"))
    assert w.segments.shape[1] == 100
    assert w.segments.shape[2] == synthetic_recording.n_neurons
    assert w.n_windows == len(w.windows)
    for name in synthetic_recording.behavior:
        assert w.behavior_per_window[name].shape[0] == w.n_windows


def test_behavior_aligned_windows(synthetic_recording):
    w = make_windows(
        synthetic_recording,
        WindowConfig(length=60, stride=10, mode="behavior", align_event="motif", min_event_gap=20),
    )
    assert w.n_windows > 0
    assert w.segments.shape[1] == 60


def test_overlapping_calcium_segmentation_preserves_context_and_centers():
    data = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    behavior = {"motif": (np.arange(100) // 10).astype(np.int64)}
    cfg = CalciumSegmentationConfig(
        core_length=20,
        stride=5,
        left_context=4,
        right_context=6,
        lag_context=8,
        taper="none",
        apply_taper=False,
    )

    windows = segment_calcium_traces(data, cfg, behavior=behavior, fps=10.0)

    assert windows.segments.shape == (14, 34, 3)
    assert windows.windows[0].start == 0
    assert windows.windows[0].stop == 34
    assert windows.metadata["core_windows"][0].start == 8
    assert windows.metadata["core_windows"][0].stop == 28
    assert windows.metadata["center_samples"][0] == 17.5
    assert np.isclose(windows.metadata["center_times"][0], 1.75)
    assert np.isclose(windows.metadata["overlap_fraction"], 1.0 - 5 / 34)
    assert windows.behavior_per_window["motif"].shape == (windows.n_windows,)


def test_overlapping_calcium_segmentation_applies_taper():
    data = np.ones((40, 2), dtype=np.float32)
    cfg = CalciumSegmentationConfig(
        core_length=10,
        stride=2,
        left_context=3,
        right_context=3,
        taper="hann",
        apply_taper=True,
    )

    windows = segment_calcium_traces(data, cfg)
    weights = windows.metadata["sample_weights"]

    assert np.isclose(weights[0], 0.0)
    assert weights.max() > 0.9
    assert np.allclose(windows.segments[0, :, 0], weights)


def test_make_overlapping_calcium_windows_from_recording(synthetic_recording):
    cfg = CalciumSegmentationConfig(core_length=50, stride=10, left_context=8, right_context=8)
    windows = make_overlapping_calcium_windows(synthetic_recording, cfg)

    assert windows.segments.shape[1] == cfg.full_length
    assert windows.segments.shape[2] == synthetic_recording.n_neurons
    assert windows.metadata["segmentation"] == "overlapping_calcium"
    for name in synthetic_recording.behavior:
        assert windows.behavior_per_window[name].shape[0] == windows.n_windows


def test_taper_variants():
    assert np.allclose(make_taper(5, "none"), 1.0)
    assert make_taper(5, "gaussian").shape == (5,)
