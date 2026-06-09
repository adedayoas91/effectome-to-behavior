"""Tests for schema, loaders, preprocessing, and windowing."""

from __future__ import annotations

import numpy as np

from effectome.data_module import (
    PreprocessConfig,
    WindowConfig,
    get_loader,
    make_windows,
    preprocess,
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
