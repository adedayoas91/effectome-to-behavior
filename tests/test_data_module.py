"""Tests for schema, loaders, preprocessing, and windowing."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.data_module import (
    ArtifactProvenance,
    CalciumSegmentationConfig,
    CommunitySeries,
    ConnectivitySeries,
    NeuralRecording,
    PreprocessConfig,
    RecordingIdentity,
    TemporalAnchor,
    WindowConfig,
    get_loader,
    make_overlapping_calcium_windows,
    make_taper,
    make_windows,
    preprocess,
    segment_calcium_traces,
)
from effectome.manifold import ManifoldArtifact, TargetSlice


def test_synthetic_loader_shapes():
    rec = get_loader("synthetic")(
        {
            "name": "synthetic",
            "dataset_id": "synthetic-benchmark",
            "recording_id": "synthetic-recording-0",
            "n_neurons": 10,
            "n_timepoints": 500,
        }
    )
    assert rec.traces.shape == (10, 500)
    assert rec.time.shape == (500,)
    assert "true_graphs" in rec.metadata
    assert rec.identity.dataset_id == "synthetic-benchmark"
    assert rec.identity.recording_id == "synthetic-recording-0"


def test_preprocess_zscore(synthetic_recording):
    pre = preprocess(synthetic_recording, PreprocessConfig(detrend=True, zscore=True))
    # z-scored traces have ~unit variance per neuron.
    assert np.allclose(pre.traces.std(axis=1), 1.0, atol=0.2)
    # input is not mutated.
    assert pre.traces is not synthetic_recording.traces
    assert pre.identity == synthetic_recording.identity


def test_sliding_windows_shapes(synthetic_recording):
    w = make_windows(synthetic_recording, WindowConfig(length=100, stride=50, mode="sliding"))
    assert w.segments.shape[1] == 100
    assert w.segments.shape[2] == synthetic_recording.n_neurons
    assert w.n_windows == len(w.windows)
    assert len(w.anchors) == w.n_windows
    for name in synthetic_recording.behavior:
        assert w.behavior_per_window[name].shape[0] == w.n_windows


def test_behavior_aligned_windows(synthetic_recording):
    w = make_windows(
        synthetic_recording,
        WindowConfig(length=60, stride=10, mode="behavior", align_event="motif", min_event_gap=20),
    )
    assert w.n_windows > 0
    assert w.segments.shape[1] == 60


def test_temporal_reference_windows_match_goal_contract():
    rec = get_loader(
        "synthetic"
    )(
        {
            "name": "synthetic",
            "dataset_id": "synthetic-benchmark",
            "recording_id": "synthetic-3000",
            "n_neurons": 8,
            "n_timepoints": 3000,
            "seed": 1,
        }
    )
    windows = make_windows(
        rec,
        WindowConfig(mode="temporal", history_length=500, target_length=15, stride=15),
    )

    assert windows.n_windows == 167
    assert windows.windows[0].start == 0
    assert windows.windows[0].stop == 500
    assert windows.anchors[0].target_start == 485
    assert windows.anchors[0].target_stop == 500
    assert windows.anchors[0].anchor_sample == 499
    assert windows.windows[1].start == 15
    assert windows.windows[1].stop == 515
    assert windows.anchors[1].target_start == 500
    assert windows.anchors[1].target_stop == 515
    assert windows.anchors[1].anchor_sample == 514
    assert windows.metadata["reference_profile"] == {
        "history_length": 500,
        "target_length": 15,
        "stride": 15,
    }


def test_temporal_windows_respect_gap_boundaries():
    t = 120
    traces = np.arange(3 * t, dtype=np.float32).reshape(3, t)
    time = np.arange(t, dtype=np.float64) / 10.0
    recording = NeuralRecording(
        traces=traces,
        time=time,
        coords=None,
        neuron_ids=np.arange(3),
        behavior={"motif": np.zeros(t, dtype=np.int64)},
        fps=10.0,
        metadata={"gap_intervals": [(40, 60)]},
        identity=RecordingIdentity(dataset="synthetic", dataset_id="gap-test", recording_id="rec-0"),
    )
    windows = make_windows(
        recording,
        WindowConfig(mode="temporal", history_length=20, target_length=5, stride=5),
    )

    assert [(w.start, w.stop) for w in windows.windows] == [
        (0, 20),
        (5, 25),
        (10, 30),
        (15, 35),
        (20, 40),
        (60, 80),
        (65, 85),
        (70, 90),
        (75, 95),
        (80, 100),
        (85, 105),
        (90, 110),
        (95, 115),
        (100, 120),
    ]
    assert all(not (window.start < 60 and window.stop > 40) for window in windows.windows)


def test_c_elegans_loader_contract(tmp_path):
    path = tmp_path / "elegans_fixture.npz"
    traces = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.savez(
        path,
        traces=traces,
        time=np.arange(6, dtype=np.float64) / 2.0,
        neuron_ids=np.array([11, 12, 13, 14]),
        behavior_motif=np.array([0, 0, 1, 1, 0, 0], dtype=np.int64),
        behavior_continuous=np.linspace(0.0, 1.0, 6, dtype=np.float32),
    )

    rec = get_loader("c_elegans")(
        {
            "name": "c_elegans",
            "path": str(path),
            "dataset_id": "elegans-ds",
            "recording_id": "animal-1",
            "behavior_keys": {
                "motif": "behavior_motif",
                "continuous": "behavior_continuous",
            },
            "keys": {"traces": "traces", "time": "time", "neuron_ids": "neuron_ids"},
        }
    )

    assert rec.identity.dataset_id == "elegans-ds"
    assert rec.identity.recording_id == "animal-1"
    assert rec.neuron_ids.tolist() == [11, 12, 13, 14]
    assert rec.metadata["source"] == "c_elegans"
    assert rec.behavior["motif"].tolist() == [0, 0, 1, 1, 0, 0]


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


def test_schema_artifacts_carry_typed_provenance_and_alignment():
    provenance = ArtifactProvenance(
        identity=RecordingIdentity(dataset="synthetic", dataset_id="ds", recording_id="rec"),
        stage="stage-4",
        source="unit-test",
    )
    anchors = [
        TemporalAnchor(
            dataset_id="ds",
            recording_id="rec",
            animal_id=None,
            session_id=None,
            segment_id="seg-0",
            context_start=0,
            context_stop=20,
            target_start=15,
            target_stop=20,
            anchor_sample=19,
            anchor_time_seconds=1.9,
            sampling_rate_hz=10.0,
        ),
        TemporalAnchor(
            dataset_id="ds",
            recording_id="rec",
            animal_id=None,
            session_id=None,
            segment_id="seg-0",
            context_start=5,
            context_stop=25,
            target_start=20,
            target_stop=25,
            anchor_sample=24,
            anchor_time_seconds=2.4,
            sampling_rate_hz=10.0,
        ),
    ]
    series = ConnectivitySeries(
        matrices=np.zeros((2, 3, 3), dtype=np.float32),
        window_starts=np.array([0, 5], dtype=np.int64),
        method="manual",
        directed=True,
        anchors=anchors,
        provenance=provenance,
    )
    community = CommunitySeries(
        labels=np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int64),
        method="temporal",
        n_communities_per_window=np.array([2, 2], dtype=np.int64),
        window_starts=np.array([0, 5], dtype=np.int64),
        anchors=anchors,
        signed=True,
        directed=True,
        resolution=1.0,
        interlayer_coupling=0.5,
        boundary_indices=np.array([0, 2], dtype=np.int64),
        provenance=provenance,
    )

    assert series.provenance.identity.recording_id == "rec"
    assert anchors[0].history_seconds == 2.0
    assert anchors[0].target_seconds == 0.5
    assert np.array_equal(community.consensus_labels, community.labels)
    assert community.boundary_indices.tolist() == [0, 2]


def test_schema_alignment_validation_rejects_mismatched_lengths():
    anchor = TemporalAnchor(
        dataset_id="ds",
        recording_id="rec",
        animal_id=None,
        session_id=None,
        segment_id=None,
        context_start=0,
        context_stop=20,
        target_start=15,
        target_stop=20,
        anchor_sample=19,
        anchor_time_seconds=1.9,
        sampling_rate_hz=10.0,
    )

    with pytest.raises(ValueError, match="window_starts must align"):
        ConnectivitySeries(
            matrices=np.zeros((1, 2, 2), dtype=np.float32),
            window_starts=np.array([3], dtype=np.int64),
            method="manual",
            directed=True,
            anchors=[anchor],
        )

    with pytest.raises(ValueError, match="target_slices length"):
        ManifoldArtifact(
            method="classical",
            behavior_key="motif",
            full_embedding=np.zeros((25, 2), dtype=np.float32),
            window_embedding=np.zeros((2, 2), dtype=np.float32),
            target_slices=[TargetSlice(15, 20)],
            target_length=5,
        )

    with pytest.raises(ValueError, match="target_slices must align"):
        ManifoldArtifact(
            method="classical",
            behavior_key="motif",
            full_embedding=np.zeros((25, 2), dtype=np.float32),
            window_embedding=np.zeros((1, 2), dtype=np.float32),
            target_slices=[TargetSlice(10, 15)],
            target_length=5,
            window_starts=np.array([0], dtype=np.int64),
            anchors=[anchor],
        )
