"""Stage 1: turn a recording into windowed ('shifted') segments for connectivity inference.

The project now distinguishes legacy full-window segmentations from typed trailing temporal anchors.
The latter implements the goal.md reference contract: history/context windows feed connectivity, while
behavior/manifold targets align to the trailing target interval inside each history window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .schema import ArtifactProvenance, NeuralRecording, TemporalAnchor, Window, WindowedSegments

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowConfig:
    """Windowing options.

    Attributes:
        length: Legacy full-window length L in samples for ``sliding``/``behavior`` modes.
        history_length: Context/history length for ``temporal`` mode. Defaults to ``length``.
        target_length: Trailing target/core length for ``temporal`` mode.
        stride: Step S between consecutive window starts in samples.
        mode: ``sliding`` for regular tiling, ``behavior`` for event-aligned windows,
            ``temporal`` for trailing history/target anchors.
        align_event: Behavior key used for alignment when mode == ``behavior``.
        min_event_gap: Minimum samples between aligned events (debounce).
        behavior_summary: How to summarize each continuous behavior variable within a window.
        respect_boundaries: Prevent windows from crossing valid-range boundaries or gaps.
        drop_incomplete_tail: Drop incomplete tail windows in temporal mode.
    """

    length: int = 100
    history_length: int | None = None
    target_length: int = 15
    stride: int = 25
    mode: Literal["sliding", "behavior", "temporal"] = "sliding"
    align_event: str = "motif"
    min_event_gap: int = 10
    behavior_summary: Literal["mean", "mode", "last"] = "mean"
    respect_boundaries: bool = True
    drop_incomplete_tail: bool = True

    @property
    def effective_history_length(self) -> int:
        return int(self.history_length if self.history_length is not None else self.length)


def _summarize(values: np.ndarray, how: str) -> float:
    if how == "mean":
        return float(np.mean(values))
    if how == "last":
        return float(values[-1])
    if how == "mode":
        vals, counts = np.unique(values, return_counts=True)
        return float(vals[np.argmax(counts)])
    raise ValueError(f"unknown behavior_summary '{how}'")


def _sliding_starts(t: int, length: int, stride: int) -> list[int]:
    if length > t:
        raise ValueError(f"window length {length} exceeds recording length {t}")
    return list(range(0, t - length + 1, stride))


def _event_starts(event: np.ndarray, length: int, min_gap: int) -> list[int]:
    onsets = np.where(np.diff((event > 0).astype(int)) == 1)[0] + 1
    starts: list[int] = []
    last = -min_gap
    half = length // 2
    for onset in onsets:
        start = max(0, onset - half)
        if start - last >= min_gap and start + length <= len(event):
            starts.append(int(start))
            last = start
    return starts


def _valid_ranges(recording: NeuralRecording) -> list[tuple[int, int]]:
    total = recording.n_timepoints
    valid_ranges = recording.metadata.get("valid_ranges")
    ranges: list[tuple[int, int]]
    if valid_ranges:
        ranges = [(int(start), int(stop)) for start, stop in valid_ranges]
    else:
        gap_intervals = [
            (int(start), int(stop)) for start, stop in recording.metadata.get("gap_intervals", [])
        ]
        if not gap_intervals:
            return [(0, total)]
        gap_intervals.sort()
        ranges = []
        cursor = 0
        for gap_start, gap_stop in gap_intervals:
            if cursor < gap_start:
                ranges.append((cursor, gap_start))
            cursor = max(cursor, gap_stop)
        if cursor < total:
            ranges.append((cursor, total))
    return [(start, stop) for start, stop in ranges if stop > start]


def _window_in_valid_range(start: int, stop: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start >= range_start and stop <= range_stop for range_start, range_stop in ranges)


def _build_anchor(
    recording: NeuralRecording,
    *,
    context_start: int,
    context_stop: int,
    target_start: int,
    target_stop: int,
    valid_ranges: list[tuple[int, int]],
) -> TemporalAnchor:
    anchor_sample = target_stop - 1
    owning_range = next(
        (
            (range_start, range_stop)
            for range_start, range_stop in valid_ranges
            if context_start >= range_start and context_stop <= range_stop
        ),
        None,
    )
    # These flags describe a *real discontinuity* adjacent to the valid segment, not whether
    # the current rolling window happens to be in the interior of that segment.  Marking every
    # interior window as a gap would incorrectly reset all temporal state/community coupling.
    gap_before = owning_range is not None and owning_range[0] > 0 and context_start == owning_range[0]
    gap_after = (
        owning_range is not None
        and owning_range[1] < recording.n_timepoints
        and context_stop == owning_range[1]
    )
    return TemporalAnchor(
        dataset_id=recording.identity.dataset_id,
        recording_id=recording.identity.recording_id,
        animal_id=recording.identity.animal_id,
        session_id=recording.identity.session_id,
        segment_id=recording.identity.segment_id,
        context_start=context_start,
        context_stop=context_stop,
        target_start=target_start,
        target_stop=target_stop,
        anchor_sample=anchor_sample,
        anchor_time_seconds=float(recording.time[anchor_sample]),
        sampling_rate_hz=recording.fps,
        gap_before=gap_before,
        gap_after=gap_after,
        original_source_indices=(context_start, context_stop),
    )


def _make_temporal_windows(recording: NeuralRecording, cfg: WindowConfig) -> WindowedSegments:
    history_length = cfg.effective_history_length
    target_length = int(cfg.target_length)
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if target_length <= 0 or target_length > history_length:
        raise ValueError("target_length must be in 1..history_length")

    x = recording.traces.T
    valid_ranges = _valid_ranges(recording) if cfg.respect_boundaries else [(0, recording.n_timepoints)]
    windows: list[Window] = []
    anchors: list[TemporalAnchor] = []
    segments: list[np.ndarray] = []

    for range_start, range_stop in valid_ranges:
        last_start = range_stop - history_length
        if last_start < range_start:
            continue
        for start in range(range_start, last_start + 1, cfg.stride):
            stop = start + history_length
            if cfg.respect_boundaries and not _window_in_valid_range(start, stop, valid_ranges):
                continue
            target_start = stop - target_length
            target_stop = stop
            windows.append(Window(start, stop))
            anchors.append(
                _build_anchor(
                    recording,
                    context_start=start,
                    context_stop=stop,
                    target_start=target_start,
                    target_stop=target_stop,
                    valid_ranges=valid_ranges,
                )
            )
            segments.append(x[start:stop])

    if not segments:
        raise ValueError(
            "no temporal windows found; check history_length/stride or valid_ranges/gap_intervals"
        )

    behavior_per_window: dict[str, np.ndarray] = {}
    for name, arr in recording.behavior.items():
        discrete = np.issubdtype(arr.dtype, np.integer)
        how = "mode" if discrete else cfg.behavior_summary
        vals = np.array([_summarize(arr[a.target_start : a.target_stop], how) for a in anchors])
        behavior_per_window[name] = np.round(vals).astype(np.int64) if discrete else vals

    metadata = {
        "window_mode": "temporal",
        "history_length": history_length,
        "target_length": target_length,
        "stride": cfg.stride,
        "valid_ranges": valid_ranges,
        "drop_incomplete_tail": cfg.drop_incomplete_tail,
        "reference_profile": {
            "history_length": 500,
            "target_length": 15,
            "stride": 15,
        },
    }
    return WindowedSegments(
        segments=np.stack(segments).astype(np.float32),
        windows=windows,
        behavior_per_window=behavior_per_window,
        n_neurons=recording.n_neurons,
        fps=recording.fps,
        metadata=metadata,
        anchors=anchors,
        provenance=ArtifactProvenance(
            identity=recording.identity,
            stage="windowing",
            source=str(recording.metadata.get("source", recording.identity.dataset)),
            units={"time": "seconds", "sample": "index"},
            axis_conventions={"segments": "window,time,neuron"},
            metadata={"window_mode": "temporal"},
        ),
    )


def _make_legacy_windows(recording: NeuralRecording, cfg: WindowConfig) -> WindowedSegments:
    t = recording.n_timepoints
    x = recording.traces.T  # (T, N)
    valid_ranges = _valid_ranges(recording) if cfg.respect_boundaries else [(0, t)]

    if cfg.mode == "sliding":
        starts = _sliding_starts(t, cfg.length, cfg.stride)
    else:
        if cfg.align_event not in recording.behavior:
            raise KeyError(f"align_event '{cfg.align_event}' not in behavior keys")
        starts = _event_starts(recording.behavior[cfg.align_event], cfg.length, cfg.min_event_gap)
        if not starts:
            raise ValueError("no behavior-aligned windows found; check align_event/min_event_gap")

    windows = [Window(start, start + cfg.length) for start in starts]
    if cfg.respect_boundaries:
        windows = [w for w in windows if _window_in_valid_range(w.start, w.stop, valid_ranges)]
        if not windows:
            raise ValueError("no windows remain after enforcing valid-range boundaries")
    segments = np.stack([x[w.start : w.stop] for w in windows]).astype(np.float32)

    behavior_per_window: dict[str, np.ndarray] = {}
    for name, arr in recording.behavior.items():
        discrete = np.issubdtype(arr.dtype, np.integer)
        how = "mode" if discrete else cfg.behavior_summary
        vals = np.array([_summarize(arr[w.start : w.stop], how) for w in windows])
        behavior_per_window[name] = np.round(vals).astype(np.int64) if discrete else vals

    anchors = [
        _build_anchor(
            recording,
            context_start=w.start,
            context_stop=w.stop,
            target_start=w.start,
            target_stop=w.stop,
            valid_ranges=valid_ranges,
        )
        for w in windows
    ]

    logger.info("Built %d windows (mode=%s, L=%d, S=%d)", len(windows), cfg.mode, cfg.length, cfg.stride)
    return WindowedSegments(
        segments=segments,
        windows=windows,
        behavior_per_window=behavior_per_window,
        n_neurons=recording.n_neurons,
        fps=recording.fps,
        metadata={"window_mode": cfg.mode, "valid_ranges": valid_ranges},
        anchors=anchors,
        provenance=ArtifactProvenance(
            identity=recording.identity,
            stage="windowing",
            source=str(recording.metadata.get("source", recording.identity.dataset)),
            units={"time": "seconds", "sample": "index"},
            axis_conventions={"segments": "window,time,neuron"},
            metadata={"window_mode": cfg.mode},
        ),
    )


def make_windows(recording: NeuralRecording, cfg: WindowConfig) -> WindowedSegments:
    """Produce windowed segments and per-window behavior summaries."""
    recording.validate()
    if cfg.mode == "temporal":
        return _make_temporal_windows(recording, cfg)
    return _make_legacy_windows(recording, cfg)
