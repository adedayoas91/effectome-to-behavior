"""Stage 1: turn a recording into windowed ('shifted') segments for connectivity inference.

The project now distinguishes legacy full-window segmentations from typed trailing temporal anchors.
The latter implements the goal.md reference contract: history/context windows feed connectivity, while
behavior/manifold targets align to the trailing target interval inside each history window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .schema import ArtifactProvenance, NeuralRecording, TemporalAnchor, Window, WindowedSegments

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowConfig:
    """Windowing options.

    Attributes:
        length: Legacy full-window length L in samples for ``sliding``/``behavior`` modes.
        history_length: Context/history length for ``temporal`` mode when using explicit
            samples. Defaults to ``length`` when no duration is provided.
        target_length: Trailing target/core length for ``temporal`` mode when using explicit
            samples. Defaults to 15 when no duration is provided.
        stride: Step S between consecutive window starts in samples. Defaults to 25 when no
            duration is provided.
        history_seconds: Optional duration-first history length for ``temporal`` mode.
        target_seconds: Optional duration-first trailing target/core length for ``temporal`` mode.
        stride_seconds: Optional duration-first cadence for ``temporal`` mode.
        mode: ``sliding`` for regular tiling, ``behavior`` for event-aligned windows,
            ``temporal`` for trailing history/target anchors.
        align_event: Behavior key used for alignment when mode == ``behavior``.
        min_event_gap: Minimum samples between aligned events (debounce).
        behavior_summary: How to summarize each continuous behavior variable within a window.
        respect_boundaries: Prevent windows from crossing valid-range boundaries or gaps.
        drop_incomplete_tail: Drop incomplete tail windows in temporal mode.
        standardize_per_window: Center and scale every neuron independently within each window.
        standardization_epsilon: Minimum standard deviation treated as non-constant.
    """

    length: int = 100
    history_length: int | None = None
    target_length: int | None = None
    stride: int | None = None
    history_seconds: float | None = None
    target_seconds: float | None = None
    stride_seconds: float | None = None
    mode: Literal["sliding", "behavior", "temporal"] = "sliding"
    align_event: str = "motif"
    min_event_gap: int = 10
    behavior_summary: Literal["mean", "mode", "last"] = "mean"
    respect_boundaries: bool = True
    drop_incomplete_tail: bool = True
    standardize_per_window: bool = False
    standardization_epsilon: float = 1.0e-8

    @property
    def effective_history_length(self) -> int:
        return int(self.history_length if self.history_length is not None else self.length)

    @property
    def effective_target_length(self) -> int:
        return int(self.target_length if self.target_length is not None else 15)

    @property
    def effective_stride(self) -> int:
        return int(self.stride if self.stride is not None else 25)

    def resolve(self, fps: float) -> WindowConfig:
        """Resolve duration-first temporal parameters to sample counts."""
        if self.mode != "temporal":
            return self
        if fps <= 0:
            raise ValueError("fps must be positive")

        history = _resolve_temporal_parameter(
            name="history",
            fps=fps,
            explicit_samples=self.history_length,
            explicit_seconds=self.history_seconds,
            default_samples=self.length,
        )
        target = _resolve_temporal_parameter(
            name="target",
            fps=fps,
            explicit_samples=self.target_length,
            explicit_seconds=self.target_seconds,
            default_samples=15,
        )
        stride = _resolve_temporal_parameter(
            name="stride",
            fps=fps,
            explicit_samples=self.stride,
            explicit_seconds=self.stride_seconds,
            default_samples=25,
        )
        if int(target["resolved_samples"]) > int(history["resolved_samples"]):
            raise ValueError("target_length must be in 1..history_length")
        return replace(
            self,
            history_length=int(history["resolved_samples"]),
            target_length=int(target["resolved_samples"]),
            stride=int(stride["resolved_samples"]),
        )


def _round_samples_half_up(raw_samples: float) -> int:
    return int(np.floor(raw_samples + 0.5))


def _resolve_temporal_parameter(
    *,
    name: str,
    fps: float,
    explicit_samples: int | None,
    explicit_seconds: float | None,
    default_samples: int,
) -> dict[str, Any]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if explicit_samples is not None:
        if explicit_samples <= 0:
            raise ValueError(f"{name}_length must be positive")
        resolved_samples = int(explicit_samples)
        request_mode = "explicit_samples"
        raw_samples_from_seconds = float(resolved_samples)
        requested_samples: int | None = resolved_samples
        requested_seconds = float(resolved_samples) / float(fps)
    elif explicit_seconds is not None:
        if explicit_seconds <= 0:
            raise ValueError(f"{name}_seconds must be positive")
        raw_samples_from_seconds = float(explicit_seconds) * float(fps)
        resolved_samples = _round_samples_half_up(raw_samples_from_seconds)
        if resolved_samples <= 0:
            raise ValueError(
                f"{name}_seconds={explicit_seconds} resolves to fewer than one sample at fps={fps}"
            )
        request_mode = "duration_first"
        requested_samples = None
        requested_seconds = float(explicit_seconds)
    else:
        if default_samples <= 0:
            raise ValueError(f"default {name} length must be positive")
        resolved_samples = int(default_samples)
        request_mode = "default_samples"
        raw_samples_from_seconds = float(resolved_samples)
        requested_samples = resolved_samples
        requested_seconds = float(resolved_samples) / float(fps)

    resolved_seconds = float(resolved_samples) / float(fps)
    return {
        "request_mode": request_mode,
        "requested_samples": requested_samples,
        "requested_seconds": requested_seconds,
        "raw_samples_from_seconds": raw_samples_from_seconds,
        "resolved_samples": resolved_samples,
        "resolved_seconds": resolved_seconds,
        "rounding_policy": "nearest_half_up",
        "rounding_error_seconds": resolved_seconds - requested_seconds,
        "rounding_error_samples": float(resolved_samples) - raw_samples_from_seconds,
    }


def _resolve_temporal_contract(recording: NeuralRecording, cfg: WindowConfig) -> dict[str, Any]:
    history = _resolve_temporal_parameter(
        name="history",
        fps=recording.fps,
        explicit_samples=cfg.history_length,
        explicit_seconds=cfg.history_seconds,
        default_samples=cfg.length,
    )
    target = _resolve_temporal_parameter(
        name="target",
        fps=recording.fps,
        explicit_samples=cfg.target_length,
        explicit_seconds=cfg.target_seconds,
        default_samples=15,
    )
    stride = _resolve_temporal_parameter(
        name="stride",
        fps=recording.fps,
        explicit_samples=cfg.stride,
        explicit_seconds=cfg.stride_seconds,
        default_samples=25,
    )
    history_length = int(history["resolved_samples"])
    target_length = int(target["resolved_samples"])
    stride_length = int(stride["resolved_samples"])
    if target_length > history_length:
        raise ValueError("target_length must be in 1..history_length")

    parameter_modes = {
        "history": str(history["request_mode"]),
        "target": str(target["request_mode"]),
        "stride": str(stride["request_mode"]),
    }
    if set(parameter_modes.values()) == {"explicit_samples"}:
        mode = "explicit_samples"
    elif "duration_first" in parameter_modes.values():
        mode = "duration_first"
    else:
        mode = "default_samples"
    return {
        "mode": mode,
        "parameter_modes": parameter_modes,
        "history": history,
        "target": target,
        "stride": stride,
        "history_length": history_length,
        "target_length": target_length,
        "stride_length": stride_length,
    }


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


def _crosses_interval(intervals: list[tuple[int, int]], start: int, stop: int) -> bool:
    return any(start < interval_stop and stop > interval_start for interval_start, interval_stop in intervals)


def _annotate_bad_frame_boundaries(
    anchors: list[TemporalAnchor],
    bad_frame_intervals: list[tuple[int, int]],
) -> list[TemporalAnchor]:
    if not anchors or not bad_frame_intervals:
        return anchors
    annotated = list(anchors)
    for idx in range(1, len(annotated)):
        previous = annotated[idx - 1]
        current = annotated[idx]
        interval_start = previous.anchor_sample + 1
        interval_stop = current.anchor_sample + 1
        if _crosses_interval(bad_frame_intervals, interval_start, interval_stop):
            annotated[idx - 1] = replace(previous, bad_frame_after=True)
            annotated[idx] = replace(current, bad_frame_before=True)
    return annotated


def _standardize_window_neurons(
    segments: np.ndarray,
    cfg: WindowConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply independent time-axis z-scoring to every neuron in every window."""
    metadata: dict[str, object] = {
        "enabled": bool(cfg.standardize_per_window),
        "scope": "window",
        "axis": "time_per_neuron",
        "ddof": 0,
        "epsilon": float(cfg.standardization_epsilon),
        "constant_neuron_policy": "center_to_zero",
        "nonfinite_policy": "preserve_nonfinite_values_and_use_finite_samples_for_scaling",
        "claim_boundary": "uses only samples inside each causal history window",
    }
    values = np.asarray(segments, dtype=np.float64)
    if not cfg.standardize_per_window:
        return values.astype(np.float32), metadata
    if cfg.standardization_epsilon <= 0:
        raise ValueError("standardization_epsilon must be positive")

    finite_mask = np.isfinite(values)
    finite_counts = finite_mask.sum(axis=1, keepdims=True)
    safe_values = np.where(finite_mask, values, np.nan)
    means = np.nanmean(safe_values, axis=1, keepdims=True)
    means = np.where(finite_counts > 0, means, 0.0)
    scales = np.nanstd(safe_values, axis=1, keepdims=True)
    scales = np.where(finite_counts > 0, scales, 0.0)
    constant = (scales < float(cfg.standardization_epsilon)) | (finite_counts == 0)
    safe_scales = np.where(constant, 1.0, scales)
    standardized = (values - means) / safe_scales
    standardized = np.where(constant, 0.0, standardized)
    standardized = np.where(finite_mask, standardized, np.nan)
    metadata["constant_window_neuron_count"] = int(constant.sum())
    metadata["nonfinite_sample_count"] = int((~finite_mask).sum())
    return standardized.astype(np.float32), metadata


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
    temporal_contract = _resolve_temporal_contract(recording, cfg)
    history_length = int(temporal_contract["history_length"])
    target_length = int(temporal_contract["target_length"])
    stride_length = int(temporal_contract["stride_length"])
    if history_length <= 0:
        raise ValueError("history_length must be positive")
    if target_length <= 0 or target_length > history_length:
        raise ValueError("target_length must be in 1..history_length")
    if stride_length <= 0:
        raise ValueError("stride must be positive")

    x = recording.traces.T
    valid_ranges = _valid_ranges(recording) if cfg.respect_boundaries else [(0, recording.n_timepoints)]
    bad_frame_intervals = [
        (int(start), int(stop)) for start, stop in recording.metadata.get("bad_frame_intervals", [])
    ]
    windows: list[Window] = []
    anchors: list[TemporalAnchor] = []
    segments: list[np.ndarray] = []

    for range_start, range_stop in valid_ranges:
        last_start = range_stop - history_length
        if last_start < range_start:
            continue
        for start in range(range_start, last_start + 1, stride_length):
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
            "no temporal windows found; check history/stride settings or valid_ranges/gap_intervals"
        )

    anchors = _annotate_bad_frame_boundaries(anchors, bad_frame_intervals)

    behavior_per_window: dict[str, np.ndarray] = {}
    for name, arr in recording.behavior.items():
        discrete = np.issubdtype(arr.dtype, np.integer)
        how = "mode" if discrete else cfg.behavior_summary
        vals = np.array([_summarize(arr[a.target_start : a.target_stop], how) for a in anchors])
        behavior_per_window[name] = np.round(vals).astype(np.int64) if discrete else vals

    standardized_segments, standardization = _standardize_window_neurons(
        np.stack(segments),
        cfg,
    )
    metadata = {
        "window_mode": "temporal",
        "temporal_contract": temporal_contract,
        "history_length": history_length,
        "target_length": target_length,
        "stride": stride_length,
        "history_seconds": float(temporal_contract["history"]["resolved_seconds"]),
        "target_seconds": float(temporal_contract["target"]["resolved_seconds"]),
        "stride_seconds": float(temporal_contract["stride"]["resolved_seconds"]),
        "valid_ranges": valid_ranges,
        "bad_frame_intervals": bad_frame_intervals,
        "drop_incomplete_tail": cfg.drop_incomplete_tail,
        "neural_standardization": standardization,
        "reference_profile": {
            "mode": str(temporal_contract["mode"]),
            "parameter_modes": dict(temporal_contract["parameter_modes"]),
            "requested": {
                "history_length": temporal_contract["history"]["requested_samples"],
                "history_seconds": temporal_contract["history"]["requested_seconds"],
                "target_length": temporal_contract["target"]["requested_samples"],
                "target_seconds": temporal_contract["target"]["requested_seconds"],
                "stride": temporal_contract["stride"]["requested_samples"],
                "stride_seconds": temporal_contract["stride"]["requested_seconds"],
            },
            "resolved": {
                "history_length": history_length,
                "history_seconds": float(temporal_contract["history"]["resolved_seconds"]),
                "target_length": target_length,
                "target_seconds": float(temporal_contract["target"]["resolved_seconds"]),
                "stride": stride_length,
                "stride_seconds": float(temporal_contract["stride"]["resolved_seconds"]),
            },
        },
        "reference_profile_mode": (
            "explicit_samples_reference"
            if temporal_contract["mode"] == "explicit_samples"
            and history_length == 500
            and target_length == 15
            and stride_length == 15
            else "non_reference"
        ),
    }
    return WindowedSegments(
        segments=standardized_segments,
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
            metadata={
                "window_mode": "temporal",
                "temporal_contract": temporal_contract,
                "bad_frame_intervals": bad_frame_intervals,
                "neural_standardization": standardization,
            },
        ),
    )


def _make_legacy_windows(recording: NeuralRecording, cfg: WindowConfig) -> WindowedSegments:
    t = recording.n_timepoints
    x = recording.traces.T  # (T, N)
    valid_ranges = _valid_ranges(recording) if cfg.respect_boundaries else [(0, t)]

    if cfg.mode == "sliding":
        starts = _sliding_starts(t, cfg.length, cfg.effective_stride)
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
    segments, standardization = _standardize_window_neurons(
        np.stack([x[w.start : w.stop] for w in windows]),
        cfg,
    )

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

    logger.info(
        "Built %d windows (mode=%s, L=%d, S=%d)",
        len(windows),
        cfg.mode,
        cfg.length,
        cfg.effective_stride,
    )
    return WindowedSegments(
        segments=segments,
        windows=windows,
        behavior_per_window=behavior_per_window,
        n_neurons=recording.n_neurons,
        fps=recording.fps,
        metadata={
            "window_mode": cfg.mode,
            "length": cfg.length,
            "stride": cfg.effective_stride,
            "valid_ranges": valid_ranges,
            "neural_standardization": standardization,
        },
        anchors=anchors,
        provenance=ArtifactProvenance(
            identity=recording.identity,
            stage="windowing",
            source=str(recording.metadata.get("source", recording.identity.dataset)),
            units={"time": "seconds", "sample": "index"},
            axis_conventions={"segments": "window,time,neuron"},
            metadata={
                "window_mode": cfg.mode,
                "neural_standardization": standardization,
            },
        ),
    )


def make_windows(recording: NeuralRecording, cfg: WindowConfig) -> WindowedSegments:
    """Produce windowed segments and per-window behavior summaries."""
    recording.validate()
    if cfg.mode == "temporal":
        return _make_temporal_windows(recording, cfg)
    return _make_legacy_windows(recording, cfg)
