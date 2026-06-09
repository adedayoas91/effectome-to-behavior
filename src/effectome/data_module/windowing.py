"""Stage 1: turn a recording into windowed ('shifted') segments for connectivity inference.

The 'shifted versions of the dataset' in the project idea are realized two ways:
  * sliding windows of length L with stride S over the full time series, and
  * (optionally) behavior-aligned windows centered on behavioral events.
Each window yields an (L, N) multivariate segment that a connectivity estimator turns into
one N x N matrix, producing the temporal sequence of connectivity matrices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .schema import NeuralRecording, Window, WindowedSegments

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowConfig:
    """Windowing options.

    Attributes:
        length: Window length L in samples.
        stride: Step S between consecutive window starts in samples.
        mode: 'sliding' for regular tiling, 'behavior' for event-aligned windows.
        align_event: Behavior key used for alignment when mode == 'behavior'.
        min_event_gap: Minimum samples between aligned events (debounce).
        behavior_summary: How to summarize each behavior variable within a window.
    """

    length: int = 100
    stride: int = 25
    mode: Literal["sliding", "behavior"] = "sliding"
    align_event: str = "motif"
    min_event_gap: int = 10
    behavior_summary: Literal["mean", "mode", "last"] = "mean"


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
    for o in onsets:
        s = max(0, o - half)
        if s - last >= min_gap and s + length <= len(event):
            starts.append(int(s))
            last = s
    return starts


def make_windows(recording: NeuralRecording, cfg: WindowConfig) -> WindowedSegments:
    """Produce windowed segments and per-window behavior summaries."""
    recording.validate()
    t = recording.n_timepoints
    x = recording.traces.T  # (T, N)

    if cfg.mode == "sliding":
        starts = _sliding_starts(t, cfg.length, cfg.stride)
    else:
        if cfg.align_event not in recording.behavior:
            raise KeyError(f"align_event '{cfg.align_event}' not in behavior keys")
        starts = _event_starts(recording.behavior[cfg.align_event], cfg.length, cfg.min_event_gap)
        if not starts:
            raise ValueError("no behavior-aligned windows found; check align_event/min_event_gap")

    windows = [Window(s, s + cfg.length) for s in starts]
    segments = np.stack([x[w.start : w.stop] for w in windows]).astype(np.float32)

    behavior_per_window: dict[str, np.ndarray] = {}
    for name, arr in recording.behavior.items():
        # Discrete behaviors are summarized by mode and kept integer; continuous by cfg.summary.
        discrete = np.issubdtype(arr.dtype, np.integer)
        how = "mode" if discrete else cfg.behavior_summary
        vals = np.array([_summarize(arr[w.start : w.stop], how) for w in windows])
        behavior_per_window[name] = np.round(vals).astype(np.int64) if discrete else vals

    logger.info("Built %d windows (mode=%s, L=%d, S=%d)", len(windows), cfg.mode, cfg.length, cfg.stride)
    return WindowedSegments(
        segments=segments,
        windows=windows,
        behavior_per_window=behavior_per_window,
        n_neurons=recording.n_neurons,
        fps=recording.fps,
    )
