"""Context-preserving overlapping segmentation for calcium traces.

The segmentation used for effectome estimation should not impose hard,
independent chunks on slow calcium dynamics. This module builds overlapping
windows with a central analysis core and optional left/right context. The CSL
estimator receives the full context window, while behavior summaries and matrix
timestamps are assigned to the center/core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .schema import NeuralRecording, Window, WindowedSegments


@dataclass(frozen=True)
class CalciumSegmentationConfig:
    """Options for overlapping calcium trace segmentation.

    Attributes:
        core_length: Number of samples assigned to the window's central analysis core.
        stride: Step between adjacent core starts. Must be smaller than the full
            context window length for overlapping windows.
        left_context: Samples prepended before the core.
        right_context: Samples appended after the core.
        lag_context: Minimum left context required for lagged CSL estimators.
        taper: Optional temporal taper applied across the full context window.
        gaussian_sigma: Width of the Gaussian taper as a fraction of window length.
        apply_taper: If true, multiply each segment by the taper weights.
        behavior_summary: Behavior summary computed over the core, not context.
    """

    core_length: int = 60
    stride: int = 10
    left_context: int = 10
    right_context: int = 10
    lag_context: int = 0
    taper: Literal["none", "hann", "gaussian"] = "hann"
    gaussian_sigma: float = 0.25
    apply_taper: bool = True
    behavior_summary: Literal["mean", "mode", "last"] = "mean"

    @property
    def effective_left_context(self) -> int:
        return max(self.left_context, self.lag_context)

    @property
    def full_length(self) -> int:
        return self.effective_left_context + self.core_length + self.right_context


def _validate_cfg(cfg: CalciumSegmentationConfig) -> None:
    if cfg.core_length <= 0:
        raise ValueError("core_length must be positive.")
    if cfg.stride <= 0:
        raise ValueError("stride must be positive.")
    if cfg.left_context < 0 or cfg.right_context < 0 or cfg.lag_context < 0:
        raise ValueError("contexts must be non-negative.")
    if cfg.stride >= cfg.full_length:
        raise ValueError("stride must be smaller than the full context window length.")
    if cfg.taper == "gaussian" and cfg.gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive.")


def _summarize(values: np.ndarray, how: str) -> float:
    if how == "mean":
        return float(np.mean(values))
    if how == "last":
        return float(values[-1])
    if how == "mode":
        vals, counts = np.unique(values, return_counts=True)
        return float(vals[np.argmax(counts)])
    raise ValueError(f"unknown behavior_summary '{how}'")


def make_taper(length: int, kind: str, gaussian_sigma: float = 0.25) -> np.ndarray:
    """Return a length-``length`` taper normalized to unit maximum."""
    if length <= 0:
        raise ValueError("length must be positive.")
    if kind == "none":
        return np.ones(length, dtype=np.float32)
    if kind == "hann":
        if length == 1:
            return np.ones(1, dtype=np.float32)
        return np.hanning(length).astype(np.float32)
    if kind == "gaussian":
        grid = np.linspace(-0.5, 0.5, length, dtype=np.float64)
        weights = np.exp(-0.5 * (grid / gaussian_sigma) ** 2)
        weights /= weights.max()
        return weights.astype(np.float32)
    raise ValueError(f"unknown taper '{kind}'")


def _as_time_by_neuron(
    data: np.ndarray,
    *,
    orientation: Literal["time_neuron", "neuron_time"],
) -> np.ndarray:
    x = np.asarray(data, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {x.shape}")
    if orientation == "time_neuron":
        return x
    if orientation == "neuron_time":
        return x.T
    raise ValueError(f"unknown orientation '{orientation}'")


def _core_starts(n_timepoints: int, cfg: CalciumSegmentationConfig) -> list[int]:
    left = cfg.effective_left_context
    last_core_start = n_timepoints - cfg.core_length - cfg.right_context
    if last_core_start < left:
        raise ValueError(
            "recording is too short for the requested core/context segmentation: "
            f"T={n_timepoints}, full_length={cfg.full_length}"
        )
    return list(range(left, last_core_start + 1, cfg.stride))


def segment_calcium_traces(
    data: np.ndarray,
    cfg: CalciumSegmentationConfig,
    *,
    behavior: dict[str, np.ndarray] | None = None,
    fps: float = 1.0,
    orientation: Literal["time_neuron", "neuron_time"] = "time_neuron",
) -> WindowedSegments:
    """Segment calcium traces into overlapping context-preserving windows.

    Parameters
    ----------
    data:
        Calcium traces in either ``(T, N)`` or ``(N, T)`` format.
    cfg:
        Segmentation configuration.
    behavior:
        Optional behavior arrays of length ``T``. Summaries are computed over
        each core window, not over padded context.
    fps:
        Sampling rate carried into the returned ``WindowedSegments``.
    orientation:
        Shape convention for ``data``.
    """
    _validate_cfg(cfg)
    x = _as_time_by_neuron(data, orientation=orientation)
    n_timepoints, n_neurons = x.shape
    core_starts = _core_starts(n_timepoints, cfg)
    left = cfg.effective_left_context
    weights = make_taper(cfg.full_length, cfg.taper, cfg.gaussian_sigma)

    windows: list[Window] = []
    core_windows: list[Window] = []
    segments = []
    for core_start in core_starts:
        start = core_start - left
        stop = core_start + cfg.core_length + cfg.right_context
        segment = x[start:stop].copy()
        if cfg.apply_taper:
            segment *= weights[:, None]
        windows.append(Window(start, stop))
        core_windows.append(Window(core_start, core_start + cfg.core_length))
        segments.append(segment)

    behavior_per_window: dict[str, np.ndarray] = {}
    if behavior:
        for name, arr in behavior.items():
            arr = np.asarray(arr)
            if arr.shape[0] != n_timepoints:
                raise ValueError(f"behavior['{name}'] length {arr.shape[0]} != T {n_timepoints}")
            discrete = np.issubdtype(arr.dtype, np.integer)
            how = "mode" if discrete else cfg.behavior_summary
            vals = np.array([_summarize(arr[w.start : w.stop], how) for w in core_windows])
            behavior_per_window[name] = np.round(vals).astype(np.int64) if discrete else vals

    centers = np.array([(w.start + w.stop - 1) / 2 for w in core_windows], dtype=np.float32)
    metadata = {
        "segmentation": "overlapping_calcium",
        "core_windows": core_windows,
        "center_samples": centers,
        "center_times": centers / float(fps),
        "sample_weights": weights,
        "taper": cfg.taper,
        "apply_taper": cfg.apply_taper,
        "core_length": cfg.core_length,
        "stride": cfg.stride,
        "left_context": left,
        "right_context": cfg.right_context,
        "lag_context": cfg.lag_context,
        "overlap_fraction": 1.0 - (cfg.stride / cfg.full_length),
    }

    return WindowedSegments(
        segments=np.stack(segments).astype(np.float32),
        windows=windows,
        behavior_per_window=behavior_per_window,
        n_neurons=n_neurons,
        fps=fps,
        metadata=metadata,
    )


def make_overlapping_calcium_windows(
    recording: NeuralRecording,
    cfg: CalciumSegmentationConfig,
) -> WindowedSegments:
    """Build overlapping calcium windows from a ``NeuralRecording``."""
    recording.validate()
    return segment_calcium_traces(
        recording.traces,
        cfg,
        behavior=recording.behavior,
        fps=recording.fps,
        orientation="neuron_time",
    )


__all__ = [
    "CalciumSegmentationConfig",
    "make_overlapping_calcium_windows",
    "make_taper",
    "segment_calcium_traces",
]
