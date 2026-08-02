"""Preprocessing of calcium traces and behavior alignment (Stage 0)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import detrend as sp_detrend

from .schema import NeuralRecording

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessConfig:
    """Stage-0 preprocessing options.

    Attributes:
        detrend: Remove a linear trend per neuron.
        zscore: Z-score each neuron's trace.
        deconvolve: Estimate spike rates with OASIS (requires the `deconv` extra).
        smooth_window: Optional moving-average smoothing window (samples, 0 = off).
        drop_low_variance: Drop neurons whose variance is below this quantile (0 = keep all).
    """

    detrend: bool = True
    zscore: bool = True
    deconvolve: bool = False
    smooth_window: int = 0
    drop_low_variance: float = 0.0


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), axis=1, arr=x)


def _deconvolve(traces: np.ndarray) -> np.ndarray:
    try:
        from oasis.functions import deconvolve
    except ImportError:
        logger.warning("oasis not installed; skipping deconvolution. Install extra 'deconv'.")
        return traces
    out = np.empty_like(traces)
    for i, row in enumerate(traces):
        _, spikes, _, _, _ = deconvolve(row.astype(np.float64))
        out[i] = spikes
    return out


def preprocess(recording: NeuralRecording, cfg: PreprocessConfig) -> NeuralRecording:
    """Return a new preprocessed recording (does not mutate the input)."""
    traces = recording.traces.astype(np.float64).copy()

    if cfg.deconvolve:
        traces = _deconvolve(traces)
    if cfg.detrend:
        traces = sp_detrend(traces, axis=1, type="linear")
    if cfg.smooth_window > 1:
        traces = _moving_average(traces, cfg.smooth_window)
    if cfg.zscore:
        mu = traces.mean(axis=1, keepdims=True)
        sd = traces.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        traces = (traces - mu) / sd

    keep = np.ones(traces.shape[0], dtype=bool)
    if cfg.drop_low_variance > 0:
        var = traces.var(axis=1)
        thresh = np.quantile(var, cfg.drop_low_variance)
        keep = var >= thresh
        logger.info("Dropping %d low-variance neurons", int((~keep).sum()))

    coords = recording.coords[keep] if recording.coords is not None else None
    out = NeuralRecording(
        traces=traces[keep].astype(np.float32),
        time=recording.time,
        coords=coords,
        neuron_ids=recording.neuron_ids[keep],
        behavior=dict(recording.behavior),
        fps=recording.fps,
        metadata={
            **recording.metadata,
            "preprocessed": True,
            "preprocess_config": {
                "detrend": cfg.detrend,
                "zscore": cfg.zscore,
                "deconvolve": cfg.deconvolve,
                "smooth_window": cfg.smooth_window,
                "drop_low_variance": cfg.drop_low_variance,
            },
        },
        identity=recording.identity,
    )
    out.validate()
    return out
