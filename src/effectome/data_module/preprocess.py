"""Preprocessing of calcium traces and behavior alignment (Stage 0)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal
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
        bundle_net_bandpass: Reproduce Bundle-Net's recording-wide forward/backward
            fourth-order Butterworth band-pass.
        bundle_net_low_fraction: Lower cutoff as the fraction used by Bundle-Net's code.
        bundle_net_high_fraction: Upper cutoff as the fraction used by Bundle-Net's code.
        bundle_net_filter_order: Butterworth order passed to each directional filter.
    """

    detrend: bool = False
    zscore: bool = False
    deconvolve: bool = False
    smooth_window: int = 0
    drop_low_variance: float = 0.0
    bundle_net_bandpass: bool = False
    bundle_net_low_fraction: float = 1.0e-10
    bundle_net_high_fraction: float = 0.05
    bundle_net_filter_order: int = 4


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


def bundle_net_bandpass(
    traces: np.ndarray,
    fps: float,
    *,
    low_fraction: float = 1.0e-10,
    high_fraction: float = 0.05,
    order: int = 4,
) -> np.ndarray:
    """Reproduce Bundle-Net commit cbefad9's forward/backward Butterworth filter.

    Bundle-Net multiplies each configured fraction by Nyquist before passing the
    resulting cutoffs to ``scipy.signal.butter(..., fs=fps)``. It then applies
    ``sosfilt`` forward and backward without padding. This is intentionally kept
    exact for reproducibility, including its endpoint transients.
    """
    values = np.asarray(traces, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"traces must have shape (neuron, time), got {values.shape}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if order <= 0:
        raise ValueError("bundle_net_filter_order must be positive")
    if not 0 <= low_fraction < high_fraction < 1:
        raise ValueError("Bundle-Net cutoff fractions must satisfy 0 <= low < high < 1")

    nyquist = float(fps) / 2.0
    cutoffs_hz = [float(low_fraction) * nyquist, float(high_fraction) * nyquist]
    sos = signal.butter(order, cutoffs_hz, "bandpass", fs=float(fps), output="sos")
    filtered = signal.sosfilt(sos, values, axis=1)
    filtered = np.flip(filtered, axis=1)
    filtered = signal.sosfilt(sos, filtered, axis=1)
    return np.flip(filtered, axis=1)


def exclude_named_neurons(
    recording: NeuralRecording,
    names: list[str] | tuple[str, ...],
    *,
    name_source: str = "raw",
) -> NeuralRecording:
    """Return a recording without exact raw or canonical neuron-name matches.

    Bundle-Net compares its exclusion list directly with each worm's
    ``NeuronNames`` values. Keeping ``name_source='raw'`` reproduces that
    behavior, including requested names that are absent from a recording.
    """
    requested = [str(name) for name in names]
    if not requested:
        return recording

    normalized_source = str(name_source).lower()
    metadata_key = {
        "raw": "raw_neuron_names",
        "canonical": "canonical_neuron_names",
    }.get(normalized_source)
    if metadata_key is None:
        raise ValueError("name_source must be 'raw' or 'canonical'")

    source_names = [str(name) for name in recording.metadata.get(metadata_key, [])]
    if len(source_names) != recording.n_neurons:
        raise ValueError(
            f"recording metadata {metadata_key!r} must contain one name per neuron"
        )
    requested_set = set(requested)
    keep = np.asarray([name not in requested_set for name in source_names], dtype=bool)
    present = [name for name in requested if name in source_names]
    absent = [name for name in requested if name not in source_names]

    metadata = dict(recording.metadata)
    for key in ("raw_neuron_names", "canonical_neuron_names"):
        values = metadata.get(key)
        if isinstance(values, list) and len(values) == recording.n_neurons:
            metadata[key] = [value for value, retain in zip(values, keep, strict=True) if retain]
    for key in ("canonical_neuron_indices", "canonical_class_ids"):
        values = metadata.get(key)
        if isinstance(values, list) and len(values) == recording.n_neurons:
            metadata[key] = [value for value, retain in zip(values, keep, strict=True) if retain]
    metadata.update(
        {
            "exclude_neuron_name_source": normalized_source,
            "requested_excluded_neuron_names": requested,
            "excluded_neuron_names": present,
            "requested_but_absent_excluded_neuron_names": absent,
        }
    )

    selected = NeuralRecording(
        traces=recording.traces[keep],
        time=recording.time,
        coords=recording.coords[keep] if recording.coords is not None else None,
        neuron_ids=recording.neuron_ids[keep],
        behavior=dict(recording.behavior),
        fps=recording.fps,
        metadata=metadata,
        identity=recording.identity,
    )
    selected.validate()
    return selected


def preprocess(recording: NeuralRecording, cfg: PreprocessConfig) -> NeuralRecording:
    """Return a new preprocessed recording (does not mutate the input)."""
    traces = recording.traces.astype(np.float64).copy()

    if cfg.bundle_net_bandpass:
        traces = bundle_net_bandpass(
            traces,
            recording.fps,
            low_fraction=cfg.bundle_net_low_fraction,
            high_fraction=cfg.bundle_net_high_fraction,
            order=cfg.bundle_net_filter_order,
        )
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
    uses_global_fit = bool(
        cfg.bundle_net_bandpass
        or cfg.deconvolve
        or cfg.detrend
        or cfg.zscore
        or cfg.drop_low_variance > 0
    )
    finite_kernel_support = max(0, int(cfg.smooth_window) - 1)
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
                "bundle_net_bandpass": cfg.bundle_net_bandpass,
                "bundle_net_low_fraction": cfg.bundle_net_low_fraction,
                "bundle_net_high_fraction": cfg.bundle_net_high_fraction,
                "bundle_net_filter_order": cfg.bundle_net_filter_order,
            },
            "preprocess_dependency": {
                "kind": (
                    "global_noncausal_iir"
                    if cfg.bundle_net_bandpass
                    else "global" if uses_global_fit else "finite"
                ),
                "past_support": finite_kernel_support,
                "future_support": finite_kernel_support,
                "claim_boundary": (
                    "Bundle-Net's forward/backward IIR filter is retrospective-only for causal claims"
                    if cfg.bundle_net_bandpass
                    else "recording-global transforms require fold-specific fitting for prospective claims"
                    if uses_global_fit
                    else "finite support is conservatively bounded for dependency purging"
                ),
            },
            "bundle_net_reference": (
                {
                    "source_repository": "https://github.com/akshey-kumar/BunDLe-Net",
                    "source_commit": "cbefad93fe98b5860c7e219e5af9a7bb0a177143",
                    "source_file": "functions.py",
                    "source_functions": ["bandpass", "preprocess_data"],
                    "actual_cutoffs_hz": [
                        cfg.bundle_net_low_fraction * recording.fps / 2.0,
                        cfg.bundle_net_high_fraction * recording.fps / 2.0,
                    ],
                    "filter_direction": "forward_then_reverse_without_padding",
                }
                if cfg.bundle_net_bandpass
                else None
            ),
        },
        identity=recording.identity,
    )
    out.validate()
    return out
