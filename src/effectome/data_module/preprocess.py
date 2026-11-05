"""Preprocessing of calcium traces and behavior alignment (Stage 0)."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy import signal
from scipy.signal import detrend as sp_detrend

from .schema import NeuralRecording, RawInterval

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


@dataclass(frozen=True)
class FittedPreprocessor:
    """Training-owned channel map and fixed scaling for prospective transforms."""

    config: PreprocessConfig
    retained_roi_ids: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    fit_intervals: tuple[RawInterval, ...]
    scale_id: str

    @classmethod
    def fit(
        cls,
        recordings: Sequence[NeuralRecording],
        allowed_intervals: Mapping[str, Sequence[RawInterval]],
        config: PreprocessConfig,
    ) -> FittedPreprocessor:
        if not recordings:
            raise ValueError("at least one training recording is required")
        if (
            config.bundle_net_bandpass
            or config.deconvolve
            or config.detrend
            or config.smooth_window > 1
        ):
            raise ValueError("prospective fitted preprocessing supports fixed channel scaling only")
        common_ids = set(int(value) for value in recordings[0].neuron_ids)
        for recording in recordings[1:]:
            common_ids &= {int(value) for value in recording.neuron_ids}
        retained = np.asarray(
            [int(value) for value in recordings[0].neuron_ids if int(value) in common_ids],
            dtype=np.int64,
        )
        if retained.size == 0:
            raise ValueError("training recordings have no common ROI panel")

        samples: list[np.ndarray] = []
        used_intervals: list[RawInterval] = []
        for recording in recordings:
            source_id = recording.identity.recording_id
            intervals = tuple(allowed_intervals.get(source_id, ()))
            if not intervals:
                raise ValueError(f"no permitted fit intervals for recording {source_id!r}")
            lookup = {int(roi): index for index, roi in enumerate(recording.neuron_ids)}
            rows = np.asarray([lookup[int(roi)] for roi in retained], dtype=int)
            for interval in intervals:
                if interval.source_id != source_id:
                    raise ValueError("fit interval source does not match its recording")
                if interval.stop > recording.n_timepoints:
                    raise ValueError("fit interval extends beyond the recording")
                block = recording.traces[rows, interval.start : interval.stop].T
                if not np.all(np.isfinite(block)):
                    raise ValueError("fitted scaling intervals must contain finite neural samples")
                samples.append(block)
                used_intervals.append(interval)
        stacked = np.concatenate(samples, axis=0)
        variances = stacked.var(axis=0)
        keep = np.ones(retained.size, dtype=bool)
        if config.drop_low_variance > 0:
            threshold = np.quantile(variances, config.drop_low_variance)
            keep = variances >= threshold
        retained = retained[keep]
        stacked = stacked[:, keep]
        means = stacked.mean(axis=0) if config.zscore else np.zeros(stacked.shape[1])
        scales = stacked.std(axis=0) if config.zscore else np.ones(stacked.shape[1])
        scales = np.where(scales > 1e-12, scales, 1.0)
        digest = sha256()
        digest.update(retained.tobytes())
        digest.update(means.tobytes())
        digest.update(scales.tobytes())
        for interval in used_intervals:
            digest.update(repr(interval).encode())
        return cls(
            config=config,
            retained_roi_ids=retained,
            means=means,
            scales=scales,
            fit_intervals=tuple(used_intervals),
            scale_id=digest.hexdigest(),
        )

    def transform(self, recording: NeuralRecording) -> NeuralRecording:
        lookup = {int(roi): index for index, roi in enumerate(recording.neuron_ids)}
        missing = [int(roi) for roi in self.retained_roi_ids if int(roi) not in lookup]
        if missing:
            raise ValueError(f"recording is missing fitted ROI IDs: {missing}")
        rows = np.asarray([lookup[int(roi)] for roi in self.retained_roi_ids], dtype=int)
        traces = (recording.traces[rows].astype(np.float64) - self.means[:, None]) / self.scales[:, None]
        metadata = dict(recording.metadata)
        metadata["fitted_preprocessing"] = {
            "scale_id": self.scale_id,
            "fit_intervals": [interval.__dict__ for interval in self.fit_intervals],
            "retained_roi_ids": self.retained_roi_ids.tolist(),
            "update_policy": "frozen",
        }
        transformed = NeuralRecording(
            traces=traces.astype(np.float32),
            time=recording.time.copy(),
            coords=None if recording.coords is None else recording.coords[rows].copy(),
            neuron_ids=self.retained_roi_ids.copy(),
            behavior={name: values.copy() for name, values in recording.behavior.items()},
            fps=recording.fps,
            metadata=metadata,
            identity=recording.identity,
            endpoint_metadata=dict(recording.endpoint_metadata),
            neural_validity=(
                None
                if recording.neural_validity is None
                else np.asarray(recording.neural_validity)[rows].copy()
            ),
            outcome_validity={
                name: values.copy() for name, values in recording.outcome_validity.items()
            },
            source_indices=(
                None if recording.source_indices is None else recording.source_indices.copy()
            ),
        )
        transformed.validate()
        return transformed

    def coefficients_to_raw(
        self,
        lagged_coefficients: np.ndarray,
        intercept: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        coefficients = np.asarray(lagged_coefficients, dtype=np.float64)
        raw = coefficients * self.scales[None, None, :] / self.scales[None, :, None]
        raw_intercept = (
            self.means
            + self.scales * np.asarray(intercept, dtype=np.float64)
            - np.sum(np.einsum("lsj,s->lj", raw, self.means), axis=0)
        )
        return raw, raw_intercept


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

    # Dataset loaders may already have applied the registered cohort selection.
    # Treat an identical request as idempotent so a downstream reference lane
    # retains the loader's record of which requested names were actually present.
    if (
        recording.metadata.get("exclude_neuron_name_source") == normalized_source
        and recording.metadata.get("requested_excluded_neuron_names") == requested
    ):
        return recording

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
        endpoint_metadata=dict(recording.endpoint_metadata),
        neural_validity=(
            None if recording.neural_validity is None else recording.neural_validity[keep]
        ),
        outcome_validity=dict(recording.outcome_validity),
        source_indices=recording.source_indices,
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
        endpoint_metadata=dict(recording.endpoint_metadata),
        neural_validity=(
            None if recording.neural_validity is None else recording.neural_validity[keep]
        ),
        outcome_validity=dict(recording.outcome_validity),
        source_indices=recording.source_indices,
    )
    out.validate()
    return out
