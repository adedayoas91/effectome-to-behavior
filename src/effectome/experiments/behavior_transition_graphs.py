"""One-sided support discovery and signed mechanism fitting for forecast origins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module.schema import Window, WindowedSegments

from .mechanism_validation import _write_json
from .structural_mechanism import AdditiveMechanism, _fit_equations, _pairs


def trailing_histories(
    neural: np.ndarray, origins: np.ndarray, length: int
) -> tuple[np.ndarray, list[Window]]:
    """Extract one window per origin, in trial-major order, including the origin."""
    if neural.ndim != 3 or origins.ndim != 2 or origins.shape[0] != len(neural):
        raise ValueError("origins must identify one matrix row per neural trial")
    if not origins.size or not np.issubdtype(origins.dtype, np.integer) or length < 3:
        raise ValueError("trailing windows need integer origins and at least three samples")
    if np.any(origins < length - 1) or np.any(origins >= neural.shape[1]):
        raise ValueError("a discovery history crosses a trial boundary")
    stop = origins.ravel() + 1
    trials = np.repeat(np.arange(len(neural)), origins.shape[1])
    columns = stop[:, None] - np.arange(length, 0, -1)
    histories = neural[trials[:, None], columns]
    raw_end = trials * neural.shape[1] + stop
    return histories, [Window(int(end - length), int(end)) for end in raw_end]


def discover_transition_supports(
    histories: np.ndarray,
    windows: list[Window],
    method: ConnectivityConfig,
    fps: float,
    evidence: Path,
    signature: str,
) -> np.ndarray:
    """Reuse native lag-one decisions and keep their per-origin evidence."""
    shape = (len(histories), histories.shape[-1], histories.shape[-1])
    if evidence.is_file():
        saved = json.loads(evidence.read_text())
        if saved["signature"] != signature:
            raise ValueError("native support evidence belongs to another study")
        matrices = np.asarray(saved["support"])
    else:
        segments = WindowedSegments(
            segments=histories,
            windows=windows,
            behavior_per_window={},
            n_neurons=histories.shape[-1],
            fps=fps,
            metadata={"scope": "one_trailing_history_per_forecast_origin"},
        )
        series = ConnectivityFactory(method.resolve(fps)).run(segments)
        matrices = np.asarray(series.matrices)
        if matrices.shape != shape or not np.isin(matrices, [0, 1]).all():
            raise ValueError("native lag-one support does not match the origin and neuron axes")
        _write_json(
            evidence,
            {
                "signature": signature,
                "support": matrices.astype(bool),
                "raw_histories": [[window.start, window.stop] for window in windows],
                "native_diagnostics": series.diagnostics,
            },
        )
    if matrices.shape != shape or not np.isin(matrices, [0, 1]).all():
        raise ValueError("saved lag-one support does not match the origin and neuron axes")
    masks = matrices.astype(bool)
    if np.any(np.diagonal(masks, axis1=1, axis2=2)):
        raise ValueError("native cross-neuron support must exclude self edges")
    return masks


def fit_transition_mechanism(histories: np.ndarray, support: np.ndarray, alpha: float) -> AdditiveMechanism:
    """Fit the shared signed additive readout with a prespecified penalty."""
    past, future = _pairs(histories)
    intercept, self_weights, cross = _fit_equations(past, future, support, alpha)
    residual = future - (intercept + past * self_weights + np.tanh(past) @ cross)
    return AdditiveMechanism(
        intercept,
        self_weights,
        cross,
        np.maximum(np.sqrt(np.mean(residual**2, axis=0)), 1e-6),
        support,
        {"fit_pairs": len(past), "ridge_alpha": alpha, "noise_source": "fitting_history_residual_rms"},
    )
