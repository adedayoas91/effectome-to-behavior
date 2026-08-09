"""Boundary-aware smoothly time-varying connectivity candidate.

This estimator is intentionally registered as a candidate, not a biological-analysis default.  It
first obtains one signed, directed, weighted estimate per history window and then solves a global
quadratic temporal-regularization problem independently inside every continuous recording segment:

    sum_k ||W_k - B_k||_F^2 + lambda * sum_k ||W_k - W_(k-1)||_F^2

The implementation supplies the sequence-level interface and a reproducible comparison candidate.
Promotion still requires the synthetic-recovery, calibration, stability, and held-out gates in
``goal.md``; smoothness alone is not evidence of better connectivity recovery.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from effectome.data_module.schema import TemporalAnchor, WindowedSegments

from .base import ConnectivityConfig, ConnectivityEstimator
from .registry import ConnectivityFactory, register_connectivity


def _anchor_key(anchor: TemporalAnchor) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        anchor.dataset_id,
        anchor.recording_id,
        anchor.animal_id,
        anchor.session_id,
        anchor.segment_id,
    )


def _continuous_slices(segments: WindowedSegments) -> list[slice]:
    n_windows = segments.n_windows
    if not segments.anchors:
        return [slice(0, n_windows)]

    boundaries = [0]
    for idx in range(1, n_windows):
        previous = segments.anchors[idx - 1]
        current = segments.anchors[idx]
        if _anchor_key(current) != _anchor_key(previous) or current.gap_before or previous.gap_after:
            boundaries.append(idx)
    boundaries.append(n_windows)
    return [slice(start, stop) for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)]


def _smooth_segment(raw: np.ndarray, temporal_lambda: float) -> tuple[np.ndarray, float]:
    length = raw.shape[0]
    if length <= 1 or temporal_lambda == 0.0:
        return np.array(raw, copy=True), float(length)

    system = np.eye(length, dtype=np.float64)
    penalty = np.zeros((length, length), dtype=np.float64)
    diagonal = np.arange(length)
    penalty[diagonal, diagonal] = 2.0
    penalty[0, 0] = penalty[-1, -1] = 1.0
    penalty[diagonal[:-1], diagonal[1:]] = -1.0
    penalty[diagonal[1:], diagonal[:-1]] = -1.0
    system += temporal_lambda * penalty

    flat = raw.reshape(length, -1).astype(np.float64)
    smoothed = np.linalg.solve(system, flat).reshape(raw.shape)
    effective_time_degrees_of_freedom = float(np.trace(np.linalg.inv(system)))
    return smoothed, effective_time_degrees_of_freedom


@register_connectivity("time_varying")
class SmoothTimeVaryingConnectivity(ConnectivityEstimator):
    """L2-smooth sequence estimator that resets at every real recording/gap boundary."""

    estimation_mode = "sequence_regularized_candidate"

    def __init__(self, cfg: ConnectivityConfig) -> None:
        super().__init__(cfg)
        options: dict[str, Any] = dict(cfg.extra)
        self.temporal_lambda = float(options.pop("temporal_lambda", 1.0))
        if self.temporal_lambda < 0:
            raise ValueError("temporal_lambda must be non-negative")
        base_name = str(options.pop("base_estimator", "cgc"))
        if base_name == "time_varying":
            raise ValueError("time_varying cannot use itself as its base_estimator")
        base_extra = options.pop("base_extra", options)
        if not isinstance(base_extra, dict):
            raise TypeError("base_extra must be a mapping")
        self.base_estimator = ConnectivityFactory(
            ConnectivityConfig(
                name=base_name,
                max_lag=cfg.max_lag,
                alpha=cfg.alpha,
                partial=cfg.partial,
                ridge=cfg.ridge,
                absolute=False,
                threshold=0.0,
                extra=base_extra,
            )
        )
        self.directed = self.base_estimator.directed
        self.weighted = self.base_estimator.weighted
        self.weight_semantics = self.base_estimator.weight_semantics
        self._sequence_diagnostics: dict[str, Any] = {}

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        """Delegate the per-window baseline fit to the configured base estimator."""
        return self.base_estimator.estimate(segment)

    def estimate_sequence(self, segments: WindowedSegments) -> np.ndarray:
        raw = np.stack(
            [self.base_estimator._postprocess(self.base_estimator.estimate(seg)) for seg in segments.segments]
        )
        smoothed = np.empty_like(raw, dtype=np.float64)
        slices = _continuous_slices(segments)
        effective_df = 0.0
        for segment_slice in slices:
            smoothed[segment_slice], segment_df = _smooth_segment(raw[segment_slice], self.temporal_lambda)
            effective_df += segment_df
        self._sequence_diagnostics = {
            "candidate_status": "candidate_not_promoted",
            "base_estimator": self.base_estimator.cfg.name,
            "temporal_penalty": "squared_first_difference",
            "temporal_lambda": self.temporal_lambda,
            "continuous_segments": len(slices),
            "effective_time_degrees_of_freedom": effective_df,
            "promotion_gate": "synthetic_recovery_prediction_calibration_stability_misspecification",
        }
        return smoothed

    def _diagnostics(self, segments: WindowedSegments, mats: np.ndarray) -> dict[str, Any]:
        diagnostics = super()._diagnostics(segments, mats)
        diagnostics.update(self._sequence_diagnostics)
        return diagnostics
