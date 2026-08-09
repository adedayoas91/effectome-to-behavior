"""Base class and config for connectivity estimators (Stage 2)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from effectome.data_module.schema import ArtifactProvenance, ConnectivitySeries, WindowedSegments

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectivityConfig:
    """Configuration shared by connectivity estimators.

    Attributes:
        name: Registry key selecting the estimator (e.g. 'correlation', 'granger', 'pcmci').
        max_lag: Maximum time lag (samples) considered for causal influence.
        alpha: Significance threshold for edge inclusion (constraint/test-based methods).
        partial: Use partial correlation instead of marginal correlation (correlation method).
        ridge: L2 regularization strength for VAR/Granger fits.
        absolute: Take absolute value of weights (treat strength, ignore sign).
        threshold: Optional magnitude threshold; entries below are zeroed (0 = off).
        extra: Method-specific options passed through to the estimator.
    """

    name: str = "correlation"
    max_lag: int = 1
    alpha: float = 0.05
    partial: bool = False
    ridge: float = 1.0
    absolute: bool = False
    threshold: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class ConnectivityEstimator(ABC):
    """Estimate one N x N connectivity matrix from an (L, N) segment.

    Convention: entry [i, j] is the (directed) influence of source neuron i on target j.
    Subclasses implement ``estimate``; ``estimate_sequence`` can be overridden by sequence-level
    estimators that share information across adjacent anchors.
    """

    directed: bool = True
    weighted: bool = True
    weight_semantics: str = "effective_influence"
    estimation_mode: str = "rolling_window"

    def __init__(self, cfg: ConnectivityConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def estimate(self, segment: np.ndarray) -> np.ndarray:
        """Return an N x N connectivity matrix for one (L, N) segment."""

    def estimate_sequence(self, segments: WindowedSegments) -> np.ndarray:
        """Default sequence implementation: estimate one matrix per window independently."""
        return np.stack([self._postprocess(self.estimate(seg)) for seg in segments.segments])

    def _postprocess(self, w: np.ndarray) -> np.ndarray:
        if self.cfg.absolute:
            w = np.abs(w)
        if self.cfg.threshold > 0:
            w = np.where(np.abs(w) >= self.cfg.threshold, w, 0.0)
        np.fill_diagonal(w, 0.0)
        return w

    def _diagnostics(self, segments: WindowedSegments, mats: np.ndarray) -> dict[str, Any]:
        sparsity = float(np.mean(np.abs(mats) < 1e-12))
        anchor_samples = [anchor.anchor_sample for anchor in segments.anchors]
        return {
            "config": {
                "name": self.cfg.name,
                "max_lag": self.cfg.max_lag,
                "alpha": self.cfg.alpha,
                "partial": self.cfg.partial,
                "ridge": self.cfg.ridge,
                "absolute": self.cfg.absolute,
                "threshold": self.cfg.threshold,
                "extra": dict(self.cfg.extra),
            },
            "directed": self.directed,
            "weighted": self.weighted,
            "signed": not self.cfg.absolute,
            "weight_semantics": self.weight_semantics,
            "estimation_mode": self.estimation_mode,
            "storage": "dense",
            "window_mode": segments.metadata.get("window_mode"),
            "input_neural_standardization": segments.metadata.get("neural_standardization"),
            "n_windows": int(mats.shape[0]),
            "n_neurons": int(mats.shape[1]),
            "anchor_samples": anchor_samples,
            "sparsity": sparsity,
        }

    def _lagged_output(self) -> np.ndarray | None:
        """Return optional lag-resolved weights aligned with the last sequence run."""
        return None

    def run(self, segments: WindowedSegments) -> ConnectivitySeries:
        """Estimate a connectivity matrix per window -> ConnectivitySeries."""
        started = perf_counter()
        mats = np.stack([self._postprocess(mat) for mat in self.estimate_sequence(segments)])
        lagged_matrices = self._lagged_output()
        if lagged_matrices is not None and lagged_matrices.shape[0] != mats.shape[0]:
            raise ValueError("lag-resolved output must align with the estimated windows")
        if lagged_matrices is not None:
            lagged_matrices = np.asarray(lagged_matrices, dtype=np.float64).copy()
            if self.cfg.absolute:
                lagged_matrices = np.abs(lagged_matrices)
            if self.cfg.threshold > 0:
                lagged_matrices = np.where(
                    np.abs(lagged_matrices) >= self.cfg.threshold,
                    lagged_matrices,
                    0.0,
                )
            diagonal = np.arange(lagged_matrices.shape[-1])
            lagged_matrices[:, :, diagonal, diagonal] = 0.0
        elapsed_seconds = perf_counter() - started
        starts = np.array([w.start for w in segments.windows])
        diagnostics = self._diagnostics(segments, mats)
        diagnostics["runtime"] = {
            "elapsed_seconds": float(elapsed_seconds),
            "seconds_per_window": float(elapsed_seconds / max(1, mats.shape[0])),
            "dense_output_bytes": int(mats.nbytes),
            "note": "Output memory only; estimator workspace and NumPy-native allocations are not included.",
        }
        logger.info("Estimated %d %s matrices (%d neurons)", len(mats), self.cfg.name, segments.n_neurons)
        return ConnectivitySeries(
            matrices=mats.astype(np.float32),
            window_starts=starts,
            method=self.cfg.name,
            directed=self.directed,
            behavior_per_window=segments.behavior_per_window,
            anchors=list(segments.anchors),
            signed=not self.cfg.absolute,
            weighted=self.weighted,
            storage="dense",
            weight_semantics=self.weight_semantics,
            diagnostics=diagnostics,
            lagged_matrices=(
                lagged_matrices.astype(np.float32)
                if lagged_matrices is not None
                else None
            ),
            provenance=ArtifactProvenance(
                identity=segments.provenance.identity,
                stage="connectivity",
                source=segments.provenance.source,
                configuration_id=self.cfg.name,
                random_seed=segments.provenance.random_seed,
                code_version=segments.provenance.code_version,
                fit_data_ids=(segments.provenance.identity.recording_id,),
                units={"weights": self.weight_semantics},
                axis_conventions={
                    "matrices": "anchor,source_neuron,target_neuron",
                    "lagged_matrices": "anchor,lag,source_neuron,target_neuron",
                },
                metadata={
                    "upstream_stage": segments.provenance.stage,
                    "input_neural_standardization": segments.metadata.get(
                        "neural_standardization"
                    ),
                },
            ),
        )
