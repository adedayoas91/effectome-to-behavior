"""Base class and config for connectivity estimators (Stage 2)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from effectome.data_module.schema import ConnectivitySeries, WindowedSegments

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
    extra: dict = field(default_factory=dict)


class ConnectivityEstimator(ABC):
    """Estimate one N x N connectivity matrix from an (L, N) segment.

    Convention: entry [i, j] is the (directed) influence of source neuron i on target j.
    Subclasses implement `estimate`; `run` maps the estimator over all windows.
    """

    directed: bool = True

    def __init__(self, cfg: ConnectivityConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def estimate(self, segment: np.ndarray) -> np.ndarray:
        """Return an N x N connectivity matrix for one (L, N) segment."""

    def _postprocess(self, w: np.ndarray) -> np.ndarray:
        if self.cfg.absolute:
            w = np.abs(w)
        if self.cfg.threshold > 0:
            w = np.where(np.abs(w) >= self.cfg.threshold, w, 0.0)
        np.fill_diagonal(w, 0.0)
        return w

    def run(self, segments: WindowedSegments) -> ConnectivitySeries:
        """Estimate a connectivity matrix per window -> ConnectivitySeries."""
        mats = np.stack([self._postprocess(self.estimate(seg)) for seg in segments.segments])
        starts = np.array([w.start for w in segments.windows])
        logger.info("Estimated %d %s matrices (%d neurons)", len(mats), self.cfg.name, segments.n_neurons)
        return ConnectivitySeries(
            matrices=mats.astype(np.float32),
            window_starts=starts,
            method=self.cfg.name,
            directed=self.directed,
            behavior_per_window=segments.behavior_per_window,
        )
