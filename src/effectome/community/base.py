"""Base class, config, and registry for community detection (Stage 4)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from effectome.data_module.schema import CommunitySeries, ConnectivitySeries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommunityConfig:
    """Configuration for community detection.

    Attributes:
        name: Registry key ('leiden', 'markov_stability', 'temporal').
        resolution: Resolution parameter (higher -> more, smaller communities).
        symmetrize: Symmetrize directed matrices before detection.
        weight_threshold: Drop edges below this absolute weight before detection.
        seed: Random seed for stochastic detectors.
        extra: Method-specific options (e.g. temporal interlayer coupling).
    """

    name: str = "leiden"
    resolution: float = 1.0
    symmetrize: bool = True
    weight_threshold: float = 0.0
    seed: int = 42
    extra: dict = field(default_factory=dict)


class CommunityDetector(ABC):
    """Partition neurons into communities for each connectivity matrix."""

    def __init__(self, cfg: CommunityConfig) -> None:
        self.cfg = cfg

    def _prepare(self, w: np.ndarray) -> np.ndarray:
        a = np.abs(w) if True else w
        if self.cfg.symmetrize:
            a = 0.5 * (a + a.T)
        if self.cfg.weight_threshold > 0:
            a = np.where(a >= self.cfg.weight_threshold, a, 0.0)
        np.fill_diagonal(a, 0.0)
        return a

    @abstractmethod
    def detect_one(self, matrix: np.ndarray) -> np.ndarray:
        """Return integer community labels, shape (N,), for one N x N matrix."""

    def run(self, series: ConnectivitySeries) -> CommunitySeries:
        """Detect communities per window -> CommunitySeries."""
        labels = np.stack([self.detect_one(self._prepare(w)) for w in series.matrices])
        counts = np.array([len(np.unique(row)) for row in labels])
        logger.info("Detected communities for %d windows (%s)", series.n_windows, self.cfg.name)
        return CommunitySeries(
            labels=labels.astype(np.int64),
            method=self.cfg.name,
            n_communities_per_window=counts,
        )


COMMUNITY_REGISTRY: dict[str, type[CommunityDetector]] = {}


def register_community(name: str):
    """Class decorator registering a `CommunityDetector` under `name`."""

    def deco(cls: type[CommunityDetector]) -> type[CommunityDetector]:
        if name in COMMUNITY_REGISTRY:
            raise ValueError(f"community '{name}' already registered")
        COMMUNITY_REGISTRY[name] = cls
        return cls

    return deco


def CommunityFactory(cfg: CommunityConfig) -> CommunityDetector:
    """Instantiate the community detector selected by `cfg.name`."""
    if cfg.name not in COMMUNITY_REGISTRY:
        raise KeyError(f"unknown community '{cfg.name}'; available: {sorted(COMMUNITY_REGISTRY)}")
    return COMMUNITY_REGISTRY[cfg.name](cfg)
