"""Base class, config, and registry for behavioral-manifold embedders (Stage 5)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManifoldConfig:
    """Configuration for manifold embedders.

    Attributes:
        name: Registry key ('classical', 'cebra', 'bunddle').
        n_dims: Output embedding dimensionality.
        method: Sub-method for 'classical' ('pca' or 'umap').
        behavior_key: Behavior variable used for supervised/contrastive embedders.
        max_iter: Training iterations for learned embedders.
        seed: Random seed.
        extra: Method-specific options.
    """

    name: str = "classical"
    n_dims: int = 3
    method: str = "pca"
    behavior_key: str = "continuous"
    max_iter: int = 2000
    seed: int = 42
    extra: dict = field(default_factory=dict)


class ManifoldEmbedder(ABC):
    """Embed neural activity (optionally with behavior) into a low-dimensional manifold."""

    def __init__(self, cfg: ManifoldConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def embed(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> np.ndarray:
        """Return an embedding of shape (T, n_dims) from neural (T, N) and behavior arrays."""


MANIFOLD_REGISTRY: dict[str, type[ManifoldEmbedder]] = {}


def register_manifold(name: str):
    """Class decorator registering a `ManifoldEmbedder` under `name`."""

    def deco(cls: type[ManifoldEmbedder]) -> type[ManifoldEmbedder]:
        if name in MANIFOLD_REGISTRY:
            raise ValueError(f"manifold '{name}' already registered")
        MANIFOLD_REGISTRY[name] = cls
        return cls

    return deco


def ManifoldFactory(cfg: ManifoldConfig) -> ManifoldEmbedder:
    """Instantiate the embedder selected by `cfg.name`."""
    if cfg.name not in MANIFOLD_REGISTRY:
        raise KeyError(f"unknown manifold '{cfg.name}'; available: {sorted(MANIFOLD_REGISTRY)}")
    return MANIFOLD_REGISTRY[cfg.name](cfg)
