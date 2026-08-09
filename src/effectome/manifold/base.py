"""Base class, config, and registry for behavioral-manifold embedders (Stage 5)."""

from __future__ import annotations

import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from effectome.data_module.schema import ArtifactProvenance, TemporalAnchor

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
        target_length: Length of the target window aligned to each connectivity anchor.
        extra: Method-specific options.
    """

    name: str = "classical"
    n_dims: int = 3
    method: str = "pca"
    behavior_key: str = "continuous"
    max_iter: int = 2000
    seed: int = 42
    target_length: int = 15
    cross_fit: bool = True
    n_folds: int = 5
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSlice:
    """Half-open [start, stop) interval used to align latent codes to anchors."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("target slices must start at a non-negative sample index")
        if self.start >= self.stop:
            raise ValueError("target slices must be ordered half-open intervals")

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class ManifoldArtifact:
    """Serialized Stage-5 output aligned to downstream Stage-6/7 windows."""

    method: str
    behavior_key: str
    full_embedding: np.ndarray
    window_embedding: np.ndarray
    target_slices: list[TargetSlice]
    target_length: int
    model_path: str | None = None
    window_starts: np.ndarray | None = None
    anchors: list[TemporalAnchor] = field(default_factory=list)
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.full_embedding.ndim != 2:
            raise ValueError(f"full_embedding must be 2D (T, D), got shape {self.full_embedding.shape}")
        if self.window_embedding.ndim != 2:
            raise ValueError(f"window_embedding must be 2D (K, D), got shape {self.window_embedding.shape}")
        if self.full_embedding.shape[1] != self.window_embedding.shape[1]:
            raise ValueError("full_embedding and window_embedding must share the same latent dimension")
        n_windows = self.window_embedding.shape[0]
        if len(self.target_slices) != n_windows:
            raise ValueError(f"target_slices length {len(self.target_slices)} != K {n_windows}")
        if self.target_length <= 0:
            raise ValueError("target_length must be positive")
        if any(ts.length != self.target_length for ts in self.target_slices):
            raise ValueError("every target slice must have length == target_length")
        full_embedding_indexing = self.metadata.get("full_embedding_indexing", "sample")
        if full_embedding_indexing not in {"sample", "target_window_end"}:
            raise ValueError("metadata['full_embedding_indexing'] must be 'sample' or 'target_window_end'")
        if (
            full_embedding_indexing == "sample"
            and self.target_slices
            and self.full_embedding.shape[0] < max(ts.stop for ts in self.target_slices)
        ):
            raise ValueError("full_embedding is shorter than one or more target slices")
        if self.window_starts is not None and self.window_starts.shape[0] != n_windows:
            raise ValueError(f"window_starts length {self.window_starts.shape[0]} != K {n_windows}")
        if self.anchors and len(self.anchors) != n_windows:
            raise ValueError(f"anchors length {len(self.anchors)} != K {n_windows}")
        for idx, anchor in enumerate(self.anchors):
            ts = self.target_slices[idx]
            if anchor.target_start != ts.start or anchor.target_stop != ts.stop:
                raise ValueError("target_slices must align with anchor target intervals")
            if self.window_starts is not None and int(self.window_starts[idx]) != anchor.context_start:
                raise ValueError("window_starts must align with anchor.context_start")


class ManifoldEmbedder(ABC):
    """Embed neural activity (optionally with behavior) into a low-dimensional manifold."""

    def __init__(self, cfg: ManifoldConfig) -> None:
        self.cfg = cfg
        self._is_fitted = False

    @abstractmethod
    def fit(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> ManifoldEmbedder:
        """Fit the manifold model from neural (T, N) and aligned behavior arrays."""

    @abstractmethod
    def transform(self, neural: np.ndarray) -> np.ndarray:
        """Transform neural activity into latent coordinates."""

    def fit_transform(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> np.ndarray:
        """Fit the model and return latent coordinates for the provided inputs."""
        return self.fit(neural, behavior).transform(neural)

    def embed(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> np.ndarray:
        """Backward-compatible one-shot API."""
        return self.fit_transform(neural, behavior)

    def transform_targets(
        self,
        neural: np.ndarray,
        target_slices: list[TargetSlice],
        behavior: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return one latent code per target slice.

        Default behavior averages sample-level embeddings across each target interval.
        Window-aware embedders (e.g. BundDLe-Net) can override this.
        """
        embedding = self.transform(neural)
        if embedding.shape[0] < max(ts.stop for ts in target_slices):
            raise ValueError("sample-level embedding is shorter than one or more target slices")
        return np.stack([embedding[ts.start : ts.stop].mean(axis=0) for ts in target_slices])

    def save(self, path: str | Path) -> Path:
        """Persist the fitted embedder via pickle."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved manifold model -> %s", p)
        return p

    @staticmethod
    def load(path: str | Path) -> ManifoldEmbedder:
        """Load a pickled embedder."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ManifoldEmbedder):
            raise TypeError(f"Loaded object from {path} is not a ManifoldEmbedder")
        return obj


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
