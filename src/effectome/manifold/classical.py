"""Classical behavioral-manifold embedders: PCA (always available) or UMAP (optional)."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

from .base import ManifoldEmbedder, register_manifold

logger = logging.getLogger(__name__)


@register_manifold("classical")
class ClassicalManifold(ManifoldEmbedder):
    """PCA or UMAP embedding of neural activity (unsupervised baseline)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._model = None

    def fit(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> ClassicalManifold:
        x = np.asarray(neural, dtype=np.float64)
        if self.cfg.method == "umap":
            try:
                import umap

                self._model = umap.UMAP(n_components=self.cfg.n_dims, random_state=self.cfg.seed)
                self._model.fit(x)
                self._is_fitted = True
                return self
            except ImportError:
                logger.warning("umap-learn not installed; falling back to PCA. Install '.[manifold]'.")

        self._model = PCA(n_components=self.cfg.n_dims, random_state=self.cfg.seed)
        self._model.fit(x)
        self._is_fitted = True
        return self

    def transform(self, neural: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("ClassicalManifold must be fitted before transform()")
        x = np.asarray(neural, dtype=np.float64)
        return np.asarray(self._model.transform(x), dtype=np.float32)
