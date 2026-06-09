"""Classical behavioral-manifold embedders: PCA (always available) or UMAP (optional)."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

from .base import ManifoldEmbedder, register_manifold

logger = logging.getLogger(__name__)


@register_manifold("classical")
class ClassicalManifold(ManifoldEmbedder):
    """PCA or UMAP embedding of neural activity (unsupervised baseline).

    PCA is the dependency-light default. UMAP (via the optional `manifold` extra) captures
    nonlinear structure; if it is unavailable the embedder falls back to PCA with a warning.
    """

    def embed(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> np.ndarray:
        x = np.asarray(neural, dtype=np.float64)
        if self.cfg.method == "umap":
            try:
                import umap

                reducer = umap.UMAP(
                    n_components=self.cfg.n_dims, random_state=self.cfg.seed
                )
                return reducer.fit_transform(x).astype(np.float32)
            except ImportError:
                logger.warning("umap-learn not installed; falling back to PCA. Install '.[manifold]'.")

        pca = PCA(n_components=self.cfg.n_dims, random_state=self.cfg.seed)
        return pca.fit_transform(x).astype(np.float32)
