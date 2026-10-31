"""Classical behavioral-manifold embedders: PCA (always available) or UMAP (optional)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.decomposition import PCA

from .base import ManifoldEmbedder, register_manifold


@register_manifold("classical")
class ClassicalManifold(ManifoldEmbedder):
    """PCA or UMAP embedding of neural activity (unsupervised baseline)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._model = None

    def fit(
        self,
        neural: np.ndarray,
        behavior: dict[str, np.ndarray],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ClassicalManifold:
        x = np.asarray(neural, dtype=np.float64)
        if self.cfg.method == "umap":
            try:
                import umap
            except ImportError as exc:
                raise ImportError(
                    "UMAP requires umap-learn. Install it with `uv pip install umap-learn`, "
                    "or explicitly select method='pca'."
                ) from exc
            self._model = umap.UMAP(n_components=self.cfg.n_dims, random_state=self.cfg.seed)
            self._model.fit(x)
            self._is_fitted = True
            return self

        self._model = PCA(n_components=self.cfg.n_dims, random_state=self.cfg.seed)
        self._model.fit(x)
        self._is_fitted = True
        return self

    def transform(self, neural: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("ClassicalManifold must be fitted before transform()")
        x = np.asarray(neural, dtype=np.float64)
        return np.asarray(self._model.transform(x), dtype=np.float32)
