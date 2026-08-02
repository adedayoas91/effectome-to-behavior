"""CEBRA joint behaviour-neural embedding (Schneider, Lee & Mathis, Nature 2023)."""

from __future__ import annotations

import numpy as np

from .base import ManifoldEmbedder, register_manifold


@register_manifold("cebra")
class CebraManifold(ManifoldEmbedder):
    """Contrastive behaviour-conditioned neural embedding via CEBRA."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._model = None

    def fit(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> CebraManifold:
        try:
            import cebra
        except ImportError as exc:
            raise ImportError(
                "CEBRA requires the 'manifold' extra: `uv pip install -e '.[manifold]'`."
            ) from exc

        x = np.asarray(neural, dtype=np.float32)
        aux = behavior.get(self.cfg.behavior_key)
        model = cebra.CEBRA(
            model_architecture="offset10-model",
            output_dimension=self.cfg.n_dims,
            max_iterations=self.cfg.max_iter,
            distance="cosine",
            device="cuda_if_available",
            verbose=False,
            **self.cfg.extra,
        )
        if aux is not None:
            model.fit(x, np.asarray(aux))
        else:
            model.fit(x)
        self._model = model
        self._is_fitted = True
        return self

    def transform(self, neural: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("CebraManifold must be fitted before transform()")
        x = np.asarray(neural, dtype=np.float32)
        return np.asarray(self._model.transform(x), dtype=np.float32)
