"""CEBRA joint behaviour-neural embedding (Schneider, Lee & Mathis, Nature 2023).

CEBRA learns a latent embedding by contrastive learning that can be conditioned on behavior
labels, producing a behavior-aligned neural manifold. We wrap it behind the `ManifoldEmbedder`
interface; CEBRA (the `manifold` extra) is imported lazily so the package installs without it.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import ManifoldEmbedder, register_manifold

logger = logging.getLogger(__name__)


@register_manifold("cebra")
class CebraManifold(ManifoldEmbedder):
    """Contrastive behaviour-conditioned neural embedding via CEBRA.

    Uses behavior variable `cfg.behavior_key` as the auxiliary/label signal. Requires the
    `manifold` extra (`uv pip install -e '.[manifold]'`).
    """

    def embed(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> np.ndarray:
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
        return np.asarray(model.transform(x), dtype=np.float32)
