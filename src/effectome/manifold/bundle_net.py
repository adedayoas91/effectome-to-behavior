"""BundDLe-Net-style manifold: preserve dynamics *and* behavior (Kumar et al., 2023).

BundDLe-Net learns a low-dimensional neural manifold whose latent trajectory is both (i)
one-step predictable (dynamics-preserving) and (ii) sufficient to decode behavior
(behavior-preserving). This module provides a compact PyTorch implementation of that joint
objective behind the `ManifoldEmbedder` interface. It is an independent re-implementation in the
spirit of the paper, not the authors' original code. Requires PyTorch (the `manifold` extra
pulls in CEBRA which brings torch; or install torch directly).
"""

from __future__ import annotations

import logging

import numpy as np

from .base import ManifoldEmbedder, register_manifold

logger = logging.getLogger(__name__)


@register_manifold("bunddle")
class BundDLeManifold(ManifoldEmbedder):
    """Joint dynamics+behavior-preserving embedding (BundDLe-Net-style).

    Trains an encoder f: x_t -> y_t with three losses:
      * behavior:  decode behavior b_t from y_t,
      * dynamics:  a latent predictor T(y_t) ~= y_{t+1} (and decoded behavior consistency),
      * smooth:    light L2 on latent steps for stable trajectories.
    """

    def embed(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> np.ndarray:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                "BundDLe-Net requires PyTorch. Install '.[manifold]' or `uv pip install torch`."
            ) from exc

        torch.manual_seed(self.cfg.seed)
        x = torch.tensor(np.asarray(neural, dtype=np.float32))
        n_in = x.shape[1]
        d = self.cfg.n_dims
        b = behavior.get(self.cfg.behavior_key)
        if b is None:
            raise ValueError(f"behavior_key '{self.cfg.behavior_key}' not found for BundDLe-Net")
        b_arr = np.asarray(b)
        is_discrete = np.issubdtype(b_arr.dtype, np.integer)
        y_behav = torch.tensor(b_arr).long() if is_discrete else torch.tensor(b_arr.astype(np.float32))
        n_classes = int(b_arr.max()) + 1 if is_discrete else 1

        encoder = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, d))
        predictor = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, d))
        decoder = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n_classes))
        params = list(encoder.parameters()) + list(predictor.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        beh_loss = nn.CrossEntropyLoss() if is_discrete else nn.MSELoss()

        x_t, x_next = x[:-1], x[1:]
        b_t = y_behav[:-1]
        for it in range(self.cfg.max_iter):
            opt.zero_grad()
            y_t = encoder(x_t)
            y_next = encoder(x_next)
            y_pred = predictor(y_t)
            logits = decoder(y_t)
            target = b_t if is_discrete else b_t.view(-1, 1)
            loss_behav = beh_loss(logits, b_t) if is_discrete else beh_loss(logits, target.float())
            loss_dyn = ((y_pred - y_next.detach()) ** 2).mean()
            loss_smooth = ((y_next - y_t) ** 2).mean()
            loss = loss_behav + loss_dyn + 0.01 * loss_smooth
            loss.backward()
            opt.step()
            if it % max(1, self.cfg.max_iter // 5) == 0:
                logger.debug("BundDLe it=%d loss=%.4f", it, float(loss))

        with torch.no_grad():
            emb = encoder(x).numpy().astype(np.float32)
        return emb
