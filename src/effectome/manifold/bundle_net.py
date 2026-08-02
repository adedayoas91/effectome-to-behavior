"""BundDLe-Net-style manifold with 15-sample target windows and one-sample-offset pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .base import ManifoldEmbedder, TargetSlice, register_manifold

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BundleTrainingBatch:
    """Prepared BundDLe inputs using one-sample-offset target windows."""

    x_t: np.ndarray
    x_next: np.ndarray
    behavior_t: np.ndarray
    behavior_next: np.ndarray


def build_bundle_training_batch(
    neural: np.ndarray,
    behavior: np.ndarray,
    target_length: int,
) -> BundleTrainingBatch:
    """Build BundDLe training windows and one-sample-offset targets.

    For target length H, the window ending at sample t is neural[t-H+1:t+1].
    Training pairs use consecutive windows ending at t and t+1 respectively.
    """
    x = np.asarray(neural, dtype=np.float32)
    b = np.asarray(behavior)
    if x.ndim != 2:
        raise ValueError(f"BundDLe expects neural with shape (T, N), got {x.shape}")
    if target_length < 2:
        raise ValueError("target_length must be at least 2")
    if x.shape[0] <= target_length:
        raise ValueError("not enough samples to build BundDLe windows")

    windows = np.stack([x[i : i + target_length] for i in range(x.shape[0] - target_length + 1)])
    return BundleTrainingBatch(
        x_t=windows[:-1],
        x_next=windows[1:],
        behavior_t=b[target_length - 1 : -1],
        behavior_next=b[target_length:],
    )


@register_manifold("bunddle")
class BundDLeManifold(ManifoldEmbedder):
    """Joint dynamics+behavior-preserving embedding using target-window codes z_k."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._torch = None
        self._encoder = None
        self._predictor = None
        self._decoder = None
        self._is_discrete = None

    def fit(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> BundDLeManifold:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                "BundDLe-Net requires PyTorch. Install '.[manifold]' or `uv pip install torch`."
            ) from exc

        b = behavior.get(self.cfg.behavior_key)
        if b is None:
            raise ValueError(f"behavior_key '{self.cfg.behavior_key}' not found for BundDLe-Net")

        batch = build_bundle_training_batch(neural, b, self.cfg.target_length)
        torch.manual_seed(self.cfg.seed)
        self._torch = torch
        n_in = batch.x_t.shape[1] * batch.x_t.shape[2]
        d = self.cfg.n_dims
        b_arr = np.asarray(b)
        is_discrete = np.issubdtype(b_arr.dtype, np.integer)
        self._is_discrete = bool(is_discrete)
        y_t = (
            torch.tensor(batch.behavior_t).long()
            if is_discrete
            else torch.tensor(batch.behavior_t.astype(np.float32))
        )
        n_classes = int(b_arr.max()) + 1 if is_discrete else 1

        encoder = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, d))
        predictor = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, d))
        decoder = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, n_classes))
        params = list(encoder.parameters()) + list(predictor.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        beh_loss = nn.CrossEntropyLoss() if is_discrete else nn.MSELoss()

        x_t = torch.tensor(batch.x_t.reshape(len(batch.x_t), -1), dtype=torch.float32)
        x_next = torch.tensor(batch.x_next.reshape(len(batch.x_next), -1), dtype=torch.float32)
        for it in range(self.cfg.max_iter):
            opt.zero_grad()
            z_t = encoder(x_t)
            z_next = encoder(x_next)
            z_pred = predictor(z_t)
            logits = decoder(z_t)
            if is_discrete:
                loss_behav = beh_loss(logits, y_t)
            else:
                loss_behav = beh_loss(logits, y_t.view(-1, 1).float())
            loss_dyn = ((z_pred - z_next.detach()) ** 2).mean()
            loss_smooth = ((z_next - z_t) ** 2).mean()
            loss = loss_behav + loss_dyn + 0.01 * loss_smooth
            loss.backward()
            opt.step()
            if it % max(1, self.cfg.max_iter // 5) == 0:
                logger.debug("BundDLe it=%d loss=%.4f", it, float(loss))

        self._encoder = encoder
        self._predictor = predictor
        self._decoder = decoder
        self._is_fitted = True
        return self

    def transform(self, neural: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._encoder is None or self._torch is None:
            raise RuntimeError("BundDLeManifold must be fitted before transform()")
        x = np.asarray(neural, dtype=np.float32)
        if x.shape[0] < self.cfg.target_length:
            raise ValueError("not enough samples to construct BundDLe target windows")
        windows = np.stack(
            [x[i : i + self.cfg.target_length] for i in range(x.shape[0] - self.cfg.target_length + 1)]
        )
        with self._torch.no_grad():
            z = self._encoder(
                self._torch.tensor(windows.reshape(len(windows), -1), dtype=self._torch.float32)
            )
        return z.numpy().astype(np.float32)

    def transform_targets(
        self,
        neural: np.ndarray,
        target_slices: list[TargetSlice],
        behavior: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if not self._is_fitted or self._encoder is None or self._torch is None:
            raise RuntimeError("BundDLeManifold must be fitted before transform_targets()")
        x = np.asarray(neural, dtype=np.float32)
        windows = np.stack([x[ts.start : ts.stop] for ts in target_slices])
        if any(ts.length != self.cfg.target_length for ts in target_slices):
            raise ValueError("BundDLe target slices must all match cfg.target_length")
        with self._torch.no_grad():
            z = self._encoder(
                self._torch.tensor(windows.reshape(len(windows), -1), dtype=self._torch.float32)
            )
        return z.numpy().astype(np.float32)
