"""Faithful BundDLe-Net port pinned to upstream commit ``cbefad93``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import ManifoldEmbedder, TargetSlice, register_manifold

logger = logging.getLogger(__name__)

_UPSTREAM_COMMIT = "cbefad93"
_DEFAULT_EPOCHS = 3000
_DEFAULT_BATCH_SIZE = 100
_DEFAULT_GAMMA = 0.9
_DEFAULT_LEARNING_RATE = 1.0e-3
_DEFAULT_GAUSSIAN_NOISE = 0.05
_DEFAULT_BEHAVIOR_CLASSES = 8


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
    """Build BundDLe training windows and one-sample-offset targets."""
    x = np.asarray(neural, dtype=np.float32)
    b = np.asarray(behavior)
    if x.ndim != 2:
        raise ValueError(f"BundDLe expects neural with shape (T, N), got {x.shape}")
    if b.ndim != 1:
        raise ValueError(f"BundDLe expects 1D behavior labels, got {b.shape}")
    if x.shape[0] != b.shape[0]:
        raise ValueError("BundDLe neural/behavior lengths must match")
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


def _resolved_epochs(cfg: Any) -> int:
    extra = getattr(cfg, "extra", {}) or {}
    if "epochs" in extra:
        epochs = int(extra["epochs"])
    elif int(getattr(cfg, "max_iter", _DEFAULT_EPOCHS)) == 2000:
        # Preserve the faithful upstream default without widening the config schema.
        epochs = _DEFAULT_EPOCHS
    else:
        epochs = int(getattr(cfg, "max_iter", _DEFAULT_EPOCHS))
    if epochs <= 0:
        raise ValueError("BundDLe epochs must be positive")
    return epochs


def _resolved_hyperparameters(cfg: Any) -> dict[str, Any]:
    extra = getattr(cfg, "extra", {}) or {}
    batch_size = int(extra.get("batch_size", _DEFAULT_BATCH_SIZE))
    gamma = float(extra.get("gamma", _DEFAULT_GAMMA))
    learning_rate = float(extra.get("learning_rate", _DEFAULT_LEARNING_RATE))
    noise_std = float(extra.get("noise_std", _DEFAULT_GAUSSIAN_NOISE))
    behavior_classes = int(extra.get("behavior_classes", _DEFAULT_BEHAVIOR_CLASSES))
    if batch_size <= 0:
        raise ValueError("BundDLe batch_size must be positive")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("BundDLe gamma must lie in [0, 1]")
    if learning_rate <= 0:
        raise ValueError("BundDLe learning_rate must be positive")
    if noise_std < 0:
        raise ValueError("BundDLe noise_std must be non-negative")
    if behavior_classes <= 0:
        raise ValueError("BundDLe behavior_classes must be positive")
    return {
        "epochs": _resolved_epochs(cfg),
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "noise_std": noise_std,
        "behavior_classes": behavior_classes,
    }


def _validate_behavior_labels(labels: np.ndarray, n_classes: int) -> np.ndarray:
    arr = np.asarray(labels)
    if arr.ndim != 1:
        raise ValueError("BundDLe behavior labels must be 1D")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("BundDLe requires integer categorical behavior labels")
    if arr.size == 0:
        raise ValueError("BundDLe requires at least one behavior label")
    if int(arr.min()) < 0 or int(arr.max()) >= n_classes:
        raise ValueError(f"BundDLe behavior labels must lie in [0, {n_classes - 1}]")
    return arr.astype(np.int64, copy=False)


def _load_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised through fit/skip behavior
        raise ImportError(
            "BundDLe-Net requires PyTorch. Install '.[manifold]' or `uv pip install torch`."
        ) from exc
    return torch, nn


def _set_deterministic_seed(torch: Any, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _module_device(module: Any) -> Any:
    return next(module.parameters()).device


def _build_model_spec(
    *,
    cfg: Any,
    n_input_features: int,
    target_length: int,
    n_neurons: int,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_family": "BundDLe-Net",
        "port_kind": "faithful_pytorch",
        "upstream_commit": _UPSTREAM_COMMIT,
        "latent_dim": int(cfg.n_dims),
        "target_length": int(target_length),
        "n_neurons": int(n_neurons),
        "n_input_features": int(n_input_features),
        "behavior_key": str(cfg.behavior_key),
        "behavior_classes": int(hyperparameters["behavior_classes"]),
        "encoder": {
            "preprocess": "flatten",
            "hidden_layers": [50, 30, 25, 10],
            "activations": ["relu", "relu", "relu", "relu"],
            "output": "linear_latent -> normalization(axis=-1) -> gaussian_noise(train_only)",
        },
        "transition": {
            "type": "residual_linear_plus_normalization",
            "formula": "z_pred = z_t + Normalization(axis=-1)(Linear(z_t))",
        },
        "behavior_head": {
            "type": "linear",
            "input": "z_next",
            "target": "behavior_next",
            "n_logits": int(hyperparameters["behavior_classes"]),
        },
        "loss": {
            "formula": "gamma * mse(z_next, z_pred) + (1 - gamma) * sparse_ce(predictor(z_next), y_next)",
            "gamma": float(hyperparameters["gamma"]),
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": float(hyperparameters["learning_rate"]),
        },
        "training": {
            "epochs": int(hyperparameters["epochs"]),
            "batch_size": int(hyperparameters["batch_size"]),
            "shuffle": False,
            "seed": int(cfg.seed),
            "gaussian_noise_std": float(hyperparameters["noise_std"]),
        },
        "window_contract": {
            "pairing": "adjacent_shifted_target_windows",
            "behavior_alignment": "final_frame_next_label",
        },
    }


@register_manifold("bunddle")
class BundDLeManifold(ManifoldEmbedder):
    """Faithful BundDLe-Net port using target-window codes and next-step behavior loss."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._encoder = None
        self._transition_linear = None
        self._behavior_head = None
        self._noise_std = _DEFAULT_GAUSSIAN_NOISE
        self.model_spec: dict[str, Any] = {}
        self.training_history: list[dict[str, float | int]] = []
        self.loss_history: list[dict[str, float | int]] = []
        self.fit_provenance: dict[str, Any] = {}

    def fit(self, neural: np.ndarray, behavior: dict[str, np.ndarray]) -> BundDLeManifold:
        torch, nn = _load_torch()
        labels = behavior.get(self.cfg.behavior_key)
        if labels is None:
            raise ValueError(f"behavior_key '{self.cfg.behavior_key}' not found for BundDLe-Net")

        hyperparameters = _resolved_hyperparameters(self.cfg)
        batch = build_bundle_training_batch(neural, labels, self.cfg.target_length)
        y_next_np = _validate_behavior_labels(batch.behavior_next, hyperparameters["behavior_classes"])
        _set_deterministic_seed(torch, int(self.cfg.seed))

        x_t = torch.tensor(batch.x_t.reshape(len(batch.x_t), -1), dtype=torch.float32)
        x_next = torch.tensor(batch.x_next.reshape(len(batch.x_next), -1), dtype=torch.float32)
        y_next = torch.tensor(y_next_np, dtype=torch.long)
        n_inputs = int(x_t.shape[1])
        encoder = nn.Sequential(
            nn.Linear(n_inputs, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, int(self.cfg.n_dims)),
        )
        transition_linear = nn.Linear(int(self.cfg.n_dims), int(self.cfg.n_dims))
        behavior_head = nn.Linear(
            int(self.cfg.n_dims), int(hyperparameters["behavior_classes"])
        )
        parameters = (
            list(encoder.parameters())
            + list(transition_linear.parameters())
            + list(behavior_head.parameters())
        )
        optimizer = torch.optim.Adam(
            parameters, lr=float(hyperparameters["learning_rate"])
        )
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()

        self.model_spec = _build_model_spec(
            cfg=self.cfg,
            n_input_features=n_inputs,
            target_length=self.cfg.target_length,
            n_neurons=batch.x_t.shape[2],
            hyperparameters=hyperparameters,
        )
        self.fit_provenance = {
            "framework": "pytorch",
            "torch_version": str(torch.__version__),
            "seed": int(self.cfg.seed),
            "deterministic_algorithms": bool(
                getattr(torch, "are_deterministic_algorithms_enabled", lambda: False)()
            ),
            "upstream_commit": _UPSTREAM_COMMIT,
            "chronological_batches": True,
            "shuffle": False,
        }

        history: list[dict[str, float | int]] = []
        batch_size = int(hyperparameters["batch_size"])
        gamma = float(hyperparameters["gamma"])
        n_samples = int(x_t.shape[0])
        noise_std = float(hyperparameters["noise_std"])

        for epoch in range(int(hyperparameters["epochs"])):
            encoder.train()
            transition_linear.train()
            behavior_head.train()
            total_loss = 0.0
            total_transition = 0.0
            total_behavior = 0.0
            batches = 0
            samples_seen = 0
            for start in range(0, n_samples, batch_size):
                stop = min(start + batch_size, n_samples)
                optimizer.zero_grad()
                z_t = encoder(x_t[start:stop])
                z_next = encoder(x_next[start:stop])
                if noise_std > 0.0:
                    z_t = z_t + torch.randn_like(z_t) * noise_std
                    z_next = z_next + torch.randn_like(z_next) * noise_std
                z_pred = z_t + transition_linear(z_t)
                logits_next = behavior_head(z_next)
                transition_loss = mse_loss(z_pred, z_next)
                behavior_loss = ce_loss(logits_next, y_next[start:stop])
                loss = gamma * transition_loss + (1.0 - gamma) * behavior_loss
                loss.backward()
                optimizer.step()

                weight = stop - start
                total_loss += float(loss.detach()) * weight
                total_transition += float(transition_loss.detach()) * weight
                total_behavior += float(behavior_loss.detach()) * weight
                batches += 1
                samples_seen += weight

            epoch_record = {
                "epoch": epoch + 1,
                "samples": samples_seen,
                "batches": batches,
                "loss": total_loss / samples_seen,
                "transition_loss": total_transition / samples_seen,
                "behavior_loss": total_behavior / samples_seen,
            }
            history.append(epoch_record)
            if (epoch + 1) % max(1, int(hyperparameters["epochs"]) // 5) == 0 or epoch == 0:
                logger.debug(
                    "BundDLe epoch=%d/%d loss=%.6f transition=%.6f behavior=%.6f",
                    epoch + 1,
                    int(hyperparameters["epochs"]),
                    epoch_record["loss"],
                    epoch_record["transition_loss"],
                    epoch_record["behavior_loss"],
                )

        encoder.eval()
        transition_linear.eval()
        behavior_head.eval()
        self._encoder = encoder
        self._transition_linear = transition_linear
        self._behavior_head = behavior_head
        self._noise_std = noise_std
        self.training_history = history
        self.loss_history = history
        self._is_fitted = True
        return self

    def _encode_windows(self, windows: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self._encoder is None:
            raise RuntimeError("BundDLeManifold must be fitted before encoding windows")
        torch, _ = _load_torch()
        encoder = self._encoder
        x = np.asarray(windows, dtype=np.float32).reshape(len(windows), -1)
        device = _module_device(encoder)
        encoder.eval()
        with torch.no_grad():
            z = encoder(torch.tensor(x, dtype=torch.float32, device=device))
        return z.detach().cpu().numpy().astype(np.float32)

    def transform(self, neural: np.ndarray) -> np.ndarray:
        x = np.asarray(neural, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"BundDLe expects neural with shape (T, N), got {x.shape}")
        if x.shape[0] < self.cfg.target_length:
            raise ValueError("not enough samples to construct BundDLe target windows")
        windows = np.stack(
            [x[i : i + self.cfg.target_length] for i in range(x.shape[0] - self.cfg.target_length + 1)]
        )
        return self._encode_windows(windows)

    def transform_targets(
        self,
        neural: np.ndarray,
        target_slices: list[TargetSlice],
        behavior: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        x = np.asarray(neural, dtype=np.float32)
        if any(ts.length != self.cfg.target_length for ts in target_slices):
            raise ValueError("BundDLe target slices must all match cfg.target_length")
        windows = np.stack([x[ts.start : ts.stop] for ts in target_slices])
        return self._encode_windows(windows)
