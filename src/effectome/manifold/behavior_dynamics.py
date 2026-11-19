"""Training-owned, continuous pointwise BunDLe-inspired behavioral encoders.

This separately named variant is not the faithful categorical/window port in
``bundle_net``. Neural intervention dynamics remain in the full observed space;
the latent transition is an auxiliary training objective only. PyTorch training
runs in a separate runtime and exports numerical weights for frozen inference.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .base import ManifoldConfig, ManifoldEmbedder, register_manifold

logger = logging.getLogger(__name__)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runtime_path(runtime_python: str | Path | None = None) -> Path:
    runtime = (
        Path(runtime_python).expanduser()
        if runtime_python is not None
        else (_repository_root() / "tools/behavior-encoder/.venv/bin/python")
    )
    if not runtime.is_file():
        raise FileNotFoundError(
            f"behavior encoder runtime is missing: {runtime}. Install it with "
            "`uv sync --project tools/behavior-encoder --python 3.12 --locked`; "
            "no PCA fallback is used."
        )
    # Preserve the venv launcher path. Resolving its symlink can bypass pyvenv.cfg.
    return runtime.absolute()


def _worker_path() -> Path:
    path = _repository_root() / "tools/behavior-encoder/worker.py"
    if not path.is_file():
        raise FileNotFoundError(f"behavior encoder worker is missing: {path}")
    return path


def _worker_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return environment


def behavior_runtime_info(runtime_python: str | Path | None = None) -> dict[str, Any]:
    """Probe the real encoder runtime without importing Torch into this process."""
    runtime = _runtime_path(runtime_python)
    process = subprocess.run(
        [str(runtime), str(_worker_path()), "--probe"],
        capture_output=True,
        text=True,
        env=_worker_environment(),
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"behavior encoder runtime probe failed: {process.stderr.strip()}")
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("behavior encoder runtime returned invalid probe JSON") from error
    required = {"torch_version", "numpy_version", "python_version", "platform", "device"}
    if not isinstance(value, dict) or not required <= value.keys():
        raise RuntimeError("behavior encoder runtime probe omitted required provenance")
    return value


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _real(value: Any, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite and numeric")
    if result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _trial_lengths(values: Sequence[int], total: int | None = None) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError("trial_lengths must explicitly identify the complete training trials")
    lengths = tuple(_integer(value, "trial_length", 3) for value in values)
    if not lengths:
        raise ValueError("at least one complete trial is required for pair sampling")
    if total is not None and (len(lengths) < 2 or sum(lengths) != total):
        raise ValueError("at least two training trial lengths must sum to the supplied neural sample count")
    return lengths


def build_behavior_pair_indices(
    trial_lengths: Sequence[int],
    max_pairs_per_trial: int = 128,
    early_samples: int = 32,
) -> dict[str, Any]:
    """Build balanced early/late adjacent pairs without crossing trial boundaries.

    Half of each trial's declared pair budget samples its early region; the other
    half samples later positions. Repeated positions are explicit design weights,
    not additional independent observations. Both temporal supports stay within
    the originating trial. Arrays are returned in trial, then early/late order.
    """
    lengths = _trial_lengths(trial_lengths)
    budget = _integer(max_pairs_per_trial, "max_pairs_per_trial", 4)
    if budget % 2:
        raise ValueError("max_pairs_per_trial must be even for balanced early/late sampling")
    early_limit = _integer(early_samples, "early_samples")
    current, next_indices, owners, times = [], [], [], []
    offset = 0
    for trial, length in enumerate(lengths):
        n_pairs = length - 1
        split = min(early_limit, max(1, n_pairs // 2))
        half = budget // 2
        early = np.linspace(0, split - 1, half).astype(int)
        late = np.linspace(split, n_pairs - 1, half).astype(int)
        selected = np.concatenate((early, late))
        current.extend((offset + selected).tolist())
        next_indices.extend((offset + selected + 1).tolist())
        owners.extend([trial] * budget)
        times.extend(selected.tolist())
        offset += length
    return {
        "current_indices": np.asarray(current, dtype=np.int64),
        "next_indices": np.asarray(next_indices, dtype=np.int64),
        "trial_ids": np.asarray(owners, dtype=np.int64),
        "time_indices": np.asarray(times, dtype=np.int64),
        "metadata": {
            "policy": "equal_early_late_pair_count_per_trial_deterministic_with_repetition",
            "max_pairs_per_trial": budget,
            "early_samples": early_limit,
            "trial_lengths": list(lengths),
            "trial_boundary_crossings": 0,
            "repeated_rows_are_independent_units": False,
        },
    }


@dataclass(frozen=True)
class BehaviorDynamicsSettings:
    latent_dim: int
    seed: int
    validation_fraction: float = 0.25
    dynamics_weights: tuple[float, ...] = (0.05, 0.2)
    hidden_widths: tuple[int, ...] = (50, 30, 25, 10)
    max_epochs: int = 200
    min_epochs: int = 20
    patience: int = 20
    batch_size: int = 128
    max_pairs_per_trial: int = 128
    early_samples: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_threads: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in (
            "latent_dim",
            "max_epochs",
            "min_epochs",
            "patience",
            "batch_size",
            "max_pairs_per_trial",
            "early_samples",
            "n_threads",
        ):
            _integer(getattr(self, name), name)
        _integer(self.seed, "seed", 0)
        if self.min_epochs > self.max_epochs:
            raise ValueError("min_epochs must not exceed max_epochs")
        if self.max_pairs_per_trial < 4 or self.max_pairs_per_trial % 2:
            raise ValueError("max_pairs_per_trial must be even and >= 4")
        if not self.hidden_widths:
            raise ValueError("hidden_widths must identify a nonempty architecture")
        for width in self.hidden_widths:
            _integer(width, "hidden_width")
        fraction = _real(self.validation_fraction, "validation_fraction")
        if fraction >= 1:
            raise ValueError("validation_fraction must lie strictly between zero and one")
        if not self.dynamics_weights or len(set(self.dynamics_weights)) != len(self.dynamics_weights):
            raise ValueError("dynamics_weights must contain distinct nonnegative penalties")
        for value in self.dynamics_weights:
            _real(value, "dynamics_weight", positive=False)
        _real(self.learning_rate, "learning_rate")
        _real(self.weight_decay, "weight_decay", positive=False)
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda or auto")


def _settings(cfg: ManifoldConfig) -> BehaviorDynamicsSettings:
    extra = cfg.extra
    epochs = extra.get("max_epochs", cfg.max_iter if cfg.max_iter != 2000 else 200)
    default_min = min(20, _integer(epochs, "max_epochs"))
    keys = {
        field
        for field in BehaviorDynamicsSettings.__dataclass_fields__
        if field
        not in {"latent_dim", "seed", "max_epochs", "min_epochs", "dynamics_weights", "hidden_widths"}
    }
    values = {key: extra[key] for key in keys if key in extra}
    return BehaviorDynamicsSettings(
        latent_dim=cfg.n_dims,
        seed=cfg.seed,
        max_epochs=epochs,
        min_epochs=extra.get("min_epochs", default_min),
        dynamics_weights=tuple(extra.get("dynamics_weights", (0.05, 0.2))),
        hidden_widths=tuple(extra.get("hidden_widths", (50, 30, 25, 10))),
        **values,
    )


def _finite_neural(neural: np.ndarray) -> np.ndarray:
    values = np.asarray(neural, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("neural must be a finite nonempty sample-by-neuron matrix")
    return values


@register_manifold("bunddle_behavior")
class BehaviorDynamicsManifold(ManifoldEmbedder):
    """Pointwise continuous-output adaptation with a training-only latent auxiliary loss."""

    def __init__(self, cfg: ManifoldConfig) -> None:
        super().__init__(cfg)
        if cfg.target_length != 1 or cfg.target_seconds is not None:
            raise ValueError("bunddle_behavior requires the explicit pointwise target_length=1 contract")
        self.settings = _settings(cfg)
        self._artifact: dict[str, Any] = {}
        self._layers: tuple[tuple[np.ndarray, np.ndarray, str], ...] = ()
        self._input_mean = np.empty(0)
        self._input_scale = np.empty(0)
        self._latent_layer = -1
        self.training_history: list[dict[str, Any]] = []
        self.fit_provenance: dict[str, Any] = {}

    def _restore(self, artifact: Mapping[str, Any]) -> None:
        copied = json.loads(json.dumps(dict(artifact), allow_nan=False))
        mean = np.asarray(copied["input_mean"], dtype=float)
        scale = np.asarray(copied["input_scale"], dtype=float)
        if (
            mean.ndim != 1
            or not mean.size
            or scale.shape != mean.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("exported input scaling must contain finite matching positive scales")
        layers = []
        width = len(mean)
        for layer in copied["layers"]:
            weight, bias = np.asarray(layer["weights"], dtype=float), np.asarray(layer["bias"], dtype=float)
            activation = layer["activation"]
            if (
                weight.ndim != 2
                or weight.shape[1] != width
                or bias.shape != (weight.shape[0],)
                or not np.isfinite(weight).all()
                or not np.isfinite(bias).all()
                or activation not in {"linear", "relu", "tanh"}
            ):
                raise ValueError("exported network layers must have finite compatible numerical parameters")
            width = weight.shape[0]
            weight.setflags(write=False)
            bias.setflags(write=False)
            layers.append((weight, bias, activation))
        index = copied["latent_layer"]
        if (
            not layers
            or type(index) is not int
            or not 0 <= index < len(layers) - 1
            or layers[index][0].shape[0] != self.cfg.n_dims
            or width != 2
        ):
            raise ValueError(
                "exported network must identify the latent layer and two-dimensional behavior head"
            )
        mean.setflags(write=False)
        scale.setflags(write=False)
        self._input_mean, self._input_scale = mean, scale
        self._layers = tuple(layers)
        self._latent_layer = index
        self._artifact = copied
        self._is_fitted = True

    def _forward(self, neural: np.ndarray, *, latent: bool = False) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("BehaviorDynamicsManifold must be fitted before frozen inference")
        values = _finite_neural(neural)
        if values.shape[1] != len(self._input_mean):
            raise ValueError("neural input must match the fitted neuron axis")
        values = (values - self._input_mean) / self._input_scale
        for index, (weight, bias, activation) in enumerate(self._layers):
            values = values @ weight.T + bias
            if activation == "relu":
                values = np.maximum(values, 0)
            elif activation == "tanh":
                values = np.tanh(values)
            if latent and index == self._latent_layer:
                break
        if not np.isfinite(values).all():
            raise ValueError("frozen behavior encoder inference became nonfinite")
        return values

    def fit(
        self,
        neural: np.ndarray,
        behavior: dict[str, np.ndarray],
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BehaviorDynamicsManifold:
        values = _finite_neural(neural)
        labels = np.asarray(behavior.get(self.cfg.behavior_key), dtype=float)
        if labels.shape != (len(values), 2) or not np.isfinite(labels).all():
            raise ValueError("continuous behavior must be finite with shape (samples, 2)")
        lengths = _trial_lengths(self.cfg.extra.get("trial_lengths", ()), len(values))
        if self.cfg.n_dims > values.shape[1]:
            raise ValueError("latent dimension must not exceed the observed neural dimension")
        pairs = build_behavior_pair_indices(
            lengths,
            self.settings.max_pairs_per_trial,
            self.settings.early_samples,
        )
        runtime = _runtime_path(self.cfg.extra.get("runtime_python"))
        worker = _worker_path()
        settings = {**asdict(self.settings), "trial_lengths": list(lengths)}
        logs = []
        with tempfile.TemporaryDirectory(prefix="effectome-behavior-") as directory:
            folder = Path(directory)
            inputs, config, outputs = folder / "inputs.npz", folder / "config.json", folder / "output.json"
            np.savez_compressed(
                inputs,
                neural=values,
                behavior=labels,
                **{
                    key: pairs[key]
                    for key in ("current_indices", "next_indices", "trial_ids", "time_indices")
                },
            )
            config.write_text(json.dumps(settings, allow_nan=False))
            process = subprocess.Popen(
                [
                    str(runtime),
                    str(worker),
                    "--input",
                    str(inputs),
                    "--config",
                    str(config),
                    "--output",
                    str(outputs),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_worker_environment(),
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    logs.append(line.rstrip())
                    logger.info("Behavior encoder: %s", line.rstrip())
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if progress_callback is not None and event.get("event") == "epoch":
                        progress_callback(int(event["epoch"]), int(event["epochs"]))
                returncode = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                if process.stdout is not None:
                    process.stdout.close()
            if returncode:
                raise RuntimeError("behavior encoder training failed: " + "\n".join(logs[-20:]))
            result = json.loads(outputs.read_text())
        self._restore(result["artifact"])
        probe = result["parity"]
        comparison = np.asarray(probe["neural"], dtype=float)
        prediction_error = float(np.max(np.abs(self.predict_behavior(comparison) - probe["prediction"])))
        latent_error = float(np.max(np.abs(self.transform(comparison) - probe["latent"])))
        if max(prediction_error, latent_error) > 1e-4:
            self._is_fitted = False
            raise RuntimeError("exported NumPy encoder does not match the trained PyTorch model within 1e-4")
        metadata = self._artifact["metadata"]
        lock = _repository_root() / "tools/behavior-encoder/uv.lock"
        metadata.update(
            {
                "worker_source_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
                "runtime_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
                if lock.is_file()
                else None,
                "input_neural_sha256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
                "input_behavior_sha256": hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest(),
                "pair_sampling": {
                    **pairs["metadata"],
                    "current_indices": pairs["current_indices"].tolist(),
                    "next_indices": pairs["next_indices"].tolist(),
                    "trial_ids": pairs["trial_ids"].tolist(),
                    "time_indices": pairs["time_indices"].tolist(),
                },
                "numpy_prediction_parity_max_abs_error": prediction_error,
                "numpy_latent_parity_max_abs_error": latent_error,
                "numpy_parity_tolerance": 1e-4,
            }
        )
        self.training_history = result["training"]["refit_history"]
        self.fit_provenance = {"runtime": result["runtime"], "ownership": result["training"]}
        return self

    def transform(self, neural: np.ndarray) -> np.ndarray:
        return self._forward(neural, latent=True)

    def predict_behavior(self, neural: np.ndarray) -> np.ndarray:
        return self._forward(neural)

    def export_artifact(self) -> dict[str, Any]:
        if not self._is_fitted:
            raise RuntimeError(
                "BehaviorDynamicsManifold must be fitted before exporting its frozen parameters"
            )
        return json.loads(json.dumps(self._artifact, allow_nan=False))

    def to_dict(self) -> dict[str, Any]:
        return self.export_artifact()

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> BehaviorDynamicsManifold:
        layers = artifact["layers"]
        index = artifact["latent_layer"]
        dimensions = len(layers[index]["bias"])
        config = ManifoldConfig(name="bunddle_behavior", n_dims=dimensions, target_length=1)
        model = cls(config)
        model._restore(artifact)
        return model
