"""Training-owned continuous behavioral measurements and frozen NumPy inference."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from effectome.experiments.distributed_behavior_controls import distributed_readout_matrix
from effectome.manifold.behavior_dynamics import build_behavior_pair_indices
from effectome.workflows.resume import content_fingerprint


def _freeze(values: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    if not array.size or not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise ValueError("frozen behavioral parameters must be finite with their declared shape")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FrozenDenseLayer:
    weights: np.ndarray
    bias: np.ndarray
    activation: str = "linear"

    def __post_init__(self) -> None:
        weights = _freeze(self.weights)
        if weights.ndim != 2:
            raise ValueError("layer weights must be output-by-input")
        if self.activation not in {"linear", "relu", "tanh"}:
            raise ValueError("unsupported frozen activation")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", _freeze(self.bias, (weights.shape[0],)))

    def apply(self, values: np.ndarray) -> np.ndarray:
        result = values @ self.weights.T + self.bias
        if self.activation == "relu":
            result = np.maximum(result, 0)
        elif self.activation == "tanh":
            result = np.tanh(result)
        if not np.isfinite(result).all():
            raise ValueError("behavioral inference became nonfinite; no clipping or filtering is applied")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"weights": self.weights.tolist(), "bias": self.bias.tolist(), "activation": self.activation}


@dataclass(frozen=True)
class FrozenBehaviorMap:
    name: str
    input_mean: np.ndarray
    input_scale: np.ndarray
    layers: tuple[FrozenDenseLayer, ...]
    latent_layer: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mean = _freeze(self.input_mean)
        if mean.ndim != 1 or not self.name:
            raise ValueError("a named readout needs a nonempty input axis")
        scale = _freeze(self.input_scale, mean.shape)
        if np.any(scale <= 0):
            raise ValueError("input scales must be positive")
        if not self.layers or any(not isinstance(layer, FrozenDenseLayer) for layer in self.layers):
            raise ValueError("readouts need explicit frozen dense layers")
        previous = len(mean)
        for layer in self.layers:
            if layer.weights.shape[1] != previous:
                raise ValueError("consecutive readout layer dimensions do not match")
            previous = layer.weights.shape[0]
        if previous != 2:
            raise ValueError("this experiment requires exactly two continuous behavioral outputs")
        if type(self.latent_layer) is not int or not -1 <= self.latent_layer < len(self.layers):
            raise ValueError("latent_layer must identify a layer or standardized raw inputs")
        object.__setattr__(self, "input_mean", mean)
        object.__setattr__(self, "input_scale", scale)
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False)))

    def _inputs(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        if (
            values.ndim < 1
            or not values.size
            or values.shape[-1] != len(self.input_mean)
            or not np.isfinite(values).all()
        ):
            raise ValueError("behavioral readout inputs must be finite and end in the neural axis")
        return (values - self.input_mean) / self.input_scale

    def predict(self, states: np.ndarray) -> np.ndarray:
        result = self._inputs(states)
        for layer in self.layers:
            result = layer.apply(result)
        return result

    def encode(self, states: np.ndarray) -> np.ndarray:
        result = self._inputs(states)
        for layer in self.layers[: self.latent_layer + 1]:
            result = layer.apply(result)
        return result

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": "continuous_behavior_map_v1",
            "name": self.name,
            "input_mean": self.input_mean.tolist(),
            "input_scale": self.input_scale.tolist(),
            "layers": [layer.to_dict() for layer in self.layers],
            "latent_layer": self.latent_layer,
            "metadata": self.metadata,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        return {**payload, "fingerprint": fingerprint}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FrozenBehaviorMap:
        payload = dict(value)
        digest = payload.pop("fingerprint", None)
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if digest != expected or payload.pop("schema_version", None) != "continuous_behavior_map_v1":
            raise ValueError("frozen behavioral artifact schema or fingerprint is invalid")
        payload["layers"] = tuple(FrozenDenseLayer(**layer) for layer in payload["layers"])
        return cls(**payload)

    @classmethod
    def from_export(cls, value: dict[str, Any], *, name: str | None = None) -> FrozenBehaviorMap:
        """Import a checked numerical worker export, never a pickled executable model."""
        return cls(
            name or value["name"],
            np.asarray(value["input_mean"]),
            np.asarray(value["input_scale"]),
            tuple(FrozenDenseLayer(**layer) for layer in value["layers"]),
            value["latent_layer"],
            {**value.get("metadata", {}), "worker_export_fingerprint": content_fingerprint(value)},
        )


def _rows(neural: np.ndarray, behavior: np.ndarray, max_pairs: int, early_samples: int) -> tuple:
    pairs = build_behavior_pair_indices([neural.shape[1]] * len(neural), max_pairs, early_samples)
    indices = np.concatenate((pairs["current_indices"], pairs["next_indices"]))
    return neural.reshape(-1, neural.shape[-1])[indices], behavior.reshape(-1, 2)[indices]


def fixed_behavior_readouts() -> dict[str, FrozenBehaviorMap]:
    reference = distributed_readout_matrix()
    n_nodes = reference.shape[1]

    def fixed(name: str, weights: np.ndarray, activation: str) -> FrozenBehaviorMap:
        return FrozenBehaviorMap(
            name,
            np.zeros(n_nodes),
            np.ones(n_nodes),
            (FrozenDenseLayer(weights, np.zeros(2), activation),),
            0,
            {"fit_scope": "predeclared_design_no_fitting", "output_units": "behavior_units"},
        )

    proxy, nuisance = np.zeros((2, n_nodes)), np.zeros((2, n_nodes))
    proxy[0, 0], proxy[1, 1] = 1, 1
    nuisance[0, 6], nuisance[1, 7] = 1, 1
    return {
        "behavior_reference": fixed("behavior_reference", reference, "tanh"),
        "constant": fixed("constant", np.zeros_like(reference), "linear"),
        "driver_proxy": fixed("driver_proxy", proxy, "tanh"),
        "nuisance_proxy": fixed("nuisance_proxy", nuisance, "tanh"),
    }


def fit_linear_behavior_readouts(
    training_trials: np.ndarray,
    behavior_trials: np.ndarray,
    *,
    ridge_alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
    validation_fraction: float = 0.25,
    max_pairs_per_trial: int = 128,
    early_samples: int = 32,
    seed: int = 42,
) -> dict[str, FrozenBehaviorMap]:
    """Select on inner whole trials, then refit every selected parameter on outer training."""
    x, y = np.asarray(training_trials, dtype=float), np.asarray(behavior_trials, dtype=float)
    if (
        x.ndim != 3
        or x.shape[0] < 4
        or x.shape[1] < 4
        or x.shape[2] < 4
        or y.shape != (*x.shape[:2], 2)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("training data need matching finite trial/time/neural and two-behavior axes")
    if (
        not 0 < validation_fraction < 1
        or not ridge_alphas
        or any(isinstance(a, bool) or not np.isfinite(a) or a <= 0 for a in ridge_alphas)
    ):
        raise ValueError("validation fraction and ridge candidates are invalid")
    split = len(x) - max(1, int(np.ceil(len(x) * validation_fraction)))
    if split < 2:
        raise ValueError("inner training needs at least two whole trials")
    fit_x, fit_y = _rows(x[:split], y[:split], max_pairs_per_trial, early_samples)
    val_x, val_y = _rows(x[split:], y[split:], max_pairs_per_trial, early_samples)
    all_x, all_y = _rows(x, y, max_pairs_per_trial, early_samples)
    output = {}

    def encoder(values: np.ndarray, targets: np.ndarray, method: str, dimension: int) -> tuple:
        scaler = StandardScaler().fit(values)
        standardized = scaler.transform(values)
        if method == "pca":
            model = PCA(n_components=dimension, svd_solver="full", random_state=seed).fit(standardized)
            weights = model.components_
            bias = -model.mean_ @ weights.T
        elif method == "pls":
            model = PLSRegression(n_components=dimension, scale=False).fit(standardized, targets)
            # Recover the fitted affine transform via its public interface, avoiding private scaler fields.
            bias = model.transform(np.zeros((1, values.shape[1])))[0]
            weights = (model.transform(np.eye(values.shape[1])) - bias).T
        else:
            weights, bias = np.eye(values.shape[1]), np.zeros(values.shape[1])
        layer = FrozenDenseLayer(weights, bias)
        return scaler, layer, layer.apply(standardized)

    for name, method, dimension in (
        ("pca_2d", "pca", 2),
        ("pca_4d", "pca", 4),
        ("pls_2d", "pls", 2),
        ("full_linear", "identity", x.shape[-1]),
    ):
        inner_scaler, inner_layer, fit_codes = encoder(fit_x, fit_y, method, dimension)
        val_codes = inner_layer.apply(inner_scaler.transform(val_x))
        scores = []
        for alpha in ridge_alphas:
            readout = Ridge(alpha=alpha).fit(fit_codes, fit_y)
            scores.append(float(np.mean((readout.predict(val_codes) - val_y) ** 2)))
        selected = int(np.argmin(scores))
        scaler, layer, codes = encoder(all_x, all_y, method, dimension)
        readout = Ridge(alpha=ridge_alphas[selected]).fit(codes, all_y)
        output[name] = FrozenBehaviorMap(
            name,
            scaler.mean_,
            scaler.scale_,
            (layer, FrozenDenseLayer(readout.coef_, readout.intercept_)),
            0,
            {
                "family": method,
                "supervision": "encoder_and_readout" if method == "pls" else "readout_only",
                "dimension": dimension,
                "candidate_ridge_alphas": list(ridge_alphas),
                "inner_validation_mse": scores,
                "selected_alpha": ridge_alphas[selected],
                "inner_training_trial_ids": list(range(split)),
                "inner_validation_trial_ids": list(range(split, len(x))),
                "refit_trial_ids": list(range(len(x))),
                "refit_rows": len(all_x),
                "fit_scope": "outer_training_only_selection_then_full_refit",
                "training_fingerprint": content_fingerprint((x, y)),
                "max_pairs_per_trial": max_pairs_per_trial,
                "early_samples": early_samples,
                "sampling": "same_balanced_within_trial_pairs_as_nonlinear_models",
                "seed": seed,
            },
        )
    return output
