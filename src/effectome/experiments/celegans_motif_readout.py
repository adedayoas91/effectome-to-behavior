"""Training-owned categorical motor motif readout with a two-dimensional bottleneck."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from effectome.experiments.behavior_readout import FrozenDenseLayer, _freeze
from effectome.workflows.resume import content_fingerprint

_SCHEMA = "celegans_motif_readout_v1"


@dataclass(frozen=True)
class MotifReadout:
    """Apply a fitted pointwise neural-to-motif map in the declared class order."""

    input_mean: np.ndarray
    input_scale: np.ndarray
    layers: tuple[FrozenDenseLayer, ...]
    n_classes: int
    classes: tuple[int, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        mean = _freeze(self.input_mean)
        scale = _freeze(self.input_scale, mean.shape)
        if mean.ndim != 1 or np.any(scale <= 0):
            raise ValueError("motif readout needs one neural input axis with positive scales")
        if type(self.n_classes) is not int or self.n_classes < 2:
            raise ValueError("n_classes must declare at least two possible motifs")
        classes = tuple(self.classes)
        if (
            len(classes) < 2
            or any(type(i) is not int or not 0 <= i < self.n_classes for i in classes)
            or len(set(classes)) != len(classes)
        ):
            raise ValueError("training classes must be distinct IDs in the declared motif vocabulary")
        layers = tuple(self.layers)
        if len(layers) != 3 or any(not isinstance(layer, FrozenDenseLayer) for layer in layers):
            raise ValueError("motif readout requires a hidden layer, bottleneck, and logit head")
        hidden, bottleneck, head = layers
        if (
            hidden.weights.shape[1] != mean.size
            or bottleneck.weights.shape != (2, hidden.weights.shape[0])
            or head.weights.shape != (len(classes), 2)
            or tuple(layer.activation for layer in layers) != ("tanh", "tanh", "linear")
        ):
            raise ValueError("invalid motif readout layer sizes or activations")
        object.__setattr__(self, "input_mean", mean)
        object.__setattr__(self, "input_scale", scale)
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False)))

    def encode(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        if values.ndim < 1 or values.shape[-1] != self.input_mean.size or not np.isfinite(values).all():
            raise ValueError("motif inference needs finite states ending in the neural axis")
        hidden, bottleneck, _ = self.layers
        standardized = (values - self.input_mean) / self.input_scale
        return bottleneck.apply(hidden.apply(standardized))

    def predict(self, states: np.ndarray) -> np.ndarray:
        logits = self.layers[-1].apply(self.encode(states))
        probabilities = np.zeros((*logits.shape[:-1], self.n_classes))
        probabilities[..., list(self.classes)] = softmax(logits, axis=-1)
        return probabilities

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": _SCHEMA,
            "input_mean": self.input_mean.tolist(),
            "input_scale": self.input_scale.tolist(),
            "layers": [layer.to_dict() for layer in self.layers],
            "n_classes": self.n_classes,
            "classes": list(self.classes),
            "metadata": json.loads(json.dumps(self.metadata, allow_nan=False)),
        }
        return {**payload, "fingerprint": content_fingerprint(payload)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MotifReadout:
        payload = dict(value)
        fingerprint = payload.pop("fingerprint", None)
        if fingerprint != content_fingerprint(payload) or payload.pop("schema_version", None) != _SCHEMA:
            raise ValueError("motif readout artifact schema or fingerprint is invalid")
        payload["layers"] = tuple(FrozenDenseLayer(**layer) for layer in payload["layers"])
        return cls(**payload)


def fit_motif_readout(
    neural: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    *,
    seed: int,
    hidden_width: int,
    alpha: float,
    max_iter: int,
) -> MotifReadout:
    """Fit scaled pointwise neural activity to training motif labels by cross-entropy."""
    if type(n_classes) is not int or n_classes < 2:
        raise ValueError("n_classes must declare at least two possible motifs")
    if type(hidden_width) is not int or hidden_width < 1 or type(max_iter) is not int or max_iter < 1:
        raise ValueError("hidden_width and max_iter must be positive integers")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    values = np.asarray(neural, dtype=float)
    motif = np.asarray(labels)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("training data must be a finite, nonempty sample-by-neuron matrix")
    if (
        motif.shape != (len(values),)
        or motif.dtype.kind not in "iu"
        or np.any((motif < 0) | (motif >= n_classes))
    ):
        raise ValueError("training labels must be aligned integer IDs in the declared motif vocabulary")
    if len(np.unique(motif)) < 2:
        raise ValueError("training observations must contain at least two distinct motifs")

    scaler = StandardScaler().fit(values)
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_width, 2),
        activation="tanh",
        solver="lbfgs",
        alpha=alpha,
        max_iter=max_iter,
        random_state=seed,
    ).fit(scaler.transform(values), motif)
    hidden = tuple(
        FrozenDenseLayer(weights.T, bias, "tanh")
        for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1], strict=True)
    )
    weights, bias = model.coefs_[-1].T, model.intercepts_[-1]
    if len(model.classes_) == 2:
        # Softmax of [0, z] matches the binary classifier's sigmoid on z.
        weights = np.concatenate((np.zeros_like(weights), weights), axis=0)
        bias = np.concatenate((np.zeros_like(bias), bias))
    return MotifReadout(
        scaler.mean_,
        scaler.scale_,
        (*hidden, FrozenDenseLayer(weights, bias)),
        n_classes,
        tuple(int(value) for value in model.classes_),
        {
            "fit_scope": "provided_training_block",
            "endpoint": "fictive_motor_motif",
            "n_samples": len(values),
            "iterations": int(model.n_iter_),
            "training_loss": float(model.loss_),
            "seed": seed,
            "hidden_width": hidden_width,
            "alpha": float(alpha),
            "max_iter": max_iter,
        },
    )
