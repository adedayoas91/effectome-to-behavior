"""Frozen binary measurements for the first learned-endpoint mechanism bridge.

Neural-space dynamics remain separate from this measurement. The PCA encoder is
unsupervised; ``pca_behavior`` adds an explicitly supervised behavioral readout.
The behavioral reference is fixed before fitting as the sign of a declared neural
coordinate, not selected from the learned representation. This linear bridge is
not evidence for nonlinear manifold learning or autonomous latent dynamics.

See ``docs/learned_manifold_validation_plan.md`` for the scientific rationale and
verified references. PCA/standardization follow the existing shared-transition
reference; affine parameters are exported directly for frozen, auditable use.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any, Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Endpoint(Protocol):
    """An instantaneous factual source and a time-first future target measure."""

    def source_states(self, initial: np.ndarray) -> np.ndarray: ...

    def target_states(self, target: np.ndarray) -> np.ndarray: ...

    def to_dict(self) -> dict[str, Any]: ...


def _finite(values: np.ndarray, *, n_nodes: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim < 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("endpoint observations must be nonempty and finite")
    if n_nodes is not None and array.shape[-1] != n_nodes:
        raise ValueError("endpoint observations must end in the fitted neural axis")
    return array


def _fingerprinted(payload: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {**payload, "fingerprint": digest}


@dataclass(frozen=True)
class CoordinateEndpoint:
    """The predeclared observed-coordinate reference or a proxy control."""

    endpoint_index: int
    threshold: float = 0.0

    def __post_init__(self) -> None:
        if type(self.endpoint_index) is not int or self.endpoint_index < 0:
            raise ValueError("endpoint_index must be a nonnegative integer")
        if (isinstance(self.threshold, bool) or not isinstance(self.threshold, Real)
                or not np.isfinite(self.threshold)):
            raise ValueError("threshold must be finite")

    def source_states(self, initial: np.ndarray) -> np.ndarray:
        values = _finite(initial)
        if self.endpoint_index >= values.shape[-1]:
            raise ValueError("endpoint_index exceeds the observed neural axis")
        return (values[..., self.endpoint_index] >= self.threshold).astype(int)

    def target_states(self, target: np.ndarray) -> np.ndarray:
        values = _finite(target)
        if values.ndim < 2:
            raise ValueError("target must have a leading nonempty time axis and final neural axis")
        if self.endpoint_index >= values.shape[-1]:
            raise ValueError("endpoint_index exceeds the observed neural axis")
        # Reproduce the old rollout's sequential additions, including boundary
        # rounding, rather than relying on NumPy's optional pairwise reduction.
        target_sum = np.zeros(values.shape[1:-1], dtype=float)
        for sample in values:
            target_sum += sample[..., self.endpoint_index]
        return (target_sum / len(values) >= self.threshold).astype(int)

    def to_dict(self) -> dict[str, Any]:
        return _fingerprinted({"schema_version": "frozen_binary_endpoint_v1", "type": "coordinate",
                               "endpoint_index": self.endpoint_index, "threshold": float(self.threshold)})


@dataclass(frozen=True)
class FrozenEndpoint:
    """A raw-unit affine encoder and an immutable binary region/readout map.

    ``code(x) = x @ encoder_weights.T + encoder_bias``. Future codes are averaged
    before applying the same frozen classifier. A coordinate-box envelope is an
    extrapolation diagnostic; it is not a statistical support guarantee.
    """

    name: str
    encoder_weights: np.ndarray
    encoder_bias: np.ndarray
    mode: str
    readout_weights: np.ndarray | None = None
    readout_intercept: float = 0.0
    centers: np.ndarray | None = None
    region_labels: np.ndarray | None = None
    code_lower: np.ndarray | None = None
    code_upper: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        weights = np.asarray(self.encoder_weights, dtype=float)
        if weights.ndim != 2 or min(weights.shape) < 1 or not np.isfinite(weights).all():
            raise ValueError("encoder_weights must be a nonempty finite code-by-neuron matrix")
        dims = weights.shape[0]
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("endpoint name must be nonempty")
        if self.mode not in {"linear", "regions", "constant"}:
            raise ValueError("unknown frozen endpoint mode")
        arrays = {"encoder_weights": weights,
                  "encoder_bias": np.asarray(self.encoder_bias, dtype=float)}
        if arrays["encoder_bias"].shape != (dims,):
            raise ValueError("encoder_bias must match the code axis")
        if (isinstance(self.readout_intercept, bool) or not isinstance(self.readout_intercept, Real)
                or not np.isfinite(self.readout_intercept)):
            raise ValueError("readout_intercept must be finite")
        if self.mode == "linear":
            arrays["readout_weights"] = np.asarray(self.readout_weights, dtype=float)
            if arrays["readout_weights"].shape != (dims,):
                raise ValueError("linear readout weights must match the code axis")
        if self.mode == "regions":
            arrays["centers"] = np.asarray(self.centers, dtype=float)
            labels = np.asarray(self.region_labels)
            if arrays["centers"].shape != (2, dims) or labels.shape != (2,) or set(labels) != {0, 1}:
                raise ValueError("regions require two code centers and a binary label permutation")
            arrays["region_labels"] = labels.astype(int)
        if (self.code_lower is None) != (self.code_upper is None):
            raise ValueError("both coordinate-envelope bounds must be supplied together")
        if self.code_lower is not None:
            arrays["code_lower"] = np.asarray(self.code_lower, dtype=float)
            arrays["code_upper"] = np.asarray(self.code_upper, dtype=float)
            if (arrays["code_lower"].shape != (dims,) or arrays["code_upper"].shape != (dims,)
                    or np.any(arrays["code_lower"] > arrays["code_upper"])):
                raise ValueError("coordinate-envelope bounds must be ordered and match code axes")
        for name, array in arrays.items():
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite")
            copied = array.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        # Separate the recorded artifact from caller-owned nested lists/dicts.
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False)))

    def codes(self, observations: np.ndarray) -> np.ndarray:
        values = _finite(observations, n_nodes=self.encoder_weights.shape[1])
        return values @ self.encoder_weights.T + self.encoder_bias

    def _states(self, codes: np.ndarray) -> np.ndarray:
        if self.mode == "constant":
            return np.zeros(codes.shape[:-1], dtype=int)
        if self.mode == "linear":
            assert self.readout_weights is not None
            return (codes @ self.readout_weights + self.readout_intercept >= 0).astype(int)
        assert self.centers is not None and self.region_labels is not None
        distances = np.sum((codes[..., None, :] - self.centers) ** 2, axis=-1)
        return self.region_labels[np.argmin(distances, axis=-1)]

    def source_states(self, initial: np.ndarray) -> np.ndarray:
        return self._states(self.codes(initial))

    def target_states(self, target: np.ndarray) -> np.ndarray:
        values = _finite(target, n_nodes=self.encoder_weights.shape[1])
        if values.ndim < 2:
            raise ValueError("target must have a leading nonempty time axis and final neural axis")
        return self._states(self.codes(values).mean(axis=0))

    def support_summary(self, observations: np.ndarray) -> dict[str, Any]:
        codes = self.codes(observations)
        if self.code_lower is None or self.code_upper is None:
            return {"diagnostic": "training_code_coordinate_box", "available": False}
        outside = np.any((codes < self.code_lower) | (codes > self.code_upper), axis=-1)
        return {"diagnostic": "training_code_coordinate_box", "available": True,
                "n_observations": int(outside.size), "outside_count": int(outside.sum()),
                "outside_fraction": float(outside.mean()),
                "support_guarantee": False, "observations_removed": 0}

    def reparameterize(self, rotation: np.ndarray) -> FrozenEndpoint:
        """Apply a consistent orthogonal coordinate change without changing events."""
        matrix = np.asarray(rotation, dtype=float)
        dims = self.encoder_weights.shape[0]
        if (matrix.shape != (dims, dims) or not np.isfinite(matrix).all()
                or not np.allclose(matrix.T @ matrix, np.eye(dims), atol=1e-10)):
            raise ValueError("coordinate change must be an orthogonal code-space matrix")
        lower, upper = None, None
        if self.code_lower is not None and self.code_upper is not None:
            positive, negative = np.maximum(matrix, 0), np.minimum(matrix, 0)
            lower = positive @ self.code_lower + negative @ self.code_upper
            upper = positive @ self.code_upper + negative @ self.code_lower
        readout = None if self.readout_weights is None else matrix @ self.readout_weights
        return replace(self, encoder_weights=matrix @ self.encoder_weights,
                       encoder_bias=matrix @ self.encoder_bias,
                       readout_weights=readout,
                       centers=None if self.centers is None else self.centers @ matrix.T,
                       code_lower=lower, code_upper=upper)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"schema_version": "frozen_binary_endpoint_v1", "type": "affine",
                                   "name": self.name, "mode": self.mode,
                                   "readout_intercept": float(self.readout_intercept),
                                   "metadata": json.loads(json.dumps(self.metadata))}
        for name in ("encoder_weights", "encoder_bias", "readout_weights", "centers", "region_labels",
                     "code_lower", "code_upper"):
            value = getattr(self, name)
            payload[name] = None if value is None else value.tolist()
        return _fingerprinted(payload)


def endpoint_from_dict(artifact: Mapping[str, Any]) -> Endpoint:
    """Restore explicit numerical parameters, rejecting an altered artifact."""
    payload = dict(artifact)
    digest = payload.pop("fingerprint", None)
    if digest != _fingerprinted(payload)["fingerprint"]:
        raise ValueError("endpoint artifact fingerprint does not match its contents")
    if payload.pop("schema_version", None) != "frozen_binary_endpoint_v1":
        raise ValueError("unknown endpoint artifact schema")
    kind = payload.pop("type", None)
    if kind == "coordinate":
        return CoordinateEndpoint(**payload)
    if kind == "affine":
        return FrozenEndpoint(**payload)
    raise ValueError("unknown endpoint artifact type")


def _window_means(trials: np.ndarray, length: int) -> np.ndarray:
    """Nonoverlapping complete windows per trial; trailing partials are excluded."""
    n_trials, trial_length, n_nodes = trials.shape
    n_windows = trial_length // length
    return trials[:, :n_windows * length].reshape(n_trials, n_windows, length, n_nodes).mean(axis=2)


def _decoder_rows(trials: np.ndarray, target_length: int) -> np.ndarray:
    # Instantaneous observations support factual source labels; complete window
    # means support future targets. Each row has equal weight (L:1 sample:window).
    return np.concatenate((trials.reshape(-1, trials.shape[-1]),
                           _window_means(trials, target_length).reshape(-1, trials.shape[-1])))


def _supervised_endpoint(
    name: str, weights: np.ndarray, bias: np.ndarray, inner: np.ndarray, validation: np.ndarray,
    training: np.ndarray, endpoint_index: int, target_length: int, decoder_cs: tuple[float, ...],
    seed: int, metadata: dict[str, Any],
) -> FrozenEndpoint:
    x_fit, x_validation = _decoder_rows(inner, target_length), _decoder_rows(validation, target_length)
    y_fit, y_validation = (x_fit[:, endpoint_index] >= 0).astype(int), (
        x_validation[:, endpoint_index] >= 0).astype(int)
    if len(np.unique(y_fit)) != 2:
        raise ValueError("inner training behavior must cover both reference states")
    codes_fit, codes_validation = x_fit @ weights.T + bias, x_validation @ weights.T + bias
    scores = []
    for penalty in decoder_cs:
        decoder = LogisticRegression(C=penalty, max_iter=1000, random_state=seed).fit(codes_fit, y_fit)
        scores.append(float(np.mean(decoder.predict(codes_validation) == y_validation)))
    selected = int(np.argmax(scores))
    all_rows = _decoder_rows(training, target_length)
    decoder = LogisticRegression(C=decoder_cs[selected], max_iter=1000, random_state=seed).fit(
        all_rows @ weights.T + bias, (all_rows[:, endpoint_index] >= 0).astype(int)
    )
    envelope = inner.reshape(-1, inner.shape[-1]) @ weights.T + bias
    return FrozenEndpoint(name, weights, bias, "linear", decoder.coef_[0], float(decoder.intercept_[0]),
                          code_lower=envelope.min(axis=0), code_upper=envelope.max(axis=0),
                          metadata={**metadata, "supervision": "behavior_labels_from_training_only",
                                    "decoder_selected_c": decoder_cs[selected],
                                    "decoder_candidate_cs": list(decoder_cs),
                                    "decoder_validation_accuracy_by_c": scores,
                                    "decoder_refit_scope": "all_supplied_outer_training_trials_fixed_encoder",
                                    "decoder_inner_rows": len(x_fit),
                                    "decoder_validation_rows": len(x_validation),
                                    "decoder_refit_rows": len(all_rows)})


def fit_endpoint_arms(
    training_trials: np.ndarray, *, endpoint_index: int, target_length: int, n_components: int = 2,
    validation_fraction: float = 0.25, decoder_cs: tuple[float, ...] = (0.1, 1.0, 10.0), seed: int = 42,
) -> dict[str, Endpoint]:
    """Fit charts using only supplied outer-training trials, never evaluation data.

    Scaler/PCA and geometric regions use inner-training trials only. Readout C is
    selected on separate whole validation trials, then only the decoder is refit
    on all supplied outer-training trials with the encoder unchanged. Decoder rows
    combine all instantaneous samples and nonoverlapping complete within-trial
    window means with equal row weights. Trailing partial windows are discarded.
    """
    trials = _finite(training_trials)
    if trials.ndim != 3 or trials.shape[0] < 2 or trials.shape[1] < 2 or trials.shape[2] < 3:
        raise ValueError("training_trials require at least two trials, two samples and three neurons")
    n_trials, trial_length, n_nodes = trials.shape
    if type(endpoint_index) is not int or not 0 <= endpoint_index < n_nodes:
        raise ValueError("endpoint_index must identify a declared reference coordinate")
    if type(target_length) is not int or not 1 <= target_length <= trial_length:
        raise ValueError("target_length must fit wholly inside training trials")
    if type(n_components) is not int or not 1 <= n_components <= n_nodes:
        raise ValueError("n_components must be a positive dimension within the neural axis")
    if (isinstance(validation_fraction, bool) or not isinstance(validation_fraction, Real)
            or not np.isfinite(validation_fraction) or not 0 < validation_fraction < 1):
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if (not isinstance(decoder_cs, (tuple, list)) or not decoder_cs
            or any(isinstance(c, bool) or not isinstance(c, Real) or not np.isfinite(c) or c <= 0
                   for c in decoder_cs) or len(set(decoder_cs)) != len(decoder_cs)):
        raise ValueError("decoder_cs must contain distinct positive finite regularization values")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    split = n_trials - max(1, int(np.ceil(n_trials * validation_fraction)))
    if split < 1 or n_components > min(split * trial_length, n_nodes):
        raise ValueError("insufficient inner-training samples or trials for the encoder")
    inner, validation = trials[:split], trials[split:]
    samples = inner.reshape(-1, n_nodes)
    scaler = StandardScaler().fit(samples)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed).fit(scaler.transform(samples))
    weights = pca.components_ / scaler.scale_
    bias = -(scaler.mean_ / scaler.scale_ + pca.mean_) @ pca.components_.T
    window_means = _window_means(inner, target_length).reshape(-1, n_nodes)
    codes = window_means @ weights.T + bias
    if np.unique(codes, axis=0).shape[0] < 2:
        raise ValueError("inner-training target codes must permit two geometric regions")
    regions = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(codes)
    labels = (window_means[:, endpoint_index] >= 0).astype(int)
    counts = np.zeros((2, 2), dtype=int)
    np.add.at(counts, (regions.labels_, labels), 1)
    rows, columns = linear_sum_assignment(-counts)
    permutation = np.empty(2, dtype=int)
    permutation[rows] = columns
    envelope = samples @ weights.T + bias
    metadata: dict[str, Any] = {
        "bridge": "linear_pointwise_PCA_not_nonlinear_manifold_validation",
        "fit_scope": "supplied_outer_training_trials_only",
        "encoder_scope": "inner_training_instantaneous_rows",
        "encoder_training_trial_ids": list(range(split)),
        "inner_validation_trial_ids": list(range(split, n_trials)),
        "encoder_sample_count": len(samples), "n_components": n_components,
        "target_length": target_length, "source_measurement": "instantaneous_factual_state",
        "target_measurement": "mean_pointwise_codes_then_frozen_classifier",
        "reference_definition": f"declared_neuron_{endpoint_index}_sign_zero",
        "training_input_sha256": hashlib.sha256(np.ascontiguousarray(trials).tobytes()).hexdigest(),
        "inner_encoder_input_sha256": hashlib.sha256(np.ascontiguousarray(inner).tobytes()).hexdigest(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "pca_center": pca.mean_.tolist(), "pca_components": pca.components_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "window_policy": "nonoverlapping_complete_within_trial_trailing_partial_excluded",
        "window_trial_boundary_crossings": 0,
        "geometry_training_window_count": len(window_means),
        "trailing_samples_excluded_per_trial": trial_length % target_length,
        "decoder_row_weighting": "equal_weight_instantaneous_samples_and_complete_window_means",
        "seed": seed,
    }
    geometry = FrozenEndpoint("pca_geometry", weights, bias, "regions", centers=regions.cluster_centers_,
                              region_labels=permutation, code_lower=envelope.min(axis=0),
                              code_upper=envelope.max(axis=0),
                              metadata={**metadata, "supervision": "training_region_name_permutation_only",
                                        "region_behavior_counts": counts.tolist()})
    full_weights = np.diag(1 / scaler.scale_)
    full_bias = -scaler.mean_ / scaler.scale_
    return {
        "behavior_reference": CoordinateEndpoint(endpoint_index), "pca_geometry": geometry,
        "pca_behavior": _supervised_endpoint(
            "pca_behavior", weights, bias, inner, validation, trials, endpoint_index, target_length,
            tuple(decoder_cs), seed, metadata),
        "full_neural_behavior": _supervised_endpoint(
            "full_neural_behavior", full_weights, full_bias, inner, validation, trials, endpoint_index,
            target_length, tuple(decoder_cs), seed,
            {**metadata, "encoder_scope": "inner_training_standardization_without_compression",
             "n_components": n_nodes}),
        "constant": FrozenEndpoint("constant", np.zeros((1, n_nodes)), np.zeros(1), "constant",
                                    metadata={"control": "collapsed_always_zero"}),
        "proxy": CoordinateEndpoint(0), "nuisance": CoordinateEndpoint(2),
    }
