"""Training-owned nonlinear mechanisms and paired outgoing-disconnection futures.

This benchmark readout estimates functions on supplied lag-one support; it does
not discover a graph. The additive tanh family is a declared favorable model,
not a claim of intervention identifiability in partially observed recordings.
See ``docs/driver_methodology_revision.md`` for the estimand and references.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive and finite")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return numeric


def _support(values: np.ndarray, n_nodes: int) -> np.ndarray:
    mask = np.asarray(values)
    if mask.dtype != np.bool_ or mask.shape != (n_nodes, n_nodes) or np.any(np.diag(mask)):
        raise ValueError("support must be a boolean square parent mask without diagonal edges")
    return mask.copy()


def _disconnected(values: Sequence[int], n_nodes: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("disconnected sources must be integer node indices")
    try:
        indices = tuple(values)
    except TypeError as error:
        raise ValueError("disconnected sources must be integer node indices") from error
    if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < n_nodes
           for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("disconnected sources must be unique integer node indices in range")
    return tuple(int(i) for i in indices)


@dataclass(frozen=True)
class MechanismFitConfig:
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    validation_fraction: float = 0.25
    noise_floor: float = 1e-6

    def __post_init__(self) -> None:
        if not isinstance(self.ridge_alphas, (tuple, list)) or not self.ridge_alphas:
            raise ValueError("ridge_alphas must contain positive candidate penalties")
        alphas = tuple(_positive(value, "ridge_alpha") for value in self.ridge_alphas)
        if len(set(alphas)) != len(alphas):
            raise ValueError("ridge_alphas must be unique")
        object.__setattr__(self, "ridge_alphas", alphas)
        if _positive(self.validation_fraction, "validation_fraction") >= 1:
            raise ValueError("validation_fraction must lie strictly between zero and one")
        _positive(self.noise_floor, "noise_floor")


@dataclass(frozen=True)
class AdditiveMechanism:
    """Raw-unit parameters; cross-weight rows are sources and columns are targets."""

    intercept: np.ndarray
    self_weights: np.ndarray
    cross_weights: np.ndarray
    noise_scale: np.ndarray
    support: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        intercept = np.asarray(self.intercept, dtype=float)
        if intercept.ndim != 1 or intercept.size == 0:
            raise ValueError("intercept must identify a nonempty node axis")
        n_nodes = len(intercept)
        mask = _support(self.support, n_nodes)
        arrays = {"intercept": intercept, "self_weights": np.asarray(self.self_weights, dtype=float),
                  "cross_weights": np.asarray(self.cross_weights, dtype=float),
                  "noise_scale": np.asarray(self.noise_scale, dtype=float)}
        for name, array in arrays.items():
            expected = (n_nodes, n_nodes) if name == "cross_weights" else (n_nodes,)
            if array.shape != expected or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {expected}")
        if np.any(arrays["noise_scale"] < 0):
            raise ValueError("noise_scale must be nonnegative")
        if np.any(arrays["cross_weights"][~mask] != 0):
            raise ValueError("cross_weights must be zero outside the declared support")
        for name, array in {**arrays, "support": mask}.items():
            frozen = array.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def mean(self, states: np.ndarray, disconnected: Sequence[int] = ()) -> np.ndarray:
        values = np.asarray(states, dtype=float)
        n_nodes = len(self.intercept)
        if values.ndim < 1 or values.shape[-1] != n_nodes or not values.size or not np.isfinite(values).all():
            raise ValueError("states must be nonempty, finite and end in the mechanism node axis")
        removed = _disconnected(disconnected, n_nodes)
        weights = self.cross_weights
        if removed:
            weights = weights.copy()
            weights[list(removed)] = 0.0
        try:
            with np.errstate(over="raise", invalid="raise"):
                result = self.intercept + values * self.self_weights + np.tanh(values) @ weights
        except FloatingPointError as error:
            raise ValueError("mechanism mean became nonfinite; no clipping is applied") from error
        if not np.isfinite(result).all():
            raise ValueError("mechanism mean became nonfinite; no clipping is applied")
        return result

    def step(self, states: np.ndarray, innovations: np.ndarray,
             disconnected: Sequence[int] = ()) -> np.ndarray:
        noise = np.asarray(innovations, dtype=float)
        mean = self.mean(states, disconnected)
        if noise.shape != mean.shape or not np.isfinite(noise).all():
            raise ValueError("standard-normal innovations must be finite and match states exactly")
        try:
            with np.errstate(over="raise", invalid="raise"):
                result = mean + noise * self.noise_scale
        except FloatingPointError as error:
            raise ValueError("mechanism step became nonfinite; no clipping is applied") from error
        if not np.isfinite(result).all():
            raise ValueError("mechanism step became nonfinite; no clipping is applied")
        return result


def _pairs(trials: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = trials.shape[-1]
    return trials[:, :-1].reshape(-1, n_nodes), trials[:, 1:].reshape(-1, n_nodes)


def _fit_equations(previous: np.ndarray, following: np.ndarray, support: np.ndarray,
                   alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = previous.shape[1]
    intercept, self_weights = np.empty(n_nodes), np.empty(n_nodes)
    cross_weights = np.zeros((n_nodes, n_nodes))
    for target in range(n_nodes):
        parents = np.flatnonzero(support[:, target])
        features = np.column_stack((previous[:, target], np.tanh(previous[:, parents])))
        scaler = StandardScaler().fit(features)
        fitted = Ridge(alpha=alpha).fit(scaler.transform(features), following[:, target])
        coefficients = fitted.coef_ / scaler.scale_
        intercept[target] = fitted.intercept_ - coefficients @ scaler.mean_
        self_weights[target] = coefficients[0]
        cross_weights[parents, target] = coefficients[1:]
    return intercept, self_weights, cross_weights


def fit_additive_mechanism(
    training_trials: np.ndarray,
    support: np.ndarray,
    config: MechanismFitConfig | None = None,
) -> AdditiveMechanism:
    """Select a penalty on inner whole trials, then refit on all supplied training trials.

    No outer held-out arrays are accepted. Independent zero-mean Gaussian noise
    scales use inner validation residual RMS, including any model error; they do
    not use simulator noise or optimistic residuals from the final fitted model.
    """
    if config is None:
        config = MechanismFitConfig()
    elif not isinstance(config, MechanismFitConfig):
        raise ValueError("config must be a MechanismFitConfig")
    trials = np.asarray(training_trials, dtype=float)
    if (trials.ndim != 3 or trials.shape[0] < 2 or trials.shape[1] < 2
            or trials.shape[2] < 1 or not np.isfinite(trials).all()):
        raise ValueError("training_trials must be finite with at least two trials and two samples")
    n_trials, length, n_nodes = trials.shape
    mask = _support(support, n_nodes)
    n_validation = max(1, int(np.ceil(config.validation_fraction * n_trials)))
    split = n_trials - n_validation
    if split < 1:
        raise ValueError("inner validation must leave at least one whole training trial")
    inner_x, inner_y = _pairs(trials[:split])
    validation_x, validation_y = _pairs(trials[split:])
    scores, residuals = [], []
    for alpha in config.ridge_alphas:
        intercept, self_weights, cross = _fit_equations(inner_x, inner_y, mask, alpha)
        residual = validation_y - (intercept + validation_x * self_weights + np.tanh(validation_x) @ cross)
        score = float(np.mean(residual**2))
        if not np.isfinite(score):
            raise ValueError("nonfinite inner validation error")
        scores.append(score)
        residuals.append(residual)
    selected = int(np.argmin(scores))
    scale = np.maximum(np.sqrt(np.mean(residuals[selected] ** 2, axis=0)), config.noise_floor)
    previous, following = _pairs(trials)
    intercept, self_weights, cross = _fit_equations(previous, following, mask, config.ridge_alphas[selected])
    return AdditiveMechanism(
        intercept, self_weights, cross, scale, mask,
        {"family": "additive_tanh_lag_one", "selected_alpha": config.ridge_alphas[selected],
         "candidate_alphas": list(config.ridge_alphas), "validation_mse_by_alpha": scores,
         "training_trial_count": n_trials, "training_pair_count": n_trials * (length - 1),
         "inner_training_trials": list(range(split)), "inner_validation_trials": list(range(split, n_trials)),
         "inner_training_pair_count": len(inner_x), "inner_validation_pair_count": len(validation_x),
         "fit_scope": "supplied_outer_training_trials_only", "trial_boundary_pairs": False,
         "noise_source": "inner_validation_residual_rms", "noise_law": "independent_zero_mean_gaussian",
         "noise_floor": config.noise_floor, "parameter_units": "original_observed_state_units"},
    )


@dataclass(frozen=True)
class MechanismRolloutConfig:
    horizon: int = 8
    target_length: int = 4
    n_rollouts: int = 256
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("horizon", "target_length", "n_rollouts", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a nonnegative seed or positive integer count")
        if self.target_length > self.horizon:
            raise ValueError("target_length must not exceed horizon")


def paired_intervention_probabilities(
    model: AdditiveMechanism,
    initial_states: np.ndarray,
    interventions: Mapping[str, Sequence[int]],
    *,
    endpoint_index: int,
    threshold: float = 0.0,
    config: MechanismRolloutConfig | None = None,
    origin_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate opposite-side future events under paired mechanism disconnections.

    Factual origins and their source sides are never changed by an intervention.
    Each origin's common random stream is independent of candidate ordering and
    batch order. Innovations are paired across branches, not independent draws.
    """
    if config is None:
        config = MechanismRolloutConfig()
    elif not isinstance(config, MechanismRolloutConfig):
        raise ValueError("config must be a MechanismRolloutConfig")
    initial = np.asarray(initial_states, dtype=float)
    if (initial.ndim != 2 or not initial.size or initial.shape[1] != len(model.intercept)
            or not np.isfinite(initial).all()):
        raise ValueError("initial_states must be finite nonempty origin-by-node observations")
    if (isinstance(endpoint_index, bool) or not isinstance(endpoint_index, (int, np.integer))
            or not 0 <= endpoint_index < initial.shape[1]):
        raise ValueError("endpoint_index must identify an observed node")
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    ids = np.arange(len(initial)) if origin_ids is None else np.asarray(origin_ids)
    if (ids.shape != (len(initial),) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or len(np.unique(ids)) != len(ids)):
        raise ValueError("origin_ids must be unique nonnegative integer origin identifiers")
    if not isinstance(interventions, Mapping) or any(
        not isinstance(name, str) or not name or name == "__factual__" for name in interventions
    ):
        raise ValueError("interventions need nonempty unit names; __factual__ is reserved")
    units = {name: _disconnected(members, initial.shape[1]) for name, members in interventions.items()}
    if any(not members for members in units.values()):
        raise ValueError("each intervention must disconnect at least one source")
    innovations = np.stack([
        np.random.default_rng(np.random.SeedSequence([config.seed, int(origin)])).normal(
            size=(config.horizon, config.n_rollouts, initial.shape[1])
        ) for origin in ids
    ], axis=1)
    sources = (initial[:, endpoint_index] >= threshold).astype(int)

    def probability(disconnected: Sequence[int]) -> np.ndarray:
        states = np.repeat(initial[:, None, :], config.n_rollouts, axis=1)
        target_sum = np.zeros((len(initial), config.n_rollouts))
        for step in range(config.horizon):
            states = model.step(states, innovations[step], disconnected)
            if step >= config.horizon - config.target_length:
                target_sum += states[:, :, endpoint_index]
        destinations = target_sum / config.target_length >= threshold
        return np.mean(destinations != sources[:, None], axis=1)

    factual = probability(())
    disconnected = {name: probability(members) for name, members in units.items()}
    return {
        "source_states": sources, "factual_transition_probability": factual,
        "disconnected_transition_probabilities": disconnected,
        "probability_reductions": {name: factual - values for name, values in disconnected.items()},
        "metadata": {"origin_ids": ids.tolist(), "endpoint_index": int(endpoint_index),
                     "threshold": float(threshold), "horizon": config.horizon,
                     "target_length": config.target_length, "n_rollouts": config.n_rollouts,
                     "seed": config.seed, "paired_innovations": True,
                     "noise_source": model.metadata.get("noise_source", "caller_supplied_model"),
                     "intervention": "disconnect_outgoing_cross_terms_retain_self_and_source_evolution",
                     "event": "future_target_mean_opposite_factual_origin_side"},
    }
