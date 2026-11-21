"""Batch existing standardized mechanism fits and factual behavior rollouts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from effectome.workflows.effectome_rollout import _validate_origins

from .continuous_behavior_rollouts import _OUTPUT_DIMENSION, ContinuousBehaviorReadout, _artifact, _predict
from .structural_mechanism import (
    AdditiveMechanism,
    MechanismRolloutConfig,
    _disconnected,
    _positive,
    _support,
)


def fit_timing_mechanisms(
    histories: np.ndarray, support: np.ndarray, alpha: float
) -> tuple[AdditiveMechanism, ...]:
    """Fit each observed window independently with the declared standardized ridge penalty.

    One scaler call treats each origin's features as distinct columns. Every
    Ridge system then uses only its own centered rows and the same fixed support.
    """
    observations = np.asarray(histories, dtype=float)
    if observations.ndim != 3 or not observations.size or observations.shape[1] < 3:
        raise ValueError("expected nonempty origin-by-time-by-node histories with at least three samples")
    if not np.isfinite(observations).all():
        raise ValueError("histories must be finite")
    n_origins, length, n_nodes = observations.shape
    mask = _support(support, n_nodes)
    penalty = _positive(alpha, "ridge_alpha")

    past, future = observations[:, :-1], observations[:, 1:]
    tanh_past = np.tanh(past)
    features = np.concatenate((past, tanh_past), axis=-1)
    feature_columns = features.transpose(1, 0, 2).reshape(length - 1, -1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_columns).reshape(length - 1, n_origins, -1).transpose(1, 0, 2)
    means = scaler.mean_.reshape(n_origins, -1)
    scales = scaler.scale_.reshape(n_origins, -1)
    future_mean = future.mean(axis=1)
    future_centered = future - future_mean[:, None, :]
    intercept = np.empty_like(future_mean)
    self_weights = np.empty_like(future_mean)
    cross = np.zeros((n_origins, n_nodes, n_nodes))

    for target in range(n_nodes):
        parents = np.flatnonzero(mask[:, target])
        columns = np.r_[target, n_nodes + parents]
        design = scaled[:, :, columns]
        design_mean = design.mean(axis=1)
        design -= design_mean[:, None, :]
        transposed = design.swapaxes(1, 2)
        gram = transposed @ design
        diagonal = np.arange(len(columns))
        gram[:, diagonal, diagonal] += penalty
        standardized = np.linalg.solve(gram, transposed @ future_centered[:, :, target, None])[:, :, 0]
        weights = standardized / scales[:, columns]
        intercept[:, target] = future_mean[:, target] - np.sum(
            standardized * design_mean + weights * means[:, columns], axis=1
        )
        self_weights[:, target] = weights[:, 0]
        cross[:, parents, target] = weights[:, 1:]

    residual = future - (intercept[:, None, :] + past * self_weights[:, None, :] + tanh_past @ cross)
    noise = np.maximum(np.sqrt(np.mean(residual**2, axis=1)), 1e-6)
    return tuple(
        AdditiveMechanism(
            bias,
            self_history,
            coupling,
            scale,
            mask,
            {"fit_pairs": length - 1, "ridge_alpha": penalty, "noise_source": "fitting_history_residual_rms"},
        )
        for bias, self_history, coupling, scale in zip(intercept, self_weights, cross, noise, strict=True)
    )


def timing_forecasts(
    mechanisms: Sequence[AdditiveMechanism],
    states: np.ndarray,
    origin_ids: np.ndarray,
    readout: ContinuousBehaviorReadout,
    config: MechanismRolloutConfig,
    *,
    output_width: int = _OUTPUT_DIMENSION,
    disconnected: Sequence[int] = (),
) -> np.ndarray:
    """Average decoded futures, optionally removing specified outgoing cross terms."""
    if not isinstance(config, MechanismRolloutConfig):
        raise ValueError("config must be a MechanismRolloutConfig")
    if type(output_width) is not int or output_width < 1:
        raise ValueError("output_width must be a positive integer")
    initial = np.asarray(states, dtype=float)
    if initial.ndim != 2 or not initial.size or len(mechanisms) != len(initial):
        raise ValueError("every origin needs one observed state and one fitted mechanism")
    if not np.isfinite(initial).all():
        raise ValueError("origin states must be finite")
    ids = np.asarray(origin_ids)
    n_origins, n_nodes = initial.shape
    removed = _disconnected(disconnected, n_nodes)
    _validate_origins(ids, n_origins)
    if any(model.intercept.shape != (n_nodes,) for model in mechanisms):
        raise ValueError("every fitted mechanism must match the observed node axis")

    artifact = _artifact(readout)
    innovations = np.stack(
        [
            np.random.default_rng(np.random.SeedSequence([config.seed, int(origin)])).normal(
                size=(config.horizon, config.n_rollouts, n_nodes)
            )
            for origin in ids
        ],
        axis=1,
    )
    intercept = np.stack([model.intercept for model in mechanisms])[:, None, :]
    self_weights = np.stack([model.self_weights for model in mechanisms])[:, None, :]
    cross = np.stack([model.cross_weights for model in mechanisms])
    if removed:
        cross[:, list(removed), :] = 0.0
    noise = np.stack([model.noise_scale for model in mechanisms])[:, None, :]
    simulated = np.repeat(initial[:, None, :], config.n_rollouts, axis=1)
    target_sum = np.zeros((n_origins, config.n_rollouts, output_width))
    try:
        with np.errstate(over="raise", invalid="raise"):
            for step, innovation in enumerate(innovations):
                simulated = intercept + simulated * self_weights + np.tanh(simulated) @ cross
                simulated += innovation * noise
                if step >= config.horizon - config.target_length:
                    simulated.setflags(write=False)
                    target_sum += _predict(readout, simulated, "behavior", output_width=output_width)
            forecast = (target_sum / config.target_length).mean(axis=1)
    except FloatingPointError as error:
        raise ValueError("mechanism forecast became nonfinite; no clipping is applied") from error
    if not np.isfinite(forecast).all():
        raise ValueError("mechanism forecast became nonfinite; no clipping is applied")
    if _artifact(readout) != artifact:
        raise ValueError("readout artifact changed during inference")
    return forecast
