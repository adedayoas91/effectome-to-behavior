"""Lag-design and normalized sparse VAR fitting helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LagDesign:
    design: np.ndarray
    response: np.ndarray
    response_indices: np.ndarray
    n_lags: int
    n_neurons: int
    segment_ids: np.ndarray | None = None


@dataclass(frozen=True)
class VARFit:
    coefficients: np.ndarray
    intercept: np.ndarray
    residual_covariance: np.ndarray
    fit_support: np.ndarray
    valid_response_indices: np.ndarray
    objective: float
    iterations: int
    converged: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    predictor_mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    predictor_scale: np.ndarray = field(default_factory=lambda: np.empty(0))
    response_mean: np.ndarray = field(default_factory=lambda: np.empty(0))
    response_scale: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def lagged_coefficients(self) -> np.ndarray:
        return self.coefficients.reshape(self.diagnostics["n_lags"], self.diagnostics["n_neurons"], -1)


def build_lag_design(
    samples: np.ndarray,
    *,
    n_lags: int,
    sample_indices: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
    masks: np.ndarray | None = None,
) -> LagDesign:
    """Construct valid one-step VAR rows without crossing sample or segment gaps."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("samples must have shape (time, neuron)")
    if n_lags < 1:
        raise ValueError("n_lags must be positive")
    n_time, n_neurons = x.shape
    indices = (
        np.arange(n_time, dtype=np.int64)
        if sample_indices is None
        else np.asarray(sample_indices, dtype=np.int64)
    )
    if indices.shape != (n_time,):
        raise ValueError("sample_indices must align with samples")
    segments = None if segment_ids is None else np.asarray(segment_ids)
    if segments is not None and segments.shape != (n_time,):
        raise ValueError("segment_ids must align with samples")
    valid_mask = np.ones_like(x, dtype=bool) if masks is None else np.asarray(masks, dtype=bool)
    if valid_mask.shape != x.shape:
        raise ValueError("masks must align with samples")

    rows: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    response_indices: list[int] = []
    response_segments: list[Any] = []
    for t in range(n_lags, n_time):
        span = slice(t - n_lags, t + 1)
        expected = indices[t] - np.arange(n_lags, -1, -1)
        if not np.array_equal(indices[span], expected):
            continue
        if segments is not None and np.any(segments[span] != segments[t]):
            continue
        block = x[span]
        block_mask = valid_mask[span]
        if not np.all(block_mask) or not np.all(np.isfinite(block)):
            continue
        rows.append(np.concatenate([x[t - lag] for lag in range(1, n_lags + 1)]))
        responses.append(x[t])
        response_indices.append(int(indices[t]))
        if segments is not None:
            response_segments.append(segments[t])

    design = np.asarray(rows, dtype=np.float64).reshape(-1, n_lags * n_neurons)
    response = np.asarray(responses, dtype=np.float64).reshape(-1, n_neurons)
    return LagDesign(
        design=design,
        response=response,
        response_indices=np.asarray(response_indices, dtype=np.int64),
        n_lags=n_lags,
        n_neurons=n_neurons,
        segment_ids=np.asarray(response_segments) if segments is not None else None,
    )


def _scales(values: np.ndarray, *, standardize: bool) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    if not standardize:
        return mean, np.ones(values.shape[1], dtype=np.float64)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def _objective(
    design: np.ndarray,
    response: np.ndarray,
    coefficients: np.ndarray,
    *,
    lambda1: float,
    lambda2: float,
    l1_mask: np.ndarray,
) -> float:
    residual = response - design @ coefficients
    loss = float(np.sum(residual * residual) / max(1, design.shape[0]))
    return loss + lambda1 * float(np.sum(np.abs(coefficients[l1_mask]))) + lambda2 * float(
        np.sum(coefficients * coefficients)
    )


def fit_sparse_var(
    lag_design: LagDesign,
    *,
    lambda1: float = 0.0,
    lambda2: float = 1e-3,
    standardize: bool = True,
    max_iter: int = 10_000,
    tol: float = 1e-6,
) -> VARFit:
    """Fit normalized sparse VAR loss with off-diagonal L1 and unpenalized intercept."""
    if lag_design.design.shape[0] == 0:
        raise ValueError("no valid lag-design rows are available")
    if lambda1 < 0 or lambda2 < 0:
        raise ValueError("regularization strengths must be nonnegative")
    if max_iter < 0:
        raise ValueError("max_iter must be nonnegative")

    raw_x = lag_design.design
    raw_y = lag_design.response
    x_mean, x_scale = _scales(raw_x, standardize=standardize)
    y_mean, y_scale = _scales(raw_y, standardize=standardize)
    x = (raw_x - x_mean) / x_scale
    y = (raw_y - y_mean) / y_scale
    m = max(1, x.shape[0])
    p, n = lag_design.n_lags, lag_design.n_neurons

    sources = np.tile(np.arange(n), p)
    targets = np.arange(n)
    l1_mask = sources[:, None] != targets[None, :]
    gram = x.T @ x + (m * lambda2) * np.eye(x.shape[1], dtype=np.float64)
    right_hand_side = x.T @ y
    if lambda1 == 0.0:
        coef_std = np.linalg.lstsq(gram, right_hand_side, rcond=None)[0]
        iterations = 1
        converged = True
    else:
        coef_std = np.linalg.lstsq(gram, right_hand_side, rcond=None)[0]
        spectral = float(np.linalg.norm(x, ord=2) ** 2)
        lipschitz = 2.0 * spectral / m + 2.0 * lambda2
        step = 1.0 / max(lipschitz, 1e-12)
        converged = False
        iterations = 0
        delta = float("inf")
        for iteration in range(1, max_iter + 1):
            gradient = (2.0 / m) * x.T @ (x @ coef_std - y) + 2.0 * lambda2 * coef_std
            proposed = coef_std - step * gradient
            proposed[l1_mask] = np.sign(proposed[l1_mask]) * np.maximum(
                np.abs(proposed[l1_mask]) - step * lambda1,
                0.0,
            )
            delta = float(np.max(np.abs(proposed - coef_std)))
            coef_std = proposed
            iterations = iteration
            if delta <= tol:
                converged = True
                break

    coefficients = (coef_std * y_scale[None, :]) / x_scale[:, None]
    intercept = y_mean - x_mean @ coefficients
    residual = raw_y - intercept[None, :] - raw_x @ coefficients
    residual_covariance = residual.T @ residual / max(1, raw_y.shape[0])
    objective = _objective(x, y, coef_std, lambda1=lambda1, lambda2=lambda2, l1_mask=l1_mask)
    fit_support = np.isfinite(raw_x).all(axis=1) & np.isfinite(raw_y).all(axis=1)
    return VARFit(
        coefficients=coefficients,
        intercept=intercept,
        residual_covariance=residual_covariance,
        fit_support=fit_support,
        valid_response_indices=lag_design.response_indices,
        objective=objective,
        iterations=iterations,
        converged=converged,
        diagnostics={
            "n_lags": p,
            "n_neurons": n,
            "lambda1": float(lambda1),
            "lambda2": float(lambda2),
            "standardize": bool(standardize),
            "loss_normalization": "mean_squared_error_by_valid_row",
            "l1_penalty": "off_diagonal_lag_coefficients_only",
            "intercept_penalty": "none",
            "status": "converged" if converged else "nonconverged",
            "proximal_residual": 0.0 if lambda1 == 0.0 else delta,
        },
        predictor_mean=x_mean,
        predictor_scale=x_scale,
        response_mean=y_mean,
        response_scale=y_scale,
    )
