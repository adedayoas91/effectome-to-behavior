"""One-sided smooth and group-fused time-varying VAR estimators."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

from effectome.data_module.schema import TemporalAnchor, WindowedSegments

from .base import ConnectivityConfig, ConnectivityEstimator
from .predictive import (
    FilterState,
    GraphEstimate,
    NeuralForecast,
    PredictiveCapabilities,
    UnsupportedForecastError,
)
from .registry import register_connectivity
from .sparse_var import _aggregate_lags
from .var_utils import LagDesign, build_lag_design

TemporalPenalty = Literal["smooth", "group_fused"]


@dataclass(frozen=True)
class TVVARFit:
    coefficients: np.ndarray
    intercept: np.ndarray
    residual_covariance: np.ndarray
    objective_components: dict[str, float]
    iterations: int
    converged: bool
    proximal_residual: float

    @property
    def lagged_coefficients(self) -> np.ndarray:
        return self.coefficients.reshape(
            int(self.objective_components["n_lags"]),
            int(self.objective_components["n_neurons"]),
            -1,
        )


@dataclass(frozen=True)
class TVVARBufferFit:
    knot_coefficients: np.ndarray
    knot_intercepts: np.ndarray
    row_knots: np.ndarray
    objective_components: dict[str, float]
    iterations: int
    converged: bool
    proximal_residual: float


def fit_tvvar_buffer(
    lag_design: LagDesign,
    *,
    knot_samples: int,
    lambda1: float = 0.0,
    lambda2: float = 1e-3,
    lambda_temporal: float = 0.0,
    lambda_intercept: float | None = None,
    penalty: TemporalPenalty = "smooth",
    max_iter: int = 2_000,
    tol: float = 1e-6,
) -> TVVARBufferFit:
    """Jointly fit all one-sided knots with one row-normalized objective."""
    if lag_design.design.shape[0] == 0:
        raise ValueError("no valid lag-design rows are available")
    if knot_samples < 1:
        raise ValueError("knot_samples must be positive")
    if penalty != "smooth":
        raise ValueError("joint buffer fitting currently supports the smooth penalty")
    if min(lambda1, lambda2, lambda_temporal) < 0:
        raise ValueError("regularization strengths must be nonnegative")
    if lambda_intercept is None:
        lambda_intercept = lambda_temporal
    if lambda_intercept < 0:
        raise ValueError("lambda_intercept must be nonnegative")

    design = np.asarray(lag_design.design, dtype=np.float64)
    response = np.asarray(lag_design.response, dtype=np.float64)
    n_rows, n_predictors = design.shape
    n_targets = response.shape[1]
    row_knots = np.arange(n_rows, dtype=int) // knot_samples
    n_knots = int(row_knots[-1]) + 1
    initial_coefficients = np.empty((n_knots, n_predictors, n_targets), dtype=np.float64)
    initial_intercepts = np.empty((n_knots, n_targets), dtype=np.float64)
    for knot in range(n_knots):
        selected = row_knots == knot
        x = design[selected]
        y = response[selected]
        augmented = np.column_stack([x, np.ones(x.shape[0])])
        fitted = np.linalg.lstsq(augmented, y, rcond=None)[0]
        initial_coefficients[knot] = fitted[:-1]
        initial_intercepts[knot] = fitted[-1]
    initial = np.concatenate([initial_coefficients.ravel(), initial_intercepts.ravel()])
    offdiag = _offdiag_mask(lag_design.n_lags, lag_design.n_neurons)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        split = n_knots * n_predictors * n_targets
        coefficients = parameters[:split].reshape(n_knots, n_predictors, n_targets)
        intercepts = parameters[split:].reshape(n_knots, n_targets)
        return coefficients, intercepts

    def objective_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients, intercepts = unpack(parameters)
        predictions = np.einsum(
            "md,mdn->mn", design, coefficients[row_knots]
        ) + intercepts[row_knots]
        residual = predictions - response
        objective = float(np.sum(residual * residual) / n_rows)
        gradient_coefficients = np.zeros_like(coefficients)
        gradient_intercepts = np.zeros_like(intercepts)
        for knot in range(n_knots):
            selected = row_knots == knot
            gradient_coefficients[knot] = (
                2.0 * design[selected].T @ residual[selected] / n_rows
            )
            gradient_intercepts[knot] = 2.0 * residual[selected].sum(axis=0) / n_rows
        objective += 0.5 * lambda2 * float(np.sum(coefficients * coefficients))
        gradient_coefficients += lambda2 * coefficients
        if lambda1:
            objective += lambda1 * float(np.sum(np.abs(coefficients[:, offdiag])))
            gradient_coefficients[:, offdiag] += lambda1 * np.sign(coefficients[:, offdiag])
        if n_knots > 1:
            coefficient_delta = coefficients[1:] - coefficients[:-1]
            intercept_delta = intercepts[1:] - intercepts[:-1]
            objective += lambda_temporal * float(np.sum(coefficient_delta**2))
            objective += lambda_intercept * float(np.sum(intercept_delta**2))
            gradient_coefficients[:-1] -= 2.0 * lambda_temporal * coefficient_delta
            gradient_coefficients[1:] += 2.0 * lambda_temporal * coefficient_delta
            gradient_intercepts[:-1] -= 2.0 * lambda_intercept * intercept_delta
            gradient_intercepts[1:] += 2.0 * lambda_intercept * intercept_delta
        gradient = np.concatenate(
            [gradient_coefficients.ravel(), gradient_intercepts.ravel()]
        )
        return objective, gradient

    result = minimize(
        objective_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "gtol": tol, "ftol": 1e-15, "maxls": 50},
    )
    coefficients, intercepts = unpack(result.x)
    objective, gradient = objective_gradient(result.x)
    residual = float(np.max(np.abs(gradient)))
    return TVVARBufferFit(
        knot_coefficients=coefficients,
        knot_intercepts=intercepts,
        row_knots=row_knots,
        objective_components={
            "objective": objective,
            "n_valid_rows": float(n_rows),
            "n_knots": float(n_knots),
        },
        iterations=int(result.nit),
        converged=bool(result.success or residual <= max(tol, 1e-6)),
        proximal_residual=residual,
    )


def _anchor_key(anchor: TemporalAnchor) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        anchor.dataset_id,
        anchor.recording_id,
        anchor.animal_id,
        anchor.session_id,
        anchor.segment_id,
    )


def _is_boundary(segments: WindowedSegments, index: int) -> bool:
    if index <= 0 or not segments.anchors:
        return True
    previous = segments.anchors[index - 1]
    current = segments.anchors[index]
    return (
        _anchor_key(previous) != _anchor_key(current)
        or previous.gap_after
        or current.gap_before
    )


def _offdiag_mask(n_lags: int, n_neurons: int) -> np.ndarray:
    sources = np.tile(np.arange(n_neurons), n_lags)
    targets = np.arange(n_neurons)
    return sources[:, None] != targets[None, :]


def _group_shrink(delta: np.ndarray, threshold: float) -> np.ndarray:
    norm = float(np.linalg.norm(delta))
    if norm <= threshold:
        return np.zeros_like(delta)
    return delta * (1.0 - threshold / norm)


def _objective_components(
    design: np.ndarray,
    response: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
    *,
    previous_coefficients: np.ndarray | None,
    previous_intercept: np.ndarray | None,
    lambda1: float,
    lambda2: float,
    lambda_temporal: float,
    lambda_intercept: float,
    penalty: TemporalPenalty,
    l1_mask: np.ndarray,
    n_lags: int,
    n_neurons: int,
) -> dict[str, float]:
    m = max(1, design.shape[0])
    residual = response - intercept[None, :] - design @ coefficients
    loss = float(np.sum(residual * residual) / m)
    l1 = float(lambda1 * np.sum(np.abs(coefficients[l1_mask])))
    ridge = float(lambda2 * np.sum(coefficients * coefficients))
    temporal = 0.0
    intercept_temporal = 0.0
    if previous_coefficients is not None:
        diff = coefficients - previous_coefficients
        if penalty == "smooth":
            temporal = float(lambda_temporal * np.sum(diff * diff))
        else:
            temporal = float(lambda_temporal * np.linalg.norm(diff))
    if previous_intercept is not None:
        idiff = intercept - previous_intercept
        intercept_temporal = float(lambda_intercept * np.sum(idiff * idiff))
    return {
        "loss": loss,
        "offdiag_l1": l1,
        "ridge": ridge,
        "temporal": temporal,
        "intercept_temporal": intercept_temporal,
        "objective": loss + l1 + ridge + temporal + intercept_temporal,
        "n_lags": float(n_lags),
        "n_neurons": float(n_neurons),
        "n_valid_rows": float(design.shape[0]),
    }


def fit_tvvar_endpoint(
    lag_design: LagDesign,
    *,
    previous_coefficients: np.ndarray | None = None,
    previous_intercept: np.ndarray | None = None,
    lambda1: float = 0.0,
    lambda2: float = 1e-3,
    lambda_temporal: float = 0.0,
    lambda_intercept: float | None = None,
    penalty: TemporalPenalty = "smooth",
    max_iter: int = 2_000,
    tol: float = 1e-6,
    standardize: bool = False,
) -> TVVARFit:
    """Fit one endpoint TVVAR knot using only the supplied lag-design rows."""
    if lag_design.design.shape[0] == 0:
        raise ValueError("no valid lag-design rows are available")
    if penalty not in {"smooth", "group_fused"}:
        raise ValueError("penalty must be 'smooth' or 'group_fused'")
    if min(lambda1, lambda2, lambda_temporal) < 0:
        raise ValueError("regularization strengths must be nonnegative")
    if lambda_intercept is None:
        lambda_intercept = lambda_temporal
    if lambda_intercept < 0:
        raise ValueError("lambda_intercept must be nonnegative")

    raw_x = np.asarray(lag_design.design, dtype=np.float64)
    raw_y = np.asarray(lag_design.response, dtype=np.float64)
    if standardize:
        x_mean = raw_x.mean(axis=0)
        x_scale = np.where(raw_x.std(axis=0) > 1e-12, raw_x.std(axis=0), 1.0)
        y_mean = raw_y.mean(axis=0)
        y_scale = np.where(raw_y.std(axis=0) > 1e-12, raw_y.std(axis=0), 1.0)
        x = (raw_x - x_mean) / x_scale
        y = (raw_y - y_mean) / y_scale
    else:
        x_mean = np.zeros(raw_x.shape[1], dtype=np.float64)
        x_scale = np.ones(raw_x.shape[1], dtype=np.float64)
        y_mean = np.zeros(raw_y.shape[1], dtype=np.float64)
        y_scale = np.ones(raw_y.shape[1], dtype=np.float64)
        x = raw_x
        y = raw_y

    m = max(1, x.shape[0])
    n_lags, n_neurons = lag_design.n_lags, lag_design.n_neurons
    l1_mask = _offdiag_mask(n_lags, n_neurons)
    x_center = x.mean(axis=0)
    y_center = y.mean(axis=0)
    centered_x = x - x_center
    centered_y = y - y_center
    previous_std = None
    previous_intercept_std = None
    if previous_coefficients is not None:
        previous_std = (previous_coefficients * x_scale[:, None]) / y_scale[None, :]
    if previous_intercept is not None:
        previous_intercept_std = (previous_intercept - y_mean) / y_scale

    system = centered_x.T @ centered_x + m * lambda2 * np.eye(x.shape[1], dtype=np.float64)
    rhs = centered_x.T @ centered_y
    if previous_std is not None and penalty == "smooth" and lambda_temporal > 0:
        system = system + m * lambda_temporal * np.eye(x.shape[1], dtype=np.float64)
        rhs = rhs + m * lambda_temporal * previous_std
    coef = np.linalg.solve(system, rhs)
    intercept = y_center - x_center @ coef
    if previous_intercept_std is not None and lambda_intercept > 0:
        weight = m * lambda_intercept
        intercept = (m * intercept + weight * previous_intercept_std) / (m + weight)

    spectral = float(np.linalg.norm(x, ord=2) ** 2)
    temporal_lipschitz = 2.0 * lambda_temporal if penalty == "smooth" else 0.0
    lipschitz = 2.0 * spectral / m + 2.0 * lambda2 + temporal_lipschitz
    step = 1.0 / max(lipschitz, 1e-12)
    converged = lambda1 == 0.0 and not (penalty == "group_fused" and lambda_temporal > 0)
    residual = 0.0
    iterations = 1 if converged else 0
    iteration_range = range(1, max_iter + 1) if not converged else range(0)
    for iteration in iteration_range:
        gradient = (2.0 / m) * centered_x.T @ (centered_x @ coef - centered_y)
        gradient += 2.0 * lambda2 * coef
        if previous_std is not None and penalty == "smooth":
            gradient += 2.0 * lambda_temporal * (coef - previous_std)
        proposed = coef - step * gradient
        if previous_std is not None and penalty == "group_fused" and lambda_temporal > 0:
            proposed = previous_std + _group_shrink(proposed - previous_std, step * lambda_temporal)
        if lambda1 > 0:
            proposed[l1_mask] = np.sign(proposed[l1_mask]) * np.maximum(
                np.abs(proposed[l1_mask]) - step * lambda1,
                0.0,
            )
        next_intercept = y_center - x_center @ proposed
        if previous_intercept_std is not None and lambda_intercept > 0:
            weight = m * lambda_intercept
            next_intercept = (m * next_intercept + weight * previous_intercept_std) / (m + weight)
        residual = max(
            float(np.max(np.abs(proposed - coef))),
            float(np.max(np.abs(next_intercept - intercept))),
        )
        coef = proposed
        intercept = next_intercept
        iterations = iteration
        if residual <= tol:
            converged = True
            break

    coefficients = (coef * y_scale[None, :]) / x_scale[:, None]
    raw_intercept = y_mean + y_scale * intercept - x_mean @ coefficients
    previous_raw_coef = previous_coefficients
    previous_raw_intercept = previous_intercept
    components = _objective_components(
        raw_x,
        raw_y,
        coefficients,
        raw_intercept,
        previous_coefficients=previous_raw_coef,
        previous_intercept=previous_raw_intercept,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda_temporal=lambda_temporal,
        lambda_intercept=lambda_intercept,
        penalty=penalty,
        l1_mask=l1_mask,
        n_lags=n_lags,
        n_neurons=n_neurons,
    )
    raw_residual = raw_y - raw_intercept[None, :] - raw_x @ coefficients
    covariance = raw_residual.T @ raw_residual / max(1, raw_y.shape[0])
    return TVVARFit(
        coefficients=coefficients,
        intercept=raw_intercept,
        residual_covariance=covariance,
        objective_components=components,
        iterations=iterations,
        converged=converged,
        proximal_residual=residual,
    )


def _options(cfg: ConnectivityConfig, penalty: TemporalPenalty) -> dict[str, Any]:
    extra = dict(cfg.extra)
    return {
        "n_lags": int(extra.get("n_lags", cfg.max_lag)),
        "lambda1": float(extra.get("lambda1", 0.0)),
        "lambda2": float(extra.get("lambda2", cfg.ridge)),
        "lambda_temporal": float(extra.get("lambda_temporal", extra.get("lambda_T", 1e-2))),
        "lambda_intercept": float(
            extra.get("lambda_intercept", extra.get("lambda_b", extra.get("lambda_temporal", 1e-2)))
        ),
        "max_iter": int(extra.get("max_iter", 2_000)),
        "tol": float(extra.get("tol", 1e-6)),
        "lag_aggregation": str(extra.get("lag_aggregation", "sum")),
        "standardize": bool(extra.get("standardize", False)),
        "penalty": penalty,
        "knot_samples": int(extra.get("knot_samples", 32)),
    }


@dataclass(frozen=True)
class TVVARModel:
    fit: TVVARFit
    options: dict[str, Any]
    capabilities: PredictiveCapabilities = PredictiveCapabilities(
        stateful_online=True, neural_forecast=True
    )

    @property
    def n_lags(self) -> int:
        return int(self.options["n_lags"])

    @property
    def n_neurons(self) -> int:
        return int(self.fit.objective_components["n_neurons"])

    @property
    def model_hash(self) -> str:
        digest = blake2b(digest_size=16)
        for value in (self.fit.coefficients, self.fit.intercept):
            digest.update(np.ascontiguousarray(value).view(np.uint8).tobytes())
        digest.update(repr(sorted(self.options.items())).encode())
        return digest.hexdigest()

    def initial_state(self, *, recording_identity: Any = None) -> FilterState:
        return FilterState(
            model_hash=self.model_hash,
            history=np.empty((0, self.n_neurons), dtype=np.float64),
            recording_identity=recording_identity,
            diagnostics={
                "coefficients": self.fit.coefficients.copy(),
                "intercept": self.fit.intercept.copy(),
                "pending_samples": np.empty((0, self.n_neurons), dtype=np.float64),
                "updates": 0,
            },
        )

    def _validate_state(self, state: FilterState) -> None:
        if state.model_hash != self.model_hash:
            raise ValueError("filter state belongs to a different fitted model")

    def observe(
        self,
        state: FilterState,
        samples: np.ndarray,
        *,
        sample_indices: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        available_inputs: np.ndarray | None = None,
    ) -> FilterState:
        self._validate_state(state)
        if available_inputs is not None:
            raise UnsupportedForecastError("TVVAR does not support exogenous inputs")
        observed = np.asarray(samples, dtype=np.float64)
        if observed.ndim != 2 or observed.shape[1] != self.n_neurons:
            raise ValueError("samples must have shape (time, n_neurons)")
        indices = None if sample_indices is None else np.asarray(sample_indices)
        if indices is not None:
            if indices.shape != (observed.shape[0],) or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("sample_indices must be aligned integers")
            indices = indices.astype(np.int64, copy=False)
            if np.any(np.diff(indices) <= 0):
                raise ValueError("sample_indices must be strictly increasing")
            if indices.size and state.sample_index is not None and indices[0] < state.sample_index:
                raise ValueError("sample_indices replay previously observed samples")
        valid = np.isfinite(observed).all(axis=1)
        if masks is not None:
            mask = np.asarray(masks, dtype=bool)
            if mask.shape != observed.shape:
                raise ValueError("masks must align with samples")
            valid &= mask.all(axis=1)

        history = np.asarray(state.history, dtype=np.float64).copy()
        coefficients = np.asarray(state.diagnostics["coefficients"], dtype=np.float64).copy()
        intercept = np.asarray(state.diagnostics["intercept"], dtype=np.float64).copy()
        pending = np.asarray(state.diagnostics["pending_samples"], dtype=np.float64).copy()
        updates = int(state.diagnostics.get("updates", 0))
        previous_index = None if state.sample_index is None else state.sample_index - 1
        needed = self.n_lags + int(self.options["knot_samples"])
        for row_number, row in enumerate(observed):
            index = row_number if indices is None else int(indices[row_number])
            gap = indices is not None and previous_index is not None and index != previous_index + 1
            if gap or not valid[row_number]:
                history = np.empty((0, self.n_neurons), dtype=np.float64)
                pending = np.empty((0, self.n_neurons), dtype=np.float64)
            if valid[row_number]:
                history = np.vstack([history, row])[-self.n_lags :]
                pending = np.vstack([pending, row])
                if pending.shape[0] == needed:
                    design = build_lag_design(pending, n_lags=self.n_lags)
                    updated = fit_tvvar_endpoint(
                        design,
                        previous_coefficients=coefficients,
                        previous_intercept=intercept,
                        lambda1=self.options["lambda1"],
                        lambda2=self.options["lambda2"],
                        lambda_temporal=self.options["lambda_temporal"],
                        lambda_intercept=self.options["lambda_intercept"],
                        penalty=self.options["penalty"],
                        max_iter=self.options["max_iter"],
                        tol=self.options["tol"],
                        standardize=self.options["standardize"],
                    )
                    if updated.converged:
                        coefficients = updated.coefficients
                        intercept = updated.intercept
                    pending = pending[-self.n_lags :]
                    updates += 1
            previous_index = index if indices is not None else None
        origin = (
            state.sample_index
            if observed.shape[0] == 0
            else (None if indices is None else int(indices[-1]) + 1)
        )
        return FilterState(
            model_hash=self.model_hash,
            history=history,
            sample_index=origin,
            recording_identity=state.recording_identity,
            diagnostics={
                "coefficients": coefficients,
                "intercept": intercept,
                "pending_samples": pending,
                "updates": updates,
                "observed_rows": int(valid.sum()),
            },
        )

    def forecast(
        self,
        state: FilterState,
        *,
        horizon_samples: int,
        known_future_inputs: np.ndarray | None = None,
    ) -> NeuralForecast:
        self._validate_state(state)
        if known_future_inputs is not None:
            raise UnsupportedForecastError("TVVAR does not support exogenous inputs")
        if horizon_samples < 1:
            raise ValueError("horizon_samples must be positive")
        if state.history.shape[0] < self.n_lags:
            return NeuralForecast(
                np.empty((0, self.n_neurons)), horizon_samples, state.sample_index,
                failure_reason="insufficient_history", model_hash=self.model_hash
            )
        coefficients = np.asarray(state.diagnostics["coefficients"], dtype=np.float64)
        intercept = np.asarray(state.diagnostics["intercept"], dtype=np.float64)
        history = state.history.copy()
        means: list[np.ndarray] = []
        for _ in range(horizon_samples):
            design_row = np.concatenate(
                [history[-lag] for lag in range(1, self.n_lags + 1)]
            )
            prediction = intercept + design_row @ coefficients
            means.append(prediction)
            history = np.vstack([history, prediction])[-self.n_lags :]
        return NeuralForecast(
            np.asarray(means), horizon_samples, state.sample_index,
            model_hash=self.model_hash,
            raw_support={"recording_identity": state.recording_identity},
        )

    def graph_at(self, state: FilterState | None = None) -> GraphEstimate:
        coefficients = self.fit.coefficients
        if state is not None:
            self._validate_state(state)
            coefficients = np.asarray(state.diagnostics["coefficients"], dtype=np.float64)
        lagged = coefficients.reshape(self.n_lags, self.n_neurons, self.n_neurons)
        return GraphEstimate(
            _aggregate_lags(lagged, self.options["lag_aggregation"]),
            lagged,
            "signed_tvvar_endpoint_coefficient_source_to_target",
            self.model_hash,
        )


def fit_tvvar_model(samples: np.ndarray, *, cfg: ConnectivityConfig) -> TVVARModel:
    penalty: TemporalPenalty = "group_fused" if cfg.name == "tvvar_fused" else "smooth"
    options = _options(cfg, penalty)
    design = build_lag_design(np.asarray(samples, dtype=np.float64), n_lags=options["n_lags"])
    fit = fit_tvvar_endpoint(
        design,
        lambda1=options["lambda1"],
        lambda2=options["lambda2"],
        lambda_temporal=0.0,
        lambda_intercept=0.0,
        penalty=penalty,
        max_iter=options["max_iter"],
        tol=options["tol"],
        standardize=options["standardize"],
    )
    return TVVARModel(fit=fit, options=options)


class _TVVARConnectivity(ConnectivityEstimator):
    directed = True
    weighted = True
    weight_semantics = "signed_tvvar_endpoint_coefficient"
    temporal_penalty: TemporalPenalty = "smooth"

    def __init__(self, cfg: ConnectivityConfig) -> None:
        super().__init__(cfg)
        self._window_lagged: list[np.ndarray] = []
        self._diagnostic_update: dict[str, Any] = {}

    @property
    def capabilities(self) -> PredictiveCapabilities:
        return PredictiveCapabilities(
            neural_forecast=True,
            stateful_online=True,
            independent_window=False,
        )

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        opts = _options(self.cfg, self.temporal_penalty)
        design = build_lag_design(segment, n_lags=opts["n_lags"])
        fit = fit_tvvar_endpoint(
            design,
            lambda1=opts["lambda1"],
            lambda2=opts["lambda2"],
            lambda_temporal=0.0,
            lambda_intercept=0.0,
            penalty=self.temporal_penalty,
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            standardize=opts["standardize"],
        )
        lagged = fit.lagged_coefficients
        self._window_lagged = [lagged]
        return _aggregate_lags(lagged, opts["lag_aggregation"])

    def estimate_sequence(self, segments: WindowedSegments) -> np.ndarray:
        opts = _options(self.cfg, self.temporal_penalty)
        self._window_lagged = []
        matrices: list[np.ndarray] = []
        components: list[dict[str, float]] = []
        statuses: list[str] = []
        previous_coefficients: np.ndarray | None = None
        previous_intercept: np.ndarray | None = None
        boundaries = 0
        for index, segment in enumerate(segments.segments):
            if _is_boundary(segments, index):
                previous_coefficients = None
                previous_intercept = None
                boundaries += 1
            design = build_lag_design(np.asarray(segment, dtype=np.float64), n_lags=opts["n_lags"])
            fit = fit_tvvar_endpoint(
                design,
                previous_coefficients=previous_coefficients,
                previous_intercept=previous_intercept,
                lambda1=opts["lambda1"],
                lambda2=opts["lambda2"],
                lambda_temporal=0.0 if previous_coefficients is None else opts["lambda_temporal"],
                lambda_intercept=0.0 if previous_intercept is None else opts["lambda_intercept"],
                penalty=self.temporal_penalty,
                max_iter=opts["max_iter"],
                tol=opts["tol"],
                standardize=opts["standardize"],
            )
            lagged = fit.lagged_coefficients
            self._window_lagged.append(lagged)
            matrices.append(_aggregate_lags(lagged, opts["lag_aggregation"]))
            components.append(fit.objective_components | {"proximal_residual": fit.proximal_residual})
            statuses.append("converged" if fit.converged else "nonconverged")
            previous_coefficients = fit.coefficients
            previous_intercept = fit.intercept
        self._diagnostic_update = {
            "adaptation_policy": "one_sided_endpoint_tvvar",
            "coefficient_knot_policy": "one_endpoint_knot_per_window",
            "piecewise_constant_coefficients": True,
            "temporal_penalty": self.temporal_penalty,
            "lambda_temporal": opts["lambda_temporal"],
            "lambda_intercept": opts["lambda_intercept"],
            "objective_components": components,
            "fit_statuses": statuses,
            "continuous_buffers": boundaries,
            "prospective_evaluation_support": "supported_one_sided_endpoint_only",
        }
        return np.asarray(matrices, dtype=np.float64)

    def _lagged_output(self) -> np.ndarray | None:
        return np.stack(self._window_lagged) if self._window_lagged else None

    def _diagnostics(self, segments: WindowedSegments, mats: np.ndarray) -> dict[str, Any]:
        diagnostics = super()._diagnostics(segments, mats)
        diagnostics.update(
            {
                "predictive_capabilities": self.capabilities.__dict__,
                "lag_design": "same_contiguous_segment_required",
                "input_penalty_policy": "known_inputs_unimplemented_shared_B_unpenalized",
                **self._diagnostic_update,
            }
        )
        return diagnostics


@register_connectivity("tvvar_smooth")
class SmoothTVVARConnectivity(_TVVARConnectivity):
    """One-sided TVVAR with squared temporal differences."""

    estimation_mode = "one_sided_smooth_tvvar"
    temporal_penalty = "smooth"


@register_connectivity("tvvar_fused")
class FusedTVVARConnectivity(_TVVARConnectivity):
    """One-sided TVVAR with group-fused coefficient differences."""

    estimation_mode = "one_sided_group_fused_tvvar"
    temporal_penalty = "group_fused"
