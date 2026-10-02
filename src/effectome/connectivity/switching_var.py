"""Observed-neuron switching VAR baseline."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

from effectome.data_module.schema import WindowedSegments

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


@dataclass(frozen=True)
class SwitchingVARFit:
    coefficients: np.ndarray
    intercept: np.ndarray
    variances: np.ndarray
    transition_matrix: np.ndarray
    initial_probs: np.ndarray
    filtered_probs: np.ndarray
    smoothed_probs: np.ndarray
    log_likelihood: float
    objective_trace: tuple[float, ...]
    rejection_reasons: tuple[str, ...]
    converged: bool


def _weighted_ridge(
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    *,
    lambda2: float,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 1e-10:
        raise ValueError("state has insufficient occupancy")
    x_mean = (weights[:, None] * design).sum(axis=0) / total
    y_mean = (weights[:, None] * response).sum(axis=0) / total
    x = design - x_mean
    y = response - y_mean
    wx = x * np.sqrt(weights[:, None])
    wy = y * np.sqrt(weights[:, None])
    system = wx.T @ wx + total * lambda2 * np.eye(design.shape[1], dtype=np.float64)
    coefficients = np.linalg.solve(system, wx.T @ wy)
    intercept = y_mean - x_mean @ coefficients
    return coefficients, intercept


def _gaussian_log_emissions(
    design: np.ndarray,
    response: np.ndarray,
    coefficients: np.ndarray,
    intercept: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    means = np.einsum("mp,cpn->mcn", design, coefficients) + intercept[None, :, :]
    residual = response[:, None, :] - means
    return -0.5 * (
        np.sum(np.log(2.0 * np.pi * variances[None, :, :]), axis=2)
        + np.sum((residual * residual) / variances[None, :, :], axis=2)
    )


def _forward_backward(
    log_emissions: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_rows, n_states = log_emissions.shape
    log_initial = np.log(initial + 1e-300)
    log_transition = np.log(transition + 1e-300)
    log_alpha = np.empty((n_rows, n_states), dtype=np.float64)
    log_beta = np.zeros((n_rows, n_states), dtype=np.float64)
    scales = np.empty(n_rows, dtype=np.float64)

    log_alpha[0] = log_initial + log_emissions[0]
    scales[0] = logsumexp(log_alpha[0])
    log_alpha[0] -= scales[0]
    for t in range(1, n_rows):
        log_alpha[t] = log_emissions[t] + logsumexp(
            log_alpha[t - 1][:, None] + log_transition,
            axis=0,
        )
        scales[t] = logsumexp(log_alpha[t])
        log_alpha[t] -= scales[t]

    for t in range(n_rows - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_transition + log_emissions[t + 1][None, :] + log_beta[t + 1][None, :],
            axis=1,
        )
        log_beta[t] -= logsumexp(log_beta[t])

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi = np.zeros((max(0, n_rows - 1), n_states, n_states), dtype=np.float64)
    for t in range(n_rows - 1):
        pair = (
            log_alpha[t][:, None]
            + log_transition
            + log_emissions[t + 1][None, :]
            + log_beta[t + 1][None, :]
        )
        pair -= logsumexp(pair)
        xi[t] = np.exp(pair)
    return gamma, xi, np.exp(log_alpha), float(scales.sum())


def _design_segments(lag_design: LagDesign) -> list[slice]:
    n_rows = lag_design.design.shape[0]
    if n_rows == 0:
        return []
    starts = [0]
    for row in range(1, n_rows):
        index_gap = lag_design.response_indices[row] != lag_design.response_indices[row - 1] + 1
        segment_gap = (
            lag_design.segment_ids is not None
            and lag_design.segment_ids[row] != lag_design.segment_ids[row - 1]
        )
        if index_gap or segment_gap:
            starts.append(row)
    starts.append(n_rows)
    return [slice(start, stop) for start, stop in zip(starts[:-1], starts[1:], strict=True)]


def _segmented_forward_backward(
    log_emissions: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    segments: list[slice],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_rows, n_states = log_emissions.shape
    gamma = np.empty((n_rows, n_states), dtype=np.float64)
    filtered = np.empty_like(gamma)
    xi_sum = np.zeros((n_states, n_states), dtype=np.float64)
    log_likelihood = 0.0
    for segment in segments:
        segment_gamma, xi, segment_filtered, segment_loglik = _forward_backward(
            log_emissions[segment], initial, transition
        )
        gamma[segment] = segment_gamma
        filtered[segment] = segment_filtered
        xi_sum += xi.sum(axis=0)
        log_likelihood += segment_loglik
    return gamma, xi_sum, filtered, float(log_likelihood)


def next_observation_weights(filtered_previous: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Mixture weights available before observing the next row."""
    weights = np.asarray(filtered_previous, dtype=np.float64) @ np.asarray(transition, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("filtered probabilities must have positive mass")
    return weights / total


def _initial_labels(response: np.ndarray, n_states: int, seed: int) -> np.ndarray:
    if n_states == 1:
        return np.zeros(response.shape[0], dtype=np.int64)
    distinct = min(n_states, response.shape[0])
    labels = KMeans(n_clusters=distinct, n_init=5, random_state=seed).fit(response).labels_
    if distinct < n_states:
        labels = labels % distinct
    return labels.astype(np.int64)


def fit_switching_var(
    lag_design: LagDesign,
    *,
    n_states: int,
    lambda2: float = 1e-3,
    eta: float = 0.5,
    variance_floor: float = 1e-6,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 42,
) -> SwitchingVARFit:
    if n_states < 1:
        raise ValueError("n_states must be at least one")
    if lag_design.design.shape[0] < n_states:
        raise ValueError("not enough valid rows for requested switching VAR states")
    if min(lambda2, eta, variance_floor) < 0:
        raise ValueError("regularization and variance floor must be nonnegative")

    design = np.asarray(lag_design.design, dtype=np.float64)
    response = np.asarray(lag_design.response, dtype=np.float64)
    n_rows, n_predictors = design.shape
    n_targets = response.shape[1]
    labels = _initial_labels(response, n_states, seed)
    gamma = np.eye(n_states, dtype=np.float64)[labels]
    if gamma.shape[1] < n_states:
        gamma = np.pad(gamma, ((0, 0), (0, n_states - gamma.shape[1])))
    transition = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)
    initial = np.full(n_states, 1.0 / n_states, dtype=np.float64)
    coefficients = np.zeros((n_states, n_predictors, n_targets), dtype=np.float64)
    intercept = np.zeros((n_states, n_targets), dtype=np.float64)
    variances = np.ones((n_states, n_targets), dtype=np.float64)
    trace: list[float] = []
    reasons: list[str] = []
    converged = False
    segments = _design_segments(lag_design)

    for iteration in range(max_iter):
        occupancy = gamma.sum(axis=0)
        for state in range(n_states):
            if occupancy[state] <= max(1e-8, 0.01 * n_rows):
                reasons.append(f"state_{state}_low_occupancy")
                weights = np.full(n_rows, 1.0 / n_rows)
            else:
                weights = gamma[:, state]
            coefficients[state], intercept[state] = _weighted_ridge(
                design,
                response,
                weights,
                lambda2=lambda2,
            )
            residual = response - intercept[state][None, :] - design @ coefficients[state]
            denom = max(float(weights.sum()), 1e-12)
            variances[state] = np.maximum(
                (weights[:, None] * residual * residual).sum(axis=0) / denom,
                variance_floor,
            )

        log_emissions = _gaussian_log_emissions(design, response, coefficients, intercept, variances)
        gamma, xi_sum, filtered, log_likelihood = _segmented_forward_backward(
            log_emissions, initial, transition, segments
        )
        objective = log_likelihood + eta * float(np.sum(np.log(transition + 1e-300)))
        trace.append(objective)
        initial = np.maximum(
            np.sum([gamma[segment.start] for segment in segments], axis=0),
            1e-12,
        )
        initial /= initial.sum()
        if sum(segment.stop - segment.start for segment in segments) > len(segments):
            counts = xi_sum + eta
            transition = counts / counts.sum(axis=1, keepdims=True)
        else:
            transition = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)
        if iteration > 0:
            prev = trace[-2]
            rel = abs(trace[-1] - prev) / max(1.0, abs(prev))
            if rel <= tol:
                converged = True
                break

    if not trace:
        reasons.append("nonconverged")
        filtered = np.empty((n_rows, n_states), dtype=np.float64)
        gamma = np.empty_like(filtered)
        log_likelihood = float("-inf")
    else:
        final_emissions = _gaussian_log_emissions(
            design, response, coefficients, intercept, variances
        )
        gamma, _, filtered, log_likelihood = _segmented_forward_backward(
            final_emissions, initial, transition, segments
        )
        final_objective = log_likelihood + eta * float(
            np.sum(np.log(transition + 1e-300))
        )
        if abs(final_objective - trace[-1]) > 1e-12:
            trace.append(final_objective)
    if not converged:
        reasons.append("nonconverged")

    if not np.all(np.isfinite(variances)):
        reasons.append("nonfinite_variance")
    if not np.allclose(transition.sum(axis=1), 1.0):
        reasons.append("invalid_transition_rows")
    return SwitchingVARFit(
        coefficients=coefficients,
        intercept=intercept,
        variances=variances,
        transition_matrix=transition,
        initial_probs=initial,
        filtered_probs=filtered,
        smoothed_probs=gamma,
        log_likelihood=log_likelihood,
        objective_trace=tuple(trace),
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        converged=converged,
    )


@dataclass(frozen=True)
class SwitchingVARModel:
    fit: SwitchingVARFit
    n_lags: int
    n_neurons: int
    lag_aggregation: str = "sum"
    capabilities: PredictiveCapabilities = PredictiveCapabilities(
        stateful_online=True, neural_forecast=True
    )

    @property
    def model_hash(self) -> str:
        digest = blake2b(digest_size=16)
        for value in (
            self.fit.coefficients,
            self.fit.intercept,
            self.fit.variances,
            self.fit.transition_matrix,
            self.fit.initial_probs,
        ):
            digest.update(np.ascontiguousarray(value).view(np.uint8).tobytes())
        return digest.hexdigest()

    def initial_state(self, *, recording_identity: Any = None) -> FilterState:
        return FilterState(
            model_hash=self.model_hash,
            history=np.empty((0, self.n_neurons), dtype=np.float64),
            recording_identity=recording_identity,
            diagnostics={"filtered_prob": self.fit.initial_probs.copy()},
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
            raise UnsupportedForecastError("switching VAR does not support exogenous inputs")
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
        filtered = np.asarray(
            state.diagnostics.get("filtered_prob", self.fit.initial_probs), dtype=np.float64
        ).copy()
        previous_index = None if state.sample_index is None else state.sample_index - 1
        for row_number, row in enumerate(observed):
            index = row_number if indices is None else int(indices[row_number])
            gap = indices is not None and previous_index is not None and index != previous_index + 1
            if gap or not valid[row_number]:
                history = np.empty((0, self.n_neurons), dtype=np.float64)
                filtered = self.fit.initial_probs.copy()
            if valid[row_number]:
                if history.shape[0] >= self.n_lags:
                    design_row = np.concatenate(
                        [history[-lag] for lag in range(1, self.n_lags + 1)]
                    )[None, :]
                    emissions = _gaussian_log_emissions(
                        design_row,
                        row[None, :],
                        self.fit.coefficients,
                        self.fit.intercept,
                        self.fit.variances,
                    )[0]
                    prior = filtered @ self.fit.transition_matrix
                    log_prob = np.log(prior + 1e-300) + emissions
                    filtered = np.exp(log_prob - logsumexp(log_prob))
                history = np.vstack([history, row])[-self.n_lags :]
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
            diagnostics={"filtered_prob": filtered, "observed_rows": int(valid.sum())},
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
            raise UnsupportedForecastError("switching VAR does not support exogenous inputs")
        if horizon_samples < 1:
            raise ValueError("horizon_samples must be positive")
        if not self.fit.converged:
            return NeuralForecast(
                np.empty((0, self.n_neurons)), horizon_samples, state.sample_index,
                failure_reason="nonconverged_fit", model_hash=self.model_hash
            )
        if state.history.shape[0] < self.n_lags:
            return NeuralForecast(
                np.empty((0, self.n_neurons)), horizon_samples, state.sample_index,
                failure_reason="insufficient_history", model_hash=self.model_hash
            )
        history = state.history.copy()
        weights = np.asarray(state.diagnostics["filtered_prob"], dtype=np.float64)
        means: list[np.ndarray] = []
        for _ in range(horizon_samples):
            weights = next_observation_weights(weights, self.fit.transition_matrix)
            coefficients = np.einsum("c,cpn->pn", weights, self.fit.coefficients)
            intercept = weights @ self.fit.intercept
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
        if not self.fit.converged:
            raise ValueError("cannot export a graph from a nonconverged fit")
        if state is None:
            weights = self.fit.filtered_probs[-1]
        else:
            self._validate_state(state)
            weights = next_observation_weights(
                np.asarray(state.diagnostics["filtered_prob"]), self.fit.transition_matrix
            )
        lagged = np.einsum("c,cpn->pn", weights, self.fit.coefficients).reshape(
            self.n_lags, self.n_neurons, self.n_neurons
        )
        return GraphEstimate(
            _aggregate_lags(lagged, self.lag_aggregation),
            lagged,
            "switching_var_preobservation_mixture_coefficient",
            self.model_hash,
        )


def fit_switching_var_model(samples: np.ndarray, *, cfg: ConnectivityConfig) -> SwitchingVARModel:
    options = SwitchingVARConnectivity(cfg)._options()
    design = build_lag_design(np.asarray(samples, dtype=np.float64), n_lags=options["n_lags"])
    fit = fit_switching_var(
        design,
        n_states=options["n_states"],
        lambda2=options["lambda2"],
        eta=options["eta"],
        variance_floor=options["variance_floor"],
        max_iter=options["max_iter"],
        tol=options["tol"],
        seed=options["seed"],
    )
    return SwitchingVARModel(fit, options["n_lags"], design.n_neurons, options["lag_aggregation"])


class SwitchingVARConnectivity(ConnectivityEstimator):
    directed = True
    weighted = True
    weight_semantics = "switching_var_preobservation_mixture_coefficient"
    estimation_mode = "observed_neuron_switching_var"

    @property
    def capabilities(self) -> PredictiveCapabilities:
        return PredictiveCapabilities(neural_forecast=True, stateful_online=True)

    def __init__(self, cfg: ConnectivityConfig) -> None:
        super().__init__(cfg)
        self._window_lagged: list[np.ndarray] = []
        self._diagnostic_update: dict[str, Any] = {}

    def _options(self) -> dict[str, Any]:
        extra = dict(self.cfg.extra)
        return {
            "n_lags": int(extra.get("n_lags", self.cfg.max_lag)),
            "n_states": int(extra.get("n_states", 2)),
            "lambda2": float(extra.get("lambda2", self.cfg.ridge)),
            "eta": float(extra.get("eta_em", 0.5)),
            "variance_floor": float(extra.get("variance_floor", 1e-6)),
            "max_iter": int(extra.get("max_iter", 100)),
            "tol": float(extra.get("tol", 1e-6)),
            "seed": int(extra.get("seed", 42)),
            "lag_aggregation": str(extra.get("lag_aggregation", "sum")),
        }

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        opts = self._options()
        design = build_lag_design(np.asarray(segment, dtype=np.float64), n_lags=opts["n_lags"])
        fit = fit_switching_var(
            design,
            n_states=opts["n_states"],
            lambda2=opts["lambda2"],
            eta=opts["eta"],
            variance_floor=opts["variance_floor"],
            max_iter=opts["max_iter"],
            tol=opts["tol"],
            seed=opts["seed"],
        )
        weights = fit.filtered_probs[-1]
        lagged = np.einsum("c,cpn->pn", weights, fit.coefficients).reshape(
            opts["n_lags"],
            design.n_neurons,
            design.n_neurons,
        )
        self._window_lagged = [lagged]
        self._diagnostic_update = {
            "last_log_likelihood": fit.log_likelihood,
            "last_rejection_reasons": list(fit.rejection_reasons),
        }
        return _aggregate_lags(lagged, opts["lag_aggregation"])

    def estimate_sequence(self, segments: WindowedSegments) -> np.ndarray:
        matrices = []
        self._window_lagged = []
        reasons: list[str] = []
        converged = []
        opts = self._options()
        for index, segment in enumerate(segments.segments):
            design = build_lag_design(np.asarray(segment, dtype=np.float64), n_lags=opts["n_lags"])
            fit = fit_switching_var(
                design,
                n_states=opts["n_states"],
                lambda2=opts["lambda2"],
                eta=opts["eta"],
                variance_floor=opts["variance_floor"],
                max_iter=opts["max_iter"],
                tol=opts["tol"],
                seed=int(self.cfg.extra.get("seed", 42)) + index,
            )
            weights = fit.filtered_probs[-1]
            lagged = np.einsum("c,cpn->pn", weights, fit.coefficients).reshape(
                opts["n_lags"],
                design.n_neurons,
                design.n_neurons,
            )
            matrices.append(_aggregate_lags(lagged, opts["lag_aggregation"]))
            self._window_lagged.append(lagged)
            reasons.extend(fit.rejection_reasons)
            converged.append(fit.converged and not fit.rejection_reasons)
        self._diagnostic_update = {
            "adaptation_policy": "independent_window_switching_var_fit",
            "candidate_states": [1, 2, 3, 4, 6],
            "n_states": int(self.cfg.extra.get("n_states", 2)),
            "rejection_reasons": sorted(set(reasons)),
            "fit_statuses": ["accepted" if item else "diagnostic_warning" for item in converged],
            "prospective_api": "filtered_only_no_smoothed_posterior",
        }
        return np.asarray(matrices, dtype=np.float64)

    def _lagged_output(self) -> np.ndarray | None:
        return np.stack(self._window_lagged) if self._window_lagged else None

    def _diagnostics(self, segments: WindowedSegments, mats: np.ndarray) -> dict[str, Any]:
        diagnostics = super()._diagnostics(segments, mats)
        diagnostics.update(
            {
                "predictive_capabilities": self.capabilities.__dict__,
                "innovation_covariance": "diagonal",
                "transition_prior": "symmetric_dirichlet_map_eta",
                **self._diagnostic_update,
            }
        )
        return diagnostics


register_connectivity("switching_var")(SwitchingVARConnectivity)


__all__ = [
    "SwitchingVARFit",
    "SwitchingVARModel",
    "fit_switching_var",
    "fit_switching_var_model",
    "next_observation_weights",
]
