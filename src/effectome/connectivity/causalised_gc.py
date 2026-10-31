"""Registered c-GC and c-GC* connectivity estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from effectome.core import CausalisedGC

from .base import ConnectivityConfig, ConnectivityEstimator
from .predictive import (
    FilterState,
    GraphEstimate,
    NeuralForecast,
    PredictiveCapabilities,
    UnsupportedForecastError,
)
from .registry import register_connectivity
from .sparse_var import SparseVARModel
from .var_utils import VARFit


@dataclass(frozen=True)
class CausalisedGCModel:
    """Full predictive cGC refit plus its separate cross-edge graph export."""

    predictor: SparseVARModel
    exported_matrix: np.ndarray
    exported_lagged_coefficients: np.ndarray
    fit_support: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def capabilities(self) -> PredictiveCapabilities:
        return PredictiveCapabilities(stateful_online=True, neural_forecast=True)

    @property
    def model_hash(self) -> str:
        return self.predictor.model_hash

    def initial_state(self, *, recording_identity: Any = None) -> FilterState:
        return self.predictor.initial_state(recording_identity=recording_identity)

    def observe(self, state: FilterState, samples: np.ndarray, **kwargs: Any) -> FilterState:
        return self.predictor.observe(state, samples, **kwargs)

    def forecast(self, state: FilterState, **kwargs: Any) -> NeuralForecast:
        return self.predictor.forecast(state, **kwargs)

    def graph_at(self, state: FilterState | None = None) -> GraphEstimate:
        if state is not None:
            self.predictor._validate_state(state)
        return GraphEstimate(
            matrix=self.exported_matrix.copy(),
            lagged_coefficients=self.exported_lagged_coefficients.copy(),
            coefficient_semantics="selected_signed_cross_coefficients_self_history_retained_in_predictor",
            model_hash=self.model_hash,
            diagnostics={
                **self.diagnostics,
                "fit_support": self.fit_support.copy(),
                "predictive_lags": self.predictor.n_lags,
            },
        )


def fit_causalised_gc_model(
    samples: np.ndarray,
    *,
    cfg: ConnectivityConfig,
    method: str | None = None,
    available_inputs: np.ndarray | None = None,
) -> CausalisedGCModel:
    if available_inputs is not None:
        raise UnsupportedForecastError("cGC does not support exogenous inputs")
    resolved = cfg.resolve(1.0)
    extra = dict(resolved.extra)
    estimator = CausalisedGC(
        n_perm=int(extra.get("n_perm", 0)),
        n_pasts=max(1, resolved.max_lag),
        n_lags=int(extra.get("n_lags", max(1, resolved.max_lag))),
        temporal=bool(extra.get("temporal", True)),
        method=method or ("cgc_star" if resolved.name == "cgc_star" else "cgc"),
        support_test=str(extra.get("support_test", "fastcgc")),
        support_ridge=float(extra.get("support_ridge", 1e-6)),
        parallel=bool(extra.get("parallel", False)),
        signed=not bool(extra.get("unsigned", False)),
        ridge_alpha=float(extra.get("ridge_alpha", resolved.ridge)),
        seed=int(extra.get("seed", 42)),
        lag_aggregation=str(extra.get("lag_aggregation", "sum")),
        min_valid_fraction=float(extra.get("min_valid_fraction", 0.95)),
    )
    estimator.fit(np.asarray(samples, dtype=np.float64).T, verbose=int(extra.get("verbose", 0)))
    exported_matrix = estimator.get_connectivity_matrix(
        simulation=bool(extra.get("simulation", True)),
        alpha=float(extra.get("alpha", resolved.alpha)),
        beta=float(extra.get(
            "beta", resolved.alpha if estimator.support_test == "fastcgc" else min(resolved.alpha, 0.01)
        )),
    )
    full_lagged = np.asarray(estimator.last_fitted_lagged_coefficients_, dtype=np.float64)
    coefficients = full_lagged.reshape(estimator.n_pasts * estimator.n_neur, estimator.n_neur)
    residual_variance = np.asarray(estimator.last_residual_variance_, dtype=np.float64)
    fit = VARFit(
        coefficients=coefficients,
        intercept=np.asarray(estimator.last_intercept_, dtype=np.float64),
        residual_covariance=np.diag(residual_variance),
        fit_support=np.asarray(estimator.last_fit_support_, dtype=bool).reshape(-1),
        valid_response_indices=np.asarray(estimator.valid_response_indices_, dtype=np.int64),
        objective=float("nan"),
        iterations=1,
        converged=True,
        diagnostics={
            "n_lags": estimator.n_pasts,
            "n_neurons": estimator.n_neur,
            "lambda1": 0.0,
            "lambda2": estimator.ridge_alpha,
            "status": "converged",
            "support_method": estimator.method,
        },
    )
    predictor = SparseVARModel(
        fit=fit,
        capabilities=PredictiveCapabilities(stateful_online=True, neural_forecast=True),
        lag_aggregation=str(extra.get("lag_aggregation", "sum")),
    )
    return CausalisedGCModel(
        predictor=predictor,
        exported_matrix=np.asarray(exported_matrix, dtype=np.float64),
        exported_lagged_coefficients=np.asarray(
            estimator.last_lagged_coefficients_[: estimator.n_lags], dtype=np.float64
        ),
        fit_support=np.asarray(estimator.last_fit_support_, dtype=bool),
        diagnostics=estimator.summary(
            alpha=float(extra.get("alpha", resolved.alpha)),
            beta=float(extra.get(
            "beta", resolved.alpha if estimator.support_test == "fastcgc" else min(resolved.alpha, 0.01)
        )),
            aggregation=str(extra.get("lag_aggregation", "sum")),
        ),
    )


class _CausalisedGCConnectivityBase(ConnectivityEstimator):
    directed = True
    weight_semantics = "signed_regularized_var_coefficient"
    method: str

    @property
    def capabilities(self) -> PredictiveCapabilities:
        return PredictiveCapabilities(independent_window=True, neural_forecast=True)

    def fit(self, samples: np.ndarray) -> CausalisedGCModel:
        return fit_causalised_gc_model(samples, cfg=self.cfg, method=self.method)

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        model = self.fit(segment)
        graph = model.graph_at()
        self._window_lagged = [graph.lagged_coefficients]
        return graph.matrix

    def estimate_sequence(self, segments) -> np.ndarray:
        self._window_diagnostics = []
        self._window_lagged = []
        matrices = []
        for segment in segments.segments:
            model = self.fit(segment)
            graph = model.graph_at()
            matrices.append(graph.matrix)
            self._window_lagged.append(graph.lagged_coefficients)
            diagnostics = dict(graph.diagnostics)
            diagnostics["predictive_snapshot"] = {
                "model_hash": model.model_hash,
                "full_lagged_coefficients": model.predictor.lagged_coefficients,
                "intercept": model.predictor.fit.intercept,
                "residual_variance": np.diag(model.predictor.fit.residual_covariance),
                "valid_response_indices": model.predictor.fit.valid_response_indices,
                "fit_support": model.fit_support,
            }
            self._window_diagnostics.append(diagnostics)
        return np.stack(matrices)

    def _lagged_output(self) -> np.ndarray | None:
        lagged = getattr(self, "_window_lagged", None)
        return np.stack(lagged) if lagged else None

    def _diagnostics(self, segments, mats):
        diagnostics = super()._diagnostics(segments, mats)
        diagnostics.update(
            {
                "support_kind": "logical_and_of_marginal_and_conditional_evidence",
                "support_description": "intersection_of_unconditional_and_conditional_evidence",
                "support_variant": self.method,
                "backend": (
                    "fastcgc"
                    if self.cfg.extra.get("support_test", "fastcgc") == "fastcgc" else "legacy"
                ),
                "signed_weight_source": "ridge_var_on_selected_support",
                "primary_tau_policy": "lagged_only_tau_ge_1",
                "lag_resolved": list(getattr(self, "_window_diagnostics", [])),
            }
        )
        return diagnostics


@register_connectivity("cgc")
class CausalisedGCConnectivity(_CausalisedGCConnectivityBase):
    """Windowed c-GC estimator."""

    method = "cgc"


@register_connectivity("cgc_star")
class CausalisedGCStarConnectivity(_CausalisedGCConnectivityBase):
    """Windowed full-conditioning c-GC* estimator."""

    method = "cgc_star"


__all__ = ["CausalisedGCModel", "fit_causalised_gc_model"]
