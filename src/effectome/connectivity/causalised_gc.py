"""Registered c-GC and c-GC* connectivity estimators."""

from __future__ import annotations

import numpy as np

from effectome.core import CausalisedGC

from .base import ConnectivityEstimator
from .registry import register_connectivity


def _run_causalised_gc(
    segment: np.ndarray,
    *,
    ridge: float,
    max_lag: int,
    alpha: float,
    extra: dict,
    method: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    estimator = CausalisedGC(
        n_perm=int(extra.get("n_perm", 0)),
        n_pasts=max(1, max_lag),
        n_lags=int(extra.get("n_lags", max(1, max_lag))),
        temporal=bool(extra.get("temporal", True)),
        method=method,
        support_test=str(extra.get("support_test", "analytic")),
        parallel=bool(extra.get("parallel", False)),
        signed=not bool(extra.get("unsigned", False)),
        ridge_alpha=float(extra.get("ridge_alpha", ridge)),
        seed=int(extra.get("seed", 42)),
        lag_aggregation=str(extra.get("lag_aggregation", "sum")),
        min_valid_fraction=float(extra.get("min_valid_fraction", 0.95)),
    )
    estimator.fit(np.asarray(segment, dtype=np.float64).T, verbose=int(extra.get("verbose", 0)))
    matrix = estimator.get_connectivity_matrix(
        simulation=bool(extra.get("simulation", True)),
        alpha=float(extra.get("alpha", alpha)),
        beta=float(extra.get("beta", min(alpha, 1e-2))),
    )
    diagnostics = estimator.summary(
        alpha=float(extra.get("alpha", alpha)),
        beta=float(extra.get("beta", min(alpha, 1e-2))),
        aggregation=str(extra.get("lag_aggregation", "sum")),
    )
    return (
        matrix,
        np.asarray(estimator.last_lagged_coefficients_[: estimator.n_lags]),
        diagnostics,
    )


class _CausalisedGCConnectivityBase(ConnectivityEstimator):
    directed = True
    weight_semantics = "signed_regularized_var_coefficient"
    method: str

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        matrix, lagged, _ = _run_causalised_gc(
            segment,
            ridge=self.cfg.ridge,
            max_lag=self.cfg.max_lag,
            alpha=self.cfg.alpha,
            extra=self.cfg.extra,
            method=self.method,
        )
        self._window_lagged = [lagged]
        return matrix

    def estimate_sequence(self, segments) -> np.ndarray:
        self._window_diagnostics = []
        self._window_lagged = []
        matrices = []
        for segment in segments.segments:
            matrix, lagged, diagnostics = _run_causalised_gc(
                segment,
                ridge=self.cfg.ridge,
                max_lag=self.cfg.max_lag,
                alpha=self.cfg.alpha,
                extra=self.cfg.extra,
                method=self.method,
            )
            matrices.append(matrix)
            self._window_lagged.append(lagged)
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
