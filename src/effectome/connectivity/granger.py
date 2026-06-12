"""Registry wrappers for causalised Granger estimators."""

from __future__ import annotations

import numpy as np

from effectome.core import CausalisedGC

from .base import ConnectivityEstimator
from .registry import register_connectivity


def _run_causalised_gc(
    segment: np.ndarray,
    *,
    max_lag: int,
    alpha: float,
    extra: dict,
    method: str,
) -> np.ndarray:
    estimator = CausalisedGC(
        n_perm=int(extra.get("n_perm", 0)),
        n_pasts=max(1, max_lag),
        n_lags=int(extra.get("n_lags", 1)),
        temporal=bool(extra.get("temporal", True)),
        method=method,
        parallel=bool(extra.get("parallel", False)),
        signed=not bool(extra.get("unsigned", False)),
    )
    estimator.fit(np.asarray(segment, dtype=np.float64).T, verbose=int(extra.get("verbose", 0)))
    return estimator.get_connectivity_matrix(
        simulation=bool(extra.get("simulation", True)),
        alpha=float(extra.get("alpha", alpha)),
        beta=float(extra.get("beta", min(alpha, 1e-3))),
    )


@register_connectivity("granger")
class GrangerConnectivity(ConnectivityEstimator):
    """Windowed c-GC wrapper over the notebook-oriented core estimator."""

    directed = True

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        return _run_causalised_gc(
            segment,
            max_lag=self.cfg.max_lag,
            alpha=self.cfg.alpha,
            extra=self.cfg.extra,
            method="cgc",
        )


@register_connectivity("granger_star")
class GrangerStarConnectivity(ConnectivityEstimator):
    """Windowed c-GC* wrapper over the notebook-oriented core estimator."""

    directed = True

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        return _run_causalised_gc(
            segment,
            max_lag=self.cfg.max_lag,
            alpha=self.cfg.alpha,
            extra=self.cfg.extra,
            method="fcgc",
        )
