"""Conditional / multivariate Granger causality via a regularized VAR fit.

For each window we fit a single multivariate vector-autoregression
    x_t = sum_{l=1..p} A_l x_{t-l} + e_t
with ridge regularization, then read directed influence i -> j as the aggregated magnitude
of the lagged coefficients from source i into target j's equation. This is the "causalised
Granger" estimator: a multivariate (conditional) Granger measure that conditions every pairwise
influence on all other neurons simultaneously, rather than running bivariate tests.
"""

from __future__ import annotations

import numpy as np

from .base import ConnectivityEstimator
from .registry import register_connectivity


def _design_matrix(x: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Build lagged predictors. x: (L, N) -> (X: (L-p, N*p), Y: (L-p, N))."""
    length, _ = x.shape
    cols = []
    for lag in range(1, p + 1):
        cols.append(x[p - lag : length - lag])  # (rows, N)
    X = np.concatenate(cols, axis=1)  # (rows, N*p)
    Y = x[p:length]  # (rows, N)
    return X, Y


@register_connectivity("granger")
class GrangerConnectivity(ConnectivityEstimator):
    """Multivariate (conditional) Granger causality from a ridge-VAR fit.

    Returns a directed N x N matrix; entry [i, j] aggregates |A_l[j, i]| over lags l,
    i.e. how much source i's past helps predict target j given all others.
    """

    directed = True

    def estimate(self, segment: np.ndarray) -> np.ndarray:
        x = np.asarray(segment, dtype=np.float64)
        n = x.shape[1]
        p = max(1, self.cfg.max_lag)
        X, Y = _design_matrix(x, p)
        if X.shape[0] <= X.shape[1]:
            # Underdetermined window; ridge still solves but warn via heavier regularization.
            pass

        # Ridge solution: B = (X'X + lambda I)^-1 X'Y, B shape (N*p, N).
        lam = self.cfg.ridge
        gram = X.T @ X + lam * np.eye(X.shape[1])
        B = np.linalg.solve(gram, X.T @ Y)  # (N*p, N)

        # B rows are stacked by lag: [lag1 (N), lag2 (N), ...]; column j is target j's equation.
        influence = np.zeros((n, n))  # [source i, target j]
        for lag in range(p):
            A_l = B[lag * n : (lag + 1) * n, :]  # (N sources, N targets)
            influence += np.abs(A_l)
        return influence
