"""Notebook-friendly causalised Granger estimators.

This module follows the structure of the markovianity_diagnostic GcStar
implementation so the same analysis style can be reused in effectome notebooks.
The main difference is that signed weights are preserved by default so the
output can support excitation/inhibition analyses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:
    from numba import jit
except Exception:  # pragma: no cover - optional runtime speedup
    def jit(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


@jit(nopython=True)
def _perm_test_numba(x: np.ndarray, y: np.ndarray, n_perm: int) -> float:
    """Circular-shift permutation p-value for correlation magnitude."""
    if x.size <= 1 or y.size <= 1 or n_perm <= 0:
        return 0.0

    count = 0
    corr_obs = np.corrcoef(x, y)[1, 0]
    low = 1
    high = x.size

    for _ in range(n_perm):
        shift = np.random.randint(low, high)
        rolled = np.hstack((x[shift:], x[:shift]))
        corr_perm = np.corrcoef(rolled, y)[1, 0]
        if np.abs(corr_perm) >= np.abs(corr_obs):
            count += 1

    return count / n_perm


def regression_residual(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return residuals after regressing ``z`` out of ``x``."""
    if z.size == 0:
        return x - np.mean(x)

    if z.ndim == 1:
        z = z[np.newaxis, :]

    design = np.vstack([z, np.ones(z.shape[1])]).T
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coef


@dataclass
class CausalisedGC:
    """c-GC / c-GC* estimator.

    Parameters mirror the markovianity_diagnostic estimator.
    ``method="cgc"`` uses the pairwise causalised conditioning set.
    ``method="fcgc"`` uses the full-conditioning c-GC* variant.
    """

    n_perm: int = 200
    n_pasts: int = 1
    n_lags: int = 1
    temporal: bool = True
    method: str = "cgc"
    parallel: bool = False
    signed: bool = True
    corr_: np.ndarray | None = None
    pval_corr_: np.ndarray | None = None
    inv_corr_: np.ndarray | None = None
    pval_inv_corr_: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.method not in {"cgc", "fcgc"}:
            raise ValueError("method must be 'cgc' or 'fcgc'.")
        if self.n_pasts < 0:
            raise ValueError("n_pasts must be non-negative.")
        if self.n_lags < 1:
            raise ValueError("n_lags must be at least 1.")
        self.logger = logging.getLogger(__name__)

    def shift_data(self, arr: np.ndarray) -> np.ndarray:
        """Create stacked lagged views of ``arr`` with shape ``(n_variables, T)``."""
        self.n_neur = arr.shape[0]
        if self.n_pasts == 0:
            return arr.copy()

        trimmed = arr[:, self.n_pasts :]
        for index in range(self.n_pasts):
            start = self.n_pasts - 1 - index
            stop = -index - 1
            trimmed = np.r_[trimmed, arr[:, start:stop]]
        return trimmed

    def get_conditioning_set(self, data: np.ndarray, i: int, j: int) -> np.ndarray:
        """Build the c-GC conditioning set for directed pair ``i -> j``."""
        shifted = self.shift_data(data.copy())
        source_index = i % self.n_neur
        excluded_history = [
            source_index + lag * self.n_neur for lag in range(i // self.n_neur)
        ]
        excluded = np.r_[np.array(excluded_history, dtype=int), [i, j]]
        return np.delete(shifted, excluded, axis=0)

    def correlation_func(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute unconditional dependence and circular-shift p-values."""
        self.n_neur = data.shape[0]
        shifted = self.shift_data(data.copy())
        corr = np.corrcoef(shifted)
        if not self.signed:
            corr = np.abs(corr)

        n_rows = shifted.shape[0]
        pvals = np.zeros((n_rows, self.n_neur))
        for i in range(n_rows):
            for j in range(self.n_neur):
                pvals[i, j] = _perm_test_numba(shifted[i, :], shifted[j, :], self.n_perm)
        return corr[:, : self.n_neur], pvals

    def inv_correlation_func(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute conditional dependence using residual correlations."""
        self.n_neur = data.shape[0]
        shifted = self.shift_data(data.copy())
        n_rows = shifted.shape[0]
        inv_corr = np.zeros((n_rows, self.n_neur))
        pvals = np.zeros((n_rows, self.n_neur))

        for i in range(n_rows):
            for j in range(self.n_neur):
                x = shifted[i]
                y = shifted[j]
                if self.method == "fcgc":
                    z = np.delete(shifted.copy(), [i, j], axis=0)
                else:
                    z = self.get_conditioning_set(data, i, j)

                x_res = regression_residual(x, z)
                y_res = regression_residual(y, z)
                corr = np.corrcoef(x_res, y_res)[1, 0]
                inv_corr[i, j] = corr if self.signed else np.abs(corr)
                pvals[i, j] = _perm_test_numba(x_res, y_res, self.n_perm)

        return inv_corr, pvals

    def fit(self, data: np.ndarray, verbose: int = 0) -> CausalisedGC:
        """Fit on ``data`` shaped ``(n_variables, T)``."""
        del verbose
        self.data = np.asarray(data, dtype=np.float64).copy()
        self.shifted_data = self.shift_data(self.data)

        if self.parallel:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                corr_future = executor.submit(self.correlation_func, self.data)
                inv_future = executor.submit(self.inv_correlation_func, self.data)
                self.corr_, self.pval_corr_ = corr_future.result()
                self.inv_corr_, self.pval_inv_corr_ = inv_future.result()
        else:
            self.corr_, self.pval_corr_ = self.correlation_func(self.data)
            self.inv_corr_, self.pval_inv_corr_ = self.inv_correlation_func(self.data)
        return self

    def get_connectivity_matrix(
        self,
        *,
        simulation: bool = True,
        alpha: float = 0.01,
        beta: float = 0.001,
    ) -> np.ndarray:
        """Construct a weighted connectivity matrix from significance masks."""
        if self.corr_ is None or self.inv_corr_ is None:
            raise RuntimeError("fit must be called before get_connectivity_matrix.")

        sig_corr = np.multiply(self.corr_, self.pval_corr_ <= alpha)
        sig_inv = np.multiply(self.inv_corr_, self.pval_inv_corr_ <= beta)
        inferred = np.logical_and(sig_corr != 0, sig_inv != 0)

        all_lags: list[np.ndarray] = []
        n_neur = inferred.shape[1]
        for lag in range(self.n_pasts + 1):
            start = lag * n_neur
            stop = (lag + 1) * n_neur
            all_lags.append(inferred[start:stop, 0:n_neur])

        if simulation:
            lag_ids = [1] if len(all_lags) > 1 else [0]
            if self.n_lags > 1:
                max_lag = min(self.n_lags, len(all_lags) - 1)
                lag_ids.extend(range(2, max_lag + 1))
        elif self.n_lags == 1:
            lag_ids = [0, 1] if len(all_lags) > 1 else [0]
        else:
            max_lag = min(self.n_lags, len(all_lags) - 1)
            lag_ids = list(range(0, max_lag + 1))

        conn = np.zeros((n_neur, n_neur), dtype=np.float64)
        for lag in lag_ids:
            start = lag * n_neur
            stop = (lag + 1) * n_neur
            weights = self.corr_[start:stop, :]
            conn += np.multiply(weights, all_lags[lag])

        self.conn_mat = conn
        np.fill_diagonal(self.conn_mat, 0.0)
        return self.conn_mat


GcStar = CausalisedGC

__all__ = ["CausalisedGC", "GcStar", "regression_residual"]
