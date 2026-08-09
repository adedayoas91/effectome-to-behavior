"""Notebook-friendly c-GC / c-GC* estimators with explicit support selection.

The primary directed effectome is lagged-only (tau >= 1). Support is selected by
intersecting unconditional and conditional evidence, while signed weights come from
a separate ridge-regularized VAR fit restricted to the selected support.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t


def _corrcoef_safe(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return 0.0
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _corr_against_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlate one sample-aligned series against every row of ``y``."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr[np.newaxis, :]
    if x_arr.shape[0] != y_arr.shape[1]:
        raise ValueError("x and y must align across the sample axis")
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean(axis=1, keepdims=True)
    x_scale = float(np.linalg.norm(x_centered))
    y_scale = np.linalg.norm(y_centered, axis=1)
    denom = x_scale * y_scale
    corr = np.zeros(y_arr.shape[0], dtype=np.float64)
    valid = denom > 1e-12
    if np.any(valid):
        corr[valid] = (y_centered[valid] @ x_centered) / denom[valid]
    return np.clip(corr, -1.0, 1.0)


def _analytic_corr_pvalues(corr: np.ndarray, *, n_obs: int, n_controls: int = 0) -> np.ndarray:
    """Two-sided analytic p-values for correlation / partial correlation."""
    dof = int(n_obs) - int(n_controls) - 2
    if dof <= 0:
        return np.ones_like(np.asarray(corr, dtype=np.float64))
    corr_arr = np.clip(np.asarray(corr, dtype=np.float64), -0.999999, 0.999999)
    statistic = np.abs(corr_arr) * np.sqrt(dof / np.maximum(1e-12, 1.0 - corr_arr * corr_arr))
    return 2.0 * student_t.sf(statistic, dof)


def _circular_shift_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Circular-shift permutation p-value for correlation magnitude."""
    if x.size <= 1 or y.size <= 1:
        raise ValueError("support inference needs at least two aligned samples")
    if n_perm <= 0:
        corr = np.clip(abs(_corrcoef_safe(x, y)), 0.0, 1.0 - 1e-12)
        dof = max(int(x.size) - 2, 1)
        statistic = corr * np.sqrt(dof / max(1.0 - corr * corr, 1e-12))
        return float(2.0 * student_t.sf(statistic, dof))

    x_centered = np.asarray(x, dtype=np.float64) - np.mean(x)
    y_centered = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 1e-12:
        return 1.0
    corr_obs = abs(float(x_centered @ y_centered) / denominator)
    # Every circular-shift correlation is available from one FFT-based circular
    # cross-correlation. Sampling its non-zero shifts preserves the configured
    # Monte-Carlo null without repeatedly allocating rolled vectors.
    circular = np.fft.ifft(
        np.conj(np.fft.fft(x_centered)) * np.fft.fft(y_centered)
    ).real
    shifts = rng.integers(1, x.size, size=n_perm)
    permuted = np.abs(circular[shifts] / denominator)
    count = int(np.sum(permuted >= corr_obs))
    return float((count + 1) / (n_perm + 1))


def regression_residual(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return residuals after regressing ``z`` out of ``x``."""
    if z.size == 0:
        return x - np.mean(x)
    if z.ndim == 1:
        z = z[np.newaxis, :]

    design = np.vstack([z, np.ones(z.shape[1])]).T
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    return x - design @ coef


def regression_residual_matrix(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Residualize every row of ``y`` against the shared conditioning set ``z``."""
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 1:
        return regression_residual(y_arr, z)[np.newaxis, :]
    if z.size == 0:
        return y_arr - y_arr.mean(axis=1, keepdims=True)
    if z.ndim == 1:
        z = z[np.newaxis, :]

    design = np.vstack([z, np.ones(z.shape[1])]).T
    coef, *_ = np.linalg.lstsq(design, y_arr.T, rcond=None)
    return (y_arr.T - design @ coef).T


def _ridge_regression(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if x.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)

    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    gram = x_centered.T @ x_centered
    penalty = max(float(alpha), 0.0) * np.eye(x_centered.shape[1], dtype=np.float64)
    rhs = x_centered.T @ y_centered
    return np.linalg.solve(gram + penalty, rhs)


@dataclass
class CausalisedGC:
    """c-GC / c-GC* estimator.

    ``method="cgc"`` uses the pairwise causalised conditioning set.
    ``method="fcgc"`` uses the full-conditioning c-GC* variant.
    """

    n_perm: int = 0
    n_pasts: int = 1
    n_lags: int = 1
    temporal: bool = True
    method: str = "cgc"
    support_test: str = "analytic"
    parallel: bool = False
    signed: bool = True
    ridge_alpha: float = 1.0
    seed: int = 42
    lag_aggregation: str = "sum"
    corr_: np.ndarray | None = None
    pval_corr_: np.ndarray | None = None
    inv_corr_: np.ndarray | None = None
    pval_inv_corr_: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.method not in {"cgc", "fcgc"}:
            raise ValueError("method must be 'cgc' or 'fcgc'.")
        if self.support_test not in {"analytic", "circular_shift"}:
            raise ValueError("support_test must be 'analytic' or 'circular_shift'.")
        if self.support_test == "circular_shift" and self.n_perm <= 0:
            raise ValueError("n_perm must be positive when support_test='circular_shift'.")
        if self.n_pasts < 1:
            raise ValueError("n_pasts must be at least 1.")
        if self.n_lags < 1 or self.n_lags > self.n_pasts:
            raise ValueError("n_lags must lie in [1, n_pasts].")
        if self.lag_aggregation not in {"sum", "mean", "max_abs"}:
            raise ValueError("lag_aggregation must be one of {'sum', 'mean', 'max_abs'}.")

    def _lagged_rows(self, data: np.ndarray) -> np.ndarray:
        rows = []
        for lag in range(1, self.n_pasts + 1):
            rows.append(data[:, self.n_pasts - lag : data.shape[1] - lag])
        return np.concatenate(rows, axis=0)

    def _targets(self, data: np.ndarray) -> np.ndarray:
        return data[:, self.n_pasts :]

    def _conditioning_indices(self, predictor_index: int) -> np.ndarray:
        all_indices = np.arange(self.n_pasts * self.n_neur, dtype=int)
        if self.method == "fcgc":
            return all_indices[all_indices != predictor_index]

        lag_index = predictor_index // self.n_neur
        source_index = predictor_index % self.n_neur
        excluded = {predictor_index}
        for previous_lag in range(lag_index):
            excluded.add(previous_lag * self.n_neur + source_index)
        return np.asarray([idx for idx in all_indices if idx not in excluded], dtype=int)

    def _compute_support_statistics(self) -> None:
        n_rows = self.shifted_data.shape[0]
        self.corr_ = np.zeros((n_rows, self.n_neur), dtype=np.float64)
        self.pval_corr_ = np.ones((n_rows, self.n_neur), dtype=np.float64)
        self.inv_corr_ = np.zeros((n_rows, self.n_neur), dtype=np.float64)
        self.pval_inv_corr_ = np.ones((n_rows, self.n_neur), dtype=np.float64)

        rng = np.random.default_rng(self.seed)
        for predictor_index in range(n_rows):
            x = self.shifted_data[predictor_index]
            cond_idx = self._conditioning_indices(predictor_index)
            z = self.shifted_data[cond_idx] if cond_idx.size else np.empty((0, x.size))
            x_res = regression_residual(x, z)
            y_residuals = regression_residual_matrix(self.target_data, z)
            corr = _corr_against_matrix(x, self.target_data)
            inv_corr = _corr_against_matrix(x_res, y_residuals)
            if not self.signed:
                corr = np.abs(corr)
                inv_corr = np.abs(inv_corr)
            self.corr_[predictor_index] = corr
            self.inv_corr_[predictor_index] = inv_corr

            if self.support_test == "analytic":
                self.pval_corr_[predictor_index] = _analytic_corr_pvalues(corr, n_obs=x.size)
                self.pval_inv_corr_[predictor_index] = _analytic_corr_pvalues(
                    inv_corr,
                    n_obs=x_res.size,
                    n_controls=int(z.shape[0]),
                )
                continue

            for target_index in range(self.n_neur):
                y = self.target_data[target_index]
                self.pval_corr_[predictor_index, target_index] = _circular_shift_pvalue(
                    x,
                    y,
                    n_perm=self.n_perm,
                    rng=rng,
                )
                self.pval_inv_corr_[predictor_index, target_index] = _circular_shift_pvalue(
                    x_res,
                    y_residuals[target_index],
                    n_perm=self.n_perm,
                    rng=rng,
                )

    def fit(self, data: np.ndarray, verbose: int = 0) -> CausalisedGC:
        """Fit on ``data`` shaped ``(n_variables, T)``."""
        del verbose
        self.data = np.asarray(data, dtype=np.float64).copy()
        self.n_neur = self.data.shape[0]
        if self.data.shape[1] <= self.n_pasts:
            raise ValueError("time series is too short for the requested lag depth")
        self.shifted_data = self._lagged_rows(self.data)
        self.target_data = self._targets(self.data)
        self.design_ = self.shifted_data.T
        self._compute_support_statistics()
        return self

    def _support_mask(self, alpha: float, beta: float) -> np.ndarray:
        if (
            self.corr_ is None
            or self.inv_corr_ is None
            or self.pval_corr_ is None
            or self.pval_inv_corr_ is None
        ):
            raise RuntimeError("fit must be called before computing support")
        resolution = 1.0 / (self.n_perm + 1) if self.n_perm > 0 else 0.0
        if self.n_perm > 0 and (alpha + 1e-15 < resolution or beta + 1e-15 < resolution):
            raise ValueError(
                "alpha and beta must be at least the Monte-Carlo p-value resolution "
                f"1/(n_perm+1)={resolution:.6g}; increase n_perm or relax the threshold"
            )
        support = (self.pval_corr_ <= alpha) & (self.pval_inv_corr_ <= beta)
        lagged_support = support.reshape(self.n_pasts, self.n_neur, self.n_neur)
        diagonal = np.arange(self.n_neur)
        lagged_support[:, diagonal, diagonal] = False
        return lagged_support

    def _lagged_coefficients(self, support: np.ndarray) -> np.ndarray:
        coefficients = np.zeros((self.n_pasts, self.n_neur, self.n_neur), dtype=np.float64)
        for target_index in range(self.n_neur):
            flat_support = support[:, :, target_index].reshape(-1)
            if not np.any(flat_support):
                continue
            coef = _ridge_regression(
                self.design_[:, flat_support],
                self.target_data[target_index],
                self.ridge_alpha,
            )
            full = np.zeros(self.design_.shape[1], dtype=np.float64)
            full[flat_support] = coef
            coefficients[:, :, target_index] = full.reshape(self.n_pasts, self.n_neur)
        if not self.signed:
            coefficients = np.abs(coefficients)
        return coefficients

    def collapse_lagged_coefficients(
        self,
        lagged_coefficients: np.ndarray,
        *,
        aggregation: str | None = None,
        n_lags: int | None = None,
    ) -> np.ndarray:
        aggregation = aggregation or self.lag_aggregation
        n_lags = int(self.n_lags if n_lags is None else n_lags)
        if n_lags < 1 or n_lags > self.n_pasts:
            raise ValueError("n_lags must lie in [1, n_pasts].")
        selected = lagged_coefficients[:n_lags]
        if aggregation == "sum":
            collapsed = selected.sum(axis=0)
        elif aggregation == "mean":
            collapsed = selected.mean(axis=0)
        elif aggregation == "max_abs":
            indices = np.argmax(np.abs(selected), axis=0, keepdims=True)
            collapsed = np.take_along_axis(selected, indices, axis=0)[0]
        else:
            raise ValueError(f"unknown aggregation '{aggregation}'")
        np.fill_diagonal(collapsed, 0.0)
        return collapsed

    def get_connectivity_matrix(
        self,
        *,
        simulation: bool = True,
        alpha: float = 0.01,
        beta: float = 0.01,
        aggregation: str | None = None,
        return_lagged: bool = False,
    ) -> np.ndarray:
        """Construct the lagged or collapsed connectivity estimate."""
        if not simulation:
            raise ValueError("tau=0 edges are not part of the primary c-GC effectome")

        support = self._support_mask(alpha, beta)
        lagged_coefficients = self._lagged_coefficients(support)
        collapsed = self.collapse_lagged_coefficients(
            lagged_coefficients,
            aggregation=aggregation,
            n_lags=self.n_lags,
        )

        self.last_support_ = support
        self.last_lagged_coefficients_ = lagged_coefficients
        self.conn_mat = collapsed
        return lagged_coefficients if return_lagged else collapsed

    def summary(self, *, alpha: float, beta: float, aggregation: str | None = None) -> dict[str, object]:
        if not hasattr(self, "last_support_"):
            self.get_connectivity_matrix(alpha=alpha, beta=beta, aggregation=aggregation)
        support = self.last_support_
        lagged = self.last_lagged_coefficients_
        resolution = 1.0 / (self.n_perm + 1) if self.n_perm > 0 else 0.0
        return {
            "method": self.method,
            "n_perm": self.n_perm if self.support_test == "circular_shift" else 0,
            "n_pasts": self.n_pasts,
            "n_lags": self.n_lags,
            "ridge_alpha": self.ridge_alpha,
            "lag_aggregation": aggregation or self.lag_aggregation,
            "primary_tau_policy": "lagged_only_tau_ge_1",
            "support_kind": "logical_and_of_marginal_and_conditional_evidence",
            "support_test": self.support_test,
            "support_test_assumption": (
                "analytic correlation calibration; autocorrelation-aware sensitivity required"
                if self.support_test == "analytic"
                else "circular shifts preserve each tested series' marginal autocorrelation"
            ),
            "signed_weight_source": "ridge_var_on_selected_support",
            "lag_support_counts": support[: self.n_lags].sum(axis=(1, 2)).astype(int).tolist(),
            "lag_nonzero_weight_counts": (
                (np.abs(lagged[: self.n_lags]) > 0.0)
                .sum(axis=(1, 2))
                .astype(int)
                .tolist()
            ),
            "support_density": float(np.mean(support[: self.n_lags])),
            "pvalue_resolution": (
                float(resolution) if self.support_test == "circular_shift" else None
            ),
            "alpha": float(alpha),
            "beta": float(beta),
        }


GcStar = CausalisedGC

__all__ = ["CausalisedGC", "GcStar", "regression_residual"]
