"""fastC-GC / fastC-GC* support selection with a separate signed VAR readout.

The primary directed effectome is lagged-only (tau >= 1). Support is selected by
intersecting unconditional and conditional evidence, while signed weights come from
a separate ridge-regularized VAR fit restricted to the selected support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import distribution

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


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float, float]:
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if x.shape[1] == 0:
        intercept = float(np.mean(y))
        residuals = y - intercept
        return np.zeros(0, dtype=np.float64), intercept, float(np.mean(residuals**2))

    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    gram = x_centered.T @ x_centered
    penalty = max(float(alpha), 0.0) * np.eye(x_centered.shape[1], dtype=np.float64)
    rhs = x_centered.T @ y_centered
    coef = np.linalg.solve(gram + penalty, rhs)
    intercept = float(y_mean - np.ravel(x_mean) @ coef)
    residuals = y - (x @ coef + intercept)
    return coef, intercept, float(np.mean(residuals**2))


def _ridge_regression(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    coef, _, _ = _ridge_fit(x, y, alpha)
    return coef


@dataclass
class CausalisedGC:
    """fastC-GC / fastC-GC* support with retained-self-history signed readout.

    ``method="cgc"`` selects fastC-GC; ``method="cgc_star"`` selects fastC-GC*.
    Explicit ``analytic`` and ``circular_shift`` tests retain the legacy
    lag-only support implementation for comparison.
    """

    n_perm: int = 0
    n_pasts: int = 1
    n_lags: int = 1
    temporal: bool = True
    method: str = "cgc"
    support_test: str = "fastcgc"
    support_ridge: float = 1e-6
    parallel: bool = False
    signed: bool = True
    ridge_alpha: float = 1.0
    seed: int = 42
    lag_aggregation: str = "sum"
    min_valid_fraction: float = 0.95
    corr_: np.ndarray | None = None
    pval_corr_: np.ndarray | None = None
    inv_corr_: np.ndarray | None = None
    pval_inv_corr_: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.method not in {"cgc", "cgc_star"}:
            raise ValueError("method must be 'cgc' or 'cgc_star'.")
        if self.support_test not in {"fastcgc", "analytic", "circular_shift"}:
            raise ValueError("support_test must be 'fastcgc', 'analytic' or 'circular_shift'.")
        if not np.isfinite(self.support_ridge) or self.support_ridge <= 0:
            raise ValueError("support_ridge must be finite and positive.")
        if self.support_test == "fastcgc" and self.n_perm != 0:
            raise ValueError("fastcgc enumerates all circular shifts; n_perm must be zero.")
        if self.support_test == "circular_shift" and self.n_perm <= 0:
            raise ValueError("n_perm must be positive when support_test='circular_shift'.")
        if self.n_pasts < 1:
            raise ValueError("n_pasts must be at least 1.")
        if self.n_lags < 1 or self.n_lags > self.n_pasts:
            raise ValueError("n_lags must lie in [1, n_pasts].")
        if self.lag_aggregation not in {"sum", "mean", "max_abs"}:
            raise ValueError("lag_aggregation must be one of {'sum', 'mean', 'max_abs'}.")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must lie in (0, 1].")

    def _lagged_rows(self, data: np.ndarray) -> np.ndarray:
        rows = []
        for lag in range(1, self.n_pasts + 1):
            rows.append(data[:, self.n_pasts - lag : data.shape[1] - lag])
        return np.concatenate(rows, axis=0)

    def _targets(self, data: np.ndarray) -> np.ndarray:
        return data[:, self.n_pasts :]

    def _valid_design_rows_mask(self, data: np.ndarray) -> np.ndarray:
        finite_by_time = np.all(np.isfinite(data), axis=0)
        if data.shape[1] <= self.n_pasts:
            return np.zeros(0, dtype=bool)
        valid_rows = np.ones(data.shape[1] - self.n_pasts, dtype=bool)
        for lag in range(self.n_pasts + 1):
            valid_rows &= finite_by_time[lag : lag + valid_rows.shape[0]]
        return valid_rows

    def _conditioning_indices(self, predictor_index: int) -> np.ndarray:
        all_indices = np.arange(self.n_pasts * self.n_neur, dtype=int)
        if self.method == "cgc_star":
            return all_indices[all_indices != predictor_index]

        lag_index = predictor_index // self.n_neur
        source_index = predictor_index % self.n_neur
        excluded = {predictor_index}
        for previous_lag in range(lag_index):
            excluded.add(previous_lag * self.n_neur + source_index)
        return np.asarray([idx for idx in all_indices if idx not in excluded], dtype=int)

    def _compute_fast_support_statistics(self) -> None:
        from fastCGC import FastGcStar

        backend_type = FastGcStar
        if self.dropped_design_rows_:
            # Upstream's public input accepts only finite recordings. Keep the
            # already aligned columns instead of joining raw samples over gaps.
            # This override is tied to the immutable dependency revision.
            from fastCGC.dependence.context import FitContext

            shifted = np.concatenate([self.target_data, self.shifted_data], axis=0)
            n_variables = self.n_neur
            n_pasts = self.n_pasts

            class MaskedFastGcStar(FastGcStar):
                def _prepare_context(self, data: np.ndarray) -> None:
                    means = shifted.mean(axis=1, keepdims=True)
                    centered = shifted - means
                    scales = np.sqrt(np.mean(centered**2, axis=1))
                    tolerance = np.finfo(float).eps * np.maximum(1.0, np.abs(means[:, 0]))
                    valid = scales > tolerance
                    standardized = np.zeros_like(centered)
                    standardized[valid] = centered[valid] / scales[valid, None]
                    allowed = valid[:, None] & valid[:n_variables][None, :]
                    diagonal = np.arange(n_variables)
                    allowed[diagonal, diagonal] = False
                    rows = np.arange(shifted.shape[0])
                    self._fit_context = FitContext(
                        data=data,
                        shifted=shifted,
                        standardized=standardized,
                        valid_rows=valid,
                        allowed_pairs=allowed,
                        n_variables=n_variables,
                        n_pasts=n_pasts,
                        n_samples=shifted.shape[1],
                        row_variable=rows % n_variables,
                        row_lag=rows // n_variables,
                        row_mean=means[:, 0],
                        row_scale=scales,
                    )
                    self.shifted_data = shifted
                    self.standardized_shifted_ = standardized
                    self.row_mean_ = means[:, 0]
                    self.row_scale_ = scales
                    self.valid_row_mask_ = valid
                    self.n_neur = n_variables
                    self._context_data_id = id(data)

            backend_type = MaskedFastGcStar

        if self.valid_design_rows_ < 2:
            raise ValueError("fastcgc support inference needs at least two valid lag-design rows")
        backend = backend_type(
            n_pasts=self.n_pasts,
            n_lags=self.n_lags,
            method="fcgc" if self.method == "cgc_star" else "cgc",
            dependence_profile="linear",
            evaluation_policy="full",
            exact_permutation=True,
            ridge=self.support_ridge,
            parallel=self.parallel,
            random_state=self.seed,
        ).fit(self.data)
        lagged = slice(self.n_neur, None)
        self.corr_ = np.asarray(backend.marginal_effect_[lagged], dtype=np.float64).copy()
        self.pval_corr_ = np.asarray(backend.marginal_pvalue_[lagged], dtype=np.float64).copy()
        self.inv_corr_ = np.asarray(backend.conditional_effect_[lagged], dtype=np.float64).copy()
        self.pval_inv_corr_ = np.asarray(backend.conditional_pvalue_[lagged], dtype=np.float64).copy()
        self.support_eligible_ = np.asarray(backend._fit_context.allowed_pairs[lagged], dtype=bool).copy()
        self.backend_metadata_ = dict(backend.backend_metadata_)
        package = distribution("fastcgc")
        self.backend_version_ = package.version
        self.backend_source_ = json.loads(package.read_text("direct_url.json") or "{}")

    def _compute_support_statistics(self) -> None:
        if self.support_test == "fastcgc":
            self._compute_fast_support_statistics()
            return
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
        if self.data.ndim != 2 or self.data.shape[0] < 1:
            raise ValueError("data must have shape (n_variables, T) with at least one variable")
        self.n_neur = self.data.shape[0]
        if self.data.shape[1] <= self.n_pasts:
            raise ValueError("time series is too short for the requested lag depth")
        self.total_design_rows_ = int(self.data.shape[1] - self.n_pasts)
        self.valid_design_rows_mask_ = self._valid_design_rows_mask(self.data)
        self.valid_design_rows_ = int(self.valid_design_rows_mask_.sum())
        self.dropped_design_rows_ = int(self.total_design_rows_ - self.valid_design_rows_)
        self.valid_row_fraction_ = self.valid_design_rows_ / max(self.total_design_rows_, 1)
        self.invalid_timepoint_count_ = int(np.size(self.data) - np.isfinite(self.data).sum())
        if self.valid_row_fraction_ + 1e-12 < self.min_valid_fraction:
            raise ValueError(
                "valid row fraction "
                f"{self.valid_row_fraction_:.3f} fell below min_valid_fraction="
                f"{self.min_valid_fraction:.3f}"
            )
        self.shifted_data = self._lagged_rows(self.data)[:, self.valid_design_rows_mask_]
        self.target_data = self._targets(self.data)[:, self.valid_design_rows_mask_]
        self.design_ = self.shifted_data.T
        self.valid_response_indices_ = np.flatnonzero(self.valid_design_rows_mask_) + self.n_pasts
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
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in (alpha, beta)):
            raise ValueError("alpha and beta must be finite and lie in [0, 1]")
        if self.support_test == "fastcgc":
            resolution = 1.0 / self.valid_design_rows_
            if any(0.0 < value and value + 1e-15 < resolution for value in (alpha, beta)):
                raise ValueError(
                    "positive alpha and beta must be at least the complete-shift p-value resolution "
                    f"1/n_valid_rows={resolution:.6g}; increase history or relax the threshold"
                )
        resolution = 1.0 / (self.n_perm + 1) if self.n_perm > 0 else 0.0
        if self.n_perm > 0 and (alpha + 1e-15 < resolution or beta + 1e-15 < resolution):
            raise ValueError(
                "alpha and beta must be at least the Monte-Carlo p-value resolution "
                f"1/(n_perm+1)={resolution:.6g}; increase n_perm or relax the threshold"
            )
        support = (self.pval_corr_ <= alpha) & (self.pval_inv_corr_ <= beta)
        if self.support_test == "fastcgc":
            support &= self.support_eligible_
        lagged_support = support.reshape(self.n_pasts, self.n_neur, self.n_neur)
        diagonal = np.arange(self.n_neur)
        lagged_support[:, diagonal, diagonal] = False
        return lagged_support

    def _lagged_coefficients(self, support: np.ndarray) -> np.ndarray:
        coefficients = np.zeros((self.n_pasts, self.n_neur, self.n_neur), dtype=np.float64)
        fitted_coefficients = np.zeros_like(coefficients)
        fit_support = np.zeros_like(support, dtype=bool)
        intercept = np.zeros(self.n_neur, dtype=np.float64)
        residual_variance = np.zeros(self.n_neur, dtype=np.float64)
        for target_index in range(self.n_neur):
            target_self_history = np.zeros((self.n_pasts, self.n_neur), dtype=bool)
            target_self_history[:, target_index] = True
            flat_support = support[:, :, target_index].reshape(-1)
            nuisance_support = target_self_history.reshape(-1)
            target_fit_support = flat_support | nuisance_support
            coef, intercept[target_index], residual_variance[target_index] = _ridge_fit(
                self.design_[:, target_fit_support],
                self.target_data[target_index],
                self.ridge_alpha,
            )
            full = np.zeros(self.design_.shape[1], dtype=np.float64)
            full[target_fit_support] = coef
            fit_support[:, :, target_index] = target_fit_support.reshape(self.n_pasts, self.n_neur)
            fitted_coefficients[:, :, target_index] = full.reshape(self.n_pasts, self.n_neur)
            exported = np.zeros_like(full)
            exported[flat_support] = full[flat_support]
            coefficients[:, :, target_index] = exported.reshape(self.n_pasts, self.n_neur)
        if not self.signed:
            coefficients = np.abs(coefficients)
        self.last_fitted_lagged_coefficients_ = fitted_coefficients
        self.last_fit_support_ = fit_support
        self.last_intercept_ = intercept
        self.last_residual_variance_ = residual_variance
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
        alpha: float | None = None,
        beta: float | None = None,
        aggregation: str | None = None,
        return_lagged: bool = False,
    ) -> np.ndarray:
        """Construct the lagged or collapsed connectivity estimate."""
        if not simulation:
            raise ValueError("tau=0 edges are not part of the primary c-GC effectome")

        default_threshold = 0.05 if self.support_test == "fastcgc" else 0.01
        alpha = default_threshold if alpha is None else alpha
        beta = default_threshold if beta is None else beta
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
        if self.support_test == "fastcgc":
            resolution = 1.0 / self.valid_design_rows_
        return {
            "method": self.method,
            "backend": "fastcgc" if self.support_test == "fastcgc" else "legacy_effectome",
            "backend_version": getattr(self, "backend_version_", None),
            "backend_source": getattr(self, "backend_source_", {}),
            "shift_time_axis": (
                "retained_lag_design_rows" if self.support_test == "fastcgc" else None
            ),
            "algorithm": (
                "fastC-GC*" if self.method == "cgc_star" else "fastC-GC"
            ) if self.support_test == "fastcgc" else self.method,
            "support_ridge": self.support_ridge if self.support_test == "fastcgc" else None,
            "support_backend_metadata": getattr(self, "backend_metadata_", {}),
            "support_expansion": (
                "present_plus_lags_with_lagged_only_export"
                if self.support_test == "fastcgc" else "lags_only"
            ),
            "n_perm": self.n_perm if self.support_test == "circular_shift" else 0,
            "n_pasts": self.n_pasts,
            "n_lags": self.n_lags,
            "ridge_alpha": self.ridge_alpha,
            "min_valid_fraction": self.min_valid_fraction,
            "lag_aggregation": aggregation or self.lag_aggregation,
            "primary_tau_policy": "lagged_only_tau_ge_1",
            "support_kind": "logical_and_of_marginal_and_conditional_evidence",
            "support_test": self.support_test,
            "nonfinite_row_policy": "drop_only_rows_whose_lagged_chain_touches_nonfinite_samples",
            "support_test_assumption": (
                "complete circular shifts require a suitable shift-invariant null law; "
                "no automatic graph-wide error control"
                if self.support_test == "fastcgc"
                else "analytic correlation calibration; autocorrelation-aware sensitivity required"
                if self.support_test == "analytic"
                else "circular shifts preserve each tested series' marginal autocorrelation"
            ),
            "signed_weight_source": "ridge_var_on_selected_support",
            "signed_refit_policy": "selected_cross_edges_plus_target_self_history_nuisance",
            "lag_support_counts": support[: self.n_lags].sum(axis=(1, 2)).astype(int).tolist(),
            "lag_fit_support_counts": (
                getattr(self, "last_fit_support_", support)[: self.n_lags]
                .sum(axis=(1, 2))
                .astype(int)
                .tolist()
            ),
            "lag_nonzero_weight_counts": (
                (np.abs(lagged[: self.n_lags]) > 0.0)
                .sum(axis=(1, 2))
                .astype(int)
                .tolist()
            ),
            "support_density": float(np.mean(support[: self.n_lags])),
            "pvalue_resolution": (
                float(resolution) if self.support_test in {"circular_shift", "fastcgc"} else None
            ),
            "total_design_rows": int(self.total_design_rows_),
            "valid_design_rows": int(self.valid_design_rows_),
            "dropped_design_rows": int(getattr(self, "dropped_design_rows_", 0)),
            "valid_row_fraction": float(getattr(self, "valid_row_fraction_", 1.0)),
            "invalid_timepoint_count": int(getattr(self, "invalid_timepoint_count_", 0)),
            "valid_response_start": (
                int(self.valid_response_indices_[0]) if self.valid_response_indices_.size else None
            ),
            "valid_response_stop": (
                int(self.valid_response_indices_[-1] + 1)
                if self.valid_response_indices_.size
                else None
            ),
            "residual_variance_mean": float(
                np.mean(getattr(self, "last_residual_variance_", np.asarray([np.nan])))
            ),
            "alpha": float(alpha),
            "beta": float(beta),
        }


__all__ = ["CausalisedGC", "regression_residual"]
