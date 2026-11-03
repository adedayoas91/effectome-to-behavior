"""Installed fastC-GC support parity and predictive-readout contracts."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import pytest
from fastCGC import FastGcStar

from effectome.connectivity import ConnectivityConfig
from effectome.connectivity.causalised_gc import fit_causalised_gc_model
from effectome.core import CausalisedGC


@pytest.mark.parametrize("method,upstream_method", [("cgc", "cgc"), ("cgc_star", "fcgc")])
def test_fast_support_matches_installed_package_lag_orientation(method, upstream_method):
    rng = np.random.default_rng(42)
    data = rng.normal(size=(3, 420))
    data[1, 2:] += 0.9 * data[0, :-2]
    upstream = FastGcStar(
        n_pasts=2,
        n_lags=2,
        method=upstream_method,
        dependence_profile="linear",
        evaluation_policy="full",
        exact_permutation=True,
        ridge=1e-6,
        random_state=7,
    ).fit(data, alpha=0.05, beta=0.05)
    upstream.get_connectivity_matrix(alpha=0.05, beta=0.05)
    estimator = CausalisedGC(n_pasts=2, n_lags=2, method=method, seed=7).fit(data)
    estimator.get_connectivity_matrix(alpha=0.05, beta=0.05)
    expected = upstream.lag_adjacency_[1:].copy()
    diagonal = np.arange(3)
    expected[:, diagonal, diagonal] = False
    np.testing.assert_array_equal(estimator.last_support_, expected)
    np.testing.assert_allclose(estimator.pval_corr_, upstream.marginal_pvalue_[3:])
    np.testing.assert_allclose(estimator.pval_inv_corr_, upstream.conditional_pvalue_[3:])
    assert estimator.last_support_[1, 0, 1]
    summary = estimator.summary(alpha=0.05, beta=0.05)
    assert summary["algorithm"] == ("fastC-GC*" if method == "cgc_star" else "fastC-GC")
    assert summary["backend_version"] == version("fastcgc")
    assert summary["support_ridge"] == 1e-6
    assert summary["pvalue_resolution"] == 1 / 418


def test_fast_support_uses_separate_signed_readout_with_self_history():
    rng = np.random.default_rng(6)
    data = np.zeros((2, 4000))
    for t in range(1, data.shape[1]):
        data[0, t] = 0.8 * data[0, t - 1] + rng.normal()
        data[1, t] = 0.65 * data[1, t - 1] - 0.3 * data[0, t - 1] + rng.normal()
    estimator = CausalisedGC(ridge_alpha=0.0, support_ridge=1e-5).fit(data[:, 300:])
    matrix = estimator.get_connectivity_matrix(alpha=1.0, beta=1.0)
    assert np.isclose(matrix[0, 1], -0.3, atol=0.04)
    assert estimator.last_fitted_lagged_coefficients_[0, 1, 1] > 0.6
    assert estimator.last_lagged_coefficients_[0, 1, 1] == 0.0
    assert np.all(np.diag(matrix) == 0.0)
    assert estimator.summary(alpha=1.0, beta=1.0)["signed_weight_source"] == (
        "ridge_var_on_selected_support"
    )


def test_fast_support_drops_only_invalid_lag_chains_without_compressing_raw_time():
    data = np.random.default_rng(8).normal(size=(3, 260))
    data[1, 100] = np.nan
    estimator = CausalisedGC(n_pasts=2, n_lags=2).fit(data)
    expected = np.r_[np.arange(2, 100), np.arange(103, 260)]
    np.testing.assert_array_equal(estimator.valid_response_indices_, expected)
    np.testing.assert_allclose(estimator.target_data, data[:, expected])
    np.testing.assert_allclose(estimator.shifted_data[:3], data[:, expected - 1])
    np.testing.assert_allclose(estimator.shifted_data[3:], data[:, expected - 2])
    estimator.get_connectivity_matrix(alpha=0.05, beta=0.05)
    summary = estimator.summary(alpha=0.05, beta=0.05)
    assert summary["dropped_design_rows"] == 3
    assert summary["pvalue_resolution"] == 1 / 255
    assert np.isfinite(estimator.pval_inv_corr_).all()


def test_fast_support_constant_and_rank_deficient_inputs_remain_finite():
    data = np.random.default_rng(9).normal(size=(3, 240))
    data[1] = data[0]
    data[2] = 0.0
    estimator = CausalisedGC(n_pasts=2, n_lags=2).fit(data)
    matrix = estimator.get_connectivity_matrix(alpha=1.0, beta=1.0)
    assert np.isfinite(matrix).all()
    assert not estimator.last_support_[:, 2, :].any()
    assert not estimator.last_support_[:, :, 2].any()


def test_fast_support_uses_complete_shift_resolution_and_explicit_zero_support():
    estimator = CausalisedGC().fit(np.random.default_rng(11).normal(size=(2, 41)))
    with pytest.raises(ValueError, match="complete-shift p-value resolution"):
        estimator.get_connectivity_matrix(alpha=0.05, beta=0.01)
    estimator.get_connectivity_matrix(alpha=0.05, beta=0.025)
    assert estimator.summary(alpha=0.05, beta=0.025)["pvalue_resolution"] == 1 / 40
    assert not estimator.get_connectivity_matrix(alpha=0.0, beta=0.0).any()


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -0.01, 1.01])
def test_fast_support_rejects_invalid_thresholds(threshold):
    estimator = CausalisedGC().fit(np.random.default_rng(12).normal(size=(2, 100)))
    with pytest.raises(ValueError, match="finite.*\\[0, 1\\]"):
        estimator.get_connectivity_matrix(alpha=threshold, beta=0.05)


def test_fast_support_rejects_monte_carlo_options_instead_of_ignoring_them():
    with pytest.raises(ValueError, match="enumerates all circular shifts"):
        CausalisedGC(n_perm=99)
    with pytest.raises(ValueError, match="support_ridge"):
        CausalisedGC(support_ridge=0.0)


def test_cgc_fit_preserves_recording_resolved_duration_lags():
    config = ConnectivityConfig(
        name="cgc",
        max_lag=1,
        max_lag_seconds=0.7,
        extra={"support_test": "analytic"},
    ).resolve(2.9046296296296297)

    fitted = fit_causalised_gc_model(
        np.random.default_rng(13).normal(size=(120, 3)),
        cfg=config,
    )

    assert fitted.diagnostics["n_pasts"] == 2
    assert fitted.diagnostics["n_lags"] == 2


def test_cgc_direct_fit_rejects_unresolved_duration_lags():
    with pytest.raises(ValueError, match="recording sampling rate"):
        fit_causalised_gc_model(
            np.random.default_rng(14).normal(size=(120, 3)),
            cfg=ConnectivityConfig(name="cgc", max_lag_seconds=0.7),
        )


def test_explicit_legacy_modes_preserve_default_thresholds():
    from effectome.connectivity import ConnectivityConfig
    from effectome.connectivity.causalised_gc import fit_causalised_gc_model

    data = np.random.default_rng(1).normal(size=(400, 8))
    direct = CausalisedGC(support_test="analytic").fit(data.T)
    default_matrix = direct.get_connectivity_matrix()
    reference_matrix = direct.get_connectivity_matrix(alpha=0.01, beta=0.01)
    np.testing.assert_array_equal(default_matrix, reference_matrix)
    config = ConnectivityConfig(name="cgc", extra={"support_test": "analytic"})
    fitted = fit_causalised_gc_model(data, cfg=config)
    expected = direct.get_connectivity_matrix(alpha=0.05, beta=0.01)
    np.testing.assert_array_equal(fitted.exported_matrix, expected)
