"""Sparse VAR predictive baseline tests for Appendix-C WP5."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.connectivity import (
    CONNECTIVITY_REGISTRY,
    ConnectivityConfig,
    ConnectivityFactory,
    UnsupportedForecastError,
)
from effectome.connectivity.sparse_var import fit_sparse_var_model
from effectome.connectivity.var_utils import LagDesign, build_lag_design, fit_sparse_var
from effectome.data_module.schema import TemporalAnchor, Window, WindowedSegments


def _ar_data(n_samples: int = 400) -> np.ndarray:
    rng = np.random.default_rng(5)
    data = np.zeros((n_samples, 2), dtype=np.float64)
    for t in range(1, n_samples):
        data[t, 0] = 0.8 * data[t - 1, 0] + rng.normal(scale=0.05)
        data[t, 1] = 0.4 * data[t - 1, 0] + 0.2 * data[t - 1, 1] + rng.normal(scale=0.05)
    return data


def _windowed(samples: np.ndarray, *, length: int = 40, step: int = 20) -> WindowedSegments:
    segments = []
    windows = []
    anchors = []
    for start in range(0, samples.shape[0] - length + 1, step):
        stop = start + length
        segments.append(samples[start:stop])
        windows.append(Window(start, stop))
        anchors.append(
            TemporalAnchor(
                dataset_id="ds",
                recording_id="rec",
                animal_id="animal",
                session_id=None,
                segment_id=None,
                context_start=start,
                context_stop=stop,
                target_start=stop - 1,
                target_stop=stop,
                anchor_sample=stop - 1,
                anchor_time_seconds=float(stop - 1),
                sampling_rate_hz=1.0,
            )
        )
    return WindowedSegments(
        segments=np.asarray(segments, dtype=np.float32),
        windows=windows,
        behavior_per_window={},
        n_neurons=samples.shape[1],
        fps=1.0,
        anchors=anchors,
    )


def test_sparse_var_registry_entries_are_available():
    assert {"sparse_var", "windowed_sparse_var"} <= set(CONNECTIVITY_REGISTRY)


def test_ridge_limit_matches_normalized_closed_form():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4))
    true_coef = rng.normal(size=(4, 2))
    y = 1.5 + x @ true_coef
    design = LagDesign(
        design=x,
        response=y,
        response_indices=np.arange(x.shape[0]),
        n_lags=2,
        n_neurons=2,
    )
    lambda2 = 0.3

    fit = fit_sparse_var(design, lambda1=0.0, lambda2=lambda2, standardize=False)

    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean(axis=0)
    system = x_centered.T @ x_centered + x.shape[0] * lambda2 * np.eye(x.shape[1])
    expected_coef = np.linalg.solve(system, x_centered.T @ y_centered)
    expected_intercept = y.mean(axis=0) - x.mean(axis=0) @ expected_coef
    assert np.allclose(fit.coefficients, expected_coef)
    assert np.allclose(fit.intercept, expected_intercept)


def test_sparse_var_self_history_forecast_uses_observed_history():
    data = _ar_data()
    cfg = ConnectivityConfig(
        name="sparse_var",
        max_lag=1,
        ridge=1e-4,
        extra={"n_lags": 1, "lambda1": 0.1, "lambda2": 1e-4, "standardize": False},
    )
    model = fit_sparse_var_model(data, cfg=cfg)
    state = model.initial_state()
    state = model.observe(state, data[-1:], sample_indices=np.array([data.shape[0] - 1]))

    forecast = model.forecast(state, horizon_samples=1)

    assert forecast.failure_reason is None
    assert forecast.mean.shape == (1, 2)
    assert np.isclose(forecast.mean[0, 0], 0.8 * data[-1, 0], atol=0.08)


def test_sparse_var_source_target_orientation_and_intercept():
    data = _ar_data()
    data[:, 1] += 2.5
    cfg = ConnectivityConfig(
        name="sparse_var",
        max_lag=1,
        ridge=1e-4,
        extra={"n_lags": 1, "lambda1": 0.0, "lambda2": 1e-4, "standardize": False},
    )

    model = fit_sparse_var_model(data, cfg=cfg)
    graph = model.graph_at()

    assert graph.lagged_coefficients[0, 0, 1] > 0.3
    assert graph.lagged_coefficients[0, 0, 1] > abs(graph.lagged_coefficients[0, 1, 0])
    assert abs(model.fit.intercept[1]) > 1.0
    assert abs(graph.diagnostics["lambda2"] - 1e-4) < 1e-12


def test_lag_design_never_crosses_sample_gaps():
    samples = np.arange(12, dtype=np.float64).reshape(6, 2)
    sample_indices = np.array([0, 1, 2, 10, 11, 12])

    design = build_lag_design(samples, n_lags=1, sample_indices=sample_indices)

    assert design.response_indices.tolist() == [1, 2, 11, 12]


def test_sparse_var_connectivity_exports_graph_and_lagged_coefficients():
    windows = _windowed(_ar_data(120), length=30, step=15)
    estimator = ConnectivityFactory(
        ConnectivityConfig(
            name="sparse_var",
            max_lag=1,
            ridge=1e-3,
            extra={"n_lags": 1, "lambda1": 0.0, "lambda2": 1e-3},
        )
    )

    series = estimator.run(windows)

    assert series.matrices.shape == (windows.n_windows, 2, 2)
    assert series.lagged_matrices is not None
    assert series.lagged_matrices.shape == (windows.n_windows, 1, 2, 2)
    assert series.diagnostics["adaptation_policy"] == "constant_fit_repeated_for_windows"
    assert series.diagnostics["predictive_capabilities"]["neural_forecast"] is True


def test_non_var_estimators_fail_forecast_explicitly():
    estimator = ConnectivityFactory(ConnectivityConfig(name="correlation"))

    with pytest.raises(UnsupportedForecastError):
        estimator.forecast(None)
