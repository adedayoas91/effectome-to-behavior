"""Observed-neuron switching VAR tests for Appendix-C WP7."""

from __future__ import annotations

import numpy as np

from effectome.connectivity import CONNECTIVITY_REGISTRY, ConnectivityConfig, ConnectivityFactory
from effectome.connectivity.sparse_var import fit_sparse_var_model
from effectome.connectivity.switching_var import fit_switching_var, next_observation_weights
from effectome.connectivity.var_utils import build_lag_design
from effectome.data_module.schema import TemporalAnchor, Window, WindowedSegments


def test_final_inference_matches_saved_parameters_and_resets():
    from effectome.connectivity.switching_var import _forward_backward, _gaussian_log_emissions
    from effectome.connectivity.var_utils import LagDesign

    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(2, 30, 1))
    ids = np.r_[np.arange(15), np.arange(30, 45)]
    fit = fit_switching_var(LagDesign(x, y, ids, 1, 1), n_states=2,
                            max_iter=100, seed=3)
    emissions = _gaussian_log_emissions(x, y, fit.coefficients, fit.intercept, fit.variances)
    left = _forward_backward(emissions[:15], fit.initial_probs, fit.transition_matrix)
    right = _forward_backward(emissions[15:], fit.initial_probs, fit.transition_matrix)
    assert np.allclose(fit.filtered_probs, np.vstack([left[2], right[2]]))
    assert np.isclose(fit.log_likelihood, left[3] + right[3])
    assert np.all(np.diff(fit.objective_trace) >= -1e-8)


def test_one_iteration_cannot_claim_convergence():
    fit = fit_switching_var(build_lag_design(_ar_data(), n_lags=1), n_states=1, max_iter=1)
    assert not fit.converged
    assert "nonconverged" in fit.rejection_reasons


def test_switching_model_filter_serialization_and_chunk_equality():
    import pickle

    from effectome.connectivity.switching_var import fit_switching_var_model

    data = np.random.default_rng(22).normal(size=(90, 2))
    cfg = ConnectivityConfig(name="switching_var", max_lag=1,
                             extra={"n_states": 1, "max_iter": 100})
    model = fit_switching_var_model(data[:50], cfg=cfg)
    state = model.initial_state()
    whole = model.observe(state, data[50:], sample_indices=np.arange(50, 90))
    partial = model.observe(state, data[50:70], sample_indices=np.arange(50, 70))
    model, partial = pickle.loads(pickle.dumps((model, partial)))
    chunk = model.observe(partial, data[70:], sample_indices=np.arange(70, 90))
    assert np.allclose(model.forecast(whole, horizon_samples=3).mean,
                       model.forecast(chunk, horizon_samples=3).mean)


def _ar_data(phi: float = 0.7, n_samples: int = 120) -> np.ndarray:
    data = np.zeros((n_samples, 2), dtype=np.float64)
    data[0] = [1.0, -0.5]
    for t in range(1, n_samples):
        data[t, 0] = phi * data[t - 1, 0] + 0.01 * np.sin(t)
        data[t, 1] = 0.35 * data[t - 1, 0] + 0.1 * data[t - 1, 1]
    return data


def _windows() -> WindowedSegments:
    segments = np.asarray([_ar_data(0.3), _ar_data(0.85)], dtype=np.float32)
    anchors = []
    windows = []
    for index, segment in enumerate(segments):
        start = index * 200
        stop = start + segment.shape[0]
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
                anchor_time_seconds=float(stop),
                sampling_rate_hz=1.0,
                gap_before=index == 1,
                gap_after=index == 0,
            )
        )
    return WindowedSegments(
        segments=segments,
        windows=windows,
        behavior_per_window={},
        n_neurons=2,
        fps=1.0,
        anchors=anchors,
    )


def test_switching_var_registry_entry_is_available():
    assert "switching_var" in CONNECTIVITY_REGISTRY


def test_single_state_matches_sparse_var_coefficients():
    data = _ar_data()
    design = build_lag_design(data, n_lags=1)
    switching = fit_switching_var(
        design,
        n_states=1,
        lambda2=1e-3,
        max_iter=20,
        seed=0,
    )
    sparse = fit_sparse_var_model(
        data,
        cfg=ConnectivityConfig(
            name="sparse_var",
            max_lag=1,
            ridge=1e-3,
            extra={"n_lags": 1, "lambda2": 1e-3, "standardize": False},
        ),
    )

    assert np.allclose(switching.coefficients[0], sparse.fit.coefficients)
    assert np.allclose(switching.transition_matrix, [[1.0]])
    assert switching.variances.min() >= 1e-6


def test_next_observation_weights_use_preobservation_filter():
    filtered = np.array([0.8, 0.2])
    transition = np.array([[0.7, 0.3], [0.1, 0.9]])

    weights = next_observation_weights(filtered, transition)

    assert np.allclose(weights, [0.58, 0.42])
    assert np.isclose(weights.sum(), 1.0)


def test_switching_var_transition_rows_and_filtered_shapes():
    data = np.vstack([_ar_data(0.25, 80), _ar_data(0.9, 80)])
    fit = fit_switching_var(
        build_lag_design(data, n_lags=1),
        n_states=2,
        lambda2=1e-3,
        max_iter=30,
        seed=3,
    )

    assert fit.filtered_probs.shape == (data.shape[0] - 1, 2)
    assert fit.smoothed_probs.shape == fit.filtered_probs.shape
    assert np.allclose(fit.transition_matrix.sum(axis=1), 1.0)
    assert np.isfinite(fit.log_likelihood)


def test_switching_var_connectivity_exports_filtered_endpoint_graphs():
    estimator = ConnectivityFactory(
        ConnectivityConfig(
            name="switching_var",
            max_lag=1,
            ridge=1e-3,
            extra={"n_lags": 1, "n_states": 1, "lambda2": 1e-3, "max_iter": 10},
        )
    )

    series = estimator.run(_windows())

    assert series.matrices.shape == (2, 2, 2)
    assert series.lagged_matrices is not None
    assert series.lagged_matrices.shape == (2, 1, 2, 2)
    assert series.diagnostics["prospective_api"] == "filtered_only_no_smoothed_posterior"
