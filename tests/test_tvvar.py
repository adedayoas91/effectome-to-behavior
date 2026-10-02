"""One-sided TVVAR regression tests for Appendix-C WP6."""

from __future__ import annotations

import numpy as np

from effectome.connectivity import CONNECTIVITY_REGISTRY, ConnectivityConfig, ConnectivityFactory
from effectome.connectivity.sparse_var import fit_sparse_var_model
from effectome.connectivity.tvvar import fit_tvvar_endpoint
from effectome.connectivity.var_utils import build_lag_design
from effectome.data_module.schema import TemporalAnchor, Window, WindowedSegments


def test_joint_buffer_normalization_and_stationarity():
    from effectome.connectivity.tvvar import fit_tvvar_buffer
    from effectome.connectivity.var_utils import LagDesign

    x = np.array([[1.0], [2.0], [3.0], [1.0], [2.0], [4.0]])
    y = np.array([[3.0], [5.0], [7.0], [0.0], [-1.0], [-3.0]])
    fit = fit_tvvar_buffer(LagDesign(x, y, np.arange(6), 1, 1),
                           knot_samples=3, lambda2=0.1, lambda_temporal=0.2,
                           lambda_intercept=0.3, tol=1e-8)
    a, b = fit.knot_coefficients[:, 0, 0], fit.knot_intercepts[:, 0]
    residual = x[:, 0] * a[fit.row_knots] + b[fit.row_knots] - y[:, 0]
    ga = np.array([2 * (x[:3, 0] @ residual[:3]) / 6,
                   2 * (x[3:, 0] @ residual[3:]) / 6]) + 0.1 * a
    ga += 0.4 * np.array([a[0] - a[1], a[1] - a[0]])
    gb = np.array([residual[:3].sum(), residual[3:].sum()]) / 3
    gb += 0.6 * np.array([b[0] - b[1], b[1] - b[0]])
    assert fit.converged
    assert max(np.max(abs(ga)), np.max(abs(gb))) < 1e-6


def test_tvvar_model_serialized_chunk_updates_match():
    import pickle

    from effectome.connectivity.tvvar import fit_tvvar_model

    rng = np.random.default_rng(90)
    data = rng.normal(size=(40, 2))
    cfg = ConnectivityConfig(name="tvvar_smooth", max_lag=1,
                             extra={"knot_samples": 8, "lambda2": 0.1})
    model = fit_tvvar_model(data[:20], cfg=cfg)
    state = model.initial_state()
    whole = model.observe(state, data[20:], sample_indices=np.arange(20, 40))
    partial = model.observe(state, data[20:30], sample_indices=np.arange(20, 30))
    model, partial = pickle.loads(pickle.dumps((model, partial)))
    chunk = model.observe(partial, data[30:], sample_indices=np.arange(30, 40))
    assert np.allclose(model.forecast(whole, horizon_samples=2).mean,
                       model.forecast(chunk, horizon_samples=2).mean)


def _ar_window(phi: float, *, n_samples: int = 80, offset: float = 0.0) -> np.ndarray:
    data = np.zeros((n_samples, 2), dtype=np.float64)
    data[0, 0] = 1.0 + offset
    data[0, 1] = -0.5
    for t in range(1, n_samples):
        data[t, 0] = phi * data[t - 1, 0] + 0.01 * np.sin(t)
        data[t, 1] = 0.25 * data[t - 1, 0] + 0.1 * data[t - 1, 1]
    return data


def _segments(windows: list[np.ndarray], *, gap_at: int | None = None) -> WindowedSegments:
    anchors = []
    window_objs = []
    for idx, window in enumerate(windows):
        start = idx * 100
        stop = start + window.shape[0]
        anchors.append(
            TemporalAnchor(
                dataset_id="ds",
                recording_id="rec-a" if gap_at is None or idx < gap_at else "rec-b",
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
                gap_before=gap_at is not None and idx == gap_at,
                gap_after=gap_at is not None and idx == gap_at - 1,
            )
        )
        window_objs.append(Window(start, stop))
    return WindowedSegments(
        segments=np.asarray(windows, dtype=np.float32),
        windows=window_objs,
        behavior_per_window={},
        n_neurons=windows[0].shape[1],
        fps=1.0,
        anchors=anchors,
    )


def test_tvvar_registry_entries_are_available():
    assert {"tvvar_smooth", "tvvar_fused"} <= set(CONNECTIVITY_REGISTRY)


def test_tvvar_one_knot_reduces_to_sparse_var_without_temporal_penalty():
    data = _ar_window(0.7)
    cfg = ConnectivityConfig(
        name="sparse_var",
        max_lag=1,
        ridge=1e-3,
        extra={"n_lags": 1, "lambda1": 0.0, "lambda2": 1e-3, "standardize": False},
    )
    sparse = fit_sparse_var_model(data, cfg=cfg).graph_at()
    design = build_lag_design(data, n_lags=1)

    tv = fit_tvvar_endpoint(
        design,
        lambda1=0.0,
        lambda2=1e-3,
        lambda_temporal=0.0,
        lambda_intercept=0.0,
        penalty="smooth",
        standardize=False,
    )

    assert np.allclose(tv.lagged_coefficients, sparse.lagged_coefficients)


def test_tvvar_objective_components_are_hand_checkable():
    samples = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    design = build_lag_design(samples, n_lags=1)

    fit = fit_tvvar_endpoint(
        design,
        previous_coefficients=np.array([[0.25]]),
        previous_intercept=np.array([0.1]),
        lambda1=0.0,
        lambda2=0.5,
        lambda_temporal=0.25,
        lambda_intercept=0.25,
        penalty="smooth",
        standardize=False,
    )

    components = fit.objective_components
    assert np.isclose(components["objective"], sum(
        components[key]
        for key in ["loss", "offdiag_l1", "ridge", "temporal", "intercept_temporal"]
    ))
    assert components["temporal"] >= 0.0
    assert components["intercept_temporal"] >= 0.0


def test_tvvar_smooth_future_tail_does_not_rewrite_issued_graphs():
    windows = [_ar_window(0.3), _ar_window(0.8), _ar_window(-0.2)]
    cfg = ConnectivityConfig(
        name="tvvar_smooth",
        max_lag=1,
        ridge=1e-3,
        extra={"n_lags": 1, "lambda_temporal": 1.0, "lambda2": 1e-3},
    )
    prefix = ConnectivityFactory(cfg).run(_segments(windows[:2]))
    full = ConnectivityFactory(cfg).run(_segments(windows))

    assert np.allclose(prefix.matrices, full.matrices[:2])
    assert np.allclose(prefix.lagged_matrices, full.lagged_matrices[:2])


def test_tvvar_resets_temporal_coupling_at_real_gaps():
    windows = [_ar_window(0.2), _ar_window(0.9), _ar_window(-0.5)]
    cfg = ConnectivityConfig(
        name="tvvar_smooth",
        max_lag=1,
        ridge=1e-3,
        extra={"n_lags": 1, "lambda_temporal": 100.0, "lambda2": 1e-3},
    )
    with_gap = ConnectivityFactory(cfg).run(_segments(windows, gap_at=2))
    last_only = ConnectivityFactory(cfg).run(_segments(windows[2:]))

    assert np.allclose(with_gap.matrices[2], last_only.matrices[0])
    assert with_gap.diagnostics["continuous_buffers"] == 2


def test_large_smooth_penalty_approaches_shared_endpoint_coefficients():
    windows = [_ar_window(0.2), _ar_window(0.9)]
    cfg = ConnectivityConfig(
        name="tvvar_smooth",
        max_lag=1,
        ridge=1e-3,
        extra={"n_lags": 1, "lambda_temporal": 1e6, "lambda2": 1e-3},
    )

    series = ConnectivityFactory(cfg).run(_segments(windows))

    assert np.allclose(series.lagged_matrices[0], series.lagged_matrices[1], atol=1e-3)


def test_tvvar_group_fused_shrinks_complete_coefficient_difference():
    windows = [_ar_window(0.2), _ar_window(0.9)]
    weak = ConnectivityFactory(
        ConnectivityConfig(
            name="tvvar_fused",
            max_lag=1,
            ridge=1e-3,
            extra={"n_lags": 1, "lambda_temporal": 0.0, "lambda2": 1e-3},
        )
    ).run(_segments(windows))
    strong = ConnectivityFactory(
        ConnectivityConfig(
            name="tvvar_fused",
            max_lag=1,
            ridge=1e-3,
            extra={"n_lags": 1, "lambda_temporal": 10.0, "lambda2": 1e-3},
        )
    ).run(_segments(windows))

    weak_diff = np.linalg.norm(weak.lagged_matrices[1] - weak.lagged_matrices[0])
    strong_diff = np.linalg.norm(strong.lagged_matrices[1] - strong.lagged_matrices[0])
    assert strong_diff < weak_diff
    assert strong.diagnostics["temporal_penalty"] == "group_fused"


def test_legacy_time_varying_smoother_is_marked_nonprospective():
    series = ConnectivityFactory(
        ConnectivityConfig(
            name="time_varying",
            max_lag=1,
            extra={"base_estimator": "correlation", "temporal_lambda": 1.0},
        )
    ).run(_segments([_ar_window(0.2), _ar_window(0.9)]))

    assert series.diagnostics["prospective_evaluation_support"] == (
        "unsupported_future_dependent_smoother"
    )
