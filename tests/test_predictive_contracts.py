"""Clock, ownership and numerical regression tests for fitted forecasts."""

import pickle
from dataclasses import replace

import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig, UnsupportedForecastError
from effectome.connectivity.sparse_var import fit_sparse_var_model
from effectome.connectivity.var_utils import LagDesign, fit_sparse_var


def _model(**kwargs):
    return fit_sparse_var_model(
        np.random.default_rng(12).normal(size=(50, 2)),
        cfg=ConnectivityConfig(name="sparse_var", max_lag=2),
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["nan", "mask", "gap"])
def test_missing_history_never_compresses_time(kind):
    model = _model()
    samples = np.ones((3, 2))
    masks = np.ones_like(samples, dtype=bool)
    ids = np.arange(3)
    if kind == "nan":
        samples[1] = np.nan
    elif kind == "mask":
        masks[1] = False
    else:
        ids = np.array([0, 1, 5])
    state = model.observe(model.initial_state(), samples, sample_indices=ids, masks=masks)
    assert model.forecast(state, horizon_samples=1).failure_reason == "insufficient_history"
    assert state.history.shape[0] == 1


def test_chunk_roundtrip_exclusive_origin_and_support():
    model = _model()
    samples = np.arange(16).reshape(8, 2)
    initial = model.initial_state(recording_identity="r")
    whole = model.observe(initial, samples, sample_indices=np.arange(10, 18))
    chunk = model.observe(initial, samples[:3], sample_indices=np.arange(10, 13))
    chunk = pickle.loads(pickle.dumps(chunk))
    chunk = model.observe(chunk, samples[3:], sample_indices=np.arange(13, 18))
    forecast = model.forecast(chunk, horizon_samples=2)
    assert forecast.origin_sample == 18
    assert forecast.raw_support["history_interval"] == (16, 18)
    assert forecast.raw_support["recording_identity"] == "r"
    assert forecast.state_hash == model.forecast(whole, horizon_samples=2).state_hash
    np.testing.assert_array_equal(forecast.mean, model.forecast(whole, horizon_samples=2).mean)


def test_observation_rejects_replay_wrong_model_and_recording():
    model = _model()
    state = model.observe(model.initial_state(recording_identity="r"), np.ones((2, 2)),
                          sample_indices=np.array([5, 6]))
    for ids in (np.array([6, 7]), np.array([8, 7]), np.array([7.5, 8.5])):
        with pytest.raises(ValueError):
            model.observe(state, np.ones((2, 2)), sample_indices=ids)
    with pytest.raises(ValueError, match="model"):
        model.forecast(replace(state, model_hash="wrong"), horizon_samples=1)
    with pytest.raises(ValueError, match="recording"):
        model.observe(state, np.ones((1, 2)), sample_indices=np.array([7]), recording_identity="other")


def test_inputs_are_explicitly_unsupported():
    model = _model()
    with pytest.raises(UnsupportedForecastError):
        _model(available_inputs=np.ones((50, 1)))
    with pytest.raises(UnsupportedForecastError):
        model.observe(model.initial_state(), np.ones((2, 2)), available_inputs=np.ones((2, 1)))
    with pytest.raises(UnsupportedForecastError):
        model.forecast(model.initial_state(), horizon_samples=1, known_future_inputs=np.ones((1, 1)))


def test_fit_owned_segments_and_duplicate_rows():
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(16, 2))
    cfg = ConnectivityConfig(name="sparse_var", max_lag=2)
    model = fit_sparse_var_model(samples, cfg=cfg, sample_indices=np.tile(np.arange(8), 2),
                                 segment_ids=np.repeat(["a", "b"], 8))
    assert model.fit.valid_response_indices.tolist() == list(range(2, 8)) * 2
    duplicate = fit_sparse_var_model(np.concatenate([samples[:8], samples[:8]]), cfg=cfg,
                                     sample_indices=np.tile(np.arange(8), 2),
                                     segment_ids=np.repeat("a", 16))
    reference = fit_sparse_var_model(samples[:8], cfg=cfg)
    np.testing.assert_allclose(duplicate.fit.coefficients, reference.fit.coefficients)


def test_zero_penalty_singular_design_and_true_stationarity():
    x = np.ones((10, 4))
    y = np.ones((10, 2)) * 3
    fit = fit_sparse_var(LagDesign(x, y, np.arange(10), 2, 2), lambda1=0, lambda2=0)
    assert fit.converged
    assert fit.diagnostics["proximal_residual"] <= 1e-6
    np.testing.assert_allclose(fit.intercept, [3, 3])


def test_nonconverged_fit_cannot_issue_forecast_or_graph():
    cfg = ConnectivityConfig(name="sparse_var", max_lag=2,
                             extra={"lambda1": 0.1, "max_iter": 0})
    model = fit_sparse_var_model(np.random.default_rng(4).normal(size=(40, 2)), cfg=cfg)
    assert not model.fit.converged
    state = model.observe(model.initial_state(), np.ones((2, 2)))
    assert model.forecast(state, horizon_samples=1).failure_reason == "nonconverged_fit"
    with pytest.raises(ValueError, match="converg"):
        model.graph_at()
