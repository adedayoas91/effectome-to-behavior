"""Numerical parity of batched timing calculations with the mechanism pipeline."""

import numpy as np
import pytest
from scipy.special import softmax

from effectome.experiments.behavior_readout import FrozenBehaviorMap, FrozenDenseLayer
from effectome.experiments.behavior_rollout_timing_kernels import fit_timing_mechanisms, timing_forecasts
from effectome.experiments.behavior_transition_graphs import fit_transition_mechanism
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig
from effectome.workflows.effectome_rollout import forecast_mechanisms


def _support() -> np.ndarray:
    return np.array([[False, False, True], [True, False, True], [False, False, False]])


def _readout() -> FrozenBehaviorMap:
    return FrozenBehaviorMap(
        "timing_test",
        np.array([0.2, 1.0, -0.1]),
        np.array([2.0, 0.4, 1.0]),
        (FrozenDenseLayer(np.array([[0.3, -0.7, 0.8], [0.0, 1.0, 0.2]]), np.zeros(2), "tanh"),),
    )


@pytest.mark.parametrize("length", [3, 256, 1024])
def test_batched_fits_match_independent_mechanisms_including_constant_columns(length):
    histories = np.random.default_rng(140).normal(size=(6, length, 3))
    histories[1, :, 0] = 0.6
    histories[4, :, 1] = 0.0
    histories[5, :, :] = [1.0, 2.0, 3.0]
    support = _support()
    models = fit_timing_mechanisms(histories, support, 0.8)

    for model, history in zip(models, histories, strict=True):
        expected = fit_transition_mechanism(history[None], support, 0.8)
        assert model.metadata == expected.metadata
        np.testing.assert_array_equal(model.support, expected.support)
        for key in ("intercept", "self_weights", "cross_weights", "noise_scale"):
            np.testing.assert_allclose(getattr(model, key), getattr(expected, key), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(models[-1].noise_scale, 1e-6)


def test_factual_timing_rollouts_share_origin_rng_across_models_and_batch_order():
    rng = np.random.default_rng(44)
    histories = rng.normal(size=(6, 55, 3))
    fitted = fit_timing_mechanisms(histories, _support(), 1.0)
    models = (fitted[0], fitted[1], fitted[2], fitted[3], fitted[1], fitted[5])
    initial = histories[:, -1]
    ids = np.array([99, 6, 48, 2, 40, 12])
    config = MechanismRolloutConfig(horizon=5, target_length=2, n_rollouts=41, seed=93)
    readout = _readout()
    expected = forecast_mechanisms(models, initial, ids, readout, {"candidate": (0,)}, config).factual
    forecast = timing_forecasts(models, initial, ids, readout, config)
    np.testing.assert_allclose(forecast, expected, rtol=0, atol=2e-15)

    batch = np.array([3, 4, 0])
    subset = timing_forecasts(tuple(models[i] for i in batch), initial[batch], ids[batch], readout, config)
    np.testing.assert_allclose(subset, forecast[batch], rtol=0, atol=2e-15)


def test_timing_inputs_preserve_window_and_origin_contracts():
    histories = np.ones((2, 3, 3))
    with pytest.raises(ValueError, match="ridge_alpha"):
        fit_timing_mechanisms(histories, _support(), 0)
    with pytest.raises(ValueError, match="at least three"):
        fit_timing_mechanisms(histories[:, :2], _support(), 1)
    with pytest.raises(ValueError, match="histories must be finite"):
        fit_timing_mechanisms(histories * np.nan, _support(), 1)

    config = MechanismRolloutConfig(n_rollouts=1)
    fitted = fit_timing_mechanisms(histories, _support(), 1)
    with pytest.raises(ValueError, match="distinct nonnegative integer"):
        timing_forecasts(fitted, histories[:, -1], np.array([4, 4]), _readout(), config)
    with pytest.raises(ValueError, match="match the observed node axis"):
        timing_forecasts(fitted, histories[:, -1, :2], np.array([4, 5]), _readout(), config)


def test_categorical_disconnect_matches_source_row_deletion_with_paired_innovations():
    class MotifProbabilities:
        def predict(self, states: np.ndarray) -> np.ndarray:
            return softmax(states, axis=-1)

        def to_dict(self) -> dict[str, str]:
            return {"type": "test_softmax"}

    histories = np.random.default_rng(777).normal(size=(3, 45, 3))
    models = fit_timing_mechanisms(histories, _support(), 1.0)
    cut_models = []
    for model in models:
        cross = model.cross_weights.copy()
        cross[1, :] = 0.0
        cut_models.append(AdditiveMechanism(model.intercept, model.self_weights, cross,
                                            model.noise_scale, model.support))
    config = MechanismRolloutConfig(horizon=2, target_length=1, n_rollouts=12, seed=29)
    states, ids, readout = histories[:, -1], np.array([10, 15, 20]), MotifProbabilities()
    cut = timing_forecasts(models, states, ids, readout, config, output_width=3, disconnected=(1,))
    expected = timing_forecasts(cut_models, states, ids, readout, config, output_width=3)
    factual = timing_forecasts(models, states, ids, readout, config, output_width=3)
    np.testing.assert_array_equal(cut, expected)
    np.testing.assert_allclose(cut.sum(axis=1), 1.0)
    assert not np.allclose(cut, factual)
