import numpy as np
import pytest

from effectome.experiments.behavior_readout import fixed_behavior_readouts
from effectome.experiments.behavior_transition_controls import (
    TransitionControlConfig,
    make_transition_control,
)
from effectome.experiments.continuous_behavior_rollouts import paired_behavior_effects
from effectome.experiments.distributed_behavior_controls import READOUT_NODES, distributed_behavior
from effectome.experiments.structural_mechanism import MechanismRolloutConfig


def test_recurrent_trial_schedule_and_exact_mechanism_change():
    config = TransitionControlConfig(n_trials=8, history_lengths=(16, 64), washout=12)
    data = make_transition_control(config, scenario="driver_switch", seed=10)
    assert data.neural.shape == (8, data.neural.shape[1], 10)
    assert data.origins.shape == data.regime_starts.shape == data.regimes.shape == (8, 3)
    np.testing.assert_array_equal(data.regimes[:, 0], data.regimes[:, 2])
    np.testing.assert_array_equal(data.regimes[:, 0], 1 - data.regimes[:, 1])
    assert data.regimes[:, 0].sum() == 4
    assert np.ptp(data.regime_starts[:, 1]) > 0
    assert np.all(data.origins - max(config.history_lengths) + 1 >= data.regime_starts + config.washout)
    assert np.all(data.origins[:, :-1] + config.horizon < data.regime_starts[:, 1:])
    assert np.all(data.origins[:, -1] + config.horizon < data.neural.shape[1])
    np.testing.assert_array_equal(data.behavior, distributed_behavior(data.neural))
    for array in (data.neural, data.behavior, data.regimes, data.origins, data.regime_starts):
        assert not array.flags.writeable

    noise_seed = np.random.SeedSequence(10).spawn(2)[1]
    noise = np.random.default_rng(noise_seed).normal(size=data.neural.shape)
    for trial in range(config.n_trials):
        switch = data.regime_starts[trial, 1]
        state = data.neural[trial, switch - 1]
        active = data.mechanisms[data.regimes[trial, 1]]
        previous = data.mechanisms[data.regimes[trial, 0]]
        np.testing.assert_allclose(data.neural[trial, switch], active.step(state, noise[trial, switch]))
        assert not np.allclose(data.neural[trial, switch], previous.step(state, noise[trial, switch]))


def test_controls_share_noise_schedule_and_exact_behavior_under_nuisance_switch():
    config = TransitionControlConfig(n_trials=8, history_lengths=(24,))
    constant = make_transition_control(config, scenario="constant", seed=20)
    nuisance = make_transition_control(config, scenario="nuisance_switch", seed=20)
    switching = make_transition_control(config, scenario="driver_switch", seed=20)
    again = make_transition_control(config, scenario="driver_switch", seed=20)
    for data in (nuisance, switching, again):
        np.testing.assert_array_equal(data.origins, constant.origins)
        np.testing.assert_array_equal(data.regimes, constant.regimes)
        np.testing.assert_array_equal(data.neural[:, :, :2], constant.neural[:, :, :2])
    np.testing.assert_array_equal(nuisance.behavior, constant.behavior)
    assert not np.array_equal(nuisance.neural[:, :, 6:], constant.neural[:, :, 6:])
    assert not np.array_equal(nuisance.mechanisms[0].support, nuisance.mechanisms[1].support)
    assert np.array_equal(constant.mechanisms[0].support, constant.mechanisms[1].support)
    assert not np.array_equal(switching.behavior, constant.behavior)
    np.testing.assert_array_equal(switching.neural, again.neural)


def test_true_outgoing_disconnection_effect_depends_on_active_driver():
    control = make_transition_control(
        TransitionControlConfig(n_trials=4, history_lengths=(16,)), scenario="driver_switch", seed=8
    )
    initial = np.zeros((1, 10))
    initial[:, :2] = 0.75
    interventions = {"driver_0": [0], "driver_1": [1], "driver_pair": [0, 1], "nuisance": [6, 7]}
    reference = fixed_behavior_readouts()["behavior_reference"]
    for regime, mechanism in enumerate(control.mechanisms):
        assert mechanism.support[regime, READOUT_NODES].all()
        assert not mechanism.support[1 - regime, READOUT_NODES].any()
        result = paired_behavior_effects(
            mechanism,
            initial,
            interventions,
            readouts={"behavior_reference": reference},
            config=MechanismRolloutConfig(n_rollouts=8),
        )
        effects = result["readouts"]["behavior_reference"]["effect_vectors"]
        active_effect = effects[f"driver_{regime}"]
        assert active_effect[0, 0] > 0.1
        np.testing.assert_array_equal(active_effect, effects["driver_pair"])
        np.testing.assert_array_equal(effects[f"driver_{1 - regime}"], np.zeros((1, 2)))
        np.testing.assert_array_equal(effects["nuisance"], np.zeros((1, 2)))


def test_invalid_timing_and_scenario_are_rejected():
    with pytest.raises(ValueError, match="history_lengths"):
        TransitionControlConfig(history_lengths=(32, 32))
    with pytest.raises(ValueError, match="forecast horizon"):
        TransitionControlConfig(horizon=3, target_length=4)
    with pytest.raises(ValueError, match="unknown transition scenario"):
        make_transition_control(TransitionControlConfig(), scenario="unexpected", seed=23)
