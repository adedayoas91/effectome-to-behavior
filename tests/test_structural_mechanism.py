"""Mechanism interventions retain state evolution and training-trial ownership."""

from dataclasses import replace

import numpy as np
import pytest

from effectome.experiments.structural_mechanism import (
    AdditiveMechanism,
    MechanismFitConfig,
    MechanismRolloutConfig,
    fit_additive_mechanism,
    paired_intervention_probabilities,
)


def _model(noise=0.08):
    weights = np.zeros((3, 3))
    weights[0, 2], weights[1, 2] = 0.75, -0.55
    return AdditiveMechanism(np.array([0.1, -0.05, 0.02]), np.array([0.65, 0.6, 0.4]),
                             weights, np.full(3, noise), weights != 0)


def _trials(model, n_trials=28, length=60, seed=7):
    rng = np.random.default_rng(seed)
    values = np.empty((n_trials, length, len(model.intercept)))
    values[:, 0] = rng.normal(size=values[:, 0].shape)
    for index in range(1, length):
        # Generate independently of AdditiveMechanism.step to exercise its fit.
        previous = values[:, index - 1]
        values[:, index] = (model.intercept + previous * model.self_weights
                            + np.tanh(previous) @ model.cross_weights
                            + rng.normal(size=previous.shape) * model.noise_scale)
    return values


def test_outgoing_disconnection_retains_self_intercept_and_source_evolution():
    model = _model(noise=0)
    states = np.array([[2.0, -1.0, 0.5]])
    factual = model.mean(states)
    changed = model.mean(states, disconnected=[0])
    np.testing.assert_array_equal(changed[:, :2], factual[:, :2])
    np.testing.assert_allclose(changed[:, 2], 0.02 + 0.4 * states[:, 2] - 0.55 * np.tanh(states[:, 1]))
    np.testing.assert_allclose(factual[:, 2] - changed[:, 2], 0.75 * np.tanh(states[:, 0]))
    np.testing.assert_allclose(model.mean(states, [0, 1]), model.intercept + states * model.self_weights)
    np.testing.assert_array_equal(states, [[2, -1, 0.5]])
    np.testing.assert_array_equal(model.cross_weights, _model(noise=0).cross_weights)


def test_fitting_recovers_raw_parameters_and_whole_trial_provenance():
    truth = _model()
    trials = _trials(truth)
    fitted = fit_additive_mechanism(trials, truth.support)
    np.testing.assert_allclose(fitted.intercept, truth.intercept, atol=0.025)
    np.testing.assert_allclose(fitted.self_weights, truth.self_weights, atol=0.04)
    np.testing.assert_allclose(fitted.cross_weights, truth.cross_weights, atol=0.07)
    np.testing.assert_allclose(fitted.noise_scale, truth.noise_scale, atol=0.015)
    assert fitted.metadata["training_pair_count"] == 28 * 59
    assert fitted.metadata["inner_training_trials"] == list(range(21))
    assert fitted.metadata["inner_validation_trials"] == list(range(21, 28))
    assert fitted.metadata["noise_source"] == "inner_validation_residual_rms"
    assert fitted.metadata["selected_alpha"] in MechanismFitConfig().ridge_alphas
    assert fitted.metadata["parameter_units"] == "original_observed_state_units"
    with pytest.raises(ValueError, match="read-only"):
        fitted.cross_weights[0, 2] = 99


def test_fit_never_manufactures_pairs_between_trials():
    levels = np.array([100, -100, 50, -50, 75, -75], dtype=float)
    trials = np.repeat(levels[:, None, None], 6, axis=1)
    fitted = fit_additive_mechanism(trials, np.zeros((1, 1), dtype=bool),
                                    MechanismFitConfig(ridge_alphas=(1e-8,)))
    assert fitted.metadata["training_pair_count"] == 30
    assert fitted.metadata["trial_boundary_pairs"] is False
    np.testing.assert_allclose(fitted.self_weights, [1], atol=1e-8)
    np.testing.assert_allclose(fitted.intercept, [0], atol=1e-8)


def test_noise_uses_inner_validation_predictions_before_the_final_refit():
    trials = np.zeros((8, 5, 1))
    trials[6:] = 4
    fitted = fit_additive_mechanism(trials, np.zeros((1, 1), dtype=bool))
    # Inner training predicts zero in held-out trials; final training can fit the
    # level-preserving process almost exactly. Its optimistic residuals are not used.
    np.testing.assert_allclose(fitted.noise_scale, [4])
    assert fitted.metadata["validation_mse_by_alpha"] == [16] * 4
    assert fitted.metadata["inner_validation_trials"] == [6, 7]


def test_fit_copies_training_inputs_and_caller_parameters():
    truth = _model()
    trials = _trials(truth)
    before = trials.copy()
    first = fit_additive_mechanism(trials, truth.support)
    second = fit_additive_mechanism(trials.copy(), truth.support.copy())
    np.testing.assert_array_equal(trials, before)
    np.testing.assert_array_equal(first.cross_weights, second.cross_weights)
    assert first.metadata == second.metadata


def test_paired_rollouts_detect_active_sources_and_exact_zero_absent_outgoing_effects():
    truth = _model(noise=0.1)
    initial = np.array([[2.0, 0, -0.2], [0, 2.0, 0.2]])
    units = {"n0": [0], "n1": [1], "endpoint": [2], "pair": [0, 1]}
    config = MechanismRolloutConfig(horizon=4, target_length=2, n_rollouts=512, seed=9)
    result = paired_intervention_probabilities(truth, initial, units, endpoint_index=2, config=config)
    assert result["probability_reductions"]["n0"][0] > 0.2
    assert result["probability_reductions"]["n1"][1] > 0.2
    np.testing.assert_array_equal(result["probability_reductions"]["endpoint"], [0, 0])
    np.testing.assert_array_equal(result["disconnected_transition_probabilities"]["endpoint"],
                                  result["factual_transition_probability"])
    removed = truth.cross_weights.copy()
    removed[[0, 1]] = 0
    without_pair = AdditiveMechanism(truth.intercept, truth.self_weights, removed, truth.noise_scale,
                                    truth.support)
    independent = paired_intervention_probabilities(
        without_pair, initial, {}, endpoint_index=2, config=config
    )
    np.testing.assert_array_equal(result["disconnected_transition_probabilities"]["pair"],
                                  independent["factual_transition_probability"])


def test_rollouts_are_reproducible_and_invariant_to_unit_and_origin_order():
    model = _model()
    initial = np.array([[2.0, 0, -0.2], [0, 2.0, 0.2], [1, 1, 0.1]])
    units = {"n0": [0], "n1": [1]}
    ids = np.array([100, 4, 78])
    config = MechanismRolloutConfig(n_rollouts=96, seed=13)
    arguments = {"endpoint_index": 2, "config": config, "origin_ids": ids}
    first = paired_intervention_probabilities(model, initial, units, **arguments)
    second = paired_intervention_probabilities(model, initial, units, **arguments)
    order = np.array([2, 0, 1])
    permuted = paired_intervention_probabilities(model, initial[order], dict(reversed(list(units.items()))),
                                                endpoint_index=2, config=config, origin_ids=ids[order])
    np.testing.assert_array_equal(first["factual_transition_probability"],
                                  second["factual_transition_probability"])
    np.testing.assert_array_equal(permuted["source_states"], first["source_states"][order])
    for unit in units:
        np.testing.assert_array_equal(first["probability_reductions"][unit],
                                      second["probability_reductions"][unit])
        np.testing.assert_array_equal(permuted["probability_reductions"][unit],
                                      first["probability_reductions"][unit][order])


def test_future_target_uses_the_declared_window_and_excludes_the_origin():
    model = AdditiveMechanism(np.zeros(1), np.array([-0.5]), np.zeros((1, 1)), np.zeros(1),
                              np.zeros((1, 1), dtype=bool))
    initial = np.ones((1, 1))
    # Exact future is [-.5, .25]. Its two-sample mean is negative, whereas the
    # endpoint alone is positive. Including the origin would also change the event.
    window = paired_intervention_probabilities(
        model, initial, {"self": [0]}, endpoint_index=0,
        config=MechanismRolloutConfig(horizon=2, target_length=2, n_rollouts=1),
    )
    endpoint = paired_intervention_probabilities(
        model, initial, {}, endpoint_index=0,
        config=MechanismRolloutConfig(horizon=2, target_length=1, n_rollouts=1),
    )
    np.testing.assert_array_equal(window["factual_transition_probability"], [1])
    np.testing.assert_array_equal(endpoint["factual_transition_probability"], [0])
    np.testing.assert_array_equal(window["probability_reductions"]["self"], [0])


def test_known_support_learned_effects_approximate_independently_generated_control():
    truth = _model(noise=0.1)
    fitted = fit_additive_mechanism(_trials(truth, n_trials=48, length=80), truth.support)
    initial = np.array([[2, 0, -0.2], [0, 2, 0.2], [0.5, 0, -0.1], [0, 0.5, 0.1]])
    units = {"n0": [0], "n1": [1], "inert": [2]}
    config = MechanismRolloutConfig(horizon=4, target_length=2, n_rollouts=2048, seed=19)
    expected = paired_intervention_probabilities(truth, initial, units, endpoint_index=2, config=config)
    actual = paired_intervention_probabilities(fitted, initial, units, endpoint_index=2, config=config)
    for unit in units:
        np.testing.assert_allclose(actual["probability_reductions"][unit],
                                   expected["probability_reductions"][unit], atol=0.07)
    assert actual["probability_reductions"]["n0"][0] > 0.2
    assert actual["probability_reductions"]["n1"][1] > 0.2
    np.testing.assert_array_equal(actual["probability_reductions"]["inert"], np.zeros(4))


@pytest.mark.parametrize("kwargs", [{"ridge_alphas": ()}, {"ridge_alphas": (0,)},
                                    {"ridge_alphas": (1, 1)}, {"validation_fraction": 1},
                                    {"validation_fraction": np.nan}, {"noise_floor": -1}])
def test_invalid_fit_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MechanismFitConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"horizon": True}, {"target_length": 9},
                                    {"n_rollouts": -1}, {"seed": -1}, {"seed": True}])
def test_invalid_rollout_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MechanismRolloutConfig(**kwargs)


def test_invalid_model_parameters_support_and_nonfinite_updates_are_rejected():
    model = _model()
    for changes in ({"support": np.eye(3, dtype=bool)}, {"support": np.zeros((3, 3))},
                    {"noise_scale": np.array([-1, 0, 0])}, {"intercept": np.array([np.nan, 0, 0])},
                    {"self_weights": np.zeros(2)}, {"cross_weights": np.ones((3, 3))}):
        with pytest.raises(ValueError):
            replace(model, **changes)
    for states in (np.array([np.nan, 0, 0]), np.zeros((2, 4)), np.zeros((0, 3))):
        with pytest.raises(ValueError):
            model.mean(states)
    for indices in ([3], [-1], [True], [0, 0], "0"):
        with pytest.raises(ValueError):
            model.mean(np.zeros(3), indices)
    with pytest.raises(ValueError, match="innovations"):
        model.step(np.zeros((2, 3)), np.zeros(3))
    huge = replace(model, self_weights=np.full(3, 1e308))
    with pytest.raises(ValueError, match="nonfinite"):
        huge.mean(np.full(3, 1e308))


def test_invalid_training_arrays_and_rollout_ownership_are_rejected():
    model = _model()
    for values in (np.zeros((4, 3)), np.zeros((1, 5, 3)), np.zeros((4, 1, 3)),
                   np.full((4, 5, 3), np.nan)):
        with pytest.raises(ValueError):
            fit_additive_mechanism(values, model.support)
    initial = np.zeros((2, 3))
    for options in ({"origin_ids": np.array([1, 1])}, {"origin_ids": np.array([-1, 2])},
                    {"origin_ids": np.array([1.0, 2.0])}, {"threshold": np.inf},
                    {"endpoint_index": 3}, {"endpoint_index": True}):
        with pytest.raises(ValueError):
            paired_intervention_probabilities(model, initial, {}, **{"endpoint_index": 2, **options})
    for units in ({"": [0]}, {"__factual__": [0]}, {"empty": []}, {"bad": [True]}):
        with pytest.raises(ValueError):
            paired_intervention_probabilities(model, initial, units, endpoint_index=2)
