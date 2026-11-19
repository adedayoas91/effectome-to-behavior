from dataclasses import replace

import numpy as np
import pytest

from effectome.experiments.distributed_behavior_controls import (
    DRIVER_NODES,
    NUISANCE_NODES,
    READOUT_NODES,
    DistributedBehaviorConfig,
    distributed_behavior,
    distributed_interventions,
    distributed_readout_matrix,
    make_distributed_behavior_control,
)
from effectome.experiments.structural_mechanism import AdditiveMechanism, fit_additive_mechanism


def _target_means(model: AdditiveMechanism, initial: np.ndarray, *, seed: int = 9, rollouts: int = 64):
    """Use common innovations to compare every declared neural disconnection."""
    noise = np.random.default_rng(seed).normal(size=(8, len(initial), rollouts, 10))
    values = {}
    neural_means = {}
    for name, sources in {"__factual__": [], **distributed_interventions()}.items():
        states = np.repeat(initial[:, None, :], rollouts, axis=1)
        target = []
        for step in range(8):
            states = model.step(states, noise[step], sources)
            if step >= 4:
                target.append(states)
        targets = np.stack(target)
        values[name] = distributed_behavior(targets).mean(axis=0)
        neural_means[name] = targets.mean(axis=0)
    return values, neural_means


def test_control_shapes_metadata_and_readonly_arrays():
    cfg = DistributedBehaviorConfig(n_trials=8, trial_length=64)
    data = make_distributed_behavior_control(cfg)
    assert data.neural_trials.shape == (8, 64, 10)
    assert data.behavior_trials.shape == (8, 64, 2)
    assert np.isfinite(data.neural_trials).all()
    np.testing.assert_array_equal(data.behavior_trials, distributed_behavior(data.neural_trials))
    assert np.max(np.abs(data.behavior_trials)) < 1
    assert np.linalg.matrix_rank(data.behavior_trials.reshape(-1, 2)) == 2
    assert data.metadata["roles"] == {"drivers": [0, 1], "readout": [2, 3, 4, 5], "nuisance": [6, 7, 8, 9]}
    assert data.metadata["valid_ranges"] == [(64 * i, 64 * (i + 1)) for i in range(8)]
    assert data.metadata["independent_trials"]
    assert not data.metadata["trial_boundary_state_carry"]
    assert data.metadata["null_design"]["nuisance_has_no_path_to_readout"]
    assert data.metadata["null_design"]["driver_nuisance_path_retained"]
    for values in (data.neural_trials, data.behavior_trials, data.mechanism.cross_weights):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 123


def test_generation_is_deterministic_and_scenarios_share_initial_states():
    cfg = DistributedBehaviorConfig(n_trials=8, trial_length=48)
    first = make_distributed_behavior_control(cfg)
    repeated = make_distributed_behavior_control(cfg)
    null = make_distributed_behavior_control(replace(cfg, scenario="null"))
    another = make_distributed_behavior_control(replace(cfg, seed=cfg.seed + 1))
    np.testing.assert_array_equal(first.neural_trials, repeated.neural_trials)
    np.testing.assert_array_equal(first.behavior_trials, repeated.behavior_trials)
    np.testing.assert_array_equal(first.neural_trials[:, 0], null.neural_trials[:, 0])
    np.testing.assert_array_equal(first.neural_trials[:, :, :2], null.neural_trials[:, :, :2])
    assert not np.array_equal(first.neural_trials, another.neural_trials)
    assert not np.array_equal(first.behavior_trials, null.behavior_trials)


@pytest.mark.parametrize("node", READOUT_NODES)
def test_both_behavior_outputs_use_every_prespecified_readout_node(node):
    point = np.zeros(10)
    point[node] = 0.4
    assert np.all(distributed_behavior(point) != 0)


@pytest.mark.parametrize("node", (*DRIVER_NODES, *NUISANCE_NODES))
def test_behavior_does_not_directly_read_driver_or_nuisance_coordinates(node):
    point = np.zeros(10)
    point[node] = 100
    np.testing.assert_array_equal(distributed_behavior(point), np.zeros(2))


def test_behavior_readout_preserves_arbitrary_leading_axes():
    states = np.random.default_rng(3).normal(size=(2, 3, 4, 10))
    assert distributed_behavior(states).shape == (2, 3, 4, 2)
    np.testing.assert_array_equal(distributed_behavior(states)[1, 2], distributed_behavior(states[1, 2]))


def test_exported_readout_is_fixed_and_cannot_change_the_generator():
    matrix = distributed_readout_matrix()
    assert matrix.shape == (2, 10)
    assert not matrix.flags.writeable
    np.testing.assert_array_equal(matrix[:, READOUT_NODES], [[0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5]])
    assert not matrix[:, (*DRIVER_NODES, *NUISANCE_NODES)].any()
    matrix.setflags(write=True)
    matrix[:] = 100
    np.testing.assert_array_equal(distributed_behavior(np.ones(10)), np.tanh(np.array([2.0, 0.0])))


@pytest.mark.parametrize(
    "states", [np.zeros(9), np.zeros((0, 10)), np.array(1), np.full((2, 10), np.nan), np.full(10, np.inf)]
)
def test_invalid_behavior_observations_rejected(states):
    with pytest.raises(ValueError):
        distributed_behavior(states)


def test_predeclared_units_include_individuals_and_fixed_groups_without_aliasing():
    units = distributed_interventions()
    assert list(units)[:10] == [f"neuron:{i}" for i in range(10)]
    assert units["group:driver_pair"] == [0, 1]
    assert units["group:nuisance_block"] == [6, 7, 8, 9]
    units["group:driver_pair"].append(5)
    assert distributed_interventions()["group:driver_pair"] == [0, 1]


def test_null_has_exactly_zero_behavioral_effects_with_responsive_nuisance():
    data = make_distributed_behavior_control(
        DistributedBehaviorConfig(scenario="null", n_trials=8, trial_length=64)
    )
    assert not data.mechanism.support[:, READOUT_NODES].any()
    assert data.metadata["behavioral_null"]
    assert data.metadata["null_design"]["all_readout_incoming_cross_terms_removed"]
    targets, neural = _target_means(data.mechanism, data.neural_trials[:, 0])
    for name in distributed_interventions():
        np.testing.assert_array_equal(targets[name], targets["__factual__"])
    nuisance_change = np.abs(neural["__factual__"][:, :, 6:] - neural["group:driver_pair"][:, :, 6:])
    assert nuisance_change.mean() > 0.05
    own_nuisance_change = np.abs(neural["__factual__"][:, :, 8:] - neural["group:nuisance_block"][:, :, 8:])
    assert own_nuisance_change.mean() > 0.01


def test_positive_driver_and_group_effects_without_nuisance_behavior_paths():
    data = make_distributed_behavior_control(DistributedBehaviorConfig(n_trials=8, trial_length=64))
    targets, _ = _target_means(data.mechanism, data.neural_trials[:, 0], rollouts=128)
    factual = targets["__factual__"].mean(axis=1)
    for name in ("neuron:0", "neuron:1", "group:driver_pair"):
        effect = factual - targets[name].mean(axis=1)
        assert np.mean(np.linalg.norm(effect, axis=-1)) > 0.05
    for node in (*READOUT_NODES, *NUISANCE_NODES):
        np.testing.assert_array_equal(targets[f"neuron:{node}"], targets["__factual__"])
    np.testing.assert_array_equal(targets["group:nuisance_block"], targets["__factual__"])


def test_known_support_mechanism_uses_within_trial_pairs_and_approximates_effects():
    data = make_distributed_behavior_control(DistributedBehaviorConfig(trial_length=512))
    fitted = fit_additive_mechanism(data.neural_trials[:12], data.mechanism.support)
    assert fitted.metadata["training_pair_count"] == 12 * 511
    assert not fitted.metadata["trial_boundary_pairs"]
    evaluation = data.neural_trials[12:]
    assert np.mean((fitted.mean(evaluation[:, :-1]) - data.mechanism.mean(evaluation[:, :-1])) ** 2) < 0.0001
    truth, _ = _target_means(data.mechanism, evaluation[:, 0])
    estimated, _ = _target_means(fitted, evaluation[:, 0])
    for name in ("neuron:0", "neuron:1", "group:driver_pair"):
        true_effect = (truth["__factual__"] - truth[name]).mean(axis=1)
        estimated_effect = (estimated["__factual__"] - estimated[name]).mean(axis=1)
        assert np.mean(np.abs(estimated_effect - true_effect)) < 0.025


@pytest.mark.parametrize(
    "values",
    [
        {"scenario": "missing"},
        {"seed": -1},
        {"seed": True},
        {"n_trials": 7},
        {"n_trials": 8.0},
        {"trial_length": 2},
        {"n_neurons": 9},
        {"n_neurons": 11},
        {"fps": 0},
        {"driver_ar": 1},
        {"readout_ar": -0.1},
        {"nuisance_ar": float("nan")},
        {"coupling": 0},
        {"nuisance_coupling": 0},
        {"noise_std": 0},
        {"noise_std": True},
        {"pulse_amplitude": -1},
        {"initial_readout": 0},
        {"pulse_probability": 0},
        {"pulse_probability": 1},
        {"initial_readout": float("inf")},
    ],
)
def test_invalid_design_rejected(values):
    with pytest.raises(ValueError):
        DistributedBehaviorConfig(**values)
