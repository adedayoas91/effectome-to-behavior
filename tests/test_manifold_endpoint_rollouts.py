"""Learned endpoints share neural futures and retain behavioral meaning checks."""

from dataclasses import dataclass

import numpy as np
import pytest

from effectome.experiments.learned_endpoints import CoordinateEndpoint
from effectome.experiments.manifold_endpoint_rollouts import (
    paired_endpoint_probabilities,
    semantics_errors,
)
from effectome.experiments.structural_mechanism import (
    AdditiveMechanism,
    MechanismRolloutConfig,
    paired_intervention_probabilities,
)


@dataclass(frozen=True)
class Coordinate:
    index: int = 2
    threshold: float = 0.0
    reverse_source: bool = False

    def source_states(self, initial):
        sources = initial[:, self.index] >= self.threshold
        return ~sources if self.reverse_source else sources

    def target_states(self, target):
        # Sequential summation reproduces the existing coordinate readout exactly.
        total = np.zeros(target.shape[1:3])
        for states in target:
            total += states[..., self.index]
        return total / len(target) >= self.threshold

    def to_dict(self):
        return {"index": self.index, "threshold": self.threshold, "reverse_source": self.reverse_source}


class Constant:
    def source_states(self, initial):
        return np.zeros(len(initial), dtype=int)

    def target_states(self, target):
        return np.zeros(target.shape[1:3], dtype=int)

    def to_dict(self):
        return {"kind": "constant"}


def _model(noise=0.08, target=2):
    weights = np.zeros((3, 3))
    weights[0, target] = 0.9
    return AdditiveMechanism(np.zeros(3), np.array([0.8, 0.6, 0.4]), weights,
                             np.full(3, noise), weights != 0)


def test_coordinate_bridge_exactly_reproduces_existing_probabilities():
    model = _model()
    initial = np.array([[2, 1, -0.1], [-2, -1, 0.1], [1, 0, 0]])
    units = {"source": [0], "pair": [0, 1], "endpoint": [2]}
    config = MechanismRolloutConfig(horizon=8, target_length=4, n_rollouts=128, seed=17)
    arguments = {"config": config, "origin_ids": np.array([5, 99, 8])}
    old = paired_intervention_probabilities(model, initial, units, endpoint_index=2, **arguments)
    new = paired_endpoint_probabilities(model, initial, units,
                                        endpoints={"behavior_reference": CoordinateEndpoint(2)}, **arguments)
    summary = new["endpoints"]["behavior_reference"]
    for key in ("source_states", "factual_transition_probability"):
        np.testing.assert_array_equal(summary[key], old[key])
    for key in ("disconnected_transition_probabilities", "probability_reductions"):
        for unit in units:
            np.testing.assert_array_equal(summary[key][unit], old[key][unit])
    assert new["events"]["behavior_reference"]["source"].shape == (3, 128)


def test_endpoint_and_branch_order_do_not_change_shared_origin_streams():
    initial = np.array([[2, 1, -0.1], [-2, -1, 0.1], [1, 0, 0]])
    before = initial.copy()
    ids = np.array([17, 9, 42])
    order = np.array([2, 0, 1])
    units = {"source": [0], "inert": [1]}
    endpoints = {"behavior_reference": Coordinate(), "proxy": Coordinate(0)}
    config = MechanismRolloutConfig(n_rollouts=32)
    first = paired_endpoint_probabilities(_model(), initial, units, endpoints=endpoints,
                                          config=config, origin_ids=ids)
    second = paired_endpoint_probabilities(_model(), initial[order], dict(reversed(list(units.items()))),
                                           endpoints=dict(reversed(list(endpoints.items()))),
                                           config=config, origin_ids=ids[order])
    np.testing.assert_array_equal(initial, before)
    for arm in endpoints:
        np.testing.assert_array_equal(second["endpoints"][arm]["source_states"],
                                      first["endpoints"][arm]["source_states"][order])
        for branch in first["events"][arm]:
            np.testing.assert_array_equal(second["events"][arm][branch], first["events"][arm][branch][order])


def test_neural_step_is_called_once_per_branch_not_per_endpoint(monkeypatch):
    calls = []
    original = AdditiveMechanism.step

    def step(model, states, innovations, disconnected=()):
        calls.append(tuple(disconnected))
        return original(model, states, innovations, disconnected)

    monkeypatch.setattr(AdditiveMechanism, "step", step)
    result = paired_endpoint_probabilities(
        _model(), np.array([[2, 0, -0.1]]), {"source": [0], "pair": [0, 1]},
        endpoints={"behavior_reference": Coordinate(), "same": Coordinate(), "proxy": Coordinate(0)},
        config=MechanismRolloutConfig(horizon=4, target_length=2, n_rollouts=8),
    )
    assert len(calls) == 4 * 3
    assert result["metadata"]["shared_neural_futures_across_endpoints"] is True
    for branch in result["events"]["same"]:
        np.testing.assert_array_equal(result["events"]["same"][branch],
                                      result["events"]["behavior_reference"][branch])


def test_endpoint_futures_are_read_only_and_exclude_the_factual_origin():
    class Inspect(Coordinate):
        def target_states(self, target):
            assert target.flags.writeable is False
            np.testing.assert_array_equal(target[:, 0, 0, 0], [-0.5, 0.25])
            return super().target_states(target)

    model = AdditiveMechanism(np.zeros(1), np.array([-0.5]), np.zeros((1, 1)), np.zeros(1),
                              np.zeros((1, 1), dtype=bool))
    result = paired_endpoint_probabilities(
        model, np.ones((1, 1)), {"self": [0]}, endpoints={"behavior_reference": Inspect(0)},
        config=MechanismRolloutConfig(horizon=2, target_length=2, n_rollouts=1),
    )
    np.testing.assert_array_equal(result["events"]["behavior_reference"]["__factual__"], [[True]])


def test_constant_and_unchanged_proxy_fail_behavior_despite_perfect_mechanism_agreement():
    result = paired_endpoint_probabilities(
        _model(noise=0), np.array([[2, 0, -0.2]]), {"driver": [0]},
        endpoints={"behavior_reference": Coordinate(), "constant": Constant(), "proxy": Coordinate(0)},
        config=MechanismRolloutConfig(horizon=4, target_length=2, n_rollouts=8),
    )
    errors = semantics_errors(result, result)
    assert errors["behavior_reference"]["end_to_end_mae"] == 0
    for arm in ("constant", "proxy"):
        assert errors[arm]["within_chart_mae"] == 0
        assert errors[arm]["end_to_end_mae"] == 1
        assert errors[arm]["semantic_mae"] == 1
        assert errors[arm]["bound_violation_count"] == 0


def test_nuisance_can_change_when_independent_behavior_has_no_effect():
    result = paired_endpoint_probabilities(
        _model(noise=0, target=1), np.array([[2, -0.2, -0.2]]), {"driver": [0]},
        endpoints={"behavior_reference": Coordinate(), "nuisance": Coordinate(1)},
        config=MechanismRolloutConfig(horizon=4, target_length=2, n_rollouts=8),
    )
    np.testing.assert_array_equal(result["endpoints"]["behavior_reference"]["probability_reductions"]["driver"],
                                  [0])
    errors = semantics_errors(result, result)
    assert errors["nuisance"]["within_chart_mae"] == 0
    assert errors["nuisance"]["end_to_end_mae"] == 1


def test_source_disagreement_is_included_in_transition_event_bound():
    result = paired_endpoint_probabilities(
        _model(noise=0), np.array([[2, 0, -0.2]]), {"driver": [0]},
        endpoints={"behavior_reference": Coordinate(), "wrong_source": Coordinate(reverse_source=True)},
        config=MechanismRolloutConfig(horizon=4, target_length=2, n_rollouts=8),
    )
    error = semantics_errors(result, result)["wrong_source"]
    assert error["source_agreement"] == 0
    assert error["end_to_end_mae"] == 2
    assert error["mean_factual_event_disagreement"] == 1
    assert error["mean_intervened_event_disagreement"] == 1
    assert error["mean_triangle_bound"] == 2
    assert error["bound_violation_count"] == 0


def test_absolute_errors_are_computed_before_averaging_and_bound_holds_for_fitted_models():
    initial = np.array([[2, 0, -0.2], [-2, 0, 0.2]])
    endpoints = {"behavior_reference": Coordinate(), "wrong_source": Coordinate(reverse_source=True)}
    config = MechanismRolloutConfig(n_rollouts=64)
    truth = paired_endpoint_probabilities(_model(), initial, {"driver": [0]},
                                          endpoints=endpoints, config=config)
    estimated = paired_endpoint_probabilities(_model(noise=0.2), initial, {"driver": [0]},
                                               endpoints=endpoints, config=config)
    errors = semantics_errors(truth, estimated)
    for value in errors.values():
        expected = np.mean(value["per_unit"]["driver"]["end_to_end_absolute_error"])
        assert value["end_to_end_mae"] == expected
        assert value["bound_violation_count"] == 0
        assert value["mean_triangle_bound"] >= value["end_to_end_mae"]


def test_opposite_effect_errors_cannot_cancel_in_primary_mae():
    weights = np.zeros((3, 3))
    weights[0, 2] = 3
    model = AdditiveMechanism(np.array([0, 0, -0.2]), np.array([0.8, 0.6, 0.4]),
                              weights, np.zeros(3), weights != 0)
    result = paired_endpoint_probabilities(
        model, np.array([[-2, 0, 4], [2, 0, 0.1]]), {"driver": [0]},
        endpoints={"behavior_reference": Coordinate(), "constant": Constant()},
        config=MechanismRolloutConfig(horizon=1, target_length=1, n_rollouts=1),
    )
    gold = result["endpoints"]["behavior_reference"]["probability_reductions"]["driver"]
    np.testing.assert_array_equal(gold, [1, -1])
    assert gold.mean() == 0
    error = semantics_errors(result, result)["constant"]
    assert error["end_to_end_mae"] == 1
    assert error["per_unit"]["driver"]["end_to_end_absolute_error"] == [1, 1]


def test_off_support_samples_are_reported_and_retained():
    result = paired_endpoint_probabilities(
        _model(noise=0), np.array([[2, 0, -0.2]]), {"driver": [0]},
        endpoints={"behavior_reference": Coordinate()},
        config=MechanismRolloutConfig(horizon=4, target_length=2, n_rollouts=8),
        training_bounds=(np.full(3, -0.5), np.full(3, 0.5)),
    )
    diagnostics = result["support_diagnostics"]
    assert diagnostics["initial_outside_by_origin"] == [True]
    assert diagnostics["branches"]["__factual__"]["fraction_outside"] == 1
    assert diagnostics["branches"]["__factual__"]["target_sample_count"] == 16
    assert result["events"]["behavior_reference"]["__factual__"].shape == (1, 8)


@pytest.mark.parametrize("options", [
    {"origin_ids": [1, 1]}, {"origin_ids": [1.0, 2.0]}, {"origin_ids": [-1, 2]},
    {"config": {}}, {"training_bounds": (np.zeros(2), np.ones(2))},
    {"training_bounds": (np.ones(3), np.zeros(3))},
    {"training_bounds": (np.zeros(3), np.full(3, np.inf))},
])
def test_invalid_input_ownership_and_bounds_are_rejected(options):
    with pytest.raises(ValueError):
        paired_endpoint_probabilities(_model(), np.zeros((2, 3)), {"driver": [0]},
                                      endpoints={"behavior_reference": Coordinate()}, **options)


@pytest.mark.parametrize("units", [{"__factual__": [0]}, {"": [0]}, {"driver": []},
                                    {"driver": [True]}, {"driver": [3]}, {"driver": [0, 0]}])
def test_invalid_interventions_are_rejected(units):
    with pytest.raises(ValueError):
        paired_endpoint_probabilities(_model(), np.zeros((2, 3)), units,
                                      endpoints={"behavior_reference": Coordinate()})


@pytest.mark.parametrize("endpoints", [{}, {"": Coordinate()}, {"missing": object()}])
def test_invalid_endpoints_are_rejected(endpoints):
    with pytest.raises(ValueError):
        paired_endpoint_probabilities(_model(), np.zeros((2, 3)), {}, endpoints=endpoints)


def test_nonbinary_or_incorrect_axis_labels_are_rejected():
    class Invalid(Coordinate):
        def source_states(self, initial):
            return np.full(len(initial), 2)

    class InvalidTarget(Coordinate):
        def target_states(self, target):
            return np.zeros(target.shape[1])

    for endpoint in (Invalid(), InvalidTarget()):
        with pytest.raises(ValueError, match="binary"):
            paired_endpoint_probabilities(_model(), np.zeros((2, 3)), {}, endpoints={"bad": endpoint})


def test_effect_comparison_rejects_mismatched_origins_and_inconsistent_saved_events():
    arguments = {"endpoints": {"behavior_reference": Coordinate()},
                 "config": MechanismRolloutConfig(n_rollouts=8)}
    truth = paired_endpoint_probabilities(_model(), np.array([[2, 0, -0.2]]), {"driver": [0]}, **arguments)
    estimated = paired_endpoint_probabilities(_model(), np.array([[2, 0, -0.2]]), {"driver": [0]},
                                               origin_ids=np.array([99]), **arguments)
    with pytest.raises(ValueError, match="share origins"):
        semantics_errors(truth, estimated)
    truth["endpoints"]["behavior_reference"]["factual_transition_probability"] = [0.1234]
    with pytest.raises(ValueError, match="inconsistent"):
        semantics_errors(truth, truth)


def test_effect_comparison_rejects_different_states_with_same_signs_and_origin_ids():
    arguments = {"endpoints": {"behavior_reference": Coordinate()},
                 "config": MechanismRolloutConfig(n_rollouts=8), "origin_ids": np.array([42])}
    truth = paired_endpoint_probabilities(_model(), np.array([[2, 1, -0.2]]), {"driver": [0]}, **arguments)
    estimated = paired_endpoint_probabilities(_model(), np.array([[3, 2, -0.3]]), {"driver": [0]},
                                               **arguments)
    np.testing.assert_array_equal(truth["endpoints"]["behavior_reference"]["source_states"],
                                  estimated["endpoints"]["behavior_reference"]["source_states"])
    assert truth["metadata"]["origin_ids"] == estimated["metadata"]["origin_ids"]
    assert (truth["metadata"]["initial_state_fingerprint"]
            != estimated["metadata"]["initial_state_fingerprint"])
    with pytest.raises(ValueError, match="initial states"):
        semantics_errors(truth, estimated)


def test_effect_comparison_rejects_different_rollout_seeds_with_identical_origins():
    arguments = {"endpoints": {"behavior_reference": Coordinate()}, "origin_ids": np.array([42])}
    initial = np.array([[2, 1, -0.2]])
    truth = paired_endpoint_probabilities(_model(), initial, {"driver": [0]},
                                          config=MechanismRolloutConfig(n_rollouts=8, seed=10), **arguments)
    estimated = paired_endpoint_probabilities(_model(), initial, {"driver": [0]},
                                               config=MechanismRolloutConfig(n_rollouts=8, seed=11),
                                               **arguments)
    assert (truth["metadata"]["initial_state_fingerprint"]
            == estimated["metadata"]["initial_state_fingerprint"])
    with pytest.raises(ValueError, match="innovations"):
        semantics_errors(truth, estimated)
