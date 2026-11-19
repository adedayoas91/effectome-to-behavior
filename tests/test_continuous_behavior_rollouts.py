"""Continuous endpoints preserve path ownership and independently defined behavior."""

import copy
import json
from dataclasses import dataclass, replace

import numpy as np
import pytest

from effectome.experiments.continuous_behavior_rollouts import (
    continuous_semantics_errors,
    paired_behavior_effects,
)
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig


@dataclass(frozen=True)
class LinearReadout:
    weights: tuple[tuple[float, ...], ...] = ((0, 1, 0), (0, 0, 1))
    bias: tuple[float, float] = (0, 0)

    def predict(self, states):
        assert states.flags.writeable is False
        return states @ np.asarray(self.weights).T + self.bias

    def to_dict(self):
        return {"type": "linear", "weights": [list(row) for row in self.weights], "bias": list(self.bias)}


class ConstantReadout:
    def predict(self, states):
        return np.zeros((*states.shape[:-1], 2))

    def to_dict(self):
        return {"type": "constant"}


def _model(noise=0.08):
    cross = np.zeros((3, 3))
    cross[0, 1], cross[0, 2] = 0.9, -0.6
    return AdditiveMechanism(np.zeros(3), np.array([0.8, 0.5, 0.4]), cross, np.full(3, noise), cross != 0)


def _initial():
    return np.array([[2, -0.2, 0.1], [-2, 0.3, -0.1], [1, 0, 0]])


def _run(model=None, **options):
    arguments = {
        "readouts": {"behavior_reference": LinearReadout(), "constant": ConstantReadout()},
        "config": MechanismRolloutConfig(n_rollouts=32),
    }
    arguments.update(options)
    return paired_behavior_effects(model or _model(), _initial(), {"driver": [0], "inert": [1]}, **arguments)


def test_one_step_vector_effect_and_magnitude_have_declared_units():
    result = paired_behavior_effects(
        _model(noise=0),
        np.array([[2, 0, 0]]),
        {"driver": [0]},
        readouts={"behavior_reference": LinearReadout()},
        config=MechanismRolloutConfig(horizon=1, target_length=1, n_rollouts=1),
    )
    expected = np.array([[0.9, -0.6]]) * np.tanh(2)
    summary = result["readouts"]["behavior_reference"]
    np.testing.assert_array_equal(summary["factual_mean"], expected)
    np.testing.assert_array_equal(summary["disconnected_means"]["driver"], np.zeros((1, 2)))
    np.testing.assert_array_equal(summary["effect_vectors"]["driver"], expected)
    np.testing.assert_allclose(
        summary["effect_magnitudes"]["driver"], np.linalg.norm(expected, axis=1) / np.sqrt(2)
    )
    assert result["metadata"]["aggregation"] == "pointwise_decode_then_target_time_mean_then_rollout_mean"
    assert "target_behavior_means" not in result


def test_affine_readouts_transform_effect_vectors_without_intervening_on_readouts():
    readouts = {
        "behavior_reference": LinearReadout(),
        "scaled": LinearReadout(((0, 3, 0), (0, 0, -2)), (5, -4)),
    }
    result = _run(readouts=readouts, retain_samples=True)
    for unit in ("driver", "inert"):
        original = result["readouts"]["behavior_reference"]["effect_vectors"][unit]
        transformed = result["readouts"]["scaled"]["effect_vectors"][unit]
        np.testing.assert_allclose(transformed, original * [3, -2], atol=1e-14)
    assert result["target_behavior_means"]["scaled"]["driver"].shape == (3, 32, 2)


def test_pointwise_nonlinear_decoding_precedes_time_averaging_and_excludes_origin():
    class PolynomialReadout:
        def predict(self, states):
            assert states.flags.writeable is False
            x = states[..., 0]
            return np.stack((x**2, x**3), axis=-1)

        def to_dict(self):
            return {"type": "polynomial"}

    model = AdditiveMechanism(
        np.zeros(1), np.array([-0.5]), np.zeros((1, 1)), np.zeros(1), np.zeros((1, 1), dtype=bool)
    )
    result = paired_behavior_effects(
        model,
        np.ones((1, 1)),
        {"self": [0]},
        readouts={"behavior_reference": PolynomialReadout()},
        config=MechanismRolloutConfig(horizon=2, target_length=2, n_rollouts=1),
        retain_samples=True,
    )
    expected = np.array([[0.15625, -0.0546875]])
    actual = result["readouts"]["behavior_reference"]["factual_mean"]
    np.testing.assert_array_equal(actual, expected)
    neural_mean = (-0.5 + 0.25) / 2
    assert not np.allclose(actual, [[neural_mean**2, neural_mean**3]])
    np.testing.assert_array_equal(result["readouts"]["behavior_reference"]["initial_behavior"], [[1, 1]])


def test_identical_readouts_share_each_neural_branch_without_additional_steps(monkeypatch):
    calls = []
    original = AdditiveMechanism.step

    def counted(model, states, innovations, disconnected=()):
        calls.append(tuple(disconnected))
        return original(model, states, innovations, disconnected)

    monkeypatch.setattr(AdditiveMechanism, "step", counted)
    result = _run(
        readouts={"behavior_reference": LinearReadout(), "same": LinearReadout()}, retain_samples=True
    )
    assert len(calls) == 3 * 8
    assert result["metadata"]["shared_neural_futures_across_readouts"]
    for branch in result["target_behavior_means"]["same"]:
        np.testing.assert_array_equal(
            result["target_behavior_means"]["same"][branch],
            result["target_behavior_means"]["behavior_reference"][branch],
        )


def test_origin_intervention_and_readout_ordering_preserve_paired_streams():
    initial = _initial()
    before = initial.copy()
    ids = np.array([90, 5, 17])
    order = np.array([2, 0, 1])
    units = {"driver": [0], "pair": [0, 1]}
    readouts = {"behavior_reference": LinearReadout(), "constant": ConstantReadout()}
    config = MechanismRolloutConfig(n_rollouts=16)
    first = paired_behavior_effects(
        _model(), initial, units, readouts=readouts, config=config, origin_ids=ids, retain_samples=True
    )
    second = paired_behavior_effects(
        _model(),
        initial[order],
        {"pair": [1, 0], "driver": [0]},
        readouts=dict(reversed(list(readouts.items()))),
        config=config,
        origin_ids=ids[order],
        retain_samples=True,
    )
    np.testing.assert_array_equal(initial, before)
    for arm in readouts:
        for branch in first["target_behavior_means"][arm]:
            np.testing.assert_array_equal(
                second["target_behavior_means"][arm][branch],
                first["target_behavior_means"][arm][branch][order],
            )
        for unit in units:
            np.testing.assert_array_equal(
                second["readouts"][arm]["effect_vectors"][unit],
                first["readouts"][arm]["effect_vectors"][unit][order],
            )


def test_empty_support_has_zero_effects_for_every_frozen_readout():
    model = _model()
    empty = AdditiveMechanism(
        model.intercept, model.self_weights, np.zeros((3, 3)), model.noise_scale, np.zeros((3, 3), dtype=bool)
    )
    result = _run(empty)
    for summary in result["readouts"].values():
        for unit in ("driver", "inert"):
            np.testing.assert_array_equal(summary["effect_vectors"][unit], np.zeros((3, 2)))
            np.testing.assert_array_equal(summary["effect_magnitudes"][unit], np.zeros(3))


def test_behavioral_null_retains_nuisance_response_and_semantic_failure():
    cross = np.zeros((3, 3))
    cross[0, 1] = 0.9
    model = AdditiveMechanism(np.zeros(3), np.array([0.8, 0.5, 0.4]), cross, np.full(3, 0.1), cross != 0)
    result = _run(
        model,
        readouts={
            "behavior_reference": LinearReadout(((0, 0, 1), (0, 0, 2))),
            "nuisance": LinearReadout(((0, 1, 0), (0, -1, 0))),
        },
    )
    np.testing.assert_array_equal(
        result["readouts"]["behavior_reference"]["effect_vectors"]["driver"], np.zeros((3, 2))
    )
    assert np.any(result["readouts"]["nuisance"]["effect_magnitudes"]["driver"] > 0)
    error = continuous_semantics_errors(result, result)["nuisance"]
    assert error["within_chart_mae"] == 0
    assert error["semantic_mae"] > 0
    assert error["end_to_end_mae"] == error["semantic_mae"]
    assert error["bound_violation_count"] == 0


def test_constant_and_driver_proxy_miss_behavior_despite_zero_within_chart_error():
    result = _run(
        readouts={
            "behavior_reference": LinearReadout(),
            "constant": ConstantReadout(),
            "proxy": LinearReadout(((1, 0, 0), (2, 0, 0))),
        }
    )
    errors = continuous_semantics_errors(result, result)
    for arm in ("constant", "proxy"):
        assert errors[arm]["within_chart_mae"] == 0
        assert errors[arm]["end_to_end_mae"] > 0
        assert errors[arm]["bound_violation_count"] == 0
    assert result["readout_diagnostics"]["constant"]["collapsed"]
    assert result["readout_diagnostics"]["constant"]["prediction_std_by_output"] == [0, 0]


def test_continuous_triangle_bound_uses_common_oracle_paths_for_fitted_neural_error():
    truth = _run()
    estimated = _run(_model(noise=0.3))
    errors = continuous_semantics_errors(truth, estimated)
    for arm, error in errors.items():
        exact = []
        for unit in ("driver", "inert"):
            gold = truth["readouts"]["behavior_reference"]["effect_vectors"][unit]
            fitted = estimated["readouts"][arm]["effect_vectors"][unit]
            exact.extend(np.mean(np.abs(fitted - gold), axis=1).tolist())
        assert error["end_to_end_mae"] == np.mean(exact)
        assert error["mean_triangle_bound"] + 1e-12 >= error["end_to_end_mae"]
        assert error["bound_violation_count"] == 0


def test_opposite_signed_origin_errors_do_not_cancel():
    initial = np.array([[2, 0, 0], [-2, 0, 0]])
    result = paired_behavior_effects(
        _model(noise=0),
        initial,
        {"driver": [0]},
        readouts={"behavior_reference": LinearReadout(), "constant": ConstantReadout()},
        config=MechanismRolloutConfig(horizon=1, target_length=1, n_rollouts=1),
    )
    gold = result["readouts"]["behavior_reference"]["effect_vectors"]["driver"]
    np.testing.assert_array_equal(gold.mean(axis=0), np.zeros(2))
    error = continuous_semantics_errors(result, result)["constant"]
    assert error["end_to_end_mae"] > 0.5
    assert error["per_unit"]["driver"]["end_to_end_absolute_error"][0] > 0.5


def test_raw_neural_support_excursions_are_reported_without_filtering():
    result = _run(training_bounds=(np.full(3, -0.01), np.full(3, 0.01)), retain_samples=True)
    support = result["support_diagnostics"]
    assert support["initial_outside_by_origin"] == [True, True, True]
    assert support["branches"]["__factual__"]["fraction_outside"] > 0
    assert support["branches"]["__factual__"]["observations_removed"] == 0
    assert support["branches"]["__factual__"]["target_sample_count"] == 3 * 32 * 4
    assert result["target_behavior_means"]["constant"]["driver"].shape == (3, 32, 2)


def test_readout_inference_never_calls_fit_and_detects_mutating_artifacts():
    class Frozen(LinearReadout):
        def fit(self, *args, **kwargs):
            raise AssertionError("no fitting during rollout")

    _run(readouts={"behavior_reference": Frozen()})

    class Mutable:
        counter = 0

        def predict(self, states):
            self.counter += 1
            return np.zeros((*states.shape[:-1], 2))

        def to_dict(self):
            return {"counter": self.counter}

    with pytest.raises(ValueError, match="changed during inference"):
        _run(readouts={"behavior_reference": Mutable()})


@pytest.mark.parametrize(
    "options",
    [
        {"origin_ids": [1, 1, 2]},
        {"origin_ids": [-1, 0, 1]},
        {"origin_ids": [1.0, 2.0, 3.0]},
        {"origin_ids": [1, 2]},
        {"config": {}},
        {"retain_samples": 1},
        {"training_bounds": (np.zeros(2), np.ones(2))},
        {"training_bounds": (np.ones(3), np.zeros(3))},
        {"training_bounds": (np.zeros(3), np.full(3, np.inf))},
    ],
)
def test_invalid_origin_config_or_envelope_inputs_are_rejected(options):
    with pytest.raises(ValueError):
        _run(**options)


@pytest.mark.parametrize(
    "units",
    [
        {},
        {"": [0]},
        {"__factual__": [0]},
        {"source": []},
        {"source": [True]},
        {"source": [3]},
        {"source": [0, 0]},
    ],
)
def test_invalid_intervention_definitions_are_rejected(units):
    with pytest.raises(ValueError):
        paired_behavior_effects(_model(), _initial(), units, readouts={"behavior_reference": LinearReadout()})


@pytest.mark.parametrize(
    "readouts",
    [{}, {"": LinearReadout()}, {"missing_reference": LinearReadout()}, {"behavior_reference": object()}],
)
def test_missing_or_invalid_readout_contracts_are_rejected(readouts):
    with pytest.raises(ValueError):
        _run(readouts=readouts)


def test_readout_predictions_and_artifacts_must_be_finite_with_two_coordinates():
    class WrongShape(ConstantReadout):
        def predict(self, states):
            return np.zeros((*states.shape[:-1], 1))

    class Nonfinite(ConstantReadout):
        def predict(self, states):
            return np.full((*states.shape[:-1], 2), np.nan)

    class BadArtifact(ConstantReadout):
        def to_dict(self):
            return {"value": np.nan}

    for readout in (WrongShape(), Nonfinite(), BadArtifact()):
        with pytest.raises(ValueError, match="finite"):
            _run(readouts={"behavior_reference": readout})


@pytest.mark.parametrize("changed", ["origins", "initial", "seed", "membership", "artifact"])
def test_effect_comparison_rejects_mismatched_conditioning_or_frozen_readout(changed):
    truth = _run()
    if changed == "origins":
        estimated = _run(origin_ids=np.array([5, 6, 7]))
    elif changed == "initial":
        estimated = paired_behavior_effects(
            _model(),
            _initial() * 2,
            {"driver": [0], "inert": [1]},
            readouts={"behavior_reference": LinearReadout(), "constant": ConstantReadout()},
            config=MechanismRolloutConfig(n_rollouts=32),
        )
    elif changed == "seed":
        estimated = _run(config=replace(MechanismRolloutConfig(n_rollouts=32), seed=43))
    elif changed == "membership":
        estimated = paired_behavior_effects(
            _model(),
            _initial(),
            {"driver": [1], "inert": [0]},
            readouts={"behavior_reference": LinearReadout(), "constant": ConstantReadout()},
            config=MechanismRolloutConfig(n_rollouts=32),
        )
    else:
        estimated = _run(
            readouts={"behavior_reference": LinearReadout(bias=(1, 0)), "constant": ConstantReadout()}
        )
    with pytest.raises(ValueError, match="share origins|frozen readout"):
        continuous_semantics_errors(truth, estimated)


def test_saved_mean_or_pathwise_discrepancy_corruption_is_rejected():
    for field in ("factual", "discrepancy", "magnitude"):
        truth = _run()
        altered = copy.deepcopy(truth)
        if field == "factual":
            altered["readouts"]["constant"]["factual_mean"][0, 0] += 1
        elif field == "discrepancy":
            altered["readout_discrepancies"]["constant"]["__factual__"][0] = -1
        else:
            altered["readouts"]["constant"]["effect_magnitudes"]["driver"][0] += 1
        with pytest.raises(ValueError, match="inconsistent"):
            continuous_semantics_errors(truth, altered)


def test_retained_sample_corruption_is_rejected():
    truth = _run(retain_samples=True)
    corrupted = copy.deepcopy(truth)
    sample = corrupted["target_behavior_means"]["behavior_reference"]["__factual__"].copy()
    sample[0, 0, 0] += 1
    corrupted["target_behavior_means"]["behavior_reference"]["__factual__"] = sample
    with pytest.raises(ValueError, match="retained samples"):
        continuous_semantics_errors(truth, corrupted)


def test_json_roundtrip_and_alternative_explicit_reference_preserve_errors():
    truth = _run(retain_samples=True)
    restored = json.loads(json.dumps(truth, default=lambda value: value.tolist()))
    assert continuous_semantics_errors(truth, truth) == continuous_semantics_errors(restored, restored)
    alternate = _run(readouts={"measured_behavior": LinearReadout()}, reference="measured_behavior")
    assert (
        continuous_semantics_errors(alternate, alternate, reference="measured_behavior")["measured_behavior"][
            "end_to_end_mae"
        ]
        == 0
    )
    with pytest.raises(ValueError, match="reference"):
        continuous_semantics_errors(alternate, alternate)
