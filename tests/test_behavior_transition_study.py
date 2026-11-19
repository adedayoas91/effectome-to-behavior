"""Check paired mechanism evidence and study-level decision ownership."""

from dataclasses import replace

import numpy as np

from effectome.experiments.behavior_transition import TransitionExperimentConfig
from effectome.experiments.behavior_transition_attribution import paired_transition_outcomes
from effectome.experiments.behavior_transition_controls import (
    TRANSITION_SCENARIOS,
    TransitionControlConfig,
    make_transition_control,
)
from effectome.experiments.behavior_transition_graphs import trailing_histories
from effectome.experiments.structural_mechanism import MechanismRolloutConfig
from synthetic.behavior_transition import SEEDS, _decision


def test_trailing_histories_never_access_next_sample_or_another_trial():
    neural = np.arange(2 * 12 * 2).reshape(2, 12, 2)
    origins = np.array([[3, 8], [4, 9]])
    values, windows = trailing_histories(neural, origins, 4)
    assert [(window.start, window.stop) for window in windows] == [(0, 4), (5, 9), (13, 17), (18, 22)]
    np.testing.assert_array_equal(
        values, neural.reshape(-1, 2)[[[0, 1, 2, 3], [5, 6, 7, 8], [13, 14, 15, 16], [18, 19, 20, 21]]]
    )


def test_paired_mechanism_evidence_retains_context_and_origin_order():
    control = make_transition_control(
        TransitionControlConfig(n_trials=2, history_lengths=(16,)),
        scenario="driver_switch",
        seed=81,
    )
    first, second = control.mechanisms
    params = MechanismRolloutConfig(n_rollouts=8, seed=99)
    states = np.repeat([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], 3, axis=0)
    origins = np.array([10, 11, 12])
    factual, effects = paired_transition_outcomes([first, second, first], states, origins, params)
    one, single_effects = paired_transition_outcomes([second], states[1:2], origins[1:2], params)
    np.testing.assert_allclose(factual[1], one[0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(effects["neuron:1"][1], single_effects["neuron:1"][0], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(effects["neuron:0"][1], [0, 0])
    np.testing.assert_array_equal(effects["neuron:1"][[0, 2]], [[0, 0], [0, 0]])
    assert np.linalg.norm(effects["neuron:0"][0]) > 0.1


def test_confirmation_requires_every_seed_and_both_comparators():
    config = replace(TransitionExperimentConfig(), practical_forecast_gain=0.005)
    rows = [
        {
            "seed": seed,
            "scenario": scenario,
            "history_samples": max(config.generator.history_lengths),
            "method": config.primary_method,
            "status": "available",
            "gain_over_history": 0.01 if scenario == "driver_switch" else 0.0,
            "gain_over_static": 0.01 if scenario == "driver_switch" else 0.0,
        }
        for scenario in TRANSITION_SCENARIOS
        for seed in SEEDS["confirmation"]
    ]
    assert _decision(rows, config, "development") is None
    assert _decision(rows, config, "confirmation")["prediction_passed"]
    assert not _decision(rows[:-1], config, "confirmation")["prediction_passed"]
    for row in rows:
        if row["scenario"] == "driver_switch":
            row["gain_over_static"] = 0.0
    assert not _decision(rows, config, "confirmation")["prediction_passed"]
