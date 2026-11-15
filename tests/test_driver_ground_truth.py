"""Driver truth is established by paired mechanism replay, never feature omission."""

from dataclasses import replace

import numpy as np
import pytest

from effectome.experiments.driver_ground_truth import (
    DRIVER_BENCHMARK_SCENARIOS,
    DriverBenchmarkConfig,
    driver_intervention_effects,
    make_driver_benchmark,
)
from effectome.experiments.transition_examples import make_transition_example_windows


@pytest.mark.parametrize("scenario", DRIVER_BENCHMARK_SCENARIOS)
def test_driver_recordings_are_reproducible_finite_and_keep_truth_out_of_behavior(scenario):
    config = DriverBenchmarkConfig(scenario=scenario, n_trials=4, seed=3)
    first = make_driver_benchmark(config)
    second = make_driver_benchmark(config)
    np.testing.assert_array_equal(first.traces, second.traces)
    assert first.traces.shape == (6, 512)
    assert np.all(np.isfinite(first.traces))
    assert first.behavior == {}
    assert first.metadata["truth_scope"] == "metadata_only_not_predictors_or_behavior"
    assert first.metadata["valid_ranges"] == [(0, 128), (128, 256), (256, 384), (384, 512)]
    assert set(first.metadata["trial_initial_states"]) == {0, 1}
    np.testing.assert_array_equal(first.metadata["true_state"], first.metadata["population_coordinate"] >= 0)
    windows = make_transition_example_windows(first, history_length=64, target_length=8, stride=16)
    assert windows.behavior_per_window == {}
    assert any(anchor.gap_before for anchor in windows.anchors)
    assert all(anchor.context_start // 128 == (anchor.context_stop - 1) // 128 for anchor in windows.anchors)


def test_measurement_stresses_leave_latent_truth_and_mechanism_effects_unchanged():
    config = DriverBenchmarkConfig(n_trials=4, seed=3)
    direct = make_driver_benchmark(config)
    measured = make_driver_benchmark(replace(config, observation_noise=0.5, calcium_decay=0.8))
    assert not np.array_equal(direct.traces, measured.traces)
    for name in ("latent_activity", "population_coordinate", "true_state"):
        np.testing.assert_array_equal(direct.metadata[name], measured.metadata[name])
    arguments = ([95], [111], {"driver": ["0"], "proxy": ["2"]})
    assert driver_intervention_effects(direct, *arguments, n_rollouts=32) == driver_intervention_effects(
        measured, *arguments, n_rollouts=32
    )


def test_active_source_specific_driver_has_a_reproducible_conditional_effect():
    recording = make_driver_benchmark(DriverBenchmarkConfig(n_trials=12, seed=3))
    origins = np.arange(12) * 128 + 95
    units = {"n0": ["0"], "n1": ["1"], "proxy": ["2"], "readout": ["3"]}
    first = driver_intervention_effects(recording, origins, origins + 16, units, n_rollouts=64, seed=9)
    repeated = driver_intervention_effects(recording, origins, origins + 16, units, n_rollouts=64, seed=9)
    assert first == repeated
    assert np.mean([row["probability_reduction"] for row in first if row["unit"] in {"n0", "n1"}]) > 0.2
    for row in first:
        trial = row["origin_sample"] // 128
        active = int(recording.metadata["trial_initial_states"][trial])
        assert row["destination_state"] == 1 - row["source_state"]
        if row["unit"] in {"proxy", "readout", f"n{1 - active}"}:
            assert row["probability_reduction"] == 0.0
            assert row["factual_probability"] == row["intervened_probability"]
    # High-cue trials cross after usable graph origins; no-cue trials provide stays.
    coordinate = recording.metadata["population_coordinate"].reshape(12, 128)
    initial = recording.metadata["trial_initial_states"]
    exits = [
        (np.flatnonzero((trace >= 0) != state)) for trace, state in zip(coordinate, initial, strict=True)
    ]
    assert any(len(exit_times) == 0 for exit_times in exits)
    assert any(len(exit_times) and 95 <= exit_times[0] <= 115 for exit_times in exits)


def test_redundant_group_joint_disconnection_exceeds_individual_disconnections():
    recording = make_driver_benchmark(DriverBenchmarkConfig(scenario="redundant_group", n_trials=12, seed=3))
    origins = np.arange(12) * 128 + 95
    effects = driver_intervention_effects(
        recording,
        origins,
        origins + 16,
        {"n0": ["0"], "n1": ["1"], "pair": ["0", "1"]},
        n_rollouts=64,
        seed=9,
    )
    means = {
        unit: np.mean([row["probability_reduction"] for row in effects if row["unit"] == unit])
        for unit in ("n0", "n1", "pair")
    }
    assert means["pair"] > max(means["n0"], means["n1"]) + 0.3
    assert recording.metadata["nominal_drivers_by_direction"]["0->1"] == ["0", "1"]


@pytest.mark.parametrize("scenario", ["correlated_bystander", "null", "hidden_confounder"])
def test_inert_bystanders_and_exogenous_transitions_have_exact_zero_disconnection_effect(scenario):
    recording = make_driver_benchmark(DriverBenchmarkConfig(scenario=scenario, n_trials=4, seed=3))
    origins = np.arange(4) * 128 + 95
    units = {"proxy": ["2"]}
    if scenario != "correlated_bystander":
        units.update({"n0": ["0"], "n1": ["1"], "pair": ["0", "1"]})
        assert recording.metadata["nominal_drivers_by_direction"] == {"0->1": [], "1->0": []}
    effects = driver_intervention_effects(recording, origins, origins + 16, units, n_rollouts=32)
    assert all(row["probability_reduction"] == 0.0 for row in effects)


def test_correlated_proxy_changes_association_without_changing_the_driving_mechanism():
    config = DriverBenchmarkConfig(n_trials=12, seed=3)
    direct = make_driver_benchmark(config)
    proxy = make_driver_benchmark(replace(config, scenario="correlated_bystander"))
    np.testing.assert_array_equal(
        direct.metadata["population_coordinate"], proxy.metadata["population_coordinate"]
    )
    active = np.repeat(proxy.metadata["trial_initial_states"], 128)
    controller = proxy.traces[active, np.arange(proxy.n_timepoints)]
    assert np.corrcoef(controller, proxy.traces[2])[0, 1] > 0.95


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario": "other"},
        {"seed": -1},
        {"seed": True},
        {"n_trials": 1},
        {"n_neurons": 5},
        {"cue_start": 112},
        {"cue_stop": 128},
        {"dt": 0.2},
        {"dt": float("nan")},
        {"driver_ar": 1.0},
        {"coupling": 0},
        {"calcium_decay": 1.0},
        {"observation_noise": -1.0},
        {"cue_probability": 0},
        {"fps": float("inf")},
    ],
)
def test_invalid_driver_benchmark_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        DriverBenchmarkConfig(**kwargs)


def test_invalid_or_boundary_crossing_interventions_are_rejected():
    recording = make_driver_benchmark(DriverBenchmarkConfig(n_trials=2))
    for origins, targets, units, options in (
        ([120], [136], {"n0": ["0"]}, {}),
        ([95], [96], {"n0": ["0"]}, {}),
        ([2], [16], {"n0": ["0"]}, {}),
        ([95.0], [111], {"n0": ["0"]}, {}),
        ([95], [111], {"unknown": ["999"]}, {}),
        ([95], [111], {"reserved": "01"}, {}),
        ([95], [111], {"__factual__": ["0"]}, {}),
        ([95], [111], {"duplicate": ["0", "0"]}, {}),
        ([95], [111], {"n0": ["0"]}, {"n_rollouts": 0}),
        ([95], [111], {"n0": ["0"]}, {"target_length": 0}),
    ):
        with pytest.raises(ValueError):
            driver_intervention_effects(recording, origins, targets, units, **options)
    assert driver_intervention_effects(recording, [], [], {"n0": ["0"]}) == []
