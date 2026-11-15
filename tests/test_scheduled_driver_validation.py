"""Scheduled trial risk sets and origin-only discovery preserve future ownership."""

from dataclasses import replace

import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.driver_ground_truth import DriverBenchmarkConfig, make_driver_benchmark
from effectome.experiments.driver_validation import DriverValidationConfig, driver_validation_windows
from effectome.workflows.cross_method import CrossMethodConfig, run_cross_method_comparison


def _config():
    return DriverValidationConfig(
        generator=DriverBenchmarkConfig(n_trials=12, n_neurons=8, trial_length=512,
                                        cue_start=288, cue_stop=480),
        history_length=256, stride=32, evaluation_fraction=.5,
        manifold_dims=1, trial_origin_offset=303,
    )


def test_scheduled_pairs_are_fixed_before_observing_cues_or_outcomes():
    config = _config()
    recording = make_driver_benchmark(config.generator)
    windows = driver_validation_windows(recording, config)
    assert windows.n_windows == 24
    assert [a.anchor_sample % 512 for a in windows.anchors] == [303, 335] * 12
    changed = make_driver_benchmark(replace(config.generator, cue_probability=.1))
    assert driver_validation_windows(changed, config).anchors == windows.anchors
    for origin, target in zip(windows.anchors[::2], windows.anchors[1::2], strict=True):
        assert origin.context_stop <= target.target_start
        assert origin.anchor_sample // 512 == target.anchor_sample // 512


def test_origin_only_discovery_has_no_invented_future_graphs():
    config = _config()
    windows = driver_validation_windows(make_driver_benchmark(config.generator), config)
    report = run_cross_method_comparison(
        windows, CrossMethodConfig(n_dims=1, n_manifold_states=2, evaluation_fraction=.5),
        method_configs={"fast_cgc": ConnectivityConfig(name="fast_cgc", alpha=.01,
                                                       extra={"beta": .005})},
        graph_origin_only=True, retain_graphs=True,
    )
    result = report["methods"]["fast_cgc"]
    assert result["status"] == "available"
    assert len(result["graph_matrices"]) == 12
    np.testing.assert_array_equal(result["graph_anchor_indices"], report["events"]["origins"])
    assert result["graph_scope"] == "prediction_origins_only"
    assert result["native_graph_states"]["status"] == "unavailable"
    assert "not estimated" in result["native_graph_states"]["reason"]
    np.testing.assert_array_equal(result["prediction"]["baseline_probabilities"],
                                  report["common_baseline"]["baseline_probabilities"])


def test_scheduled_protocol_rejects_cross_trial_target():
    with pytest.raises(ValueError, match="scheduled"):
        replace(_config(), trial_origin_offset=500)
