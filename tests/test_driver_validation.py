"""The driver benchmark preserves ownership, event semantics and failure denominators."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.driver_ground_truth import DriverBenchmarkConfig
from effectome.experiments.driver_validation import (
    DriverValidationConfig,
    _alignment,
    _minimum_class_agreement,
    driver_validation_cells,
    load_driver_validation_profile,
    run_driver_validation,
)
from effectome.experiments.driver_validation_reporting import render_driver_validation_report


def test_profiles_separate_confirmation_from_pilots_and_cover_scenarios_in_prefix():
    smoke = load_driver_validation_profile("smoke")
    confirm = load_driver_validation_profile("confirmatory")
    cells = driver_validation_cells(smoke)
    assert len(cells) == 15
    assert {c["scenario"] for c in cells[:5]} == set(smoke.scenarios)
    assert "null" in smoke.scenarios
    assert len(driver_validation_cells(confirm)) == 140
    assert set(smoke.positive_seeds).isdisjoint(confirm.positive_seeds)
    assert set(smoke.null_seeds).isdisjoint(confirm.null_seeds)
    assert smoke.estimator_profile == confirm.estimator_profile == "csl_sanity"
    assert smoke.history_length == confirm.history_length == 256
    assert smoke.generator.trial_length == confirm.generator.trial_length == 512
    with pytest.raises(ValueError, match="cue"):
        replace(smoke, history_length=320)


def test_semantic_assignment_uses_only_training_truth_and_checks_rare_outcomes():
    coordinate = np.asarray([-1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    recording = SimpleNamespace(metadata={"population_coordinate": coordinate})
    windows = SimpleNamespace(anchors=[SimpleNamespace(target_start=i, target_stop=i + 1) for i in range(6)])
    comparison = {
        "reference": {
            "fit_indices": [0, 1, 2, 3],
            "evaluation_indices": [4, 5],
            "manifold_states": [1, 1, 0, 0, 1, 0],
        }
    }
    first, accuracy, _ = _alignment(recording, windows, comparison)
    coordinate[4:] *= -1
    second, altered_accuracy, _ = _alignment(recording, windows, comparison)
    assert first == second == {0: 1, 1: 0}
    assert accuracy == 1 and altered_accuracy == 0
    predicted = np.asarray([0] * 19 + [1])
    actual = np.zeros(20, int)
    assert np.mean(predicted == actual) == 0.95
    assert _minimum_class_agreement(predicted, actual) == 0
    assert _minimum_class_agreement(np.asarray([0, 1]), np.asarray([0, 1])) == 1


def test_real_driver_run_retains_failures_oracle_units_and_exact_resume(tmp_path):
    config = DriverValidationConfig(
        scenarios=("single_driver",),
        positive_seeds=(0,),
        null_seeds=(0,),
        generator=DriverBenchmarkConfig(n_trials=8),
        history_length=64,
        estimator_profile="smoke",
        intervention_rollouts=8,
    )
    methods = {
        "partial": ConnectivityConfig(name="correlation", partial=True),
        "broken": ConnectivityConfig(name="not_registered"),
    }
    report = run_driver_validation(config, tmp_path, methods=methods)
    assert report["attempted_method_runs"] == 2
    assert report["available_method_runs"] == 1
    assert len(report["seed_rows"]) == 4
    assert report["acceptance"]["overall_status"] == "inconclusive"
    assert not report["scientific_validation_passed"]
    assert not report["biological_necessity_validated"]
    for row in report["seed_rows"]:
        if row["status"] == "available" and row["n_origins"]:
            assert set(row["driver_scores"]) == set(row["intervention_effects"])
            assert row["intervention_effects"]["neuron:2"] == 0
            assert row["intervention_effects"]["group:control_pair"] == 0
    cached = run_driver_validation(config, tmp_path, methods=methods)
    assert cached["cells"][0]["reused"]
    changed = run_driver_validation(replace(config, intervention_rollouts=9), tmp_path, methods=methods)
    assert not changed["cells"][0]["reused"]
    render_driver_validation_report(report, tmp_path / "analysis")
    assert (tmp_path / "analysis/figure-01-intervention-truth.pdf").stat().st_size > 100
    assert "broken" in (tmp_path / "analysis/numeric-summary.md").read_text()
