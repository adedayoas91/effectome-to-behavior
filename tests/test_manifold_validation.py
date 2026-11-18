from dataclasses import replace

import numpy as np
import pytest

from effectome.experiments.manifold_validation import (
    ENDPOINT_ARMS,
    manifold_validation_inventory,
    run_manifold_validation,
)
from effectome.experiments.manifold_validation_config import (
    ManifoldValidationConfig,
    load_manifold_validation_profile,
)
from effectome.experiments.mechanism_controls import MechanismControlConfig
from effectome.experiments.mechanism_validation_config import MechanismValidationConfig
from effectome.experiments.structural_mechanism import MechanismRolloutConfig
from effectome.workflows.cross_method import load_cross_method_profile


def _tiny_config():
    mechanism = MechanismValidationConfig(
        seeds=(40000,), generator=MechanismControlConfig(n_trials=8, trial_length=320),
        rollout=MechanismRolloutConfig(n_rollouts=16),
    )
    return ManifoldValidationConfig(profile="smoke", seeds=(40000,), mechanism=mechanism)


@pytest.mark.parametrize("profile, cells, native", [("smoke", 2, 72), ("development", 10, 360)])
def test_profiles_inventory_native_fits_once_per_model_not_per_endpoint(profile, cells, native):
    config = load_manifold_validation_profile(profile)
    _, methods = load_cross_method_profile("csl_sanity")
    inventory = manifold_validation_inventory(config, methods)
    assert inventory["planned_cells"] == cells
    assert inventory["native_window_fits"] == native
    assert len(inventory["endpoint_arms"]) == 7
    assert not inventory["confirmatory_executed"]
    assert not inventory["nomination_calibrated"]


@pytest.mark.parametrize("values", [
    {"seeds": (20000,)}, {"seeds": (50000,)}, {"seeds": (60000,)}, {"seeds": (70000,)},
    {"seeds": ()}, {"seeds": (40000, 40000)}, {"n_components": 0},
    {"n_components": 6}, {"decoder_cs": (1.0, 1.0)}, {"decoder_cs": (float("nan"),)},
    {"endpoint_seed": True}, {"profile": "confirmation"},
])
def test_config_keeps_confirmation_and_historical_seed_pools_reserved(values):
    with pytest.raises(ValueError):
        ManifoldValidationConfig(**values)


def test_complete_control_panel_and_resume_preserve_frozen_artifacts(tmp_path, monkeypatch):
    config = _tiny_config()
    report = run_manifold_validation(config, tmp_path, workers=1, methods={})
    assert report["execution_complete"]
    assert report["fitting_enabled"]
    assert report["all_requested_lanes_available"]
    assert report["available_method_runs"] == 6
    assert not report["scientific_validation_passed"]
    assert not report["nonlinear_manifold_validated"]
    assert not report["confirmatory_executed"]
    assert (tmp_path / "analysis/seed-metrics.csv").is_file()
    assert (tmp_path / "analysis/analysis-report.md").is_file()
    for cell in report["cells"]:
        assert cell["training_trial_ids"] == [0, 1, 2, 3]
        assert cell["evaluation_trial_ids"] == [4, 5, 6, 7]
        assert set(cell["natural_evaluation"]) == set(ENDPOINT_ARMS)
        assert cell["natural_evaluation"]["constant"]["collapsed"]
        for result in cell["methods"].values():
            for metrics in result["metrics"].values():
                assert metrics["bound_violation_count"] == 0
        oracle = cell["methods"]["oracle"]["metrics"]
        assert oracle["constant"]["within_chart_mae"] == 0
        if cell["scenario"] == "driver":
            assert oracle["constant"]["end_to_end_mae"] > 0
        empty = cell["methods"]["empty_support"]["endpoint_rollouts"]
        for endpoint in empty.values():
            for delta in endpoint["probability_reductions"].values():
                np.testing.assert_array_equal(delta, np.zeros(4))
    evidence = {path: path.read_bytes() for path in tmp_path.glob("cells/*/*") if path.is_file()}

    def forbid_refit(*args, **kwargs):
        raise AssertionError("A completed run must not refit endpoints or neural dynamics")

    monkeypatch.setattr("effectome.experiments.manifold_validation.fit_endpoint_arms", forbid_refit)
    monkeypatch.setattr("effectome.experiments.manifold_validation.fit_additive_mechanism", forbid_refit)
    resumed = run_manifold_validation(config, tmp_path, workers=1, methods={})
    assert resumed["available_method_runs"] == 6
    assert all(path.read_bytes() == original for path, original in evidence.items())
    with pytest.raises(ValueError, match="protocol/source changed"):
        run_manifold_validation(replace(config, n_components=1), tmp_path, workers=1, methods={})
    from effectome.experiments.manifold_validation import make_mechanism_control

    for sample in (1, 4 * 320, 4 * 320 + 1):
        def altered_recording(generator_config, sample=sample):
            recording = make_mechanism_control(generator_config)
            recording.traces[3, sample] += 0.25
            return recording

        monkeypatch.setattr("effectome.experiments.manifold_validation.make_mechanism_control",
                            altered_recording)
        with pytest.raises(ValueError, match="inputs changed"):
            run_manifold_validation(config, tmp_path, workers=1, methods={})
    monkeypatch.setattr("effectome.experiments.manifold_validation.make_mechanism_control",
                        make_mechanism_control)
    events = next(tmp_path.glob("cells/*/known_support-events.npz"))
    events.write_bytes(events.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="integrity"):
        run_manifold_validation(config, tmp_path, workers=1, methods={})


def test_unavailable_native_lane_is_retained_and_not_retried(tmp_path, monkeypatch):
    def fail_native(*args, **kwargs):
        raise RuntimeError("native estimator unavailable on purpose")

    monkeypatch.setattr("effectome.experiments.manifold_validation._discover_support", fail_native)
    _, methods = load_cross_method_profile("csl_sanity")
    selected = {"fast_cgc": methods["fast_cgc"]}
    report = run_manifold_validation(_tiny_config(), tmp_path, workers=1, methods=selected)
    assert report["execution_complete"]
    assert not report["all_requested_lanes_available"]
    assert report["attempted_method_runs"] == 8
    assert report["available_method_runs"] == 6
    for cell in report["cells"]:
        assert cell["methods"]["fast_cgc"]["status"] == "unavailable"
        assert "on purpose" in cell["methods"]["fast_cgc"]["reason"]
