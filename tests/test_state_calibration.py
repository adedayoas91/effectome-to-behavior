"""Raw state calibration preserves null laws, time support and uncertainty scope."""

from __future__ import annotations

import json

import numpy as np
import pytest

from effectome.data_module import WindowConfig, make_windows
from effectome.experiments.state_calibration import (
    StateCalibrationConfig,
    eligible_window_truth,
    generate_state_calibration_recording,
    run_state_calibration,
    state_calibration_profile,
    write_state_calibration_report,
)


def test_measurement_null_keeps_latent_activity_and_interaction_law_fixed():
    cfg = StateCalibrationConfig()
    constant = generate_state_calibration_recording(cfg, scenario="constant", dwell=64, seed=51)
    kinetic = generate_state_calibration_recording(cfg, scenario="kinetic_change", dwell=64, seed=51)
    variance = generate_state_calibration_recording(cfg, scenario="variance_change", dwell=64, seed=51)
    for recording in (kinetic, variance):
        np.testing.assert_array_equal(
            recording.metadata["true_lag_graphs"], constant.metadata["true_lag_graphs"]
        )
        assert recording.metadata["state_calibration"]["latent_law_constant"]
        assert not recording.metadata["state_calibration"]["discrete_regimes_present"]
    np.testing.assert_array_equal(
        kinetic.metadata["latent_neural_activity"], constant.metadata["latent_neural_activity"]
    )
    assert not np.allclose(kinetic.traces, constant.traces)
    np.testing.assert_array_equal(variance.traces[:, :480], constant.traces[:, :480])
    assert not np.allclose(variance.traces[:, 480:], constant.traces[:, 480:])


def test_regime_schedule_and_truth_do_not_compress_or_relabel_mixed_histories():
    cfg = StateCalibrationConfig()
    recording = generate_state_calibration_recording(cfg, scenario="recurrent", dwell=64, seed=52)
    windows = make_windows(recording, WindowConfig(
        mode="temporal", history_length=32, target_length=1, stride=16,
    ))
    labels, eligible = eligible_window_truth(windows, recording.metadata["true_states"])
    assert np.array_equal(np.flatnonzero(np.diff(recording.metadata["true_states"])) + 1,
                          np.arange(64, cfg.n_samples, 64))
    for index, anchor in enumerate(windows.anchors):
        history = recording.metadata["true_states"][anchor.context_start:anchor.context_stop]
        assert labels[index] == history[-1]
        assert eligible[index] == (np.unique(history).size == 1)
        assert anchor.context_stop - anchor.context_start == 32
    assert eligible.any() and (~eligible).any()


def test_drift_is_not_declared_discrete_truth_and_generator_is_reproducible():
    cfg = StateCalibrationConfig()
    a = generate_state_calibration_recording(cfg, scenario="smooth_drift", dwell=64, seed=53)
    b = generate_state_calibration_recording(cfg, scenario="smooth_drift", dwell=64, seed=53)
    np.testing.assert_array_equal(a.traces, b.traces)
    assert not a.metadata["state_calibration"]["discrete_regimes_present"]
    assert np.unique(a.metadata["temporal_weights"][:, 1]).size == cfg.n_samples


def test_development_grid_tests_histories_shorter_and_longer_than_dwells():
    cfg = state_calibration_profile("development")
    ratios = [history / dwell for history in cfg.history_lengths for dwell in cfg.dwell_lengths]
    assert min(ratios) < 1 < max(ratios)
    with pytest.raises(ValueError, match="history lengths"):
        StateCalibrationConfig(history_lengths=(1,))


def test_real_raw_pipeline_smoke_and_serialization_are_deterministic(tmp_path):
    cfg = StateCalibrationConfig(scenarios=("constant",), max_draws=1)
    report = run_state_calibration(cfg)
    repeated = run_state_calibration(cfg)
    row, same = report["draws"][0], repeated["draws"][0]
    assert row["status"] == "available", row.get("unavailable_reason")
    assert report["raw_pipeline_rerun_per_draw"] and not report["label_permutation_used"]
    assert row["raw_recording_hash"] == same["raw_recording_hash"]
    assert row["graph_hash"] == same["graph_hash"]
    assert row["selected_n_states"] == same["selected_n_states"]
    assert row["state_recovery_ari"] is None
    assert "no discrete" in row["state_recovery_unavailable_reason"]
    assert isinstance(row["false_multistate_selection"], bool)
    assert row["backend"] == "fastcgc"
    assert row["backend_version"]
    assert row["regime_claim_eligible"] is False
    assert row["realized_history_samples"] == 32
    assert np.isfinite(row["heldout_mse"])
    assert "mean_membership_entropy" in row
    json_path, csv_path = write_state_calibration_report(report, tmp_path)
    assert json.loads(json_path.read_text())["profile"] == "smoke"
    assert csv_path.read_text().count("\n") == 2


def test_raw_draw_failures_are_unavailable_instead_of_false_negative_results():
    cfg = StateCalibrationConfig(scenarios=("constant",), history_lengths=(8,), max_draws=1)
    row = run_state_calibration(cfg)["draws"][0]
    assert row["status"] == "unavailable"
    assert row["false_multistate_selection"] is None
    assert "resolution" in row["unavailable_reason"]


def test_nonrecurring_epochs_are_present_in_observed_time_after_burn_in():
    cfg = StateCalibrationConfig()
    recording = generate_state_calibration_recording(cfg, scenario="nonrecurring", dwell=64, seed=54)
    truth = recording.metadata["true_states"]
    np.testing.assert_array_equal(np.unique(truth), [0, 1, 2])
    np.testing.assert_array_equal(np.flatnonzero(np.diff(truth)) + 1, [320, 640])
    assert recording.metadata["state_calibration"]["realized_median_dwell_samples"] == 320
