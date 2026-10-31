"""Synthetic generator and validation-manifest tests for Appendix-C WP9."""

from __future__ import annotations

import numpy as np

from effectome.data_module import WindowConfig, make_windows
from effectome.experiments.validation import (
    ValidationCell,
    deterministic_seed,
    synthetic_config_for_cell,
    validation_totals,
)
from effectome.utils.synthetic import SyntheticConfig, make_synthetic_recording


def test_one_state_synthetic_generation_returns_all_zero_states():
    recording = make_synthetic_recording(
        SyntheticConfig(n_neurons=4, n_timepoints=100, n_states=1, seed=7)
    )

    assert recording.traces.shape == (4, 100)
    assert np.array_equal(recording.metadata["true_states"], np.zeros(100, dtype=np.int64))
    assert recording.metadata["true_graphs"].shape[0] == 1


def test_synthetic_generation_rejects_zero_states():
    try:
        make_synthetic_recording(SyntheticConfig(n_states=0))
    except ValueError as exc:
        assert "n_states" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected n_states=0 to fail")


def test_validation_totals_match_appendix_manifest():
    totals = validation_totals(include_grouping=True)

    assert totals["core_stress_nonlinear"] == 800
    assert totals["with_grouping"] == 920
    assert totals["independent_calibration"] == 600


def test_deterministic_seed_is_stable_and_purpose_specific():
    first = deterministic_seed(
        260926,
        suite="core",
        scenario="constant",
        graph=1,
        replicate=2,
        purpose="data",
    )
    second = deterministic_seed(
        260926,
        suite="core",
        scenario="constant",
        graph=1,
        replicate=2,
        purpose="data",
    )
    other = deterministic_seed(
        260926,
        suite="core",
        scenario="constant",
        graph=1,
        replicate=2,
        purpose="optimizer",
    )

    assert first == second
    assert first != other


def test_shared_base_edit_masks_and_burst_gaps_are_persisted():
    recording = make_synthetic_recording(
        SyntheticConfig(
            n_neurons=6,
            n_timepoints=120,
            n_states=2,
            density=0.35,
            regime_sign_flip_fraction=0.4,
            regime_sparsity_jitter=0.3,
            missing_fraction=0.2,
            missingness_mode="burst",
            missing_block_length=8,
            seed=12,
        )
    )
    edits = recording.metadata["coefficient_edit_masks"]
    expected = recording.metadata["true_lag_graphs"].shape
    assert edits["support_added"].shape == expected
    assert edits["support_removed"].shape == expected
    assert edits["sign"].shape == expected
    assert edits["magnitude"].shape == expected
    assert np.any(~recording.metadata["observation_mask"])
    assert recording.metadata["missingness"]["mode"] == "burst"
    assert recording.metadata["trajectory_diagnostics"]["finite"] is True
    assert recording.metadata["rollout_diagnostics"]["finite"] is True


def test_recording_wide_missingness_is_declared_as_hard_gaps():
    recording = make_synthetic_recording(
        SyntheticConfig(
            n_neurons=4,
            n_timepoints=160,
            n_states=1,
            missing_fraction=0.2,
            missingness_mode="gap",
            missing_block_length=16,
            seed=14,
        )
    )
    gaps = recording.metadata["gap_intervals"]
    assert gaps
    mask = recording.metadata["observation_mask"]
    for start, stop in gaps:
        assert start < stop
        assert not np.any(mask[start:stop])

    windows = make_windows(
        recording,
        WindowConfig(
            length=24,
            history_length=24,
            target_length=4,
            stride=4,
            standardize_per_window=False,
        ),
    )
    assert all(
        not (anchor.context_start < stop and anchor.context_stop > start)
        for anchor in windows.anchors
        for start, stop in gaps
    )


def test_observed_projection_truth_has_independent_replicate_metadata():
    recording = make_synthetic_recording(
        SyntheticConfig(
            n_neurons=4,
            n_timepoints=100,
            n_states=1,
            observed_projection_replicates=2,
            observed_projection_samples=120,
            seed=31,
        )
    )
    projection = recording.metadata["observed_projection_truth"]
    assert projection["coefficients"].shape == (1, 1, 4, 4)
    assert projection["replicate_count"] == 2
    assert projection["replicate_design"] == "independent_innovations"
    assert np.isfinite(projection["monte_carlo_standard_error"])
    assert recording.metadata["truth_contract"]["observed_projection"] == projection["truth_id"]


def test_conditional_null_endpoint_uses_declared_m0_variables_and_fresh_noise():
    recording = make_synthetic_recording(
        SyntheticConfig(
            n_neurons=4,
            n_timepoints=100,
            n_states=1,
            endpoint_mode="conditional_null",
            conditional_null_m0_variables=("activity_mean", "activity_variance"),
            seed=44,
        )
    )
    contract = recording.metadata["endpoint_contracts"]["conditional_null"]
    assert contract["declared_m0_variables"] == ("activity_mean", "activity_variance")
    assert contract["noise_stream"] != recording.metadata["trajectory_id"]
    assert contract["generating_function"] == "linear_m0_plus_fresh_gaussian_noise"
    assert recording.behavior["conditional_null"].shape == (100,)


def test_validation_condition_mapping_is_executable():
    cell = ValidationCell("stress", "recurrent_geometric", 2, 1, "burst_missingness")
    config = synthetic_config_for_cell(cell, master_seed=260925, profile="smoke")
    assert config.missingness_mode == "burst"
    assert config.missing_fraction > 0
    assert config.graph_seed != config.seed
    assert config.endpoint_seed not in {config.seed, config.graph_seed}
