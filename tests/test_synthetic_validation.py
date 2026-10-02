"""Synthetic generator and validation-manifest tests for Appendix-C WP9."""

from __future__ import annotations

import numpy as np

from effectome.experiments.validation import deterministic_seed, validation_totals
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
