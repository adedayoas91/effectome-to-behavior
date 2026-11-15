"""Scientific controls use training data and describe uncertainty honestly."""

import numpy as np

from effectome.experiments.transition_examples import (
    TransitionExampleConfig,
    make_transition_example,
    make_transition_example_windows,
)
from effectome.linking.transition_diagnostics import (
    frozen_markov_diagnostics,
    matched_group_controls,
    matched_transition_contrasts,
)


def test_group_controls_are_disjoint_size_matched_and_ignore_evaluation():
    data = np.random.default_rng(10).normal(size=(20, 16, 6))
    fit = np.arange(10)
    first, manifest = matched_group_controls(
        {"pair": ["0", "1"]}, list(map(str, range(6))), data, fit, seed=2
    )
    data[10:] *= 100
    second, _ = matched_group_controls({"pair": ["0", "1"]}, list(map(str, range(6))), data, fit, seed=2)
    assert first == second
    integers, _ = matched_group_controls({"pair": [0, 1]}, list(map(str, range(6))), data, fit, seed=2)
    assert integers == first
    assert not manifest["pair"]["matching_uses_evaluation"]
    for key, members in first.items():
        assert len(members) == 2
        if key != "pair":
            assert set(members).isdisjoint({"0", "1"})


def test_matched_contrast_keeps_stays_and_is_explicitly_descriptive():
    baseline = np.arange(20, dtype=float).reshape(10, 2)
    features = np.column_stack([np.arange(10), -np.arange(10)])
    sources = np.zeros(10, dtype=int)
    targets = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 2, 0])
    rows = matched_transition_contrasts(
        baseline, features, sources, targets, np.arange(4), np.arange(4, 10), {"neuron:0": [0, 1]}
    )
    first = next(row for row in rows if row["destination_state"] == 1)
    assert first["n_transition_origins"] == 2
    assert first["n_stay_origins"] == 3
    assert first["n_matched_pairs"] == 2
    assert not first["independent_event_inference"]


def test_markov_training_pairs_require_ownership_of_every_intermediate_anchor():
    windows = make_transition_example_windows(make_transition_example(TransitionExampleConfig(n_samples=512)))
    states = np.zeros(windows.n_windows, dtype=int)
    result = frozen_markov_diagnostics(
        windows,
        states,
        fit_indices=np.asarray([0, 2, 4]),
        evaluation_indices=np.arange(10, windows.n_windows),
        n_states=1,
        positive_lag=2,
    )
    assert result["training_counts"].sum() == 0
    assert result["unsupported_training_rows"].tolist() == [0]
    supported = frozen_markov_diagnostics(
        windows,
        states,
        fit_indices=np.arange(6),
        evaluation_indices=np.arange(10, windows.n_windows),
        n_states=1,
        positive_lag=2,
    )
    assert supported["training_counts"].sum() == 4
    assert all(check["status"] == "available" for check in supported["longer_horizon_checks"])
