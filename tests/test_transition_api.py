"""Fit/score separated transition API tests for Appendix-C WP8."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.dynamics import (
    TransitionConfig,
    dwell_runs,
    fit_transition_model,
    score_transition_model,
    transition_diagnostics,
)


def test_transition_fit_stores_counts_and_prior_mean():
    labels = np.array([0, 1, 1, 0, 1, 0])
    model = fit_transition_model(labels, 2, None, TransitionConfig(laplace=0.5, n_null=2))

    assert np.array_equal(model.counts, [[0.0, 2.0], [2.0, 1.0]])
    assert np.allclose(model.transition_matrix.sum(axis=1), 1.0)
    assert np.allclose(model.posterior_mean, model.transition_matrix)
    assert model.observed_exits.tolist() == [2.0, 2.0]


def test_actual_exits_eligibility_and_fixed_vocabulary_diagnostics():
    model = fit_transition_model(np.zeros(5, dtype=int), 2, None, TransitionConfig())
    assert model.observed_exits.tolist() == [0., 0.]
    assert model.row_trials.tolist() == [4., 0.]
    assert model.prior_driven_rows.tolist() == [False, True]
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    fitted = fit_transition_model(labels, 2, None, TransitionConfig(),
                                  eligible_pairs=np.array([True, False, True, True, True, True, True]))
    assert fitted.counts.sum() == 6
    diagnostic = transition_diagnostics(fitted, labels, np.array([0, 4]), cadence=0.5)
    assert diagnostic["two_step_counts"].sum() == 4
    assert diagnostic["second_order_log_scores"].shape == (4,)
    assert diagnostic["row_intervals"].shape == (2, 2, 2)
    with pytest.raises(ValueError):
        fit_transition_model(np.array([0., 0.5]), 2, None, TransitionConfig())
    with pytest.raises(ValueError):
        score_transition_model(model, labels, np.array([0]), initial_context=0)


def test_transition_score_does_not_cross_boundaries():
    train = np.array([0, 1, 1, 0])
    model = fit_transition_model(train, 2, None, TransitionConfig(laplace=0.5, n_null=2))
    test = np.array([0, 1, 0, 1])

    scored = score_transition_model(model, test, np.array([0, 2]))

    assert scored.n_scored == 2
    assert scored.log_scores.shape == (2,)


def test_transition_score_can_use_initial_context_only_when_supplied():
    model = fit_transition_model(
        np.array([0, 1, 1, 1]),
        2,
        None,
        TransitionConfig(laplace=0.5, n_null=2),
    )

    no_context = score_transition_model(model, np.array([1]), None)
    with_context = score_transition_model(model, np.array([1]), None, initial_context=0)

    assert no_context.n_scored == 0
    assert with_context.n_scored == 1


def test_dwell_runs_mark_segment_edge_censoring():
    runs = dwell_runs(np.array([0, 0, 1, 1, 1, 0]), None)

    assert runs[0].left_censored is True
    assert runs[0].right_censored is False
    assert runs[-1].right_censored is True
    assert [run.duration for run in runs] == [2, 3, 1]


def test_transition_diagnostics_returns_dwell_run_objects():
    labels = np.array([0, 1, 1, 0])
    model = fit_transition_model(labels, 2, None, TransitionConfig(laplace=1.0, n_null=2))
    diagnostics = transition_diagnostics(model, labels, None)

    assert diagnostics["prior_concentration"] == 1.0
    assert len(diagnostics["dwell_runs"]) == 3
