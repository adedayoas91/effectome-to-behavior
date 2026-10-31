"""Separate endpoint-driven sparsity selection from the default neural-dynamics objective."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import effectome.workflows.evaluation as evaluation
from effectome.data_module.schema import TemporalAnchor, Window, WindowedSegments
from effectome.linking import anchor_dependency_intervals
from effectome.workflows.evaluation import EvaluationConfig


def _windows(n=20, history=12, stride=12):
    rng = np.random.default_rng(31)
    segments = rng.normal(size=(n, history, 2))
    anchors = []
    for index in range(n):
        start, stop = index * stride, index * stride + history
        anchors.append(TemporalAnchor(
            dataset_id="fixture", recording_id="recording", animal_id="animal", session_id=None,
            segment_id=None, context_start=start, context_stop=stop, target_start=stop - 2,
            target_stop=stop, anchor_sample=stop - 1, anchor_time_seconds=float(stop - 1),
            sampling_rate_hz=1.0,
        ))
    return WindowedSegments(
        segments=segments, windows=[Window(anchor.context_start, anchor.context_stop) for anchor in anchors],
        behavior_per_window={"motif": np.arange(n) % 2}, n_neurons=2, fps=1.0, anchors=anchors,
        metadata={
            "neural_standardization": {"enabled": False},
            "preprocess_dependency": {"kind": "finite", "past_support": 0, "future_support": 0},
        },
    )


def _cfg(**changes):
    return EvaluationConfig(
        graph_selection_objective="endpoint_prediction", n_outer_folds=2,
        neural_lambda1_grid=(0.0, 0.1), decoder_c_grid=(0.1, 1.0), **changes,
    )


def _overlaps(left, right):
    return any(max(a[0], b[0]) < min(a[1], b[1]) for a in left for b in right)


def test_default_dispatch_preserves_neural_selector(monkeypatch):
    windows = _windows()
    config = EvaluationConfig()
    observed = []

    def selector(got_windows, got_origins, got_config):
        observed.append((got_windows, got_origins.copy(), got_config))
        return 0.123

    monkeypatch.setattr(evaluation, "_select_neural_lambda", selector)
    origins = np.arange(8)
    assert evaluation._select_graph_lambda(windows, origins, config) == 0.123
    assert observed[0][0] is windows
    np.testing.assert_array_equal(observed[0][1], origins)
    assert observed[0][2] is config


def test_unknown_graph_selection_objective_fails():
    with pytest.raises(ValueError, match="graph_selection_objective"):
        EvaluationConfig(graph_selection_objective="best_after_test")


def test_endpoint_inner_split_is_chronological_and_purges_future_target():
    windows = _windows(n=24, history=12, stride=6)
    config = _cfg(positive_lag=2)
    splits = evaluation._endpoint_inner_splits(windows, np.arange(20), config)
    assert splits
    for train, validation in splits:
        assert train.max() < validation.min()
        for origin in train:
            training_support = anchor_dependency_intervals(
                windows.anchors[origin], outcome_anchor=windows.anchors[origin + config.positive_lag],
            )
            for test_origin in validation:
                heldout_support = anchor_dependency_intervals(
                    windows.anchors[test_origin],
                    outcome_anchor=windows.anchors[test_origin + config.positive_lag],
                )
                assert not _overlaps(training_support, heldout_support)


def test_endpoint_selection_ignores_values_outside_outer_training_support():
    windows = _windows(n=24)
    config = _cfg()
    training_origins = np.arange(14)
    expected = evaluation._select_endpoint_parameters(windows, training_origins, config)
    changed = replace(
        windows, segments=windows.segments.copy(),
        behavior_per_window={"motif": windows.behavior_per_window["motif"].copy()},
    )
    changed.segments[15:] = np.nan
    changed.behavior_per_window["motif"][15:] = 999
    assert evaluation._select_endpoint_parameters(changed, training_origins, config) == expected


def test_endpoint_inner_vocabulary_is_training_owned_with_unknown_validation_class(monkeypatch):
    windows = _windows(n=18)
    config = _cfg()
    # Only the final inner validation block sees class 7.
    windows.behavior_per_window["motif"][11:] = 7
    original = evaluation._fit_probability_decoder
    fitted_vocabularies = []

    def spy(features, labels, **kwargs):
        fitted_vocabularies.append((tuple(kwargs["class_order"]), set(labels.tolist())))
        assert set(kwargs["class_order"]) == set(labels.tolist())
        return original(features, labels, **kwargs)

    monkeypatch.setattr(evaluation, "_fit_probability_decoder", spy)
    chosen = evaluation._select_endpoint_parameters(windows, np.arange(16), config)
    assert chosen[0] in config.neural_lambda1_grid
    assert fitted_vocabularies
    assert any(7 not in vocabulary for vocabulary, _ in fitted_vocabularies)


def test_infeasible_endpoint_selection_does_not_fallback_to_first_lambda():
    windows = _windows(n=8, history=12, stride=6)
    with pytest.raises(ValueError, match="feasible purged chronological"):
        evaluation._select_endpoint_lambda(windows, np.arange(6), _cfg(positive_lag=2))


def test_endpoint_selection_records_outer_fold_objective_and_reuses_joint_penalty(monkeypatch):
    windows = _windows(n=24)
    config = _cfg()
    monkeypatch.setattr(evaluation, "_select_endpoint_parameters", lambda *args: (0.1, 1.0))
    result = evaluation.run_primary_evaluation(windows, config)
    assert result.diagnostics["graph_selection_objective"] == "endpoint_prediction"
    for diagnostics in result.diagnostics["folds"].values():
        assert diagnostics["graph_selection_objective"] == "endpoint_prediction"
        assert diagnostics["graph_selection_metric"] == "endpoint_log_loss"
        assert diagnostics["m1_decoder_value"] == 1.0
    # The outer test's future outcome can overlap another origin's history; both are protected.
    origins = np.arange(windows.n_windows - config.positive_lag)
    for fold in result.manifest.outer_folds:
        for index in fold.train_indices:
            source = origins[index]
            train = anchor_dependency_intervals(
                windows.anchors[source], outcome_anchor=windows.anchors[source + config.positive_lag],
            )
            for test_index in fold.test_indices:
                test_source = origins[test_index]
                heldout = anchor_dependency_intervals(
                    windows.anchors[test_source],
                    outcome_anchor=windows.anchors[test_source + config.positive_lag],
                )
                assert not _overlaps(train, heldout)


def test_continuous_endpoint_selection_uses_declared_regression_grid():
    windows = _windows()
    windows.behavior_per_window["motif"] = np.linspace(0.0, 1.0, windows.n_windows)
    config = _cfg(endpoint_type="continuous", decoder_alpha_grid=(0.5, 2.0))
    sparsity, penalty = evaluation._select_endpoint_parameters(windows, np.arange(16), config)
    assert sparsity in config.neural_lambda1_grid
    assert penalty in config.decoder_alpha_grid


def test_tuning_budget_counts_joint_endpoint_grid_without_double_tuning_m1():
    config = _cfg()
    assert evaluation._tuning_candidate_count(config) == 2 * 2 + 2
    default = replace(config, graph_selection_objective="neural_dynamics")
    assert evaluation._tuning_candidate_count(default) == 2 + 4


def test_outer_test_labels_and_histories_cannot_change_endpoint_selected_models():
    windows = _windows(n=24)
    config = _cfg()
    first = evaluation.run_primary_evaluation(windows, config, selected_fold_ids=frozenset({"outer-000"}))
    changed = replace(
        windows, segments=windows.segments.copy(),
        behavior_per_window={"motif": windows.behavior_per_window["motif"].copy()},
    )
    fold = first.manifest.outer_folds[0]
    origins = np.arange(windows.n_windows - config.positive_lag)
    changed.segments[origins[list(fold.test_indices)]] *= 99.0
    changed.behavior_per_window["motif"][origins[list(fold.test_indices)] + config.positive_lag] = 7
    second = evaluation.run_primary_evaluation(changed, config, selected_fold_ids=frozenset({"outer-000"}))
    assert first.diagnostics["folds"] == second.diagnostics["folds"]


def test_endpoint_objective_uses_purged_selection_for_both_decoders(monkeypatch):
    windows = _windows(n=24)
    config = _cfg()

    def forbidden(*args, **kwargs):
        pytest.fail("index-only legacy decoder selection used by endpoint objective")

    monkeypatch.setattr(evaluation, "_select_decoder_c", forbidden)
    monkeypatch.setattr(evaluation, "_select_decoder_alpha", forbidden)
    result = evaluation.run_primary_evaluation(windows, config, selected_fold_ids=frozenset({"outer-000"}))
    assert result.diagnostics["folds"]["outer-000"]["m0_decoder_value"] in config.decoder_c_grid
    assert result.diagnostics["folds"]["outer-000"]["m1_decoder_value"] in config.decoder_c_grid
