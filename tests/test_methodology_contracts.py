"""Focused acceptance contracts for Appendix-C WP0, WP1, and WP4."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.data_module import (
    ArtifactProvenance,
    ExampleSupport,
    FittedPreprocessor,
    FoldSpec,
    NeuralRecording,
    PreprocessConfig,
    RawInterval,
    RecordingIdentity,
    TemporalAnchor,
    Window,
    WindowConfig,
    WindowedSegments,
    make_windows,
)
from effectome.linking import valid_positive_lag_origins
from effectome.utils.io import load_artifact, save_artifact
from effectome.workflows import (
    EvaluationConfig,
    combine_primary_evaluation_results,
    paired_log_loss_delta,
    primary_evaluation_fold_ids,
    run_primary_evaluation,
    score_regression_records,
)


def _anchor(index: int, *, stride: int = 7) -> TemporalAnchor:
    start = index * stride
    stop = start + 30
    return TemporalAnchor(
        dataset_id="ds",
        recording_id="rec",
        animal_id="animal",
        session_id=None,
        segment_id=None,
        context_start=start,
        context_stop=stop,
        target_start=stop - 15,
        target_stop=stop,
        anchor_sample=stop - 1,
        anchor_time_seconds=float(stop - 1),
        sampling_rate_hz=1.0,
    )


def test_versioned_support_and_fold_ownership_contracts(tmp_path):
    feature = RawInterval("rec", 0, 10)
    outcome = RawInterval("rec", 10, 12)
    support = ExampleSupport((feature,), (outcome,), available_after_sample=12)
    assert support.feature_intervals[0].overlaps(RawInterval("rec", 9, 11))
    with pytest.raises(ValueError, match="disjoint"):
        FoldSpec("fold", ("a",), ("a",))

    legacy = ArtifactProvenance(schema_version="1.0")
    path = save_artifact(legacy, tmp_path / "legacy.pkl")
    loaded = load_artifact(path)
    assert loaded.claim_eligibility == "legacy_unverified"
    with pytest.raises(ValueError, match="legacy_unverified"):
        load_artifact(path, for_confirmation=True)


def test_positive_lag_uses_exclusive_context_stop():
    anchors = [_anchor(index) for index in range(6)]
    assert valid_positive_lag_origins(6, 1, anchors).size == 0
    assert valid_positive_lag_origins(6, 2, anchors).size == 0
    np.testing.assert_array_equal(valid_positive_lag_origins(6, 3, anchors), [0, 1, 2])
    with pytest.raises(ValueError, match="requires temporal anchors"):
        valid_positive_lag_origins(6, 1, strict=True)


def test_last_valid_categorical_target_records_selected_sample():
    recording = NeuralRecording(
        traces=np.arange(8, dtype=np.float32)[None, :],
        time=np.arange(8, dtype=float),
        coords=None,
        neuron_ids=np.array([7]),
        behavior={"motif": np.array([9, 0, 1, 2, 9, 3, 4, 5])},
        fps=1.0,
        outcome_validity={"motif": np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)},
    )
    windows = make_windows(
        recording,
        WindowConfig(
            mode="temporal",
            history_length=4,
            target_length=3,
            stride=4,
            categorical_summary="last_valid",
        ),
    )
    assert windows.behavior_per_window["motif"].tolist() == [1, 5]
    assert windows.metadata["selected_behavior_samples"]["motif"] == [2, 7]


def test_fitted_preprocessing_is_training_owned_and_raw_prediction_equivalent():
    identity = RecordingIdentity(recording_id="rec")
    traces = np.array([[1, 2, 3, 4, 50, 60], [2, 4, 6, 8, 10, 12]], dtype=np.float32)
    recording = NeuralRecording(
        traces=traces,
        time=np.arange(6, dtype=float),
        coords=None,
        neuron_ids=np.array([10, 20]),
        behavior={},
        fps=1.0,
        identity=identity,
    )
    intervals = {"rec": (RawInterval("rec", 0, 4),)}
    fitted = FittedPreprocessor.fit([recording], intervals, PreprocessConfig(zscore=True))
    altered = NeuralRecording(
        traces=np.column_stack([traces[:, :4], traces[:, 4:] * 100]),
        time=recording.time,
        coords=None,
        neuron_ids=recording.neuron_ids,
        behavior={},
        fps=1.0,
        identity=identity,
    )
    repeated = FittedPreprocessor.fit([altered], intervals, PreprocessConfig(zscore=True))
    np.testing.assert_allclose(fitted.means, repeated.means)
    np.testing.assert_allclose(fitted.scales, repeated.scales)

    coefficient_std = np.array([[[0.2, -0.1], [0.3, 0.4]]])
    intercept_std = np.array([0.5, -0.2])
    coefficient_raw, intercept_raw = fitted.coefficients_to_raw(
        coefficient_std, intercept_std
    )
    x_std = np.array([0.7, -1.1])
    x_raw = fitted.means + fitted.scales * x_std
    expected = fitted.means + fitted.scales * (
        intercept_std + coefficient_std[0].T @ x_std
    )
    actual = intercept_raw + coefficient_raw[0].T @ x_raw
    np.testing.assert_allclose(actual, expected)


def _primary_windows() -> WindowedSegments:
    rng = np.random.default_rng(4)
    n_windows, length, n_neurons = 12, 20, 2
    segments = rng.normal(size=(n_windows, length, n_neurons)).astype(np.float32)
    anchors = []
    windows = []
    for index in range(n_windows):
        start = index * length
        stop = start + length
        windows.append(Window(start, stop))
        anchors.append(
            TemporalAnchor(
                dataset_id="ds",
                recording_id="rec",
                animal_id="animal",
                session_id=None,
                segment_id=None,
                context_start=start,
                context_stop=stop,
                target_start=stop - 4,
                target_stop=stop,
                anchor_sample=stop - 1,
                anchor_time_seconds=float(stop - 1),
                sampling_rate_hz=1.0,
            )
        )
    return WindowedSegments(
        segments=segments,
        windows=windows,
        behavior_per_window={"motif": np.arange(n_windows) % 2},
        n_neurons=n_neurons,
        fps=1.0,
        anchors=anchors,
    )


def test_primary_evaluator_emits_paired_probability_records():
    result = run_primary_evaluation(
        _primary_windows(),
        EvaluationConfig(
            n_outer_folds=2,
            min_train_examples=3,
            endpoint_name="motif",
            positive_lag=1,
            neural_lambda1_grid=(0.0,),
            decoder_c_grid=(1.0,),
        ),
    )
    assert result.diagnostics["status"] == "primary_probability_comparison"
    issued = [record for record in result.predictions if record.status == "issued"]
    assert issued
    assert {record.model_name for record in issued} == {"M0_activity", "M1_effectome"}
    assert all(sum(record.probabilities) == pytest.approx(1.0) for record in issued)


def test_confirmatory_config_rejects_unresolved_scientific_fields():
    with pytest.raises(ValueError, match="unresolved requirements"):
        EvaluationConfig(
            mode="confirmatory",
            endpoint_kind="synthetic",
            class_vocabulary=(0, 1),
            practical_gain_nats=0.01,
        )
    resolved = EvaluationConfig(
        mode="confirmatory",
        endpoint_kind="synthetic",
        class_vocabulary=(0, 1),
        practical_gain_nats=0.01,
        endpoint_verified=True,
        panel_mapping_id="synthetic-panel-v1",
        calibration_status="passed",
        resource_budget_id="budget-v1",
        source_manifest_hash="sha256:fixture",
        claim_to_test_map=(("incremental_predictive_value", "paired_log_loss"),),
    )
    assert resolved.mode == "confirmatory"


def test_primary_evaluator_supports_continuous_probability_records():
    windows = _primary_windows()
    windows.behavior_per_window["continuous"] = np.linspace(-1.0, 1.0, windows.n_windows)
    result = run_primary_evaluation(
        windows,
        EvaluationConfig(
            endpoint_name="continuous",
            endpoint_type="continuous",
            n_outer_folds=2,
            min_train_examples=3,
            neural_lambda1_grid=(0.0,),
            decoder_alpha_grid=(1.0,),
        ),
    )
    issued = [record for record in result.predictions if record.status == "issued"]
    assert issued
    assert all(record.predictive_scale and record.predictive_scale > 0 for record in issued)
    score = score_regression_records(
        result.predictions,
        model_name="M0_activity",
        endpoint_name="continuous",
    )
    assert np.isfinite(score.gaussian_log_loss_nats)
    delta = paired_log_loss_delta(
        result.predictions,
        baseline_model="M0_activity",
        candidate_model="M1_effectome",
        endpoint_name="continuous",
    )
    assert delta.n_matched_predictions > 0


def test_primary_evaluation_fold_checkpoints_combine_to_whole_result():
    windows = _primary_windows()
    cfg = EvaluationConfig(
        n_outer_folds=2,
        min_train_examples=3,
        endpoint_name="motif",
        neural_lambda1_grid=(0.0,),
        decoder_c_grid=(1.0,),
    )
    fold_ids = primary_evaluation_fold_ids(windows, cfg)
    chunks = [
        run_primary_evaluation(windows, cfg, selected_fold_ids=frozenset({fold_id}))
        for fold_id in fold_ids
    ]
    combined = combine_primary_evaluation_results(chunks, cfg=cfg)
    whole = run_primary_evaluation(windows, cfg)
    assert combined.predictions == whole.predictions
    assert combined.diagnostics["checkpoint_mode"] == "outer_fold"
