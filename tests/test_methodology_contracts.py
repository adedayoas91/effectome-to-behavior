"""Focused acceptance contracts for Appendix-C WP0, WP1, and WP4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from effectome.connectivity import ConnectivityConfig
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
    build_primary_inputs_from_manifest,
    combine_primary_evaluation_results,
    paired_log_loss_delta,
    primary_evaluation_fold_ids,
    run_evaluation_from_raw_manifest,
    run_evaluation_profile,
    run_evaluation_skeleton,
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
        metadata={
            "neural_standardization": {"enabled": False, "scope": "none"},
            "preprocess_dependency": {
                "kind": "finite",
                "past_support": 0,
                "future_support": 0,
            },
        },
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


def test_primary_evaluator_rejects_unverified_preprocessing_ownership():
    missing = _primary_windows()
    missing.metadata.pop("preprocess_dependency")
    with pytest.raises(ValueError, match="preprocessing provenance"):
        run_primary_evaluation(missing, EvaluationConfig(n_outer_folds=2))

    global_fit = _primary_windows()
    global_fit.metadata["preprocess_dependency"] = {"kind": "global"}
    with pytest.raises(ValueError, match="recording-global preprocessing"):
        run_primary_evaluation(global_fit, EvaluationConfig(n_outer_folds=2))


def test_observed_development_vocabulary_is_owned_by_each_outer_fold():
    windows = _primary_windows()
    cfg = EvaluationConfig(
        n_outer_folds=2,
        min_train_examples=3,
        endpoint_name="motif",
        positive_lag=1,
        neural_lambda1_grid=(0.0,),
        decoder_c_grid=(1.0,),
    )
    first = run_primary_evaluation(windows, cfg)
    fold = next(item for item in first.manifest.outer_folds if item.fold_id == "outer-000")
    origins = valid_positive_lag_origins(
        windows.n_windows,
        cfg.positive_lag,
        windows.anchors,
        strict=True,
    )
    changed_origin = int(origins[fold.test_indices[0]])
    windows.behavior_per_window["motif"][changed_origin + cfg.positive_lag] = 2
    second = run_primary_evaluation(windows, cfg)

    first_fold = [record for record in first.predictions if record.fold_id == fold.fold_id]
    second_fold = [record for record in second.predictions if record.fold_id == fold.fold_id]
    assert {record.class_order for record in first_fold} == {(0, 1)}
    assert {record.class_order for record in second_fold} == {(0, 1)}
    unaffected_first = {
        (record.origin_index, record.model_name): record.probabilities
        for record in first_fold
        if record.origin_index != changed_origin
    }
    unaffected_second = {
        (record.origin_index, record.model_name): record.probabilities
        for record in second_fold
        if record.origin_index != changed_origin
    }
    assert unaffected_second == unaffected_first
    changed = [
        record for record in second_fold if record.origin_index == changed_origin
    ]
    assert changed
    assert all(record.status == "failed" for record in changed)
    assert all(
        record.failure_reason == "outer_test_class_not_in_training_vocabulary"
        for record in changed
    )


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
        class_vocabulary_source="dataset_manifest",
        practical_gain_nats=0.01,
        endpoint_verified=True,
        panel_mapping_id="synthetic-panel-v1",
        calibration_status="passed",
        resource_budget_id="budget-v1",
        source_manifest_hash="sha256:fixture",
        claim_to_test_map=(("incremental_predictive_value", "paired_log_loss"),),
        tuning_trial_ceiling=10,
        runtime_seconds_ceiling=60.0,
        memory_megabytes_ceiling=512.0,
        null_replicates=200,
        null_families=("fixed_interaction",),
        multiplicity_family="primary_incremental_value",
    )
    assert resolved.mode == "confirmatory"
    with pytest.raises(ValueError, match="frozen fixed_development"):
        run_primary_evaluation(_primary_windows(), resolved)
    with pytest.raises(NotImplementedError, match="split inspection"):
        run_evaluation_skeleton(_primary_windows(), resolved)


def test_evaluation_mapping_and_connectivity_extras_are_strict():
    with pytest.raises(ValueError, match="unknown evaluation config keys.*typo"):
        EvaluationConfig.from_mapping({"profile": "revised_primary", "typo": 1})
    cfg = EvaluationConfig.from_mapping(
        {
            "neural_lambda1_grid": [0.0, 0.1],
            "claim_to_test_map": [["incremental", "paired_log_loss"]],
            "null_families": ["fixed_interaction"],
        }
    )
    assert cfg.neural_lambda1_grid == (0.0, 0.1)
    assert cfg.claim_to_test_map == (("incremental", "paired_log_loss"),)
    assert cfg.null_families == ("fixed_interaction",)
    assert EvaluationConfig.from_profile("revised_primary").profile == "revised_primary"

    with pytest.raises(ValueError, match="unknown extra keys.*lamdba1"):
        ConnectivityConfig(name="sparse_var", extra={"lamdba1": 0.1})

    for method in ("correlation", "pcmci", "time_varying"):
        resolved = ConnectivityConfig(name=method, max_lag_seconds=0.7).resolve(10.0)
        assert resolved.max_lag == 7
        assert "n_lags" not in resolved.extra
    assert (
        ConnectivityConfig(name="sparse_var", max_lag_seconds=0.7)
        .resolve(10.0)
        .extra["n_lags"]
        == 7
    )


def test_evaluation_results_carry_validated_semantic_provenance(tmp_path):
    windows = _primary_windows()
    cfg = EvaluationConfig(
        n_outer_folds=2,
        min_train_examples=3,
        endpoint_name="motif",
        neural_lambda1_grid=(0.0,),
        decoder_c_grid=(1.0,),
    )
    result = run_primary_evaluation(windows, cfg)
    provenance = result.provenance
    assert provenance.stage == "evaluation"
    assert provenance.config_hash
    assert provenance.split_manifest_hash
    assert provenance.source_hash
    assert provenance.code_version
    assert provenance.lock_hash
    assert provenance.metadata["endpoint_identity"]
    assert load_artifact(save_artifact(result, tmp_path / "evaluation.pkl")) == result


def test_legacy_profile_is_executable_but_not_routed_to_revised_primary():
    cfg = EvaluationConfig.from_profile("legacy_descriptive")
    result = run_evaluation_profile(_primary_windows(), cfg)
    assert result.diagnostics["status"] == "legacy_descriptive_historical_profile"
    assert result.predictions == ()
    assert result.manifest.states_enabled
    assert result.manifest.groups_enabled


def test_raw_manifest_entry_point_matches_direct_primary_runner():
    manifest = {
        "seed": 7,
        "data": {
            "name": "synthetic",
            "dataset_id": "tiny-synthetic",
            "recording_id": "tiny-0",
            "n_neurons": 2,
            "n_timepoints": 120,
            "density": 0.5,
            "max_lag": 1,
            "coupling": 0.2,
            "noise_std": 0.2,
            "n_states": 2,
            "regime_dwell": 30,
            "behavior_drivers": 1,
            "fps": 1.0,
            "seed": 7,
        },
        "preprocess": {},
        "windowing": {
            "length": 20,
            "history_length": 20,
            "target_length": 4,
            "stride": 10,
            "mode": "temporal",
            "standardize_per_window": False,
            "categorical_summary": "last_valid",
        },
        "evaluation": {
            "profile": "revised_primary",
            "mode": "development",
            "n_outer_folds": 2,
            "embargo": 0,
            "min_train_examples": 2,
            "endpoint_name": "motif",
            "categorical_summary": "last_valid",
            "class_vocabulary_source": "observed_development_only",
            "neural_lambda1_grid": [0.0],
            "decoder_c_grid": [1.0],
        },
    }
    windows, cfg = build_primary_inputs_from_manifest(manifest)
    direct = run_evaluation_profile(windows, cfg)
    handed_off = run_evaluation_from_raw_manifest(manifest)
    assert handed_off.predictions == direct.predictions
    assert handed_off.manifest == direct.manifest


def test_checked_in_synthetic_raw_manifest_runs_to_predictions():
    path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "evaluation-manifests"
        / "synthetic-smoke.yaml"
    )
    manifest = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(manifest, dict)
    result = run_evaluation_from_raw_manifest(manifest)
    assert result.diagnostics["status"] == "primary_probability_comparison"
    assert any(record.status == "issued" for record in result.predictions)


def test_raw_manifest_primary_handoff_rejects_global_learned_preprocessing():
    manifest = {
        "data": {"name": "synthetic", "n_neurons": 2, "n_timepoints": 40},
        "preprocess": {"zscore": True},
        "windowing": {"mode": "temporal", "history_length": 10, "target_length": 2},
        "evaluation": {},
    }
    with pytest.raises(ValueError, match="outer-owned"):
        build_primary_inputs_from_manifest(manifest)


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
