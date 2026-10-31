"""Frozen-cohort confirmation: independent units, complete families, and failure gates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from effectome.workflows.confirmation import attach_confirmation, preflight_confirmation
from effectome.workflows.evaluation import (
    EvaluationConfig,
    build_primary_inputs_from_manifest,
    combine_primary_evaluation_results,
    raw_manifest_source_hash,
    run_evaluation_from_raw_manifest,
    run_primary_evaluation,
)


def _cohort_manifest(n_evaluation: int = 6) -> dict:
    """Scientific gate values below are synthetic fixture attestations, never biological evidence."""
    manifest = {
        "seed": 401,
        "data": [
            {
                "name": "synthetic", "recording_id": f"rec-{index}", "animal_id": f"animal-{index}",
                "n_neurons": 2, "n_timepoints": 120, "max_lag": 1, "n_states": 1,
                "behavior_drivers": 1,
                "neuron_panel": {
                    "mapping_id": "fixture-panel", "source_indices": [0, 1],
                    "canonical_ids": ["simulator-axis-0", "simulator-axis-1"],
                    "provenance": "Synthetic fixture explicitly declares identical simulator axes.",
                },
                "fps": 1.0, "seed": 401 + index,
            }
            for index in range(n_evaluation + 1)
        ],
        "preprocess": {},
        "windowing": {
            "mode": "temporal", "history_length": 20, "target_length": 4, "stride": 20,
            "standardize_per_window": False, "categorical_summary": "last_valid",
        },
        "evaluation": {
            "mode": "confirmatory", "protocol": "fixed_development",
            "development_animal_ids": ["animal-0"],
            "evaluation_animal_ids": [f"animal-{index}" for index in range(1, n_evaluation + 1)],
            "endpoint_kind": "synthetic_fixture", "class_vocabulary": [0, 1],
            "class_vocabulary_source": "dataset_manifest", "practical_gain_nats": 0.0,
            "endpoint_verified": True, "panel_mapping_id": "fixture-panel",
            "calibration_status": "passed", "resource_budget_id": "fixture-budget",
            "source_manifest_hash": "pending", "null_replicates": 199,
            "null_families": ["animal_sign_flip"], "sign_symmetry_justified": True,
            "multiplicity_family": "fixture-family", "multiplicity_members": ["paired_log_loss"],
            "claim_to_test_map": [["predictive_performance", "paired_log_loss"]],
            "neural_lambda1_grid": [0.0], "decoder_c_grid": [1.0],
            "tuning_trial_ceiling": 3, "runtime_seconds_ceiling": 60.0,
            "memory_megabytes_ceiling": 512.0,
        },
    }
    manifest["evaluation"]["source_manifest_hash"] = raw_manifest_source_hash(manifest)
    return manifest


def _config(**overrides) -> EvaluationConfig:
    return EvaluationConfig.from_mapping({**_cohort_manifest()["evaluation"], **overrides})


def test_raw_cohort_confirmation_runs_complete_exact_null():
    result = run_evaluation_from_raw_manifest(_cohort_manifest())
    report = result.diagnostics["confirmation"]
    assert report["status"] == "computed"
    assert report["method"] == "exact"
    assert report["null_draw_count"] == 64
    assert report["p_value"] == 1.0  # Both constant-label decoders are identical.
    assert set(report["per_animal_delta"]) == {f"animal-{index}" for index in range(1, 7)}
    assert result.provenance.fit_data_ids == ("rec-0",)
    assert report["conditional_information_null_tested"] is False
    assert report["raw_process_structure_null_tested"] is False
    assert result.diagnostics["resource_accounting"]["status"] == "within_registered_budget"
    fold = result.manifest.outer_folds[0]
    assert set(fold.train_indices).isdisjoint(fold.test_indices)
    assert all(record.animal_id != "animal-0" for record in result.predictions)
    assert all(record.target_start >= record.issue_sample for record in result.predictions)


def test_null_resolution_checks_independent_animals_and_mc_draws():
    with pytest.raises(ValueError, match="finite control pool"):
        preflight_confirmation(_config(evaluation_animal_ids=tuple(f"a{i}" for i in range(5))))
    with pytest.raises(ValueError, match="Monte Carlo pool"):
        preflight_confirmation(_config(sign_flip_method="monte_carlo", null_replicates=9))
    with pytest.raises(ValueError, match="overlapping CV"):
        preflight_confirmation(_config(protocol="leave_animal_out"))
    with pytest.raises(ValueError, match="sign-symmetry"):
        preflight_confirmation(_config(sign_symmetry_justified=False))
    with pytest.raises(ValueError, match="separately registered benchmark"):
        preflight_confirmation(_config(null_families=("fixed_interaction",)))


def test_seeded_monte_carlo_is_reproducible_and_plus_one():
    manifest = _cohort_manifest()
    manifest["evaluation"]["sign_flip_method"] = "monte_carlo"
    first = run_evaluation_from_raw_manifest(manifest)
    _, cfg = build_primary_inputs_from_manifest(manifest)
    # Reuse frozen predictions; null execution never fits another model.
    second = attach_confirmation(first, cfg)
    a, b = first.diagnostics["confirmation"], second.diagnostics["confirmation"]
    assert a["null_statistics"] == b["null_statistics"]
    assert a["p_value"] == b["p_value"] == 1.0
    assert a["null_draw_count"] == 199
    assert a["p_value"] >= 1 / 200


def test_complete_family_retains_unimplemented_and_failed_members():
    manifest = _cohort_manifest(7)
    manifest["evaluation"]["multiplicity_members"] = ["paired_log_loss", "optional_unimplemented"]
    result = run_evaluation_from_raw_manifest(manifest)
    family = result.diagnostics["confirmation"]["family"]
    assert set(family) == {"paired_log_loss", "optional_unimplemented"}
    assert family["optional_unimplemented"]["p_value"] is None
    assert family["optional_unimplemented"]["q_value"] == 1.0
    assert family["optional_unimplemented"]["status"].startswith("unavailable")
    _, cfg = build_primary_inputs_from_manifest(manifest)
    broken = replace(result, predictions=(
        replace(result.predictions[0], status="failed", failure_reason="fixture_failure"),
        *result.predictions[1:],
    ))
    unavailable = attach_confirmation(broken, cfg).diagnostics["confirmation"]
    assert unavailable["status"] == "unavailable"
    assert unavailable["p_value"] is None
    assert unavailable["null_statistics"] == []
    assert unavailable["family"]["paired_log_loss"]["q_value"] == 1.0
    dropped_pair = replace(result, predictions=result.predictions[2:])
    dropped = attach_confirmation(dropped_pair, cfg).diagnostics["confirmation"]
    assert dropped["status"] == "unavailable"
    assert dropped["failure_reason"] == "registered_prediction_pairs_incomplete"
    assert dropped["p_value"] is None


def test_heldout_labels_cannot_change_tuning_or_issued_predictions():
    windows, cfg = build_primary_inputs_from_manifest(_cohort_manifest())
    # A declared two-class vocabulary avoids using the held-out class set in either fit.
    cfg = replace(cfg, class_vocabulary=(0, 1))
    first = run_primary_evaluation(windows, cfg)
    changed_labels = windows.behavior_per_window["motif"].copy()
    for index, anchor in enumerate(windows.anchors):
        if anchor.animal_id in cfg.evaluation_animal_ids:
            changed_labels[index] = 1 - changed_labels[index]
    changed = replace(windows, behavior_per_window={"motif": changed_labels})
    second = run_primary_evaluation(changed, cfg)
    assert first.diagnostics["folds"] == second.diagnostics["folds"]
    assert [record.probabilities for record in first.predictions] == [
        record.probabilities for record in second.predictions
    ]
    assert first.provenance.fit_data_ids == second.provenance.fit_data_ids == ("rec-0",)


def test_manifest_hash_cohort_identity_and_membership_are_checked():
    manifest = _cohort_manifest()
    manifest["data"][0]["seed"] += 1
    with pytest.raises(ValueError, match="source_manifest_hash"):
        build_primary_inputs_from_manifest(manifest)
    manifest = _cohort_manifest()
    manifest["data"][1]["animal_id"] = "unregistered-animal"
    manifest["evaluation"]["source_manifest_hash"] = raw_manifest_source_hash(manifest)
    with pytest.raises(ValueError, match="membership"):
        run_evaluation_from_raw_manifest(manifest)
    with pytest.raises(ValueError, match="disjoint"):
        _config(evaluation_animal_ids=("animal-0",))


def test_tuning_budget_rejects_before_any_fitting_and_runtime_breach_is_unavailable(monkeypatch):
    import effectome.workflows.evaluation as evaluation

    windows, cfg = build_primary_inputs_from_manifest(_cohort_manifest())
    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "_run_primary_evaluation", lambda *args, **kwargs: pytest.fail("fitted"))
        with pytest.raises(ValueError, match="tuning trial ceiling"):
            run_primary_evaluation(windows, replace(cfg, tuning_trial_ceiling=2))
    result = run_primary_evaluation(windows, replace(cfg, runtime_seconds_ceiling=1e-12))
    assert result.diagnostics["resource_accounting"]["failures"] == ["runtime_budget_exceeded"]
    assert result.diagnostics["confirmation"]["status"] == "unavailable"
    assert result.diagnostics["confirmation"]["p_value"] is None
    with pytest.raises(ValueError, match="atomic whole-cohort"):
        run_primary_evaluation(windows, cfg, selected_fold_ids=frozenset({"outer-000"}))
    with pytest.raises(ValueError, match="cannot be combined"):
        combine_primary_evaluation_results([result], cfg=cfg)


def test_exact_animal_sign_flip_has_expected_two_sided_tail():
    from effectome.workflows.confirmation import _sign_null

    values = np.arange(1, 7, dtype=float)
    null = _sign_null(values, method="exact", draws=199, seed=9)
    assert np.mean(null >= abs(values.mean())) == 2 / 64


@pytest.mark.parametrize("indices", [[0, 0], [0, 9], [0, 0.5]])
def test_panel_mapping_rejects_ambiguous_or_invalid_indices(indices):
    manifest = _cohort_manifest()
    manifest["data"][0]["neuron_panel"]["source_indices"] = indices
    manifest["evaluation"]["source_manifest_hash"] = raw_manifest_source_hash(manifest)
    with pytest.raises(ValueError, match="unique in-range integer"):
        build_primary_inputs_from_manifest(manifest)


def test_cohort_cannot_use_recording_local_ids_as_cross_animal_registration():
    manifest = _cohort_manifest()
    manifest["data"][0].pop("neuron_panel")
    manifest["evaluation"]["source_manifest_hash"] = raw_manifest_source_hash(manifest)
    with pytest.raises(ValueError, match="local neuron IDs do not align animals"):
        build_primary_inputs_from_manifest(manifest)


def test_canonical_panel_reorders_activity_coordinates_and_validity_together(monkeypatch):
    import importlib

    from effectome.data_module import NeuralRecording, RecordingIdentity, loaders

    preprocessing = importlib.import_module("effectome.data_module.preprocess")
    original_preprocess = preprocessing.preprocess
    captured = []
    activity = np.arange(360, dtype=np.float32).reshape(3, 120)
    coordinates = np.arange(9).reshape(3, 3)
    validity = np.ones((3, 120), dtype=bool)
    validity[2, 5] = False

    def loader(data):
        order = np.asarray([0, 1, 2] if data["recording_id"] == "rec-0" else [2, 0, 1])
        return NeuralRecording(
            traces=activity[order], time=np.arange(120, dtype=float),
            coords=coordinates[order], neuron_ids=np.arange(3),
            neural_validity=validity[order], behavior={"motif": np.ones(120, dtype=int)}, fps=1.0,
            identity=RecordingIdentity(recording_id=data["recording_id"], animal_id=data["animal_id"]),
        )

    def capture(recording, config):
        captured.append(recording)
        return original_preprocess(recording, config)

    monkeypatch.setattr(loaders, "get_loader", lambda name: loader)
    monkeypatch.setattr(preprocessing, "preprocess", capture)
    manifest = _cohort_manifest()
    for index, data in enumerate(manifest["data"]):
        data["neuron_panel"]["source_indices"] = [2, 0] if index == 0 else [0, 1]
        data["neuron_panel"]["canonical_ids"] = ["cell-C", "cell-A"]
    manifest["evaluation"]["source_manifest_hash"] = raw_manifest_source_hash(manifest)
    build_primary_inputs_from_manifest(manifest)
    for recording in captured:
        np.testing.assert_array_equal(recording.traces, activity[[2, 0]])
        np.testing.assert_array_equal(recording.coords, coordinates[[2, 0]])
        np.testing.assert_array_equal(recording.neural_validity, validity[[2, 0]])
        assert recording.neuron_ids.tolist() == ["cell-C", "cell-A"]


def test_external_source_hash_is_verified_before_load_and_rejects_changed_bytes(tmp_path):
    import hashlib

    path = tmp_path / "recording.npz"
    np.savez(
        path, traces=np.ones((2, 120), dtype=np.float32), motif=np.ones(120, dtype=int),
        continuous=np.zeros(120),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = _cohort_manifest()
    manifest["data"][0] = {
        "name": "c_elegans", "path": str(path), "recording_id": "rec-0", "animal_id": "animal-0",
        "fps": 1.0, "behavior_keys": {"motif": "motif", "continuous": "continuous"},
        "source_file_sha256": digest,
        "neuron_panel": manifest["data"][0]["neuron_panel"],
    }
    manifest["evaluation"]["source_manifest_hash"] = raw_manifest_source_hash(manifest)
    windows, _ = build_primary_inputs_from_manifest(manifest)
    contract = windows.metadata["raw_evaluation_contract"]
    assert contract["verified_source_files"]["rec-0"] == {str(path.resolve()): digest}
    path.write_bytes(path.read_bytes() + b"changed in place")
    with pytest.raises(ValueError, match="source file SHA-256 mismatch"):
        build_primary_inputs_from_manifest(manifest)


def test_directory_source_hash_requires_complete_consumed_file_coverage(tmp_path):
    import hashlib

    from effectome.workflows.confirmation import validate_source_files

    files = {"recordings": "raw.mat", "class_ids": "labels.mat", "neuron_order": "order.mat"}
    for filename in files.values():
        (tmp_path / filename).write_bytes(b"fixture")
    data = {
        "name": "bundle_net_c_elegans", "path": str(tmp_path), "files": files,
        "downloaded_sha256": {"raw.mat": hashlib.sha256(b"fixture").hexdigest()},
    }
    with pytest.raises(ValueError, match="exactly every consumed"):
        validate_source_files(data, required=True)
    data["downloaded_sha256"] = {
        filename: hashlib.sha256(b"fixture").hexdigest() for filename in files.values()
    }
    assert len(validate_source_files(data, required=True)) == 3


def test_direct_confirmation_rejects_missing_raw_source_panel_contract():
    windows, cfg = build_primary_inputs_from_manifest(_cohort_manifest())
    altered = replace(windows, metadata={key: value for key, value in windows.metadata.items()
                                         if key != "raw_evaluation_contract"})
    with pytest.raises(ValueError, match="raw-source and canonical-panel contract"):
        run_primary_evaluation(altered, cfg)
