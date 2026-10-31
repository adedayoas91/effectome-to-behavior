"""Outer-owned notebook features must survive changes confined to held-out data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

import notebooks._shared as shared
from effectome.attribution.scoring import CandidateDriverResult, DriverScore
from effectome.connectivity import ConnectivityConfig
from effectome.data_module import PreprocessConfig, WindowConfig, make_windows, preprocess
from effectome.data_module.schema import NeuralRecording
from effectome.utils.io import save_artifact
from effectome.workflows.feature_ownership import (
    chronological_feature_split,
    fit_owned_feature_artifacts,
    load_or_fit_owned_features,
)


def _raw_windows():
    rng = np.random.default_rng(19)
    n_samples = 640
    recording = NeuralRecording(
        traces=rng.normal(size=(3, n_samples)),
        time=np.arange(n_samples, dtype=float),
        neuron_ids=np.arange(3),
        fps=1.0,
        coords=None,
        behavior={"continuous": np.sin(np.arange(n_samples) * 0.05)},
    )
    recording = preprocess(
        recording,
        PreprocessConfig(
            detrend=False, zscore=False, smooth_window=0, deconvolve=False, drop_low_variance=0.0
        ),
    )
    windows = make_windows(
        recording,
        WindowConfig(
            mode="temporal", history_length=24, target_length=4, stride=12, standardize_per_window=False
        ),
    )
    return recording, windows


@pytest.mark.parametrize("method", ["cgc", "tvvar_smooth", "sparse_var"])
def test_owned_fits_and_training_outputs_ignore_future_changes(method):
    _, windows = _raw_windows()
    fit, evaluation = chronological_feature_split(windows)
    cfg = ConnectivityConfig(name=method, max_lag=1, extra={"standardize": False} if method != "cgc" else {})
    kwargs = dict(
        connectivity_cfg=cfg,
        fit_indices=fit,
        evaluation_indices=evaluation,
        fit_scope_id="outer-0",
        n_dims=2,
        n_roles=2,
    )
    first = fit_owned_feature_artifacts(windows, **kwargs)
    changed = deepcopy(windows)
    changed.segments[evaluation] *= 17
    changed.behavior_per_window["continuous"][evaluation] += 400
    second = fit_owned_feature_artifacts(changed, **kwargs)
    for key in ("scale_mean", "scale_scale", "role_centroids"):
        np.testing.assert_allclose(first.training_parameters[key], second.training_parameters[key])
    np.testing.assert_allclose(first.connectivity.matrices[fit], second.connectivity.matrices[fit])
    np.testing.assert_allclose(first.manifold.window_embedding[fit], second.manifold.window_embedding[fit])
    np.testing.assert_array_equal(first.community.labels[fit], second.community.labels[fit])
    assert first.connectivity.provenance.fit_scope_id == first.manifold.provenance.fit_scope_id == "outer-0"


def test_owned_features_reject_index_disjoint_raw_overlap():
    _, windows = _raw_windows()
    with pytest.raises(ValueError, match="raw sample"):
        fit_owned_feature_artifacts(
            windows,
            connectivity_cfg=ConnectivityConfig(name="cgc", max_lag=1),
            fit_indices=np.arange(20),
            evaluation_indices=np.arange(20, 30),
            fit_scope_id="overlap",
        )


def test_notebook_requires_produced_matching_ownership(tmp_path):
    run = shared.make_run(tmp_path, "run", "cgc")
    with pytest.raises(ValueError, match="explicit outer ownership"):
        shared.run_candidate_drivers(run, recording_path=tmp_path / "recording.pkl", behavior_key="x")


def test_no_candidate_is_unavailable_without_substituting_top_node(tmp_path, monkeypatch):
    recording, windows = _raw_windows()
    recording_path, windows_path = tmp_path / "recording.pkl", tmp_path / "windows.pkl"
    save_artifact(recording, recording_path)
    save_artifact(windows, windows_path)
    run = shared.make_run(tmp_path / "runs", "owned", "cgc")
    owned = shared.run_owned_feature_artifacts(
        run,
        windows_path=windows_path,
        connectivity_cfg=ConnectivityConfig(name="cgc", max_lag=1),
        fit_scope_id="outer-0",
        n_dims=2,
        n_roles=2,
    )
    ownership = dict(
        fit_indices=owned.fit_indices,
        evaluation_indices=owned.evaluation_indices,
        fit_scope_id=owned.fit_scope_id,
    )
    result = CandidateDriverResult(
        [], "continuous", "future_behavior", 95.0, 0, fit_scope_id="outer-0", candidate_nodes=()
    )
    run.execute("candidate_drivers", lambda: result, config=ownership)
    monkeypatch.setattr(
        shared,
        "fit_linear_surrogate",
        lambda *args, **kwargs: pytest.fail("no candidate must not fit a surrogate"),
    )
    sensitivity = shared.run_feature_sensitivity(
        run,
        recording_path=recording_path,
        behavior_key="continuous",
        **ownership,
        multiplicity_family=("continuous:feature-edit",),
        hypothesis_id="continuous:feature-edit",
    )
    assert sensitivity.node_indices == []
    assert sensitivity.fail_safe_reason == "no_preliminary_candidate_nodes"
    assert sensitivity.effect_size is None
    assert not sensitivity.biological_intervention_identified
    with pytest.raises(ValueError, match="frozen outer-owned"):
        shared.run_candidate_drivers(
            run,
            recording_path=recording_path,
            behavior_key="continuous",
            **{**ownership, "fit_scope_id": "different"},
        )


def test_cli_bundle_cache_tracks_data_and_producer_settings(tmp_path):
    _, windows = _raw_windows()
    path, bundle = tmp_path / "windows.pkl", tmp_path / "owned.pkl"
    save_artifact(windows, path)
    cfg = ConnectivityConfig(name="cgc", max_lag=1)
    first = load_or_fit_owned_features(path, bundle, cfg, {"n_dims": 2})
    second = load_or_fit_owned_features(path, bundle, cfg, {"n_dims": 1})
    assert first.manifold.window_embedding.shape[1] == 2
    assert second.manifold.window_embedding.shape[1] == 1
    assert first.training_parameters["input_signature"] != second.training_parameters["input_signature"]
    windows.segments[-1] *= 2
    save_artifact(windows, path)
    third = load_or_fit_owned_features(path, bundle, cfg, {"n_dims": 1})
    assert third.training_parameters["input_signature"] != second.training_parameters["input_signature"]


def test_strict_notebook_surrogate_executes_with_frozen_training_candidate(tmp_path):
    recording, windows = _raw_windows()
    recording_path, windows_path = tmp_path / "recording.pkl", tmp_path / "windows.pkl"
    save_artifact(recording, recording_path)
    save_artifact(windows, windows_path)
    run = shared.make_run(tmp_path / "runs", "strict", "cgc")
    owned = shared.run_owned_feature_artifacts(
        run,
        windows_path=windows_path,
        connectivity_cfg=ConnectivityConfig(name="cgc", max_lag=1),
        fit_scope_id="outer-0",
        n_dims=2,
        n_roles=2,
    )
    # Freeze a declared synthetic candidate to exercise the surrogate handoff,
    # independently of whether the unrelated random fixture passes screening.
    score = DriverScore(0, 1.0, 1.0, 0.0, 0.1, 0.0, True)
    screen = CandidateDriverResult(
        [score], "continuous", "future_behavior", 95.0, 2, fit_scope_id="outer-0", candidate_nodes=(0,)
    )
    run.execute("candidate_drivers", lambda: screen, config={"synthetic_fixture": True})
    result = shared.run_feature_sensitivity(
        run,
        recording_path=recording_path,
        behavior_key="continuous",
        n_folds=3,
        fit_indices=owned.fit_indices,
        evaluation_indices=owned.evaluation_indices,
        fit_scope_id="outer-0",
        multiplicity_family=("continuous:feature-edit",),
        hypothesis_id="continuous:feature-edit",
        min_skill=-1e6,
        min_absolute_skill=-1e6,
        max_distribution_shift=1e6,
        dose_scales=[1.0, 0.0],
    )
    model, _ = shared.require_stage(run, "surrogate_model")
    assert model.metadata["strict_outer_ownership"]
    assert model.metadata["candidate_nodes"] == [0]
    assert model.metadata["fit_indices"] == owned.fit_indices.tolist()
    assert model.metadata["multiplicity_family"] == ["continuous:feature-edit"]
    assert result.node_indices == [0]
    assert not result.biological_intervention_identified


def test_confirmatory_notebook_checkpoints_whole_cohort_once(tmp_path, monkeypatch):
    @dataclass
    class ConfirmationFixture:
        mode: str = "confirmatory"

    _, windows = _raw_windows()
    windows_path = tmp_path / "windows.pkl"
    save_artifact(windows, windows_path)
    run = shared.make_run(tmp_path / "runs", "confirmation", "primary")
    calls = []

    def evaluate(owned_windows, config):
        calls.append((owned_windows.n_windows, config.mode))
        return {"whole_cohort": True}

    monkeypatch.setattr(shared, "run_evaluation_profile", evaluate)
    monkeypatch.setattr(
        shared,
        "primary_evaluation_fold_ids",
        lambda *args, **kwargs: pytest.fail("confirmation must not split into fold chunks"),
    )
    for _ in range(2):
        result = shared.run_primary_probability_evaluation(
            run, windows_path=windows_path, evaluation_cfg=ConfirmationFixture()
        )
        assert result == {"whole_cohort": True}
    assert calls == [(windows.n_windows, "confirmatory")]
