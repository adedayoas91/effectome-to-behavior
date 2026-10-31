"""Attribution and intervention tests for Stage 7."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.attribution import community_switch_rates, qualify_candidate_drivers, signed_node_roles
from effectome.data_module.schema import ArtifactProvenance, CommunitySeries, ConnectivitySeries
from effectome.intervention import (
    fit_linear_surrogate,
    run_predictor_feature_sensitivity,
    run_virtual_perturbation,
)
from effectome.manifold import ManifoldArtifact, TargetSlice


def _toy_lane_artifacts() -> tuple[ConnectivitySeries, CommunitySeries, ManifoldArtifact, np.ndarray]:
    n_windows = 60
    n_nodes = 3
    signal = np.sin(np.linspace(0, 4 * np.pi, n_windows))
    matrices = np.zeros((n_windows, n_nodes, n_nodes), dtype=np.float32)
    matrices[:, 0, 1] = signal
    matrices[:, 1, 2] = 0.1 * np.cos(np.linspace(0, 2 * np.pi, n_windows))
    matrices[:, 2, 0] = 0.05

    behavior = np.roll(signal, 1)
    behavior[0] = 0.0
    manifold_window = np.stack([0.2 * signal, np.zeros_like(signal)], axis=1).astype(np.float32)

    series = ConnectivitySeries(
        matrices=matrices,
        window_starts=np.arange(n_windows),
        method="toy",
        directed=True,
        behavior_per_window={"continuous": behavior.astype(np.float32)},
    )
    community = CommunitySeries(
        labels=np.column_stack(
            [
                np.zeros(n_windows, dtype=int),
                np.zeros(n_windows, dtype=int),
                np.arange(n_windows) % 2,
            ]
        ),
        method="toy",
        n_communities_per_window=np.full(n_windows, 2),
    )
    manifold = ManifoldArtifact(
        method="toy",
        behavior_key="continuous",
        full_embedding=manifold_window,
        window_embedding=manifold_window,
        target_slices=[TargetSlice(i, i + 1) for i in range(n_windows)],
        target_length=1,
        window_starts=np.arange(n_windows),
    )
    return series, community, manifold, behavior.astype(np.float32)


def _assign_fit_scope(
    series: ConnectivitySeries,
    manifold: ManifoldArtifact,
    fit_scope_id: str,
    community: CommunitySeries | None = None,
) -> None:
    series.provenance = ArtifactProvenance(stage="connectivity", fit_scope_id=fit_scope_id)
    manifold.provenance = ArtifactProvenance(stage="manifold", fit_scope_id=fit_scope_id)
    series.diagnostics["fit_scope_id"] = fit_scope_id
    manifold.metadata["fit_scope_id"] = fit_scope_id
    if community is not None:
        community.provenance = ArtifactProvenance(
            stage="community",
            fit_scope_id=fit_scope_id,
        )
        community.metadata["fit_scope_id"] = fit_scope_id


def test_signed_node_roles_shapes():
    series, _, _, _ = _toy_lane_artifacts()
    roles = signed_node_roles(series)
    assert set(roles) == {
        "outgoing_pos",
        "outgoing_neg",
        "incoming_pos",
        "incoming_neg",
        "net_outgoing",
        "net_incoming",
    }
    assert roles["net_outgoing"].shape == (series.n_windows, series.n_neurons)


def test_attribution_baseline_is_category_code_invariant():
    from effectome.attribution.scoring import _baseline_features

    labels = np.array([0, 1, 2, 1, 0])
    latent = np.zeros((5, 1))
    first = _baseline_features(labels, latent)
    second = _baseline_features(np.array([90, -7, 12])[labels], latent)
    np.testing.assert_allclose(first @ first.T, second @ second.T)


def test_frozen_surrogate_ignores_outer_test_for_fitting_and_matching():
    series, _, manifold, behavior = _toy_lane_artifacts()
    kwargs = dict(fit_indices=np.arange(40), evaluation_indices=np.arange(40, 60),
                  candidate_nodes=[0], endpoint="behavior", min_skill=-1e9,
                  max_shift=1e9)
    first = fit_linear_surrogate(series, manifold, behavior, **kwargs)
    series.matrices[40:] *= 100
    behavior[40:] += 100
    second = fit_linear_surrogate(series, manifold, behavior, **kwargs)
    np.testing.assert_allclose(first.model.coef_, second.model.coef_)
    assert first.metadata["matching_strengths"] == second.metadata["matching_strengths"]
    assert first.metadata["candidate_nodes"] == second.metadata["candidate_nodes"]
    assert first.metadata["fit_origins"] == list(range(39))


def test_strict_surrogate_requires_explicit_aligned_outer_ownership():
    series, _, manifold, behavior = _toy_lane_artifacts()
    _assign_fit_scope(series, manifold, "outer-fold-0")
    common = dict(
        fit_indices=np.arange(40),
        evaluation_indices=np.arange(40, 60),
        fit_scope_id="outer-fold-0",
        strict_outer_ownership=True,
        candidate_nodes=[0],
        multiplicity_family=["continuous:feature-edit"],
        hypothesis_id="continuous:feature-edit",
        endpoint="behavior",
        min_skill=-1e9,
        min_absolute_skill=-1.0,
        max_shift=1e9,
        control_resolution_policy="warn",
    )

    with pytest.warns(RuntimeWarning, match="cannot attain"):
        model = fit_linear_surrogate(series, manifold, behavior, **common)

    assert model.metadata["strict_outer_ownership"] is True
    assert model.metadata["fit_scope_id"] == "outer-fold-0"
    assert model.metadata["outer_ownership_verified"] is True
    assert model.validation.increment_interval is not None
    assert model.validation.mean_absolute_skill is not None
    assert model.validation.evaluation_absolute_skill is not None
    assert model.validation.evaluation_increment is not None

    with pytest.raises(ValueError, match="explicit fit_indices and evaluation_indices"):
        fit_linear_surrogate(
            series,
            manifold,
            behavior,
            fit_scope_id="outer-fold-0",
            strict_outer_ownership=True,
            candidate_nodes=[0],
            multiplicity_family=["continuous:feature-edit"],
            hypothesis_id="continuous:feature-edit",
            min_absolute_skill=0.0,
            max_shift=5.0,
        )

    manifold.provenance = ArtifactProvenance(stage="manifold", fit_scope_id="outer-fold-1")
    manifold.metadata["fit_scope_id"] = "outer-fold-1"
    with pytest.raises(ValueError, match="fit_scope_id"):
        fit_linear_surrogate(series, manifold, behavior, **common)


def test_strict_surrogate_must_pass_the_owned_held_out_skill_gate():
    series, _, manifold, behavior = _toy_lane_artifacts()
    _assign_fit_scope(series, manifold, "outer-fold-0")
    behavior[40:] = np.random.default_rng(4).normal(0.0, 10.0, size=20)

    model = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        fit_indices=np.arange(40),
        evaluation_indices=np.arange(40, 60),
        fit_scope_id="outer-fold-0",
        strict_outer_ownership=True,
        candidate_nodes=[0],
        multiplicity_family=["continuous:feature-edit"],
        hypothesis_id="continuous:feature-edit",
        endpoint="behavior",
        min_skill=0.0,
        min_absolute_skill=0.0,
        max_shift=1e9,
    )

    assert model.validation.mean_skill > 0.0
    assert model.validation.evaluation_absolute_skill < 0.0
    assert model.validation.status == "invalid"
    assert model.validation.fail_safe_reason == "held_out_absolute_skill_gate_failed"


def test_strict_surrogate_uses_the_registered_increment_interval_gate(monkeypatch):
    series, _, manifold, behavior = _toy_lane_artifacts()
    _assign_fit_scope(series, manifold, "outer-fold-0")
    monkeypatch.setattr(
        "effectome.intervention.surrogate._fold_mean_interval",
        lambda *_args, **_kwargs: (-0.1, 0.5),
    )

    model = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        fit_indices=np.arange(40),
        evaluation_indices=np.arange(40, 60),
        fit_scope_id="outer-fold-0",
        strict_outer_ownership=True,
        candidate_nodes=[],
        multiplicity_family=["continuous:feature-edit"],
        hypothesis_id="continuous:feature-edit",
        endpoint="behavior",
        min_skill=0.0,
        min_absolute_skill=-1e9,
        max_shift=1e9,
    )

    assert model.validation.status == "invalid"
    assert model.validation.fail_safe_reason == "increment_uncertainty_gate_failed"


def test_training_frozen_attribution_is_outer_test_invariant():
    series, community, manifold, behavior = _toy_lane_artifacts()
    first = qualify_candidate_drivers(
        series,
        community,
        manifold,
        behavior,
        "continuous",
        fit_indices=np.arange(40),
        evaluation_indices=np.arange(40, 60),
        fit_scope_id="outer-fold-0",
    )
    series.matrices[40:] *= 100
    manifold.window_embedding[40:] += 100
    behavior[40:] += 100
    second = qualify_candidate_drivers(
        series,
        community,
        manifold,
        behavior,
        "continuous",
        fit_indices=np.arange(40),
        evaluation_indices=np.arange(40, 60),
        fit_scope_id="outer-fold-0",
    )

    assert first.candidate_nodes == second.candidate_nodes
    assert first.category_vocabulary == second.category_vocabulary
    np.testing.assert_allclose(
        [score.predictive_gain for score in first.scores],
        [score.predictive_gain for score in second.scores],
    )


def test_strict_attribution_requires_aligned_community_ownership():
    series, community, manifold, behavior = _toy_lane_artifacts()
    _assign_fit_scope(series, manifold, "outer-fold-0")
    kwargs = dict(
        fit_indices=np.arange(40),
        evaluation_indices=np.arange(40, 60),
        fit_scope_id="outer-fold-0",
        require_feature_ownership=True,
    )

    with pytest.raises(ValueError, match="community"):
        qualify_candidate_drivers(
            series,
            community,
            manifold,
            behavior,
            "continuous",
            **kwargs,
        )

    _assign_fit_scope(series, manifold, "outer-fold-0", community)
    result = qualify_candidate_drivers(
        series,
        community,
        manifold,
        behavior,
        "continuous",
        **kwargs,
    )
    assert result.provenance["feature_ownership_verified"] is True
    assert result.provenance["community_fit_scope_id"] == "outer-fold-0"


def test_finite_controls_are_unique_and_resolution_is_unavailable():
    series, _, manifold, behavior = _toy_lane_artifacts()
    model = fit_linear_surrogate(series, manifold, behavior, endpoint="behavior", min_skill=-1e9)
    result = run_virtual_perturbation(model, series, manifold, behavior, [0], n_controls=1000)
    assert len(result.control_effects) == 2
    assert result.control_pvalue is None
    assert result.provenance["control_status"] == "insufficient_control_resolution"
    assert result.provenance["minimum_attainable_pvalue"] == pytest.approx(1 / 3)
    assert result.control_pvalue_status == "unavailable"
    assert result.effect_status == "available"


def test_feature_sensitivity_uses_the_registered_multiplicity_threshold(monkeypatch):
    import effectome.intervention.surrogate as surrogate_module

    series, _, manifold, behavior = _toy_lane_artifacts()
    expanded = np.zeros((series.n_windows, 64, 64), dtype=np.float32)
    expanded[:, : series.n_neurons, : series.n_neurons] = series.matrices
    expanded_series = ConnectivitySeries(
        matrices=expanded,
        window_starts=np.array(series.window_starts, copy=True),
        method=series.method,
        directed=series.directed,
        behavior_per_window=series.behavior_per_window,
    )
    model = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        endpoint="behavior",
        min_skill=-1e9,
    )
    model.metadata.update(
        {
            "candidate_nodes": [0],
            "eligible_unique_control_sets": [[node] for node in range(1, 64)],
            "multiplicity_adjusted_alpha": 0.02,
        }
    )
    monkeypatch.setattr(
        model,
        "predict",
        lambda features: np.zeros((features.shape[0], 1), dtype=float),
    )

    def fake_effect(_model, _series, _manifold, _behavior, nodes, scale, _endpoint, _baseline):
        multiplier = 1.0 - float(scale)
        if nodes == [0]:
            return 2.0 * multiplier
        if nodes == [1]:
            return 3.0 * multiplier
        return 0.5 * multiplier

    monkeypatch.setattr(surrogate_module, "_mean_endpoint_effect", fake_effect)
    result = run_predictor_feature_sensitivity(
        model,
        expanded_series,
        manifold,
        behavior,
        node_indices=[0],
        endpoint="behavior",
        dose_scales=[0.5, 0.0],
        n_controls=63,
    )

    assert result.control_pvalue == pytest.approx(2 / 64)
    assert result.control_pvalue < 0.05
    assert result.control_pvalue > 0.02
    assert result.validation_status == "model_sensitivity_unconfirmed"


def test_strict_control_resolution_fails_before_feature_edits():
    series, _, manifold, behavior = _toy_lane_artifacts()
    _assign_fit_scope(series, manifold, "outer-fold-0")
    model = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        fit_indices=np.arange(40),
        evaluation_indices=np.arange(40, 60),
        fit_scope_id="outer-fold-0",
        strict_outer_ownership=True,
        candidate_nodes=[0],
        multiplicity_family=["continuous:feature-edit"],
        hypothesis_id="continuous:feature-edit",
        endpoint="behavior",
        min_skill=-1e9,
        min_absolute_skill=-1.0,
        max_shift=1e9,
        control_resolution_policy="fail",
    )

    assert model.validation.status == "invalid"
    assert model.validation.fail_safe_reason == "control_resolution_unattainable"
    result = run_predictor_feature_sensitivity(
        model,
        series,
        manifold,
        behavior,
        node_indices=[0],
        endpoint="behavior",
        hypothesis_id="continuous:feature-edit",
    )
    assert result.status == "invalid"
    assert result.effect_size is None
    assert result.effect_status == "unavailable"
    assert result.fail_safe_reason == "control_resolution_unattainable"


def test_missing_learned_feature_ownership_fails_closed():
    series, _, manifold, behavior = _toy_lane_artifacts()
    model = fit_linear_surrogate(series, manifold, behavior, require_feature_ownership=True)
    assert model.validation.status == "invalid"
    assert model.validation.fail_safe_reason == "learned_feature_ownership_unverified"


def test_absolute_skill_gate_cannot_be_bypassed_by_relative_gain():
    series, _, manifold, behavior = _toy_lane_artifacts()
    model = fit_linear_surrogate(series, manifold, behavior, min_skill=-1e9, min_absolute_skill=1.1)
    assert model.validation.status == "invalid"
    assert model.validation.absolute_skills


def test_fit_and_evaluation_ownership_must_be_disjoint():
    series, _, manifold, behavior = _toy_lane_artifacts()
    with pytest.raises(ValueError, match="disjoint"):
        fit_linear_surrogate(series, manifold, behavior,
                             fit_indices=np.arange(40), evaluation_indices=np.arange(30, 60))


def test_switch_rates_exclude_recording_boundary_changes():
    community = CommunitySeries(
        labels=np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=int),
        method="toy",
        n_communities_per_window=np.ones(4, dtype=int),
        boundary_indices=np.array([0, 2], dtype=int),
    )
    assert np.array_equal(community_switch_rates(community), np.zeros(2))


def test_candidate_driver_requires_predictive_gain_not_just_switching():
    series, community, manifold, behavior = _toy_lane_artifacts()
    result = qualify_candidate_drivers(
        series,
        community,
        manifold,
        behavior,
        "continuous",
        lag=1,
        n_folds=5,
        embargo=2,
        seed=0,
        matched_control_percentile=80.0,
    )
    best = result.scores[0]
    by_node = {score.node: score for score in result.scores}
    assert result.status == "preliminary_predictive_screen"
    assert best.node == 0
    assert by_node[0].is_candidate
    assert by_node[0].candidate_stage == "preliminary_predictive_candidate"
    assert not by_node[0].is_validated_driver
    assert by_node[0].provenance["claim_boundary"] == "screening_only_not_a_validated_driver"
    assert not by_node[2].is_candidate  # switches communities but carries no predictive outgoing effect
    assert by_node[0].predictive_gain > by_node[2].predictive_gain


def test_surrogate_validation_and_graded_virtual_perturbation():
    series, _, manifold, behavior = _toy_lane_artifacts()
    surrogate = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        lag=1,
        ridge_alpha=1.0,
        n_folds=5,
        embargo=2,
        min_skill=-0.05,
    )
    assert surrogate.validation.status == "valid"
    assert surrogate.validation.mean_skill > -0.05

    perturb = run_virtual_perturbation(
        surrogate,
        series,
        manifold,
        behavior,
        node_indices=[0],
        scale=0.0,
        n_controls=8,
        endpoint="behavior",
        seed=0,
        dose_scales=[0.75, 0.5, 0.25, 0.0],
    )
    assert perturb.status == "valid"
    assert len(perturb.control_effects) == 2
    assert perturb.effect_size != 0.0
    assert len(perturb.dose_response) == 4
    assert perturb.analysis_kind == "predictor_feature_sensitivity"
    assert perturb.biological_intervention_identified is False
    assert perturb.provenance["claim_boundary"] == "predictor_feature_sensitivity_not_biological_intervention"
    assert perturb.control_pvalue is None
    assert perturb.provenance["control_status"] == "insufficient_control_resolution"
    assert perturb.candidate_stage != "validated_candidate_driver"
    assert perturb.candidate_stage != "counterfactually_supported_candidate"
    assert all(point.sham_effect == 0.0 for point in perturb.dose_response)
    assert {point.scale for point in perturb.dose_response} == {0.75, 0.5, 0.25, 0.0}


def test_invalid_surrogate_fails_safe():
    series, _, manifold, behavior = _toy_lane_artifacts()
    surrogate = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        lag=1,
        ridge_alpha=1.0,
        n_folds=5,
        embargo=2,
        min_skill=10.0,
    )
    assert surrogate.validation.status == "invalid"
    perturb = run_virtual_perturbation(surrogate, series, manifold, behavior, node_indices=[0])
    assert perturb.status == "invalid"
    assert perturb.effect_size is None
    assert perturb.control_pvalue is None
    assert perturb.validation_status == "unavailable"
    assert perturb.fail_safe_reason == "surrogate_validation_failed"
    assert perturb.effect_status == "unavailable"
    assert perturb.control_pvalue_status == "unavailable"


def test_estimated_zero_is_distinct_from_unavailable_for_serialization():
    series, _, manifold, behavior = _toy_lane_artifacts()
    surrogate = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        endpoint="behavior",
        min_skill=-1e9,
    )

    result = run_predictor_feature_sensitivity(
        surrogate,
        series,
        manifold,
        behavior,
        node_indices=[0],
        scale=1.0,
        dose_scales=[1.0],
        endpoint="behavior",
    )

    assert result.effect_size == pytest.approx(0.0)
    assert result.effect_status == "available"
    assert result.dose_response[0].effect_status == "available"
    assert result.provenance["serialization_contract"] == "none_is_unavailable_zero_is_estimated"


def test_exact_linear_feature_edit_remains_non_interventional():
    series, _, manifold, behavior = _toy_lane_artifacts()
    surrogate = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        endpoint="behavior",
        min_skill=-1e9,
    )
    result = run_predictor_feature_sensitivity(
        surrogate,
        series,
        manifold,
        behavior,
        node_indices=[0],
        endpoint="behavior",
        dose_scales=[1.0, 0.75, 0.5, 0.25, 0.0],
    )

    assert result.dose_response_monotonic is True
    assert all(point.sham_effect == pytest.approx(0.0) for point in result.dose_response)
    assert result.analysis_kind == "predictor_feature_sensitivity"
    assert result.biological_intervention_identified is False
    assert result.candidate_stage != "counterfactually_supported_candidate"
    assert "not_biological_intervention" in result.provenance["claim_boundary"]


def test_manifold_surrogate_one_hot_encodes_integer_behavior_covariate():
    series, _, manifold, behavior = _toy_lane_artifacts()
    categorical = np.digitize(behavior, bins=[-0.3, 0.3]).astype(np.int64)
    second_axis_manifold = ManifoldArtifact(
        method=manifold.method,
        behavior_key="motif",
        full_embedding=np.column_stack(
            [np.zeros(len(behavior), dtype=np.float32), manifold.window_embedding[:, 0]]
        ),
        window_embedding=np.column_stack(
            [np.zeros(len(behavior), dtype=np.float32), manifold.window_embedding[:, 0]]
        ),
        target_slices=manifold.target_slices,
        target_length=manifold.target_length,
        window_starts=manifold.window_starts,
    )

    surrogate = fit_linear_surrogate(
        series,
        second_axis_manifold,
        categorical,
        endpoint="manifold",
        lag=1,
        n_folds=5,
        embargo=2,
        min_skill=-1e9,
    )

    assert surrogate.endpoint == "manifold"
    assert surrogate.validation.status == "valid"
    assert surrogate.metadata["target_dim"] == 2
    assert surrogate.metadata["behavior_covariate_encoding"] == "categorical_one_hot"
    assert surrogate.metadata["behavior_categories"] == [0, 1, 2]
    assert surrogate.feature_shape == (series.n_windows - 1, series.n_neurons**2 + 2 + 3)
    assert surrogate.model is not None
    assert surrogate.model.coef_.shape[0] == 2

    perturb = run_virtual_perturbation(
        surrogate,
        series,
        second_axis_manifold,
        categorical,
        node_indices=[0],
        endpoint="manifold",
        scale=0.0,
        n_controls=4,
        dose_scales=[0.5, 0.0],
    )
    assert perturb.status == "valid"
    assert perturb.endpoint == "manifold"
    assert perturb.provenance["surrogate_endpoint"] == "manifold"
    assert perturb.provenance["effect_metric"] == "mean_euclidean_latent_displacement"
    assert perturb.effect_size != 0.0
    assert perturb.biological_intervention_identified is False


@pytest.mark.parametrize("endpoint", ["behavior", "joint"])
def test_integer_behavior_cannot_be_used_as_regression_target(endpoint: str):
    series, _, manifold, behavior = _toy_lane_artifacts()
    categorical = np.digitize(behavior, bins=[-0.3, 0.3]).astype(np.int64)

    with pytest.raises(ValueError, match="integer behavior labels are categorical"):
        fit_linear_surrogate(series, manifold, categorical, endpoint=endpoint)


def test_behavior_only_surrogate_retains_continuous_endpoint_path():
    series, _, manifold, behavior = _toy_lane_artifacts()
    surrogate = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        endpoint="behavior",
        min_skill=-1e9,
    )

    assert surrogate.endpoint == "behavior"
    assert surrogate.metadata["target_dim"] == 1
    perturb = run_virtual_perturbation(
        surrogate,
        series,
        manifold,
        behavior,
        node_indices=[0],
        endpoint="behavior",
        n_controls=2,
    )
    assert perturb.status == "valid"


def test_virtual_perturbation_rejects_endpoint_not_fitted_by_surrogate():
    series, _, manifold, behavior = _toy_lane_artifacts()
    surrogate = fit_linear_surrogate(
        series,
        manifold,
        behavior,
        endpoint="manifold",
        min_skill=-1e9,
    )

    with pytest.raises(ValueError, match="incompatible with a surrogate fitted"):
        run_virtual_perturbation(
            surrogate,
            series,
            manifold,
            behavior,
            node_indices=[0],
            endpoint="behavior",
        )
