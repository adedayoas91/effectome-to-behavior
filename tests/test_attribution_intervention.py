"""Attribution and intervention tests for Stage 7."""

from __future__ import annotations

import numpy as np

from effectome.attribution import community_switch_rates, qualify_candidate_drivers, signed_node_roles
from effectome.data_module.schema import CommunitySeries, ConnectivitySeries
from effectome.intervention import fit_linear_surrogate, run_virtual_perturbation
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
    assert len(perturb.control_effects) == 8
    assert perturb.effect_size != 0.0
    assert len(perturb.dose_response) == 4
    assert perturb.provenance["claim_boundary"] == "model_based_counterfactual_not_biological_causation"
    assert perturb.control_pvalue >= 1.0 / 9.0
    assert perturb.candidate_stage != "validated_candidate_driver"
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
    assert perturb.control_pvalue == 1.0
    assert perturb.validation_status == "unvalidated_counterfactual"
    assert perturb.fail_safe_reason == "surrogate_validation_failed"
