"""Role geometry uses each signed in/out coordinate once, with explicit legacy opt-in."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from effectome.community import CommunityConfig, CommunityFactory
from effectome.community.temporal import _signed_role_features, fit_temporal_role_model
from effectome.data_module.schema import ConnectivitySeries


def _series():
    rng = np.random.default_rng(13)
    matrices = rng.normal(size=(6, 4, 4))
    diagonal = np.arange(4)
    matrices[:, diagonal, diagonal] = 0.0
    return ConnectivitySeries(
        matrices=matrices,
        window_starts=np.arange(6) * 4,
        method="fixture",
        directed=True,
        signed=True,
        diagnostics={"node_ids": ["a", "b", "c", "d"]},
    )


def _config(representation="signed_in_out", **extra):
    return CommunityConfig(
        name="temporal", seed=7,
        extra={
            "n_communities": 2, "mode": "prospective", "n_runs": 2,
            "feature_representation": representation, **extra,
        },
    )


def test_signed_profiles_preserve_neighbor_axis_and_exclude_self_coefficients():
    matrix = np.array([[91.0, 2.0, -3.0], [4.0, 92.0, 5.0], [6.0, -7.0, 93.0]])
    profiles = _signed_role_features(matrix)
    np.testing.assert_array_equal(profiles, [
        [0.0, 2.0, -3.0, 0.0, 4.0, 6.0],
        [4.0, 0.0, 5.0, 2.0, 0.0, -7.0],
        [6.0, -7.0, 0.0, -3.0, 5.0, 0.0],
    ])
    changed = matrix.copy()
    np.fill_diagonal(changed, -10000)
    np.testing.assert_array_equal(_signed_role_features(changed), profiles)


def test_expanded_representation_is_explicit_and_preserves_legacy_layout():
    matrix = np.array([[0.0, 2.0], [-3.0, 0.0]])
    legacy = _signed_role_features(matrix, feature_representation="legacy_expanded")
    np.testing.assert_array_equal(legacy[0], [
        0, 2, 0, -3, 0, 2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 3, 2, -3,
    ])
    assert legacy.shape == (2, 18)
    assert _signed_role_features(matrix).shape == (2, 4)


def test_unknown_role_representation_fails_explicitly():
    with pytest.raises(ValueError, match="feature_representation"):
        _signed_role_features(np.zeros((3, 3)), feature_representation="unweighted_magic")
    with pytest.raises(ValueError, match="feature_representation"):
        CommunityFactory(_config("unweighted_magic")).run(_series())


@pytest.mark.parametrize("representation", ["signed_in_out", "legacy_expanded"])
def test_model_projection_and_result_record_role_representation(representation):
    series = _series()
    model = fit_temporal_role_model(series, _config(representation), fit_indices=np.arange(3))
    projected = model.filter(series)
    result = CommunityFactory(_config(representation)).run(series)
    assert model.feature_representation == representation
    assert model.graph_schema[-1] == representation
    assert model.grouping_target == projected.grouping_target == result.metadata["grouping_target"] == "roles"
    assert projected.feature_representation == result.metadata["feature_representation"] == representation
    assert all(state.feature_representation == representation for state in projected.state_snapshots)
    assert result.metadata["actual_backend"] == "sklearn.KMeans_with_temporal_role_assignment"


def test_model_rejects_filter_state_from_other_representation():
    series = _series()
    modern = fit_temporal_role_model(series, _config(), fit_indices=np.arange(3))
    legacy = fit_temporal_role_model(series, _config("legacy_expanded"), fit_indices=np.arange(3))
    with pytest.raises(ValueError, match="feature representation mismatch"):
        modern.filter(series, state=legacy.initial_state())


def test_training_role_geometry_and_issued_prefix_ignore_future_changes():
    series = _series()
    changed = replace(series, matrices=series.matrices.copy())
    changed.matrices[3:] *= -100
    config = _config(online_learning_rate=0.0)
    first = fit_temporal_role_model(series, config, fit_indices=np.arange(3))
    second = fit_temporal_role_model(changed, config, fit_indices=np.arange(3))
    np.testing.assert_array_equal(first.feature_mean, second.feature_mean)
    np.testing.assert_array_equal(first.feature_scale, second.feature_scale)
    np.testing.assert_array_equal(first.initial_centroids, second.initial_centroids)
    np.testing.assert_array_equal(first.filter(series).labels[:3], second.filter(changed).labels[:3])


def test_one_role_is_a_valid_undifferentiated_result():
    series = _series()
    result = CommunityFactory(_config(n_communities=1)).run(series)
    assert not result.labels.any()
    assert np.all(result.n_communities_per_window == 1)
    assert result.metadata["grouping_target"] == "roles"


def test_lag_resolved_profiles_distinguish_opposing_edges_hidden_by_collapse():
    series = _series()
    lagged = np.stack([series.matrices, -series.matrices], axis=1)
    resolved = replace(
        series, matrices=lagged.sum(axis=1), lagged_matrices=lagged,
        diagnostics={**series.diagnostics, "physical_lags": [0.1, 0.2]},
    )
    collapsed = replace(resolved, lagged_matrices=None)
    modern = fit_temporal_role_model(resolved, _config(), fit_indices=np.arange(3))
    flat = fit_temporal_role_model(collapsed, _config(n_communities=1), fit_indices=np.arange(3))
    assert modern.feature_source == "lag_resolved"
    assert modern.lag_count == 2
    assert modern.physical_lags == (0.1, 0.2)
    assert modern.initial_centroids.shape[1] == 16
    assert np.any(modern.feature_scale != 1.0)
    np.testing.assert_array_equal(flat.initial_centroids, np.zeros((1, 8)))
    assert flat.feature_source == "collapsed_lag"
    projection = modern.filter(resolved)
    assert projection.lag_count == 2
    assert projection.physical_lags == (0.1, 0.2)
    result = CommunityFactory(_config()).run(resolved)
    assert result.metadata["lag_count"] == 2
    assert result.metadata["feature_source"] == "lag_resolved"
    with pytest.raises(ValueError, match="graph schema mismatch"):
        modern.filter(collapsed)
    changed_units = replace(resolved, diagnostics={**resolved.diagnostics, "physical_lags": [0.2, 0.4]})
    with pytest.raises(ValueError, match="graph schema mismatch"):
        modern.filter(changed_units)


def test_legacy_expanded_mode_explicitly_uses_collapsed_graphs():
    series = _series()
    lagged = np.stack([series.matrices, -series.matrices], axis=1)
    resolved = replace(series, lagged_matrices=lagged)
    model = fit_temporal_role_model(resolved, _config("legacy_expanded"), fit_indices=np.arange(3))
    assert model.feature_source == "collapsed_lag"
    assert model.lag_count == 1
    assert model.physical_lags is None


def test_fitting_role_model_never_reads_heldout_graph_values():
    series = _series()
    changed = replace(series, matrices=series.matrices.copy())
    changed.matrices[3:] = np.nan
    fitted = fit_temporal_role_model(changed, _config(), fit_indices=np.arange(3))
    reference = fit_temporal_role_model(series, _config(), fit_indices=np.arange(3))
    np.testing.assert_array_equal(fitted.initial_centroids, reference.initial_centroids)
