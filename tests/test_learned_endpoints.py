"""Learned measurements own training rows and stay frozen during interventions."""

import json
from dataclasses import replace

import numpy as np
import pytest

from effectome.experiments.learned_endpoints import (
    CoordinateEndpoint,
    FrozenEndpoint,
    _decoder_rows,
    _window_means,
    endpoint_from_dict,
    fit_endpoint_arms,
)


def _training(seed=7):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(8, 29, 6))
    values[..., 5] = 0.7 * values[..., 0] - 0.5 * values[..., 1] + 0.3 * values[..., 5]
    return values


@pytest.fixture
def arms():
    return fit_endpoint_arms(_training(), endpoint_index=5, target_length=4)


def test_coordinate_reference_keeps_initial_sign_and_future_mean():
    endpoint = CoordinateEndpoint(1)
    initial = np.array([[9, -0.1], [-3, 0], [2, 1]])
    future = np.array([[[0, -0.5], [0, 0.2]], [[0, 0.25], [0, -0.1]]])
    np.testing.assert_array_equal(endpoint.source_states(initial), [0, 1, 1])
    np.testing.assert_array_equal(endpoint.target_states(future), [0, 1])
    assert endpoint.to_dict()["type"] == "coordinate"


def test_window_means_never_join_different_trials_or_use_partial_windows():
    trials = np.stack((np.full((7, 3), -10.0), np.full((7, 3), 10.0)))
    means = _window_means(trials, 3)
    assert means.shape == (2, 2, 3)
    np.testing.assert_array_equal(means[0], -10)
    np.testing.assert_array_equal(means[1], 10)
    assert _decoder_rows(trials, 3).shape == (18, 3)
    assert not np.any(_decoder_rows(trials, 3) == 0)


def test_fit_owns_inner_encoder_and_keeps_validation_out_of_geometric_fit(arms):
    changed = _training()
    changed[6:] *= -100
    second = fit_endpoint_arms(changed, endpoint_index=5, target_length=4)
    for name in ("pca_geometry", "pca_behavior", "full_neural_behavior"):
        first, other = arms[name], second[name]
        assert isinstance(first, FrozenEndpoint) and isinstance(other, FrozenEndpoint)
        np.testing.assert_array_equal(first.encoder_weights, other.encoder_weights)
        np.testing.assert_array_equal(first.encoder_bias, other.encoder_bias)
        assert first.metadata["encoder_training_trial_ids"] == list(range(6))
        assert first.metadata["inner_validation_trial_ids"] == [6, 7]
        assert first.metadata["encoder_sample_count"] == 6 * 29
        assert first.metadata["geometry_training_window_count"] == 6 * 7
        assert first.metadata["window_trial_boundary_crossings"] == 0
    first, other = arms["pca_geometry"], second["pca_geometry"]
    np.testing.assert_array_equal(first.centers, other.centers)
    np.testing.assert_array_equal(first.region_labels, other.region_labels)
    for name in ("pca_behavior", "full_neural_behavior"):
        assert (arms[name].metadata["decoder_refit_scope"]
                == "all_supplied_outer_training_trials_fixed_encoder")
        assert arms[name].metadata["decoder_refit_rows"] == 8 * (29 + 7)


def test_only_training_is_supplied_and_prediction_never_fits_or_mutates(arms):
    rng = np.random.default_rng(99)
    unseen = rng.normal(size=(4, 7, 11, 6)) * 80
    for endpoint in arms.values():
        before = endpoint.to_dict()
        assert endpoint.target_states(unseen).shape == (7, 11)
        assert endpoint.source_states(unseen[0]).shape == (7, 11)
        assert endpoint.to_dict() == before


def test_exported_raw_affine_codes_match_recorded_pca_parameters(arms):
    endpoint = arms["pca_behavior"]
    assert isinstance(endpoint, FrozenEndpoint)
    metadata = endpoint.metadata
    sample = _training(88)[0]
    scaled = (sample - metadata["scaler_mean"]) / metadata["scaler_scale"]
    expected = (scaled - metadata["pca_center"]) @ np.asarray(metadata["pca_components"]).T
    np.testing.assert_allclose(endpoint.codes(sample), expected, atol=1e-14)


@pytest.mark.parametrize("name", ["pca_geometry", "pca_behavior", "full_neural_behavior"])
def test_frozen_arrays_are_copies_and_read_only(arms, name):
    endpoint = arms[name]
    assert isinstance(endpoint, FrozenEndpoint)
    with pytest.raises(ValueError, match="read-only"):
        endpoint.encoder_weights[0, 0] = 99
    payload = endpoint.to_dict()
    payload["encoder_weights"][0][0] = 12345
    assert endpoint.encoder_weights[0, 0] != 12345


@pytest.mark.parametrize("name", ["pca_geometry", "pca_behavior", "full_neural_behavior"])
def test_consistent_orthogonal_coordinate_changes_preserve_events(arms, name):
    endpoint = arms[name]
    assert isinstance(endpoint, FrozenEndpoint)
    rng = np.random.default_rng(18)
    rotation, _ = np.linalg.qr(rng.normal(size=(endpoint.encoder_weights.shape[0],) * 2))
    rotated = endpoint.reparameterize(rotation)
    source, target = rng.normal(size=(50, 6)), rng.normal(size=(4, 50, 9, 6))
    np.testing.assert_array_equal(rotated.source_states(source), endpoint.source_states(source))
    np.testing.assert_array_equal(rotated.target_states(target), endpoint.target_states(target))
    assert rotated.to_dict()["fingerprint"] != endpoint.to_dict()["fingerprint"]
    with pytest.raises(ValueError, match="orthogonal"):
        endpoint.reparameterize(rotation * 2)


def test_constant_control_collapses_and_proxy_is_predeclared(arms):
    values = _training(99)
    np.testing.assert_array_equal(arms["constant"].source_states(values), 0)
    np.testing.assert_array_equal(arms["constant"].target_states(values), 0)
    np.testing.assert_array_equal(arms["proxy"].source_states(values), values[..., 0] >= 0)
    np.testing.assert_array_equal(arms["nuisance"].source_states(values), values[..., 2] >= 0)


def test_json_roundtrip_retains_exact_frozen_events_and_hashes(arms):
    values = _training(25)
    for endpoint in arms.values():
        artifact = json.loads(json.dumps(endpoint.to_dict()))
        restored = endpoint_from_dict(artifact)
        assert restored.to_dict() == artifact
        np.testing.assert_array_equal(restored.source_states(values), endpoint.source_states(values))
        np.testing.assert_array_equal(restored.target_states(values), endpoint.target_states(values))
        artifact["fingerprint"] = "bad"
        with pytest.raises(ValueError, match="fingerprint"):
            endpoint_from_dict(artifact)


def test_coordinate_box_reports_extrapolation_without_dropping_values(arms):
    endpoint = arms["pca_behavior"]
    assert isinstance(endpoint, FrozenEndpoint)
    summary = endpoint.support_summary(np.full((9, 6), 1e6))
    assert summary["outside_count"] == 9
    assert summary["outside_fraction"] == 1
    assert summary["observations_removed"] == 0
    assert summary["support_guarantee"] is False
    assert endpoint.source_states(np.full((9, 6), 1e6)).shape == (9,)


@pytest.mark.parametrize("kwargs", [
    {"endpoint_index": True}, {"endpoint_index": -1}, {"endpoint_index": 7},
    {"target_length": 0}, {"target_length": 30}, {"n_components": 0}, {"n_components": True},
    {"n_components": 7}, {"validation_fraction": 1}, {"validation_fraction": np.nan},
    {"decoder_cs": ()}, {"decoder_cs": (-1,)}, {"decoder_cs": (1, 1)},
    {"seed": -1}, {"seed": True},
])
def test_invalid_fit_declarations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_endpoint_arms(_training(), **{"endpoint_index": 5, "target_length": 4, **kwargs})


@pytest.mark.parametrize("values", [np.zeros((2, 6)), np.zeros((1, 4, 6)),
                                   np.zeros((4, 4, 2)), np.full((8, 29, 6), np.nan)])
def test_invalid_training_arrays_are_rejected(values):
    with pytest.raises(ValueError):
        fit_endpoint_arms(values, endpoint_index=5, target_length=4)


def test_invalid_endpoint_parameters_and_observations_are_rejected(arms):
    for index in (-1, True):
        with pytest.raises(ValueError):
            CoordinateEndpoint(index)
    with pytest.raises(ValueError):
        CoordinateEndpoint(0, threshold=np.inf)
    endpoint = arms["pca_geometry"]
    assert isinstance(endpoint, FrozenEndpoint)
    for changes in ({"encoder_weights": np.zeros((0, 6))}, {"encoder_bias": np.zeros(3)},
                    {"region_labels": [0, 0]}, {"centers": np.zeros((2, 3))},
                    {"code_lower": np.full(2, np.inf)}, {"mode": "unknown"}):
        with pytest.raises(ValueError):
            replace(endpoint, **changes)
    for values in (np.zeros((0, 6)), np.zeros((3, 7)), np.full((4, 6), np.inf)):
        with pytest.raises(ValueError):
            endpoint.source_states(values)
    with pytest.raises(ValueError, match="time axis"):
        endpoint.target_states(np.ones(6))
    with pytest.raises(ValueError):
        CoordinateEndpoint(10).source_states(np.ones((3, 6)))
