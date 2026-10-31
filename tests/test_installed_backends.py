"""Exercise maintained connectivity, community and distance-clustering packages."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from effectome.community import CommunityConfig, CommunityFactory
from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.dynamics.metrics import kmedoids
from effectome.experiments import analyze_with_jpcmciplus


def _lagged_recording(seed):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(240, 3))
    data[1:, 1] = 0.9 * data[:-1, 0] + 0.1 * rng.normal(size=239)
    return data


@pytest.mark.parametrize("name", ["pcmci", "jpcmci"])
def test_installed_tigramite_recovers_directed_lagged_dependence(name):
    data = _lagged_recording(17)
    config = ConnectivityConfig(
        name=name, max_lag=1,
        extra={"recordings": {0: data, 1: _lagged_recording(18)}} if name == "jpcmci" else {},
    )
    estimator = ConnectivityFactory(config)
    matrix = estimator.estimate(data)
    assert matrix.shape == (3, 3)
    assert matrix[0, 1] > 0.9
    assert matrix[1, 0] == 0.0
    assert not np.diag(matrix).any()
    assert estimator.weight_semantics == "signed_mci_dependence_statistic"


def test_jpcmci_notebook_adapter_supplies_system_classifications():
    matrix = analyze_with_jpcmciplus([_lagged_recording(17), _lagged_recording(18)], [1])[1]
    assert matrix[0, 1] > 0.9


def test_jpcmci_rejects_incomplete_context_classification():
    estimator = ConnectivityFactory(ConnectivityConfig(
        name="jpcmci", max_lag=1, extra={"node_classification": {0: "system"}},
    ))
    with pytest.raises(ValueError, match="classify every variable"):
        estimator.estimate(_lagged_recording(1))


def test_pcmci_adapter_rejects_unresolved_contemporaneous_edges():
    estimator = ConnectivityFactory(ConnectivityConfig(name="pcmci", extra={"tau_min": 0}))
    with pytest.raises(ValueError, match="positive lags"):
        estimator.estimate(_lagged_recording(2))


def test_leiden_keeps_lower_triangle_directed_edges():
    matrix = np.zeros((4, 4))
    matrix[1, 0] = 3
    matrix[3, 2] = 3
    estimator = CommunityFactory(CommunityConfig(name="leiden", symmetrize=False))
    labels = estimator.detect_one(matrix)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    assert estimator._last_backend == "leidenalg.RBConfigurationVertexPartition"


def test_leiden_missing_package_fails_instead_of_substituting_greedy(monkeypatch):
    original_import = builtins.__import__

    def without_leiden(name, *args, **kwargs):
        if name == "leidenalg":
            raise ImportError("simulated missing backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_leiden)
    estimator = CommunityFactory(CommunityConfig(name="leiden"))
    with pytest.raises(ImportError, match="requires installed igraph and leidenalg"):
        estimator.detect_one(np.zeros((3, 3)))


def test_leiden_rejects_negative_rb_weights():
    estimator = CommunityFactory(CommunityConfig(name="leiden", use_absolute=False))
    with pytest.raises(ValueError, match="nonnegative weights"):
        estimator.detect_one(np.array([[0.0, -1.0], [-1.0, 0.0]]))


def test_installed_kmedoids_preserves_seeded_partition_and_medoid_semantics():
    points = np.array([0.0, 0.1, 0.2, 10.0, 10.1, 10.2])
    distance = np.abs(points[:, None] - points[None, :])
    labels, medoids = kmedoids(distance, 2, seed=4)
    assert len(set(medoids)) == 2
    assert len(set(labels[:3])) == len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    np.testing.assert_array_equal(labels, np.argmin(distance[:, medoids], axis=1))
    repeated, repeated_medoids = kmedoids(distance, 2, seed=4)
    np.testing.assert_array_equal(labels, repeated)
    np.testing.assert_array_equal(medoids, repeated_medoids)
