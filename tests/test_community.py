"""Community detection: registry, shapes, and temporal label alignment."""

from __future__ import annotations

import numpy as np

from effectome.community import COMMUNITY_REGISTRY, CommunityConfig, CommunityFactory
from effectome.connectivity import ConnectivityConfig, ConnectivityFactory


def _series(windows):
    return ConnectivityFactory(ConnectivityConfig(name="granger", absolute=True)).run(windows)


def test_registry_has_methods():
    assert {"leiden", "markov_stability", "temporal"} <= set(COMMUNITY_REGISTRY)


def test_static_community_shapes(windows):
    series = _series(windows)
    det = CommunityFactory(CommunityConfig(name="leiden"))
    com = det.run(series)
    assert com.labels.shape == (series.n_windows, series.n_neurons)
    assert (com.n_communities_per_window >= 1).all()


def test_temporal_alignment_runs(windows):
    series = _series(windows)
    det = CommunityFactory(CommunityConfig(name="temporal"))
    com = det.run(series)
    assert com.labels.shape == (series.n_windows, series.n_neurons)
    # labels are integers and well-formed
    assert np.issubdtype(com.labels.dtype, np.integer)
