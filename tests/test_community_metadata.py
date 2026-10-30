"""Community backend/transform metadata tests for Appendix-C WP8."""

from __future__ import annotations

import numpy as np

from effectome.community import CommunityConfig, CommunityFactory
from effectome.data_module.schema import ArtifactProvenance, ConnectivitySeries


def _series() -> ConnectivitySeries:
    matrices = np.asarray(
        [
            [
                [0.0, -1.0, 0.2],
                [0.5, 0.0, -0.3],
                [0.1, 0.4, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    return ConnectivitySeries(
        matrices=matrices,
        window_starts=np.array([0]),
        method="signed_directed_fixture",
        directed=True,
        signed=True,
    )


def test_community_records_actual_transform_and_backend():
    communities = CommunityFactory(
        CommunityConfig(name="leiden", use_absolute=True, symmetrize=True)
    ).run(_series())

    metadata = communities.metadata
    assert metadata["input_transform"] == {
        "use_absolute": True,
        "symmetrize": True,
        "weight_threshold": 0.0,
    }
    assert metadata["supported_input"] == "symmetric_nonnegative"
    assert metadata["actual_backend"] in {
        "networkx.greedy_modularity",
        "leidenalg.RBConfigurationVertexPartition",
    }


def test_markov_stability_is_labeled_as_proxy_backend():
    communities = CommunityFactory(
        CommunityConfig(name="markov_stability", extra={"markov_time": 2})
    ).run(_series())

    assert communities.metadata["actual_backend"] == (
        "markov_stability_proxy.networkx_greedy_modularity"
    )


def test_full_series_community_fit_does_not_inherit_outer_ownership():
    series = _series()
    series.provenance = ArtifactProvenance(
        stage="connectivity",
        fit_scope_id="outer-fold-0",
        split_manifest_hash="split-hash",
        analysis_mode="development",
    )

    communities = CommunityFactory(
        CommunityConfig(name="leiden", use_absolute=True, symmetrize=True)
    ).run(series)

    assert communities.provenance.stage == "community"
    assert communities.provenance.fit_scope_id is None
    assert communities.provenance.split_manifest_hash is None
    assert communities.provenance.metadata["upstream_fit_scope_id"] == "outer-fold-0"
    assert communities.provenance.metadata["ownership_contract"] == (
        "full_series_community_fit_requires_separate_outer_scope"
    )
    assert communities.provenance.metadata["upstream_stage"] == "connectivity"
