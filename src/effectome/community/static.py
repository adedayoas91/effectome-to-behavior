"""Static per-window community detection: Leiden (preferred) or greedy modularity (fallback)."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from .base import CommunityDetector, register_community

logger = logging.getLogger(__name__)


def _labels_from_sets(communities: list[set[int]], n: int) -> np.ndarray:
    labels = np.zeros(n, dtype=np.int64)
    for c, nodes in enumerate(communities):
        for v in nodes:
            labels[v] = c
    return labels


@register_community("leiden")
class LeidenCommunity(CommunityDetector):
    """Leiden community detection via igraph+leidenalg; falls back to greedy modularity.

    Leiden optimizes modularity (with a resolution parameter) and guarantees well-connected
    communities. Without the optional `community` extra installed, this transparently falls back
    to NetworkX greedy modularity so the pipeline still runs.
    """

    def detect_one(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        try:
            import igraph as ig
            import leidenalg as la

            sources, targets = np.nonzero(np.triu(matrix, k=1))
            weights = matrix[sources, targets]
            g = ig.Graph(n=n, edges=list(zip(sources.tolist(), targets.tolist(), strict=False)))
            g.es["weight"] = weights.tolist()
            part = la.find_partition(
                g,
                la.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=self.cfg.resolution,
                seed=self.cfg.seed,
            )
            return np.asarray(part.membership, dtype=np.int64)
        except ImportError:
            logger.debug("leidenalg not installed; using greedy modularity fallback")
            g = nx.from_numpy_array(matrix)
            communities = nx.community.greedy_modularity_communities(
                g, weight="weight", resolution=self.cfg.resolution
            )
            return _labels_from_sets([set(c) for c in communities], n)
