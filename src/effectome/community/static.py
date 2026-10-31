"""Static community detection using the installed Leiden implementation."""

from __future__ import annotations

import numpy as np

from .base import CommunityDetector, register_community


def _labels_from_sets(communities: list[set[int]], n: int) -> np.ndarray:
    labels = np.zeros(n, dtype=np.int64)
    for c, nodes in enumerate(communities):
        for v in nodes:
            labels[v] = c
    return labels


@register_community("leiden")
class LeidenCommunity(CommunityDetector):
    """Leiden RB modularity via installed igraph/leidenalg on nonnegative weights."""

    def detect_one(self, matrix: np.ndarray) -> np.ndarray:
        try:
            import igraph as ig
            import leidenalg as la
        except ImportError as exc:
            raise ImportError(
                "Leiden requires installed igraph and leidenalg; run uv sync --extra community "
                "alongside your other environment extras."
            ) from exc
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("Leiden RB requires finite nonnegative weights; use use_absolute=True")
        directed = not self.cfg.symmetrize
        selected = matrix if directed else np.triu(matrix, k=1)
        sources, targets = np.nonzero(selected)
        graph = ig.Graph(
            n=matrix.shape[0],
            edges=list(zip(sources.tolist(), targets.tolist(), strict=False)),
            directed=directed,
        )
        graph.es["weight"] = matrix[sources, targets].tolist()
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=self.cfg.resolution,
            seed=self.cfg.seed,
        )
        self._last_backend = "leidenalg.RBConfigurationVertexPartition"
        return np.asarray(partition.membership, dtype=np.int64)
