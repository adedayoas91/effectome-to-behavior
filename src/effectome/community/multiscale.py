"""Multi-scale community detection via a Markov-stability surrogate.

Markov stability views communities as regions where a random walker stays trapped for a
Markov time t. We approximate it by running modularity on the t-step transition matrix of the
graph: small t recovers many fine communities, large t recovers a few coarse ones. The Markov
time is taken from `cfg.extra['markov_time']`. This keeps the community partition consistent
with the random-walk/Markov framing used for the transition dynamics in Stage 3.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .base import CommunityDetector, register_community
from .static import _labels_from_sets


@register_community("markov_stability")
class MarkovStabilityCommunity(CommunityDetector):
    """Modularity on the t-step random-walk transition matrix (Markov-stability proxy)."""

    def detect_one(self, matrix: np.ndarray) -> np.ndarray:
        n = matrix.shape[0]
        markov_time = int(self.cfg.extra.get("markov_time", 1))

        deg = matrix.sum(axis=1)
        deg[deg == 0] = 1.0
        p = matrix / deg[:, None]  # row-stochastic random walk
        pt = np.linalg.matrix_power(p, max(1, markov_time))

        # Flow graph: symmetric co-visitation weights pi_i * Pt_ij.
        pi = deg / deg.sum()
        flow = pi[:, None] * pt
        flow = 0.5 * (flow + flow.T)
        np.fill_diagonal(flow, 0.0)

        g = nx.from_numpy_array(flow)
        communities = nx.community.greedy_modularity_communities(
            g, weight="weight", resolution=self.cfg.resolution
        )
        return _labels_from_sets([set(c) for c in communities], n)
