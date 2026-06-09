"""Temporal/evolving community tracking across windows.

Per-window partitions are detected independently, then community labels are made temporally
consistent by matching each window's communities to the previous window's via maximum node
overlap (greedy bipartite matching). The result is a set of communities whose membership
*evolves* over time, so a community keeps a stable identity even as neurons join and leave it.
This realizes the 'evolving communities via a Markov process' idea by yielding label tracks
whose changes can themselves be modeled as a transition process.
"""

from __future__ import annotations

import numpy as np

from effectome.data_module.schema import CommunitySeries, ConnectivitySeries

from .base import CommunityConfig, CommunityDetector, register_community
from .static import LeidenCommunity


def _match_labels(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Relabel `cur` so its communities align with `prev` by maximum node overlap."""
    prev_ids = np.unique(prev)
    cur_ids = np.unique(cur)
    overlap = np.zeros((len(cur_ids), len(prev_ids)))
    for i, c in enumerate(cur_ids):
        for j, p in enumerate(prev_ids):
            overlap[i, j] = np.sum((cur == c) & (prev == p))

    mapping: dict[int, int] = {}
    used: set[int] = set()
    order = np.argsort(-overlap.max(axis=1))
    next_free = int(prev.max()) + 1
    for i in order:
        c = int(cur_ids[i])
        ranked = np.argsort(-overlap[i])
        assigned = None
        for j in ranked:
            if overlap[i, j] == 0:
                break
            cand = int(prev_ids[j])
            if cand not in used:
                assigned = cand
                used.add(cand)
                break
        if assigned is None:
            assigned = next_free
            next_free += 1
        mapping[c] = assigned
    return np.array([mapping[int(x)] for x in cur], dtype=np.int64)


@register_community("temporal")
class TemporalCommunity(CommunityDetector):
    """Detect per-window communities then align labels across time for stable tracks."""

    def __init__(self, cfg: CommunityConfig) -> None:
        super().__init__(cfg)
        base_cfg = CommunityConfig(
            name="leiden",
            resolution=cfg.resolution,
            symmetrize=cfg.symmetrize,
            weight_threshold=cfg.weight_threshold,
            seed=cfg.seed,
        )
        self._base = LeidenCommunity(base_cfg)

    def detect_one(self, matrix: np.ndarray) -> np.ndarray:
        return self._base.detect_one(matrix)

    def run(self, series: ConnectivitySeries) -> CommunitySeries:
        raw = [self.detect_one(self._prepare(w)) for w in series.matrices]
        aligned = [raw[0]]
        for k in range(1, len(raw)):
            aligned.append(_match_labels(aligned[-1], raw[k]))
        labels = np.stack(aligned).astype(np.int64)
        counts = np.array([len(np.unique(row)) for row in labels])
        return CommunitySeries(
            labels=labels, method=self.cfg.name, n_communities_per_window=counts
        )
