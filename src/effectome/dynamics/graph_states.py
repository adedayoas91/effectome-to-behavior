"""Cluster the sequence of connectivity matrices into recurring 'connectivity states'.

The dynamic effectome {W_1..W_K} is treated as a trajectory in graph space. We discretize it
into a small number of recurring states (regimes) by clustering the vectorized matrices. The
state label sequence becomes the input to the transition model (Stage 3b). Model selection over
the number of states K is supported via silhouette score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from effectome.data_module.schema import ConnectivitySeries

from .metrics import vectorize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphStateConfig:
    """Configuration for connectivity-state discovery.

    Attributes:
        n_states: Number of states K (ignored if select_k is True).
        select_k: If True, pick K in [k_min, k_max] by best silhouette score.
        k_min: Minimum K when selecting.
        k_max: Maximum K when selecting.
        standardize: Z-score features before clustering.
        seed: Random seed for KMeans.
    """

    n_states: int = 3
    select_k: bool = False
    k_min: int = 2
    k_max: int = 8
    standardize: bool = True
    seed: int = 42


@dataclass
class GraphStateModel:
    """Result of connectivity-state clustering.

    Attributes:
        labels: State label per window, shape (K_windows,).
        centroids: State centroid matrices, shape (n_states, N, N).
        n_states: Number of states.
        silhouette: Silhouette score of the chosen clustering.
        window_starts: Window start indices (carried through for alignment).
    """

    labels: np.ndarray
    centroids: np.ndarray
    n_states: int
    silhouette: float
    window_starts: np.ndarray


def _prepare(features: np.ndarray, standardize: bool) -> np.ndarray:
    if not standardize:
        return features
    mu = features.mean(axis=0, keepdims=True)
    sd = features.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (features - mu) / sd


def fit_graph_states(series: ConnectivitySeries, cfg: GraphStateConfig) -> GraphStateModel:
    """Cluster connectivity matrices into states; return labels + centroid graphs."""
    n, n_neurons = series.n_windows, series.n_neurons
    feats = _prepare(vectorize(series.matrices), cfg.standardize)

    if cfg.select_k:
        best = None
        for k in range(cfg.k_min, min(cfg.k_max, n - 1) + 1):
            km = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10).fit(feats)
            score = silhouette_score(feats, km.labels_) if k < n else -1.0
            logger.info("K=%d silhouette=%.3f", k, score)
            if best is None or score > best[0]:
                best = (score, k, km)
        silhouette, n_states, km = best  # type: ignore[misc]
    else:
        n_states = cfg.n_states
        km = KMeans(n_clusters=n_states, random_state=cfg.seed, n_init=10).fit(feats)
        silhouette = silhouette_score(feats, km.labels_) if 1 < n_states < n else 0.0

    labels = km.labels_.astype(np.int64)
    centroids = np.stack(
        [series.matrices[labels == s].mean(axis=0) for s in range(n_states)]
    ).reshape(n_states, n_neurons, n_neurons)

    logger.info("Fit %d connectivity states (silhouette=%.3f)", n_states, silhouette)
    return GraphStateModel(
        labels=labels,
        centroids=centroids,
        n_states=n_states,
        silhouette=float(silhouette),
        window_starts=series.window_starts,
    )
