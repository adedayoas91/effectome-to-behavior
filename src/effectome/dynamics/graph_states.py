"""Cluster the sequence of connectivity matrices into recurring 'connectivity states'.

The dynamic effectome {W_1..W_K} is treated as a trajectory in graph space, and clustering it
into recurring states (regimes) is *Fréchet quantization*: pick a codebook and assignment that
minimize summed squared distance under a chosen graph metric (see ``metrics.py``). The metric
choice is not cosmetic:

* ``frobenius`` / ``cosine`` -- flat geometry, vectorized matrices, arithmetic-mean centroids
  (the original behavior; Frobenius k-means is the special case in the manuscript).
* ``log_euclidean`` -- Riemannian SPD geometry with a closed-form barycenter (expm of the mean
  matrix-log), avoiding the determinant "swelling" bias of Euclidean averaging.
* ``affine_invariant`` / ``gromov_wasserstein`` -- distance-matrix clustering via k-medoids,
  whose centroids are discrete Fréchet barycenters (medoid graphs); Gromov-Wasserstein compares
  graphs relationally, without assuming node correspondence.

The state label sequence becomes the input to the transition model (Stage 3b). Model selection
over the number of states K is supported via silhouette score in the active geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from effectome.data_module.schema import ConnectivitySeries

from .metrics import (
    expm_sym,
    kmedoids,
    log_euclidean_features,
    logm_spd,
    pairwise_distances,
    vectorize,
)

logger = logging.getLogger(__name__)

FEATURE_METRICS = ("frobenius", "cosine", "log_euclidean")
DISTANCE_METRICS = ("affine_invariant", "gromov_wasserstein")


@dataclass(frozen=True)
class GraphStateConfig:
    """Configuration for connectivity-state discovery.

    Attributes:
        n_states: Number of states K (ignored if select_k is True).
        select_k: If True, pick K in [k_min, k_max] by best silhouette score.
        k_min: Minimum K when selecting.
        k_max: Maximum K when selecting.
        standardize: Z-score features before clustering (feature-metric path only).
        metric: Graph-space geometry. One of FEATURE_METRICS or DISTANCE_METRICS.
        eps: Eigenvalue floor for SPD projection (log_euclidean / affine_invariant).
        gw_epsilon: Entropic regularization for Gromov-Wasserstein.
        gw_max_iter: Outer iterations for Gromov-Wasserstein.
        seed: Random seed for KMeans / k-medoids.
    """

    n_states: int = 3
    select_k: bool = False
    k_min: int = 2
    k_max: int = 8
    standardize: bool = True
    metric: str = "frobenius"
    eps: float = 1e-6
    gw_epsilon: float = 0.05
    gw_max_iter: int = 200
    seed: int = 42


@dataclass
class GraphStateModel:
    """Result of connectivity-state clustering.

    Attributes:
        labels: State label per window, shape (K_windows,).
        centroids: State centroid matrices, shape (n_states, N, N). Fréchet means
            (arithmetic / log-Euclidean) or medoid graphs, depending on metric.
        n_states: Number of states.
        silhouette: Silhouette score of the chosen clustering (in the active geometry).
        metric: Graph metric used for clustering.
        window_starts: Window start indices (carried through for alignment).
    """

    labels: np.ndarray
    centroids: np.ndarray
    n_states: int
    silhouette: float
    metric: str
    window_starts: np.ndarray


def _prepare(features: np.ndarray, standardize: bool) -> np.ndarray:
    if not standardize:
        return features
    mu = features.mean(axis=0, keepdims=True)
    sd = features.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (features - mu) / sd


def _features_for(matrices: np.ndarray, metric: str, cfg: GraphStateConfig) -> np.ndarray:
    """Build the feature matrix whose Euclidean geometry equals the requested metric."""
    if metric == "log_euclidean":
        feats = log_euclidean_features(matrices, cfg.eps)
    elif metric == "cosine":
        feats = vectorize(matrices)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
    else:  # frobenius
        feats = vectorize(matrices)
    return _prepare(feats, cfg.standardize)


def _centroids(
    matrices: np.ndarray, labels: np.ndarray, n_states: int, metric: str, eps: float
) -> np.ndarray:
    """Per-state centroid graph: log-Euclidean / arithmetic Fréchet mean."""
    n_neurons = matrices.shape[1]
    out = []
    for s in range(n_states):
        members = matrices[labels == s]
        if members.shape[0] == 0:
            out.append(np.zeros((n_neurons, n_neurons)))
        elif metric == "log_euclidean":
            mean_log = np.mean([logm_spd(w, eps) for w in members], axis=0)
            out.append(expm_sym(mean_log))
        else:
            out.append(members.mean(axis=0))
    return np.stack(out).reshape(n_states, n_neurons, n_neurons)


def _fit_feature_path(matrices: np.ndarray, cfg: GraphStateConfig) -> tuple[np.ndarray, int, float]:
    """KMeans in a Euclidean feature space (frobenius / cosine / log_euclidean)."""
    n = matrices.shape[0]
    feats = _features_for(matrices, cfg.metric, cfg)
    ks = range(cfg.k_min, min(cfg.k_max, n - 1) + 1) if cfg.select_k else [cfg.n_states]
    best = None
    for k in ks:
        km = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10).fit(feats)
        score = silhouette_score(feats, km.labels_) if 1 < k < n else 0.0
        logger.info("metric=%s K=%d silhouette=%.3f", cfg.metric, k, score)
        if best is None or score > best[0]:
            best = (score, k, km.labels_.astype(np.int64))
    score, k, labels = best  # type: ignore[misc]
    return labels, k, float(score)


def _fit_distance_path(matrices: np.ndarray, cfg: GraphStateConfig) -> tuple[np.ndarray, int, float]:
    """K-medoids on a precomputed distance matrix (affine_invariant / gromov_wasserstein)."""
    n = matrices.shape[0]
    dist = pairwise_distances(
        matrices, cfg.metric, eps=cfg.eps, gw_epsilon=cfg.gw_epsilon, gw_max_iter=cfg.gw_max_iter
    )
    ks = range(cfg.k_min, min(cfg.k_max, n - 1) + 1) if cfg.select_k else [cfg.n_states]
    best = None
    for k in ks:
        labels, _ = kmedoids(dist, k, seed=cfg.seed)
        valid = 1 < len(np.unique(labels)) < n
        score = silhouette_score(dist, labels, metric="precomputed") if valid else 0.0
        logger.info("metric=%s K=%d silhouette=%.3f", cfg.metric, k, score)
        if best is None or score > best[0]:
            best = (score, k, labels.astype(np.int64))
    score, k, labels = best  # type: ignore[misc]
    return labels, k, float(score)


def fit_graph_states(series: ConnectivitySeries, cfg: GraphStateConfig) -> GraphStateModel:
    """Cluster connectivity matrices into states; return labels + centroid graphs.

    Dispatches on ``cfg.metric`` between a Euclidean feature path (KMeans) and a
    distance-matrix path (k-medoids). See module docstring for the geometries.
    """
    matrices = series.matrices
    if cfg.metric in FEATURE_METRICS:
        labels, n_states, silhouette = _fit_feature_path(matrices, cfg)
    elif cfg.metric in DISTANCE_METRICS:
        labels, n_states, silhouette = _fit_distance_path(matrices, cfg)
    else:
        raise ValueError(
            f"unknown metric '{cfg.metric}'; choices: {FEATURE_METRICS + DISTANCE_METRICS}"
        )

    centroids = _centroids(matrices, labels, n_states, cfg.metric, cfg.eps)
    logger.info(
        "Fit %d connectivity states (metric=%s, silhouette=%.3f)", n_states, cfg.metric, silhouette
    )
    return GraphStateModel(
        labels=labels,
        centroids=centroids,
        n_states=n_states,
        silhouette=float(silhouette),
        metric=cfg.metric,
        window_starts=series.window_starts,
    )
