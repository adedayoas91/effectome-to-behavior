"""Training-owned medoid adapters for the two binary-support graph geometries.

Jaccard preserves neuron identity; directed WL compares topology under relabelling.
Neither supplies a causal-equivalence or latent-regime certificate. See
``docs/graph_similarity_methodology.md`` for the estimand and reference audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import silhouette_score

from effectome.data_module.schema import ConnectivitySeries

from .graph_similarity import as_lagged_support, graph_similarity_distances
from .metrics import kmedoids

if TYPE_CHECKING:
    from .graph_states import GraphStateConfig, GraphStateModel


def _supports(series: ConnectivitySeries | np.ndarray) -> np.ndarray:
    from .graph_states import _lagged_tensor_from_series

    if not isinstance(series, ConnectivitySeries) or not series.directed:
        raise ValueError("support-similarity assignment requires a directed ConnectivitySeries")
    tensor = _lagged_tensor_from_series(series)
    if len(tensor) == 0:
        return tensor.astype(bool)
    return np.stack([as_lagged_support(graph) for graph in tensor])


def _initial_medoids(distance: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Seeded farthest-first points avoid duplicate graph classes at initialization.

    This is a project initialization choice for the installed alternating solver,
    not an optimality guarantee or a replacement clustering algorithm.
    """
    chosen = [int(np.random.default_rng(seed).integers(len(distance)))]
    while len(chosen) < count:
        nearest = distance[:, chosen].min(axis=1)
        candidate = int(np.argmax(nearest))
        if nearest[candidate] <= 1e-12:
            raise ValueError("state count exceeds distinguishable training graph classes under this metric")
        chosen.append(candidate)
    return np.asarray(chosen, dtype=int)


def assign_similarity_states(
    model: GraphStateModel, series: ConnectivitySeries | np.ndarray,
) -> np.ndarray:
    """Compare new graphs only to frozen lag-resolved training medoids."""
    from .graph_states import graph_schema

    schema = graph_schema(series, strict=True)
    assert schema is not None  # strict=True either supplies the registered schema or raises.
    if schema != model.graph_schema:
        raise ValueError("registered graph schema mismatch")
    if model.lagged_medoid_matrices is None:
        raise ValueError("lag-resolved medoids are required for graph-similarity assignment")
    supports = _supports(series)
    if len(supports) == 0:
        return np.empty(0, dtype=np.int64)
    distances = graph_similarity_distances(
        supports, model.lagged_medoid_matrices, metric=model.metric,
        wl_iterations=model.wl_iterations, lag_labels=schema[1],
    )
    return np.argmin(distances, axis=1).astype(np.int64)


def fit_similarity_states(
    series: ConnectivitySeries,
    cfg: GraphStateConfig,
    train_indices: np.ndarray,
    boundary_indices: np.ndarray,
    *,
    predictive_losses: dict[int, float] | None = None,
    feature_weights: np.ndarray | None = None,
) -> GraphStateModel:
    """Fit graph medoids within training data; diagnostic silhouette never selects K."""
    from .graph_states import GraphStateModel, graph_schema

    if cfg.feature_representation != "binary_lag_offdiag" or cfg.assignment != "kmedoids":
        raise ValueError("support-similarity metrics require binary_lag_offdiag and kmedoids")
    if cfg.standardize or cfg.pca_components is not None or feature_weights is not None:
        raise ValueError("support similarities do not accept standardization, PCA or feature weights")
    schema = graph_schema(series, strict=True)
    assert schema is not None
    supports = _supports(series)
    training = supports[train_indices]
    count = cfg.n_states
    if cfg.select_k:
        candidates = [k for k in (1, 2, 3, 4, 6) if k <= len(training)]
        if predictive_losses is None or any(
            k not in predictive_losses or not np.isfinite(predictive_losses[k]) for k in candidates
        ):
            raise ValueError("select_k requires caller-owned inner predictive losses for every candidate")
        count = min(candidates, key=lambda k: (predictive_losses[k], k))
    if type(count) is not int or not 1 <= count <= len(training):
        raise ValueError("state count is infeasible for the training windows")
    distances = graph_similarity_distances(
        training, metric=cfg.metric, wl_iterations=cfg.wl_iterations, lag_labels=schema[1],
    )
    initial = _initial_medoids(distances, count, cfg.seed)
    _, local_medoids = kmedoids(distances, count, seed=cfg.seed, initial_medoids=initial)
    medoid_indices = train_indices[local_medoids].copy()
    medoids = supports[medoid_indices].copy()
    all_distances = graph_similarity_distances(
        supports, medoids, metric=cfg.metric, wl_iterations=cfg.wl_iterations, lag_labels=schema[1],
    )
    labels = np.argmin(all_distances, axis=1).astype(np.int64)
    train_labels = labels[train_indices]
    score = (
        float(silhouette_score(distances, train_labels, metric="precomputed"))
        if 1 < len(np.unique(train_labels)) < len(training) else float("nan")
    )
    lag_centroids = np.stack([
        training[train_labels == state].mean(axis=0) for state in range(count)
    ])
    medoids.setflags(write=False)
    medoid_indices.setflags(write=False)
    centroids = np.asarray(series.matrices)[medoid_indices].copy()
    return GraphStateModel(
        labels=labels, centroids=centroids, n_states=count, silhouette=score,
        metric=cfg.metric, window_starts=np.asarray(series.window_starts, dtype=int).copy(),
        medoid_matrices=centroids.copy(), lagged_medoid_matrices=medoids,
        medoid_indices=medoid_indices, fit_indices=train_indices.copy(),
        boundary_indices=boundary_indices, feature_representation=cfg.feature_representation,
        assignment="kmedoids", lag_count=training.shape[1], n_nodes=training.shape[2],
        lag_centroids=lag_centroids, graph_schema=schema,
        selection_loss=None if predictive_losses is None else predictive_losses.get(count),
        membership_probabilities=np.eye(count, dtype=float)[labels],
        effective_algorithm="kmedoids_alternating",
        effective_distance=("directed_lagged_support_jaccard" if cfg.metric == "edge_jaccard"
                            else "normalized_directed_lagged_wl_kernel_distance"),
        requested_metric=cfg.metric, analysis_scope=cfg.analysis_scope, wl_iterations=cfg.wl_iterations,
    )
