"""Cluster the sequence of connectivity matrices into recurring dataset-global states.

The dynamic effectome {W_1..W_K} is treated as a trajectory in graph space, and clustering it
into recurring states (regimes) is *Fréchet quantization*: pick a codebook and assignment that
minimize summed squared distance under a chosen graph metric (see ``metrics.py``).

This implementation is dataset-global and fit/transform-aware:

* fit-time standardization is learned only on the fit subset and reused for held-out assignment;
* state centroids/medoids are learned once and reused by :meth:`GraphStateModel.transform`;
* recording/gap boundaries are inferred from anchor metadata or window starts and carried into
  downstream transition modeling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score

from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor

from .metrics import expm_sym, kmedoids, log_euclidean_features, logm_spd, pairwise_distances, vectorize

logger = logging.getLogger(__name__)

FEATURE_METRICS = ("frobenius", "cosine", "log_euclidean")
KERNEL_METRICS = ("causal_kernel",)
DISTANCE_METRICS = ("affine_invariant", "gromov_wasserstein")


@dataclass(frozen=True)
class GraphStateConfig:
    """Configuration for connectivity-state discovery.

    Attributes:
        n_states: Number of states K (ignored if select_k is True).
        select_k: If True, pick K in [k_min, k_max] by best silhouette score.
        k_min: Minimum K when selecting.
        k_max: Maximum K when selecting.
        standardize: Z-score features before clustering (fit subset only).
        metric: Graph-space geometry. One of FEATURE_METRICS, KERNEL_METRICS, DISTANCE_METRICS.
        eps: Eigenvalue floor for SPD projection (log_euclidean / affine_invariant).
        gw_epsilon: Entropic regularization for Gromov-Wasserstein.
        gw_max_iter: Outer iterations for Gromov-Wasserstein.
        kernel_gamma: RBF bandwidth multiplier for the causal-kernel affinity.
        boundary_gap_factor: Gaps larger than this multiple of the nominal stride start a new
            transition segment when explicit recording ids are unavailable.
        seed: Random seed for KMeans / spectral clustering / k-medoids.
    """

    n_states: int = 3
    select_k: bool = False
    k_min: int = 2
    k_max: int = 8
    standardize: bool = True
    metric: str = "causal_kernel"
    eps: float = 1e-6
    gw_epsilon: float = 0.05
    gw_max_iter: int = 200
    kernel_gamma: float = 1.0
    boundary_gap_factor: float = 3.0
    seed: int = 42


@dataclass
class GraphStateModel:
    """Result of dataset-global connectivity-state clustering.

    Attributes:
        labels: State label per window, shape (K_windows,).
        centroids: State centroid/medoid matrices, shape (n_states, N, N).
        n_states: Number of states.
        silhouette: Silhouette-like score of the chosen clustering on the fit subset.
        metric: Graph metric used for clustering.
        window_starts: Window start indices for the series labels.
        feature_centroids: Optional centroids in feature space used for held-out assignment.
        feature_mean: Fit-time feature mean for held-out standardization.
        feature_scale: Fit-time feature std for held-out standardization.
        medoid_matrices: Optional medoid graphs used for distance-based assignment.
        fit_indices: Indices used to fit the state model.
        boundary_indices: Start indices of independent transition segments.
    """

    labels: np.ndarray
    centroids: np.ndarray
    n_states: int
    silhouette: float
    metric: str
    window_starts: np.ndarray
    feature_centroids: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    medoid_matrices: np.ndarray | None = None
    fit_indices: np.ndarray | None = None
    boundary_indices: np.ndarray | None = None
    eps: float = 1e-6
    gw_epsilon: float = 0.05
    gw_max_iter: int = 200

    def transform(
        self,
        series: ConnectivitySeries | np.ndarray,
        *,
        window_starts: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assign labels to new matrices without refitting the state model."""
        matrices = series.matrices if isinstance(series, ConnectivitySeries) else np.asarray(series)
        starts = series.window_starts if isinstance(series, ConnectivitySeries) else window_starts
        labels = assign_graph_states(self, matrices)
        if starts is not None:
            self.window_starts = np.asarray(starts, dtype=int)
        return labels


@dataclass(frozen=True)
class _Standardizer:
    mean: np.ndarray | None
    scale: np.ndarray | None

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            return features
        return (features - self.mean) / self.scale


def _series_metadata(series: ConnectivitySeries) -> dict:
    metadata = getattr(series, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}


def _boundary_indices_from_anchors(anchors: list[TemporalAnchor]) -> np.ndarray:
    """Infer segment starts from typed temporal anchors."""
    if not anchors:
        return np.asarray([0], dtype=int)

    def _key(anchor: TemporalAnchor) -> tuple[str, str, str | None, str | None, str | None]:
        return (
            anchor.dataset_id,
            anchor.recording_id,
            anchor.animal_id,
            anchor.session_id,
            anchor.segment_id,
        )

    boundaries = [0]
    for idx in range(1, len(anchors)):
        prev = anchors[idx - 1]
        cur = anchors[idx]
        if _key(cur) != _key(prev) or cur.gap_before or prev.gap_after:
            boundaries.append(idx)
    return np.asarray(boundaries, dtype=int)


def infer_boundary_indices(
    window_starts: np.ndarray,
    *,
    recording_ids: np.ndarray | None = None,
    gap_factor: float = 3.0,
) -> np.ndarray:
    """Infer independent transition segments from pooled window starts / recording ids."""
    starts = np.asarray(window_starts, dtype=int)
    if starts.ndim != 1:
        raise ValueError("window_starts must be a 1D array")
    boundaries = [0]
    if starts.size <= 1:
        return np.asarray(boundaries, dtype=int)

    if recording_ids is not None:
        rec = np.asarray(recording_ids)
        if rec.shape[0] != starts.shape[0]:
            raise ValueError("recording_ids must match window_starts length")
        for idx in range(1, len(starts)):
            if rec[idx] != rec[idx - 1]:
                boundaries.append(idx)
        return np.asarray(boundaries, dtype=int)

    diffs = np.diff(starts)
    positive = diffs[diffs > 0]
    nominal_stride = float(np.median(positive)) if positive.size else 1.0
    gap_threshold = max(gap_factor * nominal_stride, nominal_stride + 1.0)
    for idx, delta in enumerate(diffs, start=1):
        if delta <= 0 or delta > gap_threshold:
            boundaries.append(idx)
    return np.asarray(boundaries, dtype=int)


def boundary_indices_for_series(
    series: ConnectivitySeries,
    *,
    gap_factor: float = 3.0,
) -> np.ndarray:
    """Infer boundary indices from typed anchors first, then fall back to provenance or starts."""
    anchors = list(getattr(series, "anchors", []))
    if anchors:
        if len(anchors) != series.n_windows:
            raise ValueError("series.anchors must align 1:1 with series windows")
        return _boundary_indices_from_anchors(anchors)

    metadata = _series_metadata(series)
    recording_ids = metadata.get("recording_ids")
    return infer_boundary_indices(
        series.window_starts,
        recording_ids=None if recording_ids is None else np.asarray(recording_ids),
        gap_factor=gap_factor,
    )


def _fit_standardizer(features: np.ndarray, standardize: bool) -> tuple[np.ndarray, _Standardizer]:
    if not standardize:
        return features, _Standardizer(None, None)
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    return (features - mean) / scale, _Standardizer(mean=mean, scale=scale)


def _causal_kernel_raw_features(matrices: np.ndarray) -> np.ndarray:
    pos = np.maximum(matrices, 0.0)
    neg = np.maximum(-matrices, 0.0)
    return np.concatenate(
        [
            vectorize(matrices),
            vectorize(pos),
            vectorize(neg),
            pos.sum(axis=2),
            neg.sum(axis=2),
            pos.sum(axis=1),
            neg.sum(axis=1),
        ],
        axis=1,
    )


def _rbf_affinity(features: np.ndarray, gamma: float) -> np.ndarray:
    sq = np.sum(features**2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (features @ features.T), 0.0)
    scale = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
    eff_gamma = gamma / max(scale, 1e-12)
    affinity = np.exp(-eff_gamma * d2)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def _raw_features_for(matrices: np.ndarray, metric: str, cfg: GraphStateConfig) -> np.ndarray:
    if metric == "log_euclidean":
        return log_euclidean_features(matrices, cfg.eps)
    if metric == "cosine":
        feats = vectorize(matrices)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms
    if metric == "frobenius":
        return vectorize(matrices)
    if metric == "causal_kernel":
        return _causal_kernel_raw_features(matrices)
    raise ValueError(f"metric '{metric}' does not use direct features")


def _assign_feature_labels(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    d2 = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1).astype(np.int64)


def _centroids(
    matrices: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    metric: str,
    eps: float,
    medoid_matrices: np.ndarray | None = None,
) -> np.ndarray:
    """Per-state centroid graph: arithmetic/log-Euclidean mean or medoid graph."""
    n_neurons = matrices.shape[1]
    out = []
    for state in range(n_states):
        members = matrices[labels == state]
        if medoid_matrices is not None:
            out.append(medoid_matrices[state])
        elif members.shape[0] == 0:
            out.append(np.zeros((n_neurons, n_neurons)))
        elif metric == "log_euclidean":
            mean_log = np.mean([logm_spd(w, eps) for w in members], axis=0)
            out.append(expm_sym(mean_log))
        else:
            out.append(members.mean(axis=0))
    return np.stack(out).reshape(n_states, n_neurons, n_neurons)


def _fit_kernel_path(
    train_matrices: np.ndarray,
    cfg: GraphStateConfig,
) -> tuple[np.ndarray, np.ndarray, _Standardizer, int, float]:
    """Spectral clustering on signed directed effectome features."""
    n = train_matrices.shape[0]
    raw = _raw_features_for(train_matrices, "causal_kernel", cfg)
    feats, standardizer = _fit_standardizer(raw, cfg.standardize)
    affinity = _rbf_affinity(feats, cfg.kernel_gamma)
    ks = range(cfg.k_min, min(cfg.k_max, n - 1) + 1) if cfg.select_k else [cfg.n_states]
    best = None
    for k in ks:
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=cfg.seed,
        ).fit(affinity)
        labels = model.labels_.astype(np.int64)
        score = silhouette_score(feats, labels) if 1 < len(np.unique(labels)) < n else 0.0
        logger.info("metric=%s K=%d silhouette=%.3f", cfg.metric, k, score)
        if best is None or score > best[0]:
            centroids = np.stack([feats[labels == state].mean(axis=0) for state in range(k)])
            best = (score, k, labels, centroids)
    score, k, labels, centroids = best  # type: ignore[misc]
    return labels, centroids, standardizer, int(k), float(score)


def _fit_feature_path(
    train_matrices: np.ndarray,
    cfg: GraphStateConfig,
) -> tuple[np.ndarray, np.ndarray, _Standardizer, int, float]:
    """KMeans in a Euclidean feature space (frobenius / cosine / log_euclidean)."""
    n = train_matrices.shape[0]
    raw = _raw_features_for(train_matrices, cfg.metric, cfg)
    feats, standardizer = _fit_standardizer(raw, cfg.standardize)
    ks = range(cfg.k_min, min(cfg.k_max, n - 1) + 1) if cfg.select_k else [cfg.n_states]
    best = None
    for k in ks:
        km = KMeans(n_clusters=k, random_state=cfg.seed, n_init=10).fit(feats)
        score = silhouette_score(feats, km.labels_) if 1 < k < n else 0.0
        logger.info("metric=%s K=%d silhouette=%.3f", cfg.metric, k, score)
        if best is None or score > best[0]:
            best = (score, k, km.labels_.astype(np.int64), km.cluster_centers_)
    score, k, labels, feature_centroids = best  # type: ignore[misc]
    return labels, feature_centroids, standardizer, int(k), float(score)


def _fit_distance_path(
    train_matrices: np.ndarray,
    cfg: GraphStateConfig,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """K-medoids on a precomputed distance matrix (affine_invariant / gromov_wasserstein)."""
    n = train_matrices.shape[0]
    dist = pairwise_distances(
        train_matrices,
        cfg.metric,
        eps=cfg.eps,
        gw_epsilon=cfg.gw_epsilon,
        gw_max_iter=cfg.gw_max_iter,
    )
    ks = range(cfg.k_min, min(cfg.k_max, n - 1) + 1) if cfg.select_k else [cfg.n_states]
    best = None
    for k in ks:
        labels, medoid_indices = kmedoids(dist, k, seed=cfg.seed)
        valid = 1 < len(np.unique(labels)) < n
        score = silhouette_score(dist, labels, metric="precomputed") if valid else 0.0
        logger.info("metric=%s K=%d silhouette=%.3f", cfg.metric, k, score)
        if best is None or score > best[0]:
            best = (score, k, labels.astype(np.int64), train_matrices[medoid_indices])
    score, k, labels, medoid_matrices = best  # type: ignore[misc]
    return labels, medoid_matrices, int(k), float(score)


def assign_graph_states(model: GraphStateModel, matrices: np.ndarray) -> np.ndarray:
    """Assign matrices to the nearest learned state without refitting."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if model.metric in FEATURE_METRICS or model.metric in KERNEL_METRICS:
        raw = _raw_features_for(
            matrices,
            model.metric,
            GraphStateConfig(
                metric=model.metric,
                eps=model.eps,
                gw_epsilon=model.gw_epsilon,
                gw_max_iter=model.gw_max_iter,
            ),
        )
        standardizer = _Standardizer(model.feature_mean, model.feature_scale)
        feats = standardizer.transform(raw)
        if model.feature_centroids is None:
            raise ValueError("feature_centroids are required for feature/kernel state assignment")
        return _assign_feature_labels(feats, model.feature_centroids)

    if model.metric in DISTANCE_METRICS:
        if model.medoid_matrices is None:
            raise ValueError("medoid_matrices are required for distance-based state assignment")
        dist = np.zeros((matrices.shape[0], model.medoid_matrices.shape[0]))
        for i, matrix in enumerate(matrices):
            for state, medoid in enumerate(model.medoid_matrices):
                dist[i, state] = pairwise_distances(
                    np.stack([matrix, medoid]),
                    model.metric,
                    eps=model.eps,
                    gw_epsilon=model.gw_epsilon,
                    gw_max_iter=model.gw_max_iter,
                )[0, 1]
        return np.argmin(dist, axis=1).astype(np.int64)

    raise ValueError(f"unknown metric '{model.metric}'")


def fit_graph_states(
    series: ConnectivitySeries,
    cfg: GraphStateConfig,
    *,
    fit_indices: np.ndarray | None = None,
) -> GraphStateModel:
    """Fit dataset-global connectivity states and assign held-out windows without refitting."""
    matrices = np.asarray(series.matrices, dtype=np.float64)
    n_windows = matrices.shape[0]
    train_idx = np.arange(n_windows, dtype=int) if fit_indices is None else np.asarray(fit_indices, dtype=int)
    train_matrices = matrices[train_idx]

    boundary_indices = boundary_indices_for_series(series, gap_factor=cfg.boundary_gap_factor)

    feature_centroids = None
    feature_mean = None
    feature_scale = None
    medoid_matrices = None

    if cfg.metric in FEATURE_METRICS:
        train_labels, feature_centroids, standardizer, n_states, silhouette = _fit_feature_path(
            train_matrices, cfg
        )
        all_feats = standardizer.transform(_raw_features_for(matrices, cfg.metric, cfg))
        labels = _assign_feature_labels(all_feats, feature_centroids)
        labels[train_idx] = train_labels
        feature_mean, feature_scale = standardizer.mean, standardizer.scale
    elif cfg.metric in KERNEL_METRICS:
        train_labels, feature_centroids, standardizer, n_states, silhouette = _fit_kernel_path(
            train_matrices, cfg
        )
        all_feats = standardizer.transform(_raw_features_for(matrices, cfg.metric, cfg))
        labels = _assign_feature_labels(all_feats, feature_centroids)
        labels[train_idx] = train_labels
        feature_mean, feature_scale = standardizer.mean, standardizer.scale
    elif cfg.metric in DISTANCE_METRICS:
        train_labels, medoid_matrices, n_states, silhouette = _fit_distance_path(train_matrices, cfg)
        temp_model = GraphStateModel(
            labels=train_labels,
            centroids=medoid_matrices,
            n_states=n_states,
            silhouette=silhouette,
            metric=cfg.metric,
            window_starts=series.window_starts[train_idx],
            medoid_matrices=medoid_matrices,
            eps=cfg.eps,
            gw_epsilon=cfg.gw_epsilon,
            gw_max_iter=cfg.gw_max_iter,
        )
        labels = assign_graph_states(temp_model, matrices)
        labels[train_idx] = train_labels
    else:
        raise ValueError(
            f"unknown metric '{cfg.metric}'; choices: {FEATURE_METRICS + KERNEL_METRICS + DISTANCE_METRICS}"
        )

    centroids = _centroids(matrices, labels, n_states, cfg.metric, cfg.eps, medoid_matrices=medoid_matrices)
    logger.info(
        "Fit %d dataset-global connectivity states (metric=%s, silhouette=%.3f, fit_windows=%d/%d)",
        n_states,
        cfg.metric,
        silhouette,
        len(train_idx),
        n_windows,
    )
    return GraphStateModel(
        labels=labels.astype(np.int64),
        centroids=centroids,
        n_states=n_states,
        silhouette=float(silhouette),
        metric=cfg.metric,
        window_starts=np.asarray(series.window_starts, dtype=int),
        feature_centroids=(
            None if feature_centroids is None else np.asarray(feature_centroids, dtype=np.float64)
        ),
        feature_mean=None if feature_mean is None else np.asarray(feature_mean, dtype=np.float64),
        feature_scale=None if feature_scale is None else np.asarray(feature_scale, dtype=np.float64),
        medoid_matrices=None if medoid_matrices is None else np.asarray(medoid_matrices, dtype=np.float64),
        fit_indices=train_idx,
        boundary_indices=boundary_indices,
        eps=cfg.eps,
        gw_epsilon=cfg.gw_epsilon,
        gw_max_iter=cfg.gw_max_iter,
    )
