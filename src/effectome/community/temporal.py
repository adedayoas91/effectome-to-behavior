"""Temporally regularized community inference across the dynamic effectome.

The primary ``temporal`` method is a dependency-free joint objective over node roles across
time: each neuron's signed directed role vector is clustered into a shared community vocabulary
while an interlayer penalty discourages gratuitous switching between adjacent windows. This gives
one community label per neuron per layer without relying on independent per-window partitions plus
greedy relabeling.

Two modes are supported:

* ``retrospective``: full-sequence alternating optimization with Viterbi smoothing.
* ``prospective``: one-sided online assignment / centroid updates that do not use future layers.

The legacy greedy label-matching baseline is kept as ``temporal_greedy``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from effectome.data_module.schema import CommunitySeries, ConnectivitySeries
from effectome.dynamics.graph_states import boundary_indices_for_series

from .base import CommunityConfig, CommunityDetector, build_community_series, register_community
from .static import LeidenCommunity


def _match_labels(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Relabel ``cur`` so its communities align with ``prev`` by maximum node overlap."""
    prev_ids = np.unique(prev)
    cur_ids = np.unique(cur)
    overlap = np.zeros((len(cur_ids), len(prev_ids)))
    for i, cur_id in enumerate(cur_ids):
        for j, prev_id in enumerate(prev_ids):
            overlap[i, j] = np.sum((cur == cur_id) & (prev == prev_id))

    mapping: dict[int, int] = {}
    used: set[int] = set()
    order = np.argsort(-overlap.max(axis=1))
    next_free = int(prev.max()) + 1
    for i in order:
        cur_id = int(cur_ids[i])
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
        mapping[cur_id] = assigned
    return np.array([mapping[int(x)] for x in cur], dtype=np.int64)


def _signed_role_features(matrix: np.ndarray) -> np.ndarray:
    """Per-node signed directed features preserving in/out roles and sign structure."""
    w = np.asarray(matrix, dtype=np.float64)
    pos = np.maximum(w, 0.0)
    neg = np.maximum(-w, 0.0)
    out_signed = w
    in_signed = w.T
    out_strength = np.stack([pos.sum(axis=1), neg.sum(axis=1)], axis=1)
    in_strength = np.stack([pos.sum(axis=0), neg.sum(axis=0)], axis=1)
    balance = np.stack(
        [
            out_strength[:, 0] - out_strength[:, 1],
            in_strength[:, 0] - in_strength[:, 1],
        ],
        axis=1,
    )
    return np.concatenate(
        [out_signed, in_signed, pos, neg, pos.T, neg.T, out_strength, in_strength, balance],
        axis=1,
    )


def _standardize_node_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    scale = flat.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    standardized = (flat - mean) / scale
    return standardized.reshape(features.shape), mean, scale


def _sqdist(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)


def _coupled_step_mask(t_steps: int, boundary_indices: np.ndarray | None) -> np.ndarray:
    mask = np.ones(max(t_steps - 1, 0), dtype=bool)
    if boundary_indices is None:
        return mask
    for idx in np.asarray(boundary_indices, dtype=int):
        if 0 < idx < t_steps:
            mask[idx - 1] = False
    return mask


def _viterbi_assign(
    unary: np.ndarray,
    penalty: float,
    *,
    boundary_indices: np.ndarray | None = None,
) -> np.ndarray:
    t_steps, n_states = unary.shape
    cost = np.empty((t_steps, n_states), dtype=np.float64)
    back = np.zeros((t_steps, n_states), dtype=np.int64)
    cost[0] = unary[0]
    coupled_steps = _coupled_step_mask(t_steps, boundary_indices)
    for t in range(1, t_steps):
        if not coupled_steps[t - 1]:
            best_prev = int(np.argmin(cost[t - 1]))
            back[t].fill(best_prev)
            cost[t] = unary[t] + cost[t - 1, best_prev]
            continue
        switch = cost[t - 1][:, None] + penalty * (1.0 - np.eye(n_states))
        back[t] = np.argmin(switch, axis=0)
        cost[t] = unary[t] + switch[back[t], np.arange(n_states)]
    path = np.empty(t_steps, dtype=np.int64)
    path[-1] = int(np.argmin(cost[-1]))
    for t in range(t_steps - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _objective(
    features: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    penalty: float,
    *,
    boundary_indices: np.ndarray | None = None,
) -> float:
    total = 0.0
    coupled_steps = _coupled_step_mask(features.shape[0], boundary_indices)
    for neuron in range(features.shape[1]):
        unary = _sqdist(features[:, neuron, :], centroids)
        total += float(unary[np.arange(features.shape[0]), labels[:, neuron]].sum())
        if coupled_steps.size:
            switches = labels[1:, neuron] != labels[:-1, neuron]
            total += float(penalty * np.sum(switches[coupled_steps]))
    return total


def _update_centroids(features: np.ndarray, labels: np.ndarray, n_states: int) -> np.ndarray:
    flat_features = features.reshape(-1, features.shape[-1])
    flat_labels = labels.reshape(-1)
    centroids = np.zeros((n_states, features.shape[-1]), dtype=np.float64)
    for state in range(n_states):
        members = flat_features[flat_labels == state]
        if len(members) == 0:
            centroids[state] = flat_features[state % len(flat_features)]
        else:
            centroids[state] = members.mean(axis=0)
    return centroids


def _align_run_labels(reference: np.ndarray, labels: np.ndarray, n_states: int) -> np.ndarray:
    cost = np.zeros((n_states, n_states), dtype=np.float64)
    ref = reference.reshape(-1)
    cur = labels.reshape(-1)
    for i in range(n_states):
        for j in range(n_states):
            cost[i, j] = -float(np.sum((ref == i) & (cur == j)))
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(col): int(row) for row, col in zip(rows, cols, strict=False)}
    aligned = np.vectorize(lambda x: mapping.get(int(x), int(x)))(labels)
    return aligned.astype(np.int64)


def _coassignment_matrix(runs: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(runs)
    return np.mean(stack[:, :, :, None] == stack[:, :, None, :], axis=0)


def _flexibility(labels: np.ndarray, boundary_indices: np.ndarray | None = None) -> np.ndarray:
    if labels.shape[0] <= 1:
        return np.zeros(labels.shape[1], dtype=np.float64)
    coupled_steps = _coupled_step_mask(labels.shape[0], boundary_indices)
    if not np.any(coupled_steps):
        return np.zeros(labels.shape[1], dtype=np.float64)
    return np.mean((labels[1:] != labels[:-1])[coupled_steps], axis=0)


@dataclass(frozen=True)
class _TemporalSettings:
    n_communities: int
    temporal_penalty: float
    n_runs: int
    max_iter: int
    mode: str
    init_window_count: int
    online_learning_rate: float


class _TemporalRegularizedMixin:
    cfg: CommunityConfig

    def _settings(self) -> _TemporalSettings:
        extra = self.cfg.extra
        return _TemporalSettings(
            n_communities=int(extra.get("n_communities", 3)),
            temporal_penalty=float(extra.get("temporal_penalty", 1.5)),
            n_runs=max(1, int(extra.get("n_runs", 4))),
            max_iter=max(1, int(extra.get("max_iter", 20))),
            mode=str(extra.get("mode", "retrospective")),
            init_window_count=max(1, int(extra.get("init_window_count", 4))),
            online_learning_rate=float(extra.get("online_learning_rate", 0.25)),
        )

    def _prepare_signed(self, matrix: np.ndarray) -> np.ndarray:
        out = np.array(matrix, copy=True, dtype=np.float64)
        if self.cfg.weight_threshold > 0:
            out = np.where(np.abs(out) >= self.cfg.weight_threshold, out, 0.0)
        np.fill_diagonal(out, 0.0)
        return out

    def _feature_tensor(self, series: ConnectivitySeries) -> np.ndarray:
        settings = self._settings()
        matrices = np.stack([self._prepare_signed(w) for w in series.matrices])
        feats = np.stack([_signed_role_features(w) for w in matrices])
        if settings.mode == "prospective":
            # A prospective label at layer k may not depend on k+1.  Initialize scaling and
            # centroids from the first observed layer only; init_window_count remains a retained
            # compatibility setting but is not allowed to introduce a future-looking warm-up.
            base = feats[0].reshape(-1, feats.shape[-1])
            mean = base.mean(axis=0, keepdims=True)
            scale = base.std(axis=0, keepdims=True)
            scale[scale == 0] = 1.0
            return ((feats.reshape(-1, feats.shape[-1]) - mean) / scale).reshape(feats.shape)
        standardized, _, _ = _standardize_node_features(feats)
        return standardized

    def _fit_retro_run(
        self,
        features: np.ndarray,
        seed: int,
        *,
        boundary_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        settings = self._settings()
        flat = features.reshape(-1, features.shape[-1])
        km = KMeans(n_clusters=settings.n_communities, random_state=seed, n_init=10)
        labels = km.fit_predict(flat).reshape(features.shape[0], features.shape[1]).astype(np.int64)
        centroids = km.cluster_centers_.astype(np.float64)
        for _ in range(settings.max_iter):
            updated = np.zeros_like(labels)
            for neuron in range(features.shape[1]):
                unary = _sqdist(features[:, neuron, :], centroids)
                updated[:, neuron] = _viterbi_assign(
                    unary,
                    settings.temporal_penalty,
                    boundary_indices=boundary_indices,
                )
            new_centroids = _update_centroids(features, updated, settings.n_communities)
            if np.array_equal(updated, labels) and np.allclose(new_centroids, centroids):
                labels, centroids = updated, new_centroids
                break
            labels, centroids = updated, new_centroids
        return (
            labels,
            centroids,
            _objective(
                features,
                labels,
                centroids,
                settings.temporal_penalty,
                boundary_indices=boundary_indices,
            ),
        )

    def _fit_prospective_run(
        self,
        features: np.ndarray,
        seed: int,
        *,
        boundary_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        settings = self._settings()
        km = KMeans(n_clusters=settings.n_communities, random_state=seed, n_init=10)
        km.fit(features[0])
        centroids = km.cluster_centers_.astype(np.float64)
        labels = np.zeros((features.shape[0], features.shape[1]), dtype=np.int64)
        counts = np.ones(settings.n_communities, dtype=np.float64)
        boundary_starts = set(
            np.asarray([] if boundary_indices is None else boundary_indices, dtype=int).tolist()
        )
        for t in range(features.shape[0]):
            unary = _sqdist(features[t], centroids)
            if t == 0 or t in boundary_starts:
                labels[t] = np.argmin(unary, axis=1)
            else:
                switch = settings.temporal_penalty * (
                    np.arange(settings.n_communities)[None, :] != labels[t - 1][:, None]
                )
                labels[t] = np.argmin(unary + switch, axis=1)
            for state in range(settings.n_communities):
                members = features[t, labels[t] == state]
                if len(members) == 0:
                    continue
                batch = members.mean(axis=0)
                lr = min(1.0, settings.online_learning_rate / counts[state])
                centroids[state] = (1.0 - lr) * centroids[state] + lr * batch
                counts[state] += len(members)
        return (
            labels,
            centroids,
            _objective(
                features,
                labels,
                centroids,
                settings.temporal_penalty,
                boundary_indices=boundary_indices,
            ),
        )

    def _fit_runs(
        self,
        features: np.ndarray,
        *,
        boundary_indices: np.ndarray | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        settings = self._settings()
        runs: list[np.ndarray] = []
        scores: list[float] = []
        for run_idx in range(settings.n_runs):
            seed = self.cfg.seed + run_idx
            if settings.mode == "prospective":
                labels, _, score = self._fit_prospective_run(
                    features,
                    seed,
                    boundary_indices=boundary_indices,
                )
            elif settings.mode == "retrospective":
                labels, _, score = self._fit_retro_run(
                    features,
                    seed,
                    boundary_indices=boundary_indices,
                )
            else:
                raise ValueError(f"unknown temporal mode '{settings.mode}'")
            runs.append(labels)
            scores.append(score)
            if progress_callback is not None:
                progress_callback(run_idx + 1, settings.n_runs)
        return runs, scores

    def _package(
        self,
        series: ConnectivitySeries,
        runs: list[np.ndarray],
        scores: list[float],
        *,
        boundary_indices: np.ndarray | None = None,
    ) -> CommunitySeries:
        settings = self._settings()
        ref_idx = int(np.argmin(scores))
        reference = runs[ref_idx]
        aligned = [reference]
        for idx, labels in enumerate(runs):
            if idx == ref_idx:
                continue
            aligned.append(_align_run_labels(reference, labels, settings.n_communities))
        stack = np.stack(aligned)
        consensus = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=settings.n_communities).argmax(),
            0,
            stack,
        ).astype(np.int64)
        counts = np.array([len(np.unique(row)) for row in consensus])
        coupled_steps = _coupled_step_mask(consensus.shape[0], boundary_indices)
        switching = consensus[1:] != consensus[:-1]
        if switching.size and coupled_steps.size:
            switching = switching & coupled_steps[:, None]
        switching_rate = float(np.mean(switching[coupled_steps])) if np.any(coupled_steps) else 0.0
        agreements = [
            np.max(np.bincount(stack[:, t, n], minlength=settings.n_communities)) / len(aligned)
            for t in range(consensus.shape[0])
            for n in range(consensus.shape[1])
        ]
        return build_community_series(
            series=series,
            labels=consensus,
            method=self.cfg.name,
            n_communities_per_window=counts,
            extras={
                "resolution": float(self.cfg.resolution),
                "interlayer_coupling": float(settings.temporal_penalty),
                "boundary_indices": (
                    None if boundary_indices is None else np.asarray(boundary_indices, dtype=int)
                ),
                "mode": settings.mode,
                "flexibility": _flexibility(consensus, boundary_indices=boundary_indices),
                "switching": switching,
                "switching_rate": switching_rate,
                "coassignment": _coassignment_matrix(aligned),
                "stability": float(np.mean(agreements)) if agreements else 1.0,
                "objective_scores": np.asarray(scores, dtype=np.float64),
                "n_runs": len(runs),
                "metadata": {
                    "connectivity_diagnostics": dict(series.diagnostics),
                    "temporal_mode": settings.mode,
                    "temporal_penalty": float(settings.temporal_penalty),
                },
            },
        )


@register_community("temporal")
class TemporalCommunity(_TemporalRegularizedMixin, CommunityDetector):
    """Joint temporally regularized community detection on signed directed node roles."""

    def detect_one(self, matrix: np.ndarray) -> np.ndarray:  # pragma: no cover - run() is primary API
        raise NotImplementedError("temporal communities are inferred jointly across windows")

    def run(
        self,
        series: ConnectivitySeries,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> CommunitySeries:
        features = self._feature_tensor(series)
        boundary_indices = boundary_indices_for_series(series)
        runs, scores = self._fit_runs(
            features,
            boundary_indices=boundary_indices,
            progress_callback=progress_callback,
        )
        return self._package(series, runs, scores, boundary_indices=boundary_indices)


@register_community("temporal_greedy")
class TemporalGreedyCommunity(CommunityDetector):
    """Legacy baseline: per-window Leiden followed by greedy overlap-based relabeling."""

    def __init__(self, cfg: CommunityConfig) -> None:
        super().__init__(cfg)
        base_cfg = CommunityConfig(
            name="leiden",
            resolution=cfg.resolution,
            symmetrize=cfg.symmetrize,
            use_absolute=cfg.use_absolute,
            weight_threshold=cfg.weight_threshold,
            seed=cfg.seed,
        )
        self._base = LeidenCommunity(base_cfg)

    def detect_one(self, matrix: np.ndarray) -> np.ndarray:
        return self._base.detect_one(matrix)

    def run(
        self,
        series: ConnectivitySeries,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> CommunitySeries:
        boundary_indices = set(boundary_indices_for_series(series).tolist())
        raw = []
        for index, matrix in enumerate(series.matrices):
            raw.append(self.detect_one(self._prepare(matrix)))
            if progress_callback is not None:
                progress_callback(index + 1, series.n_windows)
        aligned = [raw[0]]
        for idx in range(1, len(raw)):
            if idx in boundary_indices:
                aligned.append(raw[idx])
            else:
                aligned.append(_match_labels(aligned[-1], raw[idx]))
        labels = np.stack(aligned).astype(np.int64)
        counts = np.array([len(np.unique(row)) for row in labels])
        return build_community_series(
            series=series,
            labels=labels,
            method=self.cfg.name,
            n_communities_per_window=counts,
        )


__all__ = [
    "TemporalCommunity",
    "TemporalGreedyCommunity",
    "_match_labels",
    "_signed_role_features",
]
