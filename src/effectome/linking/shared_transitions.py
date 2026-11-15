"""Training-owned common manifold targets for comparing effectome estimators.

Regions are explicit exploratory geometric bins, not evidence for neural regimes.
The same reference and event rows must be reused by every method in a comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from effectome.data_module.schema import TemporalAnchor, WindowedSegments

from .decoding import valid_positive_lag_origins


@dataclass(frozen=True)
class SharedTransitionConfig:
    n_dims: int = 3
    n_manifold_states: int = 3
    positive_lag: int = 1
    seed: int = 42
    neural_history_samples: int = 2

    def __post_init__(self) -> None:
        for name in ("n_dims", "n_manifold_states", "positive_lag", "neural_history_samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TransitionEventTable:
    """Rows refer to globally indexed window origins and future target windows."""

    origins: np.ndarray
    targets: np.ndarray
    source_states: np.ndarray
    target_states: np.ndarray
    horizon_seconds: np.ndarray
    train_rows: np.ndarray
    evaluation_rows: np.ndarray
    groups: np.ndarray


@dataclass(frozen=True)
class SharedTransitionReference:
    embeddings: np.ndarray
    manifold_states: np.ndarray
    baseline_features: np.ndarray
    fit_indices: np.ndarray
    evaluation_indices: np.ndarray
    events: TransitionEventTable
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    pca_components: np.ndarray
    state_centers: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def _indices(values: np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if (
        arr.ndim != 1
        or arr.size == 0
        or not np.issubdtype(arr.dtype, np.integer)
        or np.any(arr < 0)
        or np.any(arr >= n)
        or np.any(np.diff(arr) <= 0)
    ):
        raise ValueError(f"{name} must be nonempty strictly increasing integer indices")
    return arr.astype(int)


def _source(anchor: TemporalAnchor) -> tuple[object, ...]:
    return (anchor.dataset_id, anchor.animal_id, anchor.session_id, anchor.recording_id, anchor.segment_id)


def _overlap(left: TemporalAnchor, right: TemporalAnchor) -> bool:
    return _source(left) == _source(right) and max(left.context_start, right.context_start) < min(
        left.context_stop, right.context_stop
    )


def _partitions(
    windows: WindowedSegments, fit_indices: np.ndarray, evaluation_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(windows.anchors) != windows.n_windows:
        raise ValueError("shared transitions require typed raw temporal anchors")
    fit = _indices(fit_indices, windows.n_windows, "fit_indices")
    evaluation = _indices(evaluation_indices, windows.n_windows, "evaluation_indices")
    if fit[-1] >= evaluation[0]:
        raise ValueError("shared transitions require chronological disjoint outer ownership")
    dependency = windows.metadata.get("preprocess_dependency", {})
    if (
        not isinstance(dependency, dict)
        or dependency.get("kind") not in {"identity", "finite"}
        or dependency.get("past_support", 0) != 0
        or dependency.get("future_support", 0) != 0
    ):
        raise ValueError("shared transitions require identity-preprocessed raw windows")
    standardization = windows.metadata.get("neural_standardization", {})
    if not isinstance(standardization, dict) or standardization.get("enabled", True):
        raise ValueError("shared transitions reject per-window standardization")
    for a in fit:
        for b in evaluation:
            if _overlap(windows.anchors[int(a)], windows.anchors[int(b)]):
                raise ValueError("fit and evaluation histories overlap in raw sample space")
    if any(not windows.anchors[int(i)].valid for i in fit):
        raise ValueError("fit_indices must exclude invalid anchors")
    if any(anchor.target_stop != anchor.context_stop for anchor in windows.anchors):
        raise ValueError("shared transitions require trailing targets ending at history availability")
    return fit, evaluation


def _continuity_groups(anchors: list[TemporalAnchor]) -> np.ndarray:
    groups = np.zeros(len(anchors), dtype=int)
    for index in range(1, len(anchors)):
        previous, current = anchors[index - 1], anchors[index]
        boundary = (
            _source(current) != _source(previous)
            or not current.valid
            or not previous.valid
            or current.gap_before
            or previous.gap_after
            or current.bad_frame_before
            or previous.bad_frame_after
            or current.anchor_time_seconds <= previous.anchor_time_seconds
        )
        groups[index] = groups[index - 1] + int(boundary)
    return groups


def _baseline(
    embeddings: np.ndarray,
    states: np.ndarray,
    scaled: np.ndarray,
    windows: WindowedSegments,
    n_states: int,
    groups: np.ndarray,
    neural_history_samples: int,
) -> tuple[np.ndarray, list[str]]:
    velocity = np.zeros_like(embeddings)
    dwell = np.zeros(len(embeddings))
    for index in range(1, len(embeddings)):
        if groups[index] != groups[index - 1]:
            continue
        delta = windows.anchors[index].anchor_time_seconds - windows.anchors[index - 1].anchor_time_seconds
        velocity[index] = (embeddings[index] - embeddings[index - 1]) / delta
        if states[index] == states[index - 1]:
            dwell[index] = dwell[index - 1] + delta
    target_means = np.stack(
        [
            segment[
                anchor.target_start - anchor.context_start : anchor.target_stop - anchor.context_start
            ].mean(0)
            for segment, anchor in zip(scaled, windows.anchors, strict=True)
        ]
    )
    recent_samples = np.concatenate([scaled[:, -lag - 1, :] for lag in range(neural_history_samples)], axis=1)
    baseline = np.column_stack(
        [
            np.eye(n_states)[states],
            embeddings,
            velocity,
            np.linalg.norm(velocity, axis=1),
            dwell,
            target_means,
            scaled.mean(axis=1),
            scaled.std(axis=1),
            recent_samples,
        ]
    )
    names = (
        [f"current_state_{i}" for i in range(n_states)]
        + [f"position_{i}" for i in range(embeddings.shape[1])]
        + [f"past_velocity_{i}" for i in range(embeddings.shape[1])]
        + ["past_speed", "state_dwell_seconds"]
        + [f"target_activity_mean_{i}" for i in range(windows.n_neurons)]
        + [f"history_activity_mean_{i}" for i in range(windows.n_neurons)]
        + [f"history_activity_std_{i}" for i in range(windows.n_neurons)]
        + [
            f"neural_history_lag_{lag}_{i}"
            for lag in range(neural_history_samples)
            for i in range(windows.n_neurons)
        ]
    )
    return baseline, names


def fit_shared_transition_reference(
    windows: WindowedSegments,
    *,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    config: SharedTransitionConfig,
) -> SharedTransitionReference:
    """Fit one frozen reference and issue strict positive-lag events.

    PCA fits complete rows from training histories, weighted by their appearances
    in those histories. Targets are the mean PCA coordinates of the declared
    target interval. This is an explicitly named reference representation, not a
    BunDLe target encoder. No observed behavior is used to define the regions.
    """
    fit, evaluation = _partitions(windows, fit_indices, evaluation_indices)
    if config.neural_history_samples > windows.segments.shape[1]:
        raise ValueError("declared neural history samples exceed the stored history length")
    if not np.all(np.isfinite(windows.segments)):
        raise ValueError(
            "shared transitions require finite neural histories; declare a causal missing-data policy"
        )
    training = np.asarray(windows.segments[fit], dtype=float).reshape(-1, windows.n_neurons)
    if training.size == 0 or config.n_dims > min(training.shape):
        raise ValueError("insufficient complete training samples for the declared PCA dimensions")
    scaler = StandardScaler().fit(training)
    scaled = (np.asarray(windows.segments, dtype=float) - scaler.mean_) / scaler.scale_
    pca = PCA(n_components=config.n_dims, svd_solver="full").fit(scaler.transform(training))
    targets = []
    for segment, anchor in zip(scaled, windows.anchors, strict=True):
        start, stop = anchor.target_start - anchor.context_start, anchor.target_stop - anchor.context_start
        if start < 0 or stop > len(segment) or start >= stop:
            raise ValueError("anchor target interval must align with stored neural history")
        targets.append(pca.transform(segment[start:stop]).mean(axis=0))
    embeddings = np.asarray(targets)
    if config.n_manifold_states > np.unique(embeddings[fit], axis=0).shape[0]:
        raise ValueError("insufficient distinct training targets for the declared manifold regions")
    regions = KMeans(n_clusters=config.n_manifold_states, n_init=10, random_state=config.seed).fit(
        embeddings[fit]
    )
    states = regions.predict(embeddings)
    groups = _continuity_groups(windows.anchors)
    ownership = np.full(windows.n_windows, -1, dtype=int)
    ownership[fit], ownership[evaluation] = 0, 1
    ownership_runs = np.r_[0, np.cumsum(ownership[1:] != ownership[:-1])]
    groups = groups * (windows.n_windows + 1) + ownership_runs
    baseline, names = _baseline(
        embeddings,
        states,
        scaled,
        windows,
        config.n_manifold_states,
        groups,
        config.neural_history_samples,
    )
    origins = valid_positive_lag_origins(windows.n_windows, config.positive_lag, windows.anchors, strict=True)
    future = origins + config.positive_lag
    continuous = groups[origins] == groups[future]
    train_mask = np.isin(origins, fit) & np.isin(future, fit) & continuous
    evaluation_mask = np.isin(origins, evaluation) & np.isin(future, evaluation) & continuous
    # Explicitly purge complete outcome histories as well as origin histories.
    # With both endpoints assigned to disjoint partitions this normally changes
    # nothing, but keeps the event contract safe if the split policy evolves.
    for row in np.flatnonzero(train_mask):
        for candidate in evaluation:
            if any(
                _overlap(windows.anchors[int(i)], windows.anchors[int(candidate)])
                for i in (origins[row], future[row])
            ):
                train_mask[row] = False
                break
    keep = train_mask | evaluation_mask
    train_mask, evaluation_mask = train_mask[keep], evaluation_mask[keep]
    origins, future = origins[keep], future[keep]
    if np.sum(train_mask) < 2 or not np.any(evaluation_mask):
        raise ValueError("insufficient train/evaluation events after strict future-support purging")
    horizon = np.asarray(
        [
            windows.anchors[int(b)].anchor_time_seconds - windows.anchors[int(a)].anchor_time_seconds
            for a, b in zip(origins, future, strict=True)
        ]
    )
    if np.any(horizon <= 0) or not np.all(np.isfinite(horizon)):
        raise ValueError("transition horizons must be positive finite physical times")
    if not np.allclose(horizon, horizon[0], rtol=1e-7, atol=1e-9):
        raise ValueError("transition events require one fixed physical horizon across the shared comparison")
    events = TransitionEventTable(
        origins,
        future,
        states[origins],
        states[future],
        horizon,
        np.flatnonzero(train_mask),
        np.flatnonzero(evaluation_mask),
        groups[origins],
    )
    return SharedTransitionReference(
        embeddings,
        states,
        baseline,
        fit,
        evaluation,
        events,
        scaler.mean_.copy(),
        scaler.scale_.copy(),
        pca.components_.copy(),
        regions.cluster_centers_.copy(),
        {
            "representation": "training_pca_target_mean",
            "region_definition": "training_kmeans_geometric",
            "region_count_is_explicit": True,
            "regime_claim_eligible": False,
            "claim_eligibility": "development_only",
            "fit_uses_evaluation": False,
            "raw_support_purged": True,
            "physical_horizon_seconds": float(horizon[0]),
            "scaling_weighting": "training_history_window_weighted",
            "baseline_feature_names": names,
            "baseline_behavior_covariates": "not_used",
            "baseline_neural_history_samples": config.neural_history_samples,
            "pca_mean": pca.mean_.copy(),
            "seed": config.seed,
        },
    )


def _contingency(left: np.ndarray, right: np.ndarray, n_left: int, n_right: int) -> np.ndarray:
    counts = np.zeros((n_left, n_right), dtype=int)
    np.add.at(counts, (left, right), 1)
    return counts


def _conditional_entropy(counts: np.ndarray) -> float | None:
    total = int(counts.sum())
    if total == 0:
        return None
    row_totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals > 0
    )
    present = counts > 0
    return float(-np.sum(counts[present] * np.log(probabilities[present])) / total)


def state_correspondence(
    effectome_labels: np.ndarray,
    manifold_labels: np.ndarray,
    *,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    future_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    """Describe many-to-many state occupancy without claiming biological equivalence.

    ``future_targets`` optionally supplies an outcome code aligned with every
    window; -1 marks an unavailable future. These are outcome labels, not window
    indices. Observed conditional entropies are descriptive, not significance tests.
    """
    effectome, manifold = np.asarray(effectome_labels), np.asarray(manifold_labels)
    if effectome.ndim != 1 or manifold.shape != effectome.shape or effectome.size == 0:
        raise ValueError("state label sequences must be nonempty and aligned")
    if any(
        not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) for values in (effectome, manifold)
    ):
        raise ValueError("state labels must be nonnegative integers")
    fit = _indices(fit_indices, len(effectome), "fit_indices")
    evaluation = _indices(evaluation_indices, len(effectome), "evaluation_indices")
    if np.intersect1d(fit, evaluation).size:
        raise ValueError("fit and evaluation ownership must be disjoint")
    n_effectome, n_manifold = int(effectome.max()) + 1, int(manifold.max()) + 1
    n_train_effectome, n_train_manifold = int(effectome[fit].max()) + 1, int(manifold[fit].max()) + 1
    train = _contingency(effectome[fit], manifold[fit], n_train_effectome, n_train_manifold)
    heldout = _contingency(effectome[evaluation], manifold[evaluation], n_effectome, n_manifold)
    mapping = np.full(n_train_effectome, -1, dtype=int)
    observed = train.sum(axis=1) > 0
    mapping[observed] = train[observed].argmax(axis=1)
    seen = effectome[evaluation] < n_train_effectome
    covered = np.zeros(len(evaluation), dtype=bool)
    covered[seen] = mapping[effectome[evaluation][seen]] >= 0
    result: dict[str, Any] = {
        "training_contingency": train,
        "heldout_contingency": heldout,
        "training_effectome_to_manifold": mapping,
        "heldout_entropy_effectome_given_manifold_nats": _conditional_entropy(heldout.T),
        "heldout_entropy_manifold_given_effectome_nats": _conditional_entropy(heldout),
        "heldout_effectome_count_per_manifold": (heldout > 0).sum(axis=0),
        "heldout_manifold_count_per_effectome": (heldout > 0).sum(axis=1),
        "training_count": int(len(fit)),
        "evaluation_count": int(len(evaluation)),
        "heldout_mapping_coverage": float(np.mean(covered)),
        "mapping_uses_evaluation": False,
        "analysis_scope": "descriptive_many_to_many",
        "regime_claim_eligible": False,
        "biological_equivalence_established": False,
    }
    if future_targets is not None:
        future = np.asarray(future_targets)
        if (
            future.shape != effectome.shape
            or not np.issubdtype(future.dtype, np.integer)
            or np.any(future < -1)
        ):
            raise ValueError("future_targets must be aligned integer outcome labels, with -1 for unavailable")
        available = evaluation[future[evaluation] >= 0]
        n_future = int(future[available].max()) + 1 if available.size else 0
        outcome_counts = _contingency(effectome[available], future[available], n_effectome, n_future)
        result.update(
            {
                "heldout_effectome_future_contingency": outcome_counts,
                "heldout_future_count_per_effectome": (outcome_counts > 0).sum(axis=1),
                "heldout_entropy_future_given_effectome_nats": _conditional_entropy(outcome_counts),
                "heldout_future_evaluation_count": int(available.size),
            }
        )
    return result
