"""Decode behavior from connectivity features using purged blocked CV."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

from effectome.data_module.schema import CommunitySeries, ConnectivitySeries, TemporalAnchor


def connectivity_features(series: ConnectivitySeries) -> np.ndarray:
    """Vectorize each window's connectivity matrix into a feature row -> (K, N*N)."""
    return series.matrices.reshape(series.n_windows, -1)


def state_features(state_labels: np.ndarray, n_states: int) -> np.ndarray:
    """One-hot encode connectivity-state labels -> (K, n_states)."""
    oh = np.zeros((len(state_labels), n_states))
    oh[np.arange(len(state_labels)), state_labels] = 1.0
    return oh


def community_features(series: CommunitySeries) -> np.ndarray:
    """Summarize each window's partition: [n_communities, modularity-free size entropy]."""
    feats = []
    for row in series.labels:
        _, counts = np.unique(row, return_counts=True)
        p = counts / counts.sum()
        entropy = -np.sum(p * np.log(p + 1e-12))
        feats.append([len(counts), entropy])
    return np.asarray(feats)


def community_reconfiguration_features(series: CommunitySeries) -> np.ndarray:
    """Represent when and which neurons changed temporal community allegiance.

    The first column is the fraction of neurons switching at an anchor; remaining
    columns retain neuron-resolved switching indicators. Temporal community methods
    already zero transitions at declared recording/gap boundaries.
    """

    k, n = series.labels.shape
    switching = series.switching
    if switching is None:
        switching_arr = np.zeros((k, n), dtype=float)
        if k > 1:
            switching_arr[1:] = series.labels[1:] != series.labels[:-1]
    else:
        switch = np.asarray(switching, dtype=float)
        if switch.shape != (max(k - 1, 0), n):
            raise ValueError("community switching must align with adjacent windows and neurons")
        switching_arr = np.zeros((k, n), dtype=float)
        if k > 1:
            switching_arr[1:] = switch
    switching_fraction = switching_arr.mean(axis=1, keepdims=True)
    return np.concatenate([switching_fraction, switching_arr], axis=1)


def _anchor_source_key(anchor: TemporalAnchor) -> tuple[str, str | None, str | None, str, str | None]:
    return (
        anchor.dataset_id,
        anchor.animal_id,
        anchor.session_id,
        anchor.recording_id,
        anchor.segment_id,
    )


def _anchor_group_key(
    anchor: TemporalAnchor, group_by: str
) -> tuple[str, str | None, str | None, str, str | None]:
    if group_by == "segment":
        return _anchor_source_key(anchor)
    if group_by == "animal":
        return (
            anchor.dataset_id,
            anchor.animal_id or anchor.recording_id,
            None,
            "",
            None,
        )
    if group_by == "recording":
        return (
            anchor.dataset_id,
            anchor.animal_id,
            anchor.session_id,
            anchor.recording_id,
            None,
        )
    raise ValueError(f"unknown group_by '{group_by}'")


def anchor_group_labels(anchors: Sequence[TemporalAnchor], group_by: str = "recording") -> np.ndarray:
    """Return stable group labels for grouped CV from typed temporal anchors."""
    labels = []
    for anchor in anchors:
        labels.append(
            "|".join("" if part is None else str(part) for part in _anchor_group_key(anchor, group_by))
        )
    return np.asarray(labels, dtype=object)


def anchor_continuity_labels(
    anchors: Sequence[TemporalAnchor],
    group_by: str = "recording",
) -> np.ndarray:
    """Return group labels split at every hard-gap or sparse-invalid transition."""

    base = anchor_group_labels(anchors, group_by=group_by)
    continuity = np.empty(len(anchors), dtype=object)
    segment = 0
    for index, anchor in enumerate(anchors):
        if index > 0:
            previous = anchors[index - 1]
            if (
                base[index] != base[index - 1]
                or anchor.gap_before
                or previous.gap_after
                or anchor.bad_frame_before
                or previous.bad_frame_after
            ):
                segment += 1
        continuity[index] = (base[index], segment)
    return continuity


def valid_positive_lag_origins(
    n_samples: int,
    lag: int,
    anchors: Sequence[TemporalAnchor] | None = None,
) -> np.ndarray:
    """Return origins whose strictly future target remains in one continuous source segment."""
    if lag <= 0:
        raise ValueError("lag must be strictly positive")
    if anchors is None:
        return np.arange(max(0, n_samples - lag), dtype=int)
    if len(anchors) != n_samples:
        raise ValueError("anchors must align 1:1 with samples")

    origins: list[int] = []
    for origin in range(n_samples - lag):
        path = anchors[origin : origin + lag + 1]
        same_source = all(_anchor_source_key(anchor) == _anchor_source_key(path[0]) for anchor in path)
        crosses_gap = (
            any(anchor.gap_after or anchor.bad_frame_after for anchor in path[:-1])
            or any(anchor.gap_before or anchor.bad_frame_before for anchor in path[1:])
        )
        if same_source and not crosses_gap:
            origins.append(origin)
    return np.asarray(origins, dtype=int)


@dataclass(frozen=True)
class DependencySupport:
    """Raw-sample support outside an anchor's recorded context.

    ``lag_extension`` is zero when all lagged predictors are constructed inside the
    stored context window.  Non-causal preprocessing (for example, centered
    smoothing) must declare its past and future kernel support explicitly.
    """

    lag_extension: int = 0
    preprocessing_past: int = 0
    preprocessing_future: int = 0

    def __post_init__(self) -> None:
        if min(self.lag_extension, self.preprocessing_past, self.preprocessing_future) < 0:
            raise ValueError("dependency support values must be non-negative")

    @property
    def total_past(self) -> int:
        return self.lag_extension + self.preprocessing_past


def _intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def anchor_dependency_intervals(
    anchor: TemporalAnchor,
    support: DependencySupport | None = None,
    outcome_anchor: TemporalAnchor | None = None,
) -> tuple[tuple[int, int], ...]:
    """Return the exact half-open raw intervals used by one model row.

    The feature interval is the connectivity history expanded by any lag or
    preprocessing support.  A positive-lag outcome contributes only its target
    interval, not the future anchor's entire history.
    """
    dep = support or DependencySupport()
    feature = (
        max(0, int(anchor.context_start) - dep.total_past),
        int(anchor.context_stop) + dep.preprocessing_future,
    )
    if outcome_anchor is None:
        return (feature,)
    if _anchor_source_key(anchor) != _anchor_source_key(outcome_anchor):
        raise ValueError("feature and outcome anchors must belong to the same source segment")
    outcome = (int(outcome_anchor.target_start), int(outcome_anchor.target_stop))
    return (feature, outcome)


def _dependencies_overlap(
    left: TemporalAnchor,
    right: TemporalAnchor,
    support: DependencySupport,
    left_outcome: TemporalAnchor | None = None,
    right_outcome: TemporalAnchor | None = None,
) -> bool:
    if _anchor_source_key(left) != _anchor_source_key(right):
        return False
    left_intervals = anchor_dependency_intervals(left, support, left_outcome)
    right_intervals = anchor_dependency_intervals(right, support, right_outcome)
    return any(_intervals_overlap(a, b) for a in left_intervals for b in right_intervals)


def _apply_anchor_overlap_purge(
    train: np.ndarray,
    test: np.ndarray,
    anchors: Sequence[TemporalAnchor],
    support: DependencySupport,
    outcome_anchors: Sequence[TemporalAnchor] | None = None,
) -> np.ndarray:
    test_by_source: dict[
        tuple[str, str | None, str | None, str, str | None], list[tuple[int, TemporalAnchor]]
    ] = {}
    for idx in test:
        anchor = anchors[int(idx)]
        test_by_source.setdefault(_anchor_source_key(anchor), []).append((int(idx), anchor))

    kept: list[int] = []
    for idx in train:
        anchor = anchors[int(idx)]
        overlaps = any(
            _dependencies_overlap(
                anchor,
                test_anchor,
                support,
                None if outcome_anchors is None else outcome_anchors[int(idx)],
                None if outcome_anchors is None else outcome_anchors[test_idx],
            )
            for test_idx, test_anchor in test_by_source.get(_anchor_source_key(anchor), [])
        )
        if not overlaps:
            kept.append(int(idx))
    return np.asarray(kept, dtype=int)


def _apply_group_embargo(
    train: np.ndarray,
    test: np.ndarray,
    embargo: int,
    groups: np.ndarray,
) -> np.ndarray:
    if embargo <= 0:
        return train
    test_by_group: dict[object, np.ndarray] = {}
    for group in np.unique(groups[test]):
        test_by_group[group] = test[groups[test] == group]
    kept: list[int] = []
    for idx in train:
        same_group_test = test_by_group.get(groups[int(idx)])
        if same_group_test is None or np.min(np.abs(same_group_test - idx)) > embargo:
            kept.append(int(idx))
    return np.asarray(kept, dtype=int)


def purged_blocked_splits(
    n_samples: int,
    n_splits: int,
    embargo: int = 0,
    anchors: Sequence[TemporalAnchor] | None = None,
    groups: Sequence[object] | None = None,
    dependency_support: DependencySupport | None = None,
    outcome_anchors: Sequence[TemporalAnchor] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return blocked folds purged by raw dependency overlap and index embargo."""
    if n_samples < 4:
        raise ValueError("need at least 4 samples for blocked CV")
    indices = np.arange(n_samples)
    if anchors is not None and len(anchors) != n_samples:
        raise ValueError("anchors must align 1:1 with samples")
    if outcome_anchors is not None:
        if anchors is None:
            raise ValueError("outcome_anchors require feature anchors")
        if len(outcome_anchors) != n_samples:
            raise ValueError("outcome_anchors must align 1:1 with samples")

    groups_arr = np.asarray(groups if groups is not None else np.zeros(n_samples, dtype=int), dtype=object)
    if groups_arr.shape[0] != n_samples:
        raise ValueError("groups must align 1:1 with samples")

    unique_groups = list(dict.fromkeys(groups_arr.tolist()))
    if len(unique_groups) > 1:
        effective_splits = max(2, min(n_splits, len(unique_groups)))
        group_blocks = np.array_split(np.asarray(unique_groups, dtype=object), effective_splits)
        test_blocks = [indices[np.isin(groups_arr, block)] for block in group_blocks if len(block) > 0]
    else:
        effective_splits = max(2, min(n_splits, n_samples // 2))
        test_blocks = [block for block in np.array_split(indices, effective_splits) if len(block) > 0]

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for test in test_blocks:
        mask = np.ones(n_samples, dtype=bool)
        mask[test] = False
        train = indices[mask]
        if anchors is not None:
            train = _apply_anchor_overlap_purge(
                train,
                test,
                anchors,
                dependency_support or DependencySupport(),
                outcome_anchors=outcome_anchors,
            )
        train = _apply_group_embargo(train, test, embargo, groups_arr)
        if len(train) >= 2:
            splits.append((train, test))
    if len(splits) < 2:
        raise ValueError("unable to construct at least two purged blocked splits")
    return splits


@dataclass
class DecodeResult:
    """Cross-validated decoding score for one feature set."""

    feature_set: str
    behavior_key: str
    task: str
    mean_score: float
    std_score: float
    n_folds: int
    embargo: int


@dataclass
class IncrementalDecodeResult:
    baseline_score: float
    full_score: float
    gain: float
    n_folds: int
    embargo: int


def _score_predictions(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if task == "classification":
        return float(accuracy_score(y_true, y_pred))
    return float(r2_score(y_true, y_pred))


def _model_for(behavior: np.ndarray, seed: int):
    is_discrete = np.issubdtype(np.asarray(behavior).dtype, np.integer)
    if is_discrete:
        return LogisticRegression(max_iter=1000, random_state=seed), behavior, "classification"
    return Ridge(alpha=1.0), behavior.astype(float), "regression"


def _fit_predict(
    model: LogisticRegression | Ridge,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    task: str,
) -> np.ndarray:
    """Fit with train-only scaling, or return a constant for a single-class fold."""
    if task == "classification" and len(np.unique(y_train)) < 2:
        return np.full(len(x_test), y_train[0], dtype=y_train.dtype)
    mean = np.asarray(x_train, dtype=float).mean(axis=0, keepdims=True)
    scale = np.asarray(x_train, dtype=float).std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    train_scaled = (np.asarray(x_train, dtype=float) - mean) / scale
    test_scaled = (np.asarray(x_test, dtype=float) - mean) / scale
    model.fit(train_scaled, y_train)
    return np.asarray(model.predict(test_scaled))


def decode_behavior(
    features: np.ndarray,
    behavior: np.ndarray,
    feature_set: str,
    behavior_key: str,
    n_folds: int = 5,
    seed: int = 42,
    embargo: int = 0,
    anchors: Sequence[TemporalAnchor] | None = None,
    groups: Sequence[object] | None = None,
    dependency_support: DependencySupport | None = None,
    outcome_anchors: Sequence[TemporalAnchor] | None = None,
) -> DecodeResult:
    """Cross-validate a decoder of `behavior` from `features` with purged blocked splits."""
    model, y, task = _model_for(behavior, seed)
    scores = []
    for train, test in purged_blocked_splits(
        len(y),
        n_folds,
        embargo=embargo,
        anchors=anchors,
        groups=groups,
        dependency_support=dependency_support,
        outcome_anchors=outcome_anchors,
    ):
        pred = _fit_predict(model, features[train], y[train], features[test], task)
        scores.append(_score_predictions(task, y[test], pred))
    arr = np.asarray(scores, dtype=float)
    return DecodeResult(
        feature_set=feature_set,
        behavior_key=behavior_key,
        task=task,
        mean_score=float(arr.mean()),
        std_score=float(arr.std()),
        n_folds=int(len(arr)),
        embargo=int(embargo),
    )


def incremental_decode_behavior(
    baseline_features: np.ndarray,
    extra_features: np.ndarray,
    behavior: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
    embargo: int = 0,
    anchors: Sequence[TemporalAnchor] | None = None,
    groups: Sequence[object] | None = None,
    dependency_support: DependencySupport | None = None,
    outcome_anchors: Sequence[TemporalAnchor] | None = None,
) -> IncrementalDecodeResult:
    """Quantify gain from `extra_features` over autoregressive baseline features."""
    model, y, task = _model_for(behavior, seed)
    baseline_scores: list[float] = []
    full_scores: list[float] = []
    full = np.concatenate([baseline_features, extra_features], axis=1)
    for train, test in purged_blocked_splits(
        len(y),
        n_folds,
        embargo=embargo,
        anchors=anchors,
        groups=groups,
        dependency_support=dependency_support,
        outcome_anchors=outcome_anchors,
    ):
        baseline_pred = _fit_predict(
            model,
            baseline_features[train],
            y[train],
            baseline_features[test],
            task,
        )
        baseline_scores.append(_score_predictions(task, y[test], baseline_pred))

        full_pred = _fit_predict(model, full[train], y[train], full[test], task)
        full_scores.append(_score_predictions(task, y[test], full_pred))

    base = float(np.mean(baseline_scores))
    full_score = float(np.mean(full_scores))
    return IncrementalDecodeResult(
        baseline_score=base,
        full_score=full_score,
        gain=full_score - base,
        n_folds=len(full_scores),
        embargo=int(embargo),
    )
