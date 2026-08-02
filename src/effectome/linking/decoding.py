"""Decode behavior from connectivity features using purged blocked CV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score

from effectome.data_module.schema import CommunitySeries, ConnectivitySeries


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


def purged_blocked_splits(
    n_samples: int, n_splits: int, embargo: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return contiguous test blocks with neighboring train samples purged by `embargo`."""
    if n_samples < 4:
        raise ValueError("need at least 4 samples for blocked CV")
    effective_splits = max(2, min(n_splits, n_samples // 2))
    indices = np.arange(n_samples)
    blocks = np.array_split(indices, effective_splits)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        if len(block) == 0:
            continue
        lo = max(0, int(block[0]) - embargo)
        hi = min(n_samples, int(block[-1]) + embargo + 1)
        mask = np.ones(n_samples, dtype=bool)
        mask[lo:hi] = False
        train = indices[mask]
        test = block
        if len(train) >= max(2, len(np.unique(test))):
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
    """Fit a model unless a classification fold is single-class, then predict the constant class."""
    if task == "classification" and len(np.unique(y_train)) < 2:
        return np.full(len(x_test), y_train[0], dtype=y_train.dtype)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test))


def decode_behavior(
    features: np.ndarray,
    behavior: np.ndarray,
    feature_set: str,
    behavior_key: str,
    n_folds: int = 5,
    seed: int = 42,
    embargo: int = 0,
) -> DecodeResult:
    """Cross-validate a decoder of `behavior` from `features` with purged blocked splits."""
    model, y, task = _model_for(behavior, seed)
    scores = []
    for train, test in purged_blocked_splits(len(y), n_folds, embargo=embargo):
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
) -> IncrementalDecodeResult:
    """Quantify gain from `extra_features` over autoregressive baseline features."""
    model, y, task = _model_for(behavior, seed)
    baseline_scores: list[float] = []
    full_scores: list[float] = []
    full = np.concatenate([baseline_features, extra_features], axis=1)
    for train, test in purged_blocked_splits(len(y), n_folds, embargo=embargo):
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
