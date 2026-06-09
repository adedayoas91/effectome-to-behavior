"""Decode behavior from connectivity features (tests H5: causal > correlation).

Builds per-window feature vectors from the dynamic effectome (vectorized matrices, connectivity
-state one-hot, or community summaries) and cross-validates a decoder predicting the per-window
behavior. Comparing decoders trained on causal vs correlation connectivity quantifies whether
causal structure carries more behavioral information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from effectome.data_module.schema import CommunitySeries, ConnectivitySeries

from .stats import discretize


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


@dataclass
class DecodeResult:
    """Cross-validated decoding score for one feature set."""

    feature_set: str
    behavior_key: str
    task: str
    mean_score: float
    std_score: float
    n_folds: int


def decode_behavior(
    features: np.ndarray,
    behavior: np.ndarray,
    feature_set: str,
    behavior_key: str,
    n_folds: int = 5,
    seed: int = 42,
) -> DecodeResult:
    """Cross-validate a decoder of `behavior` from `features`.

    Classification (accuracy) if behavior is integer-valued, else regression (R^2).
    """
    is_discrete = np.issubdtype(np.asarray(behavior).dtype, np.integer)
    if is_discrete:
        model = LogisticRegression(max_iter=1000)
        scoring = "accuracy"
        y = behavior
        task = "classification"
    else:
        model = Ridge(alpha=1.0)
        scoring = "r2"
        y = behavior.astype(float)
        task = "regression"

    n_splits = min(n_folds, len(np.unique(discretize(y))) if is_discrete else n_folds, len(y) // 2)
    n_splits = max(2, n_splits)
    scores = cross_val_score(model, features, y, cv=n_splits, scoring=scoring)
    return DecodeResult(
        feature_set=feature_set,
        behavior_key=behavior_key,
        task=task,
        mean_score=float(scores.mean()),
        std_score=float(scores.std()),
        n_folds=int(n_splits),
    )
