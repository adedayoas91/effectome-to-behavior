"""Statistical primitives for linking connectivity dynamics to behavior.

Provides surrogate-null generators and association measures used to test whether connectivity
states/communities carry behavioral information beyond chance (project hypotheses H4, H5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score


def discretize(x: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Quantile-bin a continuous array into integer labels."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        return x
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.digitize(x, edges)


def state_behavior_mi(states: np.ndarray, behavior: np.ndarray, n_bins: int = 5) -> float:
    """Mutual information between a state sequence and a (binned) behavior sequence."""
    b = discretize(behavior, n_bins)
    return float(mutual_info_score(states, b))


def time_shuffle_null(
    states: np.ndarray, behavior: np.ndarray, n_null: int, seed: int, n_bins: int = 5
) -> np.ndarray:
    """MI null by independently permuting behavior in time (breaks temporal correspondence)."""
    rng = np.random.default_rng(seed)
    b = discretize(behavior, n_bins)
    return np.array([mutual_info_score(states, rng.permutation(b)) for _ in range(n_null)])


@dataclass
class AssociationResult:
    """Association between a discrete sequence and behavior with a surrogate p-value."""

    statistic: float
    null_mean: float
    null_std: float
    p_value: float
    z_score: float


def association_with_null(
    states: np.ndarray, behavior: np.ndarray, n_null: int = 1000, seed: int = 42, n_bins: int = 5
) -> AssociationResult:
    """Mutual-information association of `states` with `behavior` vs a time-shuffle null."""
    stat = state_behavior_mi(states, behavior, n_bins)
    null = time_shuffle_null(states, behavior, n_null, seed, n_bins)
    p = float((null >= stat).mean())
    std = float(null.std()) or 1e-12
    return AssociationResult(
        statistic=stat,
        null_mean=float(null.mean()),
        null_std=float(null.std()),
        p_value=p,
        z_score=float((stat - null.mean()) / std),
    )


def partition_stability(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted mutual information between two community partitions (reproducibility check)."""
    return float(adjusted_mutual_info_score(labels_a, labels_b))
