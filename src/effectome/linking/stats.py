"""Statistical primitives for linking connectivity dynamics to behavior.

Provides surrogate-null generators and association measures used to test whether connectivity
states/communities carry behavioral information beyond chance while preserving temporal structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score


def _continuous_slices(n_samples: int, groups: np.ndarray | None) -> list[slice]:
    if groups is None:
        return [slice(0, n_samples)]
    group_arr = np.asarray(groups, dtype=object)
    if group_arr.shape[0] != n_samples:
        raise ValueError("groups must align with the input sequence")
    boundaries = [0]
    boundaries.extend((np.flatnonzero(group_arr[1:] != group_arr[:-1]) + 1).tolist())
    boundaries.append(n_samples)
    return [slice(start, stop) for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)]


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


def circular_shift_null(
    states: np.ndarray,
    behavior: np.ndarray,
    n_null: int,
    seed: int,
    n_bins: int = 5,
    min_shift: int = 1,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """MI null by circularly shifting behavior, preserving autocorrelation structure."""
    if len(states) != len(behavior):
        raise ValueError("states and behavior must have the same length")
    if len(states) < 3:
        raise ValueError("need at least 3 samples for a circular-shift null")

    rng = np.random.default_rng(seed)
    b = discretize(behavior, n_bins)
    slices = _continuous_slices(len(b), groups)
    out = []
    for _ in range(n_null):
        surrogate = np.array(b, copy=True)
        for segment in slices:
            length = segment.stop - segment.start
            valid_shifts = np.arange(min_shift, length)
            if valid_shifts.size:
                surrogate[segment] = np.roll(b[segment], int(rng.choice(valid_shifts)))
        out.append(mutual_info_score(states, surrogate))
    return np.asarray(out)


def block_shuffle_null(
    states: np.ndarray,
    behavior: np.ndarray,
    n_null: int,
    seed: int,
    block_length: int,
    n_bins: int = 5,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """MI null by permuting contiguous behavior blocks rather than individual samples."""
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    rng = np.random.default_rng(seed)
    b = discretize(behavior, n_bins)
    slices = _continuous_slices(len(b), groups)
    out = []
    for _ in range(n_null):
        shuffled = np.array(b, copy=True)
        for segment in slices:
            segment_values = b[segment]
            starts = list(range(0, len(segment_values), block_length))
            blocks = [segment_values[start : start + block_length] for start in starts]
            order = rng.permutation(len(blocks))
            shuffled[segment] = np.concatenate([blocks[i] for i in order])[: len(segment_values)]
        out.append(mutual_info_score(states, shuffled))
    return np.asarray(out)


@dataclass
class AssociationResult:
    """Association between a discrete sequence and behavior with a surrogate p-value."""

    statistic: float
    null_mean: float
    null_std: float
    p_value: float
    z_score: float
    null_kind: str = "circular_shift"


@dataclass
class ConfidenceInterval:
    low: float
    high: float


def bootstrap_mean_ci(
    x: np.ndarray, seed: int = 42, n_boot: int = 500, alpha: float = 0.05
) -> ConfidenceInterval:
    """Percentile bootstrap CI for the sample mean."""
    arr = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return ConfidenceInterval(
        low=float(np.quantile(means, alpha / 2)),
        high=float(np.quantile(means, 1.0 - alpha / 2)),
    )


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return BH-adjusted q-values in original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / (i + 1)
        prev = min(prev, val)
        q[i] = prev
    out = np.empty(m, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def association_with_null(
    states: np.ndarray,
    behavior: np.ndarray,
    n_null: int = 1000,
    seed: int = 42,
    n_bins: int = 5,
    null_kind: str = "circular_shift",
    block_length: int | None = None,
    groups: np.ndarray | None = None,
) -> AssociationResult:
    """Mutual-information association of `states` with `behavior` vs a temporally aware null."""
    stat = state_behavior_mi(states, behavior, n_bins)
    if null_kind == "block_shuffle":
        block_length = block_length or max(2, len(states) // 10)
        null = block_shuffle_null(states, behavior, n_null, seed, block_length, n_bins, groups=groups)
    else:
        null = circular_shift_null(states, behavior, n_null, seed, n_bins, groups=groups)
        null_kind = "circular_shift"
    p = float((int(np.sum(null >= stat)) + 1) / (len(null) + 1))
    std = float(null.std()) or 1e-12
    return AssociationResult(
        statistic=stat,
        null_mean=float(null.mean()),
        null_std=float(null.std()),
        p_value=p,
        z_score=float((stat - null.mean()) / std),
        null_kind=null_kind,
    )


def partition_stability(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Adjusted mutual information between two community partitions (reproducibility check)."""
    return float(adjusted_mutual_info_score(labels_a, labels_b))
