"""Statistical primitives for linking connectivity dynamics to behavior.

Provides surrogate-null generators and association measures used to test whether connectivity
states/communities carry behavioral information beyond chance while preserving temporal structure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    progress_callback: Callable[[int, int], None] | None = None,
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
    for index in range(n_null):
        surrogate = np.array(b, copy=True)
        for segment in slices:
            length = segment.stop - segment.start
            valid_shifts = np.arange(min_shift, length)
            if valid_shifts.size:
                surrogate[segment] = np.roll(b[segment], int(rng.choice(valid_shifts)))
        out.append(mutual_info_score(states, surrogate))
        if progress_callback is not None:
            progress_callback(index + 1, n_null)
    return np.asarray(out)


def block_shuffle_null(
    states: np.ndarray,
    behavior: np.ndarray,
    n_null: int,
    seed: int,
    block_length: int,
    n_bins: int = 5,
    groups: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """MI null by permuting contiguous behavior blocks rather than individual samples."""
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    rng = np.random.default_rng(seed)
    b = discretize(behavior, n_bins)
    slices = _continuous_slices(len(b), groups)
    out = []
    for index in range(n_null):
        shuffled = np.array(b, copy=True)
        for segment in slices:
            segment_values = b[segment]
            starts = list(range(0, len(segment_values), block_length))
            blocks = [segment_values[start : start + block_length] for start in starts]
            order = rng.permutation(len(blocks))
            shuffled[segment] = np.concatenate([blocks[i] for i in order])[: len(segment_values)]
        out.append(mutual_info_score(states, shuffled))
        if progress_callback is not None:
            progress_callback(index + 1, n_null)
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


@dataclass(frozen=True)
class RegisteredFamily:
    """A frozen secondary-testing family that retains unavailable attempts."""

    members: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.members or len(self.members) != len(set(self.members)):
            raise ValueError("registered family members must be non-empty and unique")

    def adjust(self, results: Mapping[str, tuple[float | None, str]]) -> dict[str, dict]:
        if set(results) != set(self.members):
            raise ValueError("results must contain every registered family member exactly once")
        correction_values: list[float] = []
        for name in self.members:
            p_value = results[name][0]
            correction_values.append(1.0 if p_value is None else float(p_value))
        correction_input = np.asarray(correction_values, dtype=float)
        q_values = benjamini_yekutieli(correction_input)
        adjusted: dict[str, dict] = {}
        for index, name in enumerate(self.members):
            p_value, status = results[name]
            adjusted[name] = {
                "p_value": p_value,
                "q_value": float(q_values[index]),
                "status": status,
            }
        return adjusted


@dataclass(frozen=True)
class RawBlockSample:
    """Aligned raw-process block sample with explicit artificial joins."""

    arrays: dict[str, np.ndarray]
    source_indices: np.ndarray
    segment_ids: np.ndarray
    join_before: np.ndarray

    def valid_origins(self, *, history: int, horizon: int) -> np.ndarray:
        if history < 0 or horizon < 0:
            raise ValueError("history and horizon must be non-negative")
        n_samples = self.source_indices.size
        valid = np.zeros(n_samples, dtype=bool)
        for origin in range(history, n_samples - horizon):
            support = self.segment_ids[origin - history : origin + horizon + 1]
            valid[origin] = support.size > 0 and np.all(support == support[0])
        return valid


def raw_block_resample(
    arrays: Mapping[str, Sequence],
    *,
    block_length: int,
    seed: int,
) -> RawBlockSample:
    """Resample contiguous source blocks while carrying all aligned channels together."""
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    converted = {name: np.asarray(values) for name, values in arrays.items()}
    if not converted:
        raise ValueError("at least one aligned array is required")
    lengths = {values.shape[0] for values in converted.values()}
    if len(lengths) != 1:
        raise ValueError("all arrays must share their first dimension")
    n_samples = lengths.pop()
    if n_samples == 0:
        raise ValueError("cannot resample empty arrays")

    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    while sum(block.size for block in sampled) < n_samples:
        width = min(block_length, n_samples)
        start = int(rng.integers(0, n_samples - width + 1))
        sampled.append(np.arange(start, start + width, dtype=np.int64))
    source_indices = np.concatenate(sampled)[:n_samples]

    segment_ids = np.empty(n_samples, dtype=np.int64)
    join_before = np.zeros(n_samples, dtype=bool)
    cursor = 0
    for segment, block in enumerate(sampled):
        stop = min(cursor + block.size, n_samples)
        segment_ids[cursor:stop] = segment
        if cursor:
            join_before[cursor] = True
        cursor = stop
        if cursor == n_samples:
            break
    return RawBlockSample(
        arrays={name: values[source_indices] for name, values in converted.items()},
        source_indices=source_indices,
        segment_ids=segment_ids,
        join_before=join_before,
    )


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


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """Return BY-adjusted q-values in original order, including ties and empty families."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return np.asarray([], dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    m = ranked.size
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    adjusted_ranked = ranked * m * harmonic / np.arange(1, m + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    out = np.empty_like(adjusted_ranked)
    out[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return out


def classification_log_loss(
    probabilities: np.ndarray,
    observed: np.ndarray,
    *,
    class_order: np.ndarray | list,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    obs = np.asarray(observed)
    classes = list(class_order)
    if probs.ndim != 2 or probs.shape[0] != obs.shape[0] or probs.shape[1] != len(classes):
        raise ValueError("probabilities must align with observations and class_order")
    totals = probs.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0) or np.any(totals <= 0.0):
        raise ValueError("probabilities must be finite, nonnegative and have positive row mass")
    normalized = probs / totals
    indices = np.array([classes.index(item) for item in obs], dtype=int)
    return float(-np.mean(np.log(np.clip(normalized[np.arange(len(obs)), indices], 1e-12, 1.0))))


def brier_score(
    probabilities: np.ndarray,
    observed: np.ndarray,
    *,
    class_order: np.ndarray | list,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    obs = np.asarray(observed)
    classes = list(class_order)
    if probs.ndim != 2 or probs.shape[0] != obs.shape[0] or probs.shape[1] != len(classes):
        raise ValueError("probabilities must align with observations and class_order")
    totals = probs.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(probs)) or np.any(probs < 0.0) or np.any(totals <= 0.0):
        raise ValueError("probabilities must be finite, nonnegative and have positive row mass")
    normalized = probs / totals
    target = np.zeros_like(normalized)
    for row, item in enumerate(obs):
        target[row, classes.index(item)] = 1.0
    return float(np.mean(np.sum((normalized - target) ** 2, axis=1)))


def plus_one_monte_carlo_pvalue(n_extreme: int, n_draws: int) -> float:
    if n_draws < 1 or n_extreme < 0 or n_extreme > n_draws:
        raise ValueError("expected 0 <= n_extreme <= n_draws and n_draws >= 1")
    return float((n_extreme + 1) / (n_draws + 1))


def require_monte_carlo_resolution(n_draws: int, alpha_min: float) -> None:
    if alpha_min <= 0 or alpha_min >= 1:
        raise ValueError("alpha_min must lie in (0, 1)")
    if n_draws + 1 < 1.0 / alpha_min:
        raise ValueError("Monte Carlo pool is too small for the requested alpha_min")


def association_with_null(
    states: np.ndarray,
    behavior: np.ndarray,
    n_null: int = 1000,
    seed: int = 42,
    n_bins: int = 5,
    null_kind: str = "circular_shift",
    block_length: int | None = None,
    groups: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> AssociationResult:
    """Mutual-information association of `states` with `behavior` vs a temporally aware null."""
    stat = state_behavior_mi(states, behavior, n_bins)
    if null_kind == "block_shuffle":
        block_length = block_length or max(2, len(states) // 10)
        null = block_shuffle_null(
            states,
            behavior,
            n_null,
            seed,
            block_length,
            n_bins,
            groups=groups,
            progress_callback=progress_callback,
        )
    else:
        null = circular_shift_null(
            states,
            behavior,
            n_null,
            seed,
            n_bins,
            groups=groups,
            progress_callback=progress_callback,
        )
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
