"""Boundary-safe Markov transition models for connectivity states."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import beta

from .graph_states import infer_boundary_indices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitionConfig:
    """Configuration for transition modeling."""

    laplace: float = 1.0
    n_null: int = 500
    null_mode: str = "block"
    null_block_length: int = 0
    heldout_fraction: float = 0.3
    seed: int = 42


@dataclass
class TransitionModel:
    transition_matrix: np.ndarray
    stationary: np.ndarray
    dwell_times: np.ndarray
    log_likelihood: float
    heldout_transition_log_likelihood: float
    heldout_baseline_log_likelihood: float
    heldout_delta_log_likelihood: float
    heldout_n_transitions: int
    null_delta_log_likelihoods: np.ndarray
    p_value: float
    boundary_indices: np.ndarray
    segment_lengths: np.ndarray
    counts: np.ndarray | None = None
    observed_exits: np.ndarray | None = None
    row_trials: np.ndarray | None = None
    prior_driven_rows: np.ndarray | None = None
    eligible_pairs: np.ndarray | None = None
    initial_probs: np.ndarray | None = None
    occupancy: np.ndarray | None = None
    prior_concentration: float = 1.0
    posterior_mean: np.ndarray | None = None
    per_transition_log_scores: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass(frozen=True)
class TransitionScore:
    log_scores: np.ndarray
    n_scored: int
    total_log_score: float
    skipped: int


@dataclass(frozen=True)
class DwellRun:
    state: int
    start: int
    stop: int
    duration: int
    left_censored: bool
    right_censored: bool


def _segment_slices(n_items: int, boundary_indices: np.ndarray | None) -> list[slice]:
    boundaries = [0] if boundary_indices is None or len(boundary_indices) == 0 else list(boundary_indices)
    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    boundaries = sorted(set(int(x) for x in boundaries if 0 <= int(x) < n_items))
    boundaries.append(n_items)
    return [slice(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _counts(
    labels: np.ndarray,
    n_states: int,
    boundary_indices: np.ndarray | None = None,
    eligible_pairs: np.ndarray | None = None,
) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        for offset, (a, b) in enumerate(zip(seg[:-1], seg[1:], strict=False), start=slc.start):
            if eligible_pairs is not None and not eligible_pairs[offset]:
                continue
            counts[int(a), int(b)] += 1.0
    return counts


def _initial_counts(labels: np.ndarray, n_states: int, boundary_indices: np.ndarray | None) -> np.ndarray:
    initial = np.zeros(n_states, dtype=np.float64)
    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        if len(seg):
            initial[int(seg[0])] += 1.0
    return initial


def _row_stochastic(counts: np.ndarray, laplace: float) -> np.ndarray:
    m = counts + laplace
    denom = m.sum(axis=1, keepdims=True)
    zero_rows = denom.squeeze(-1) == 0
    denom[zero_rows] = 1.0
    out = m / denom
    if np.any(zero_rows):
        out[zero_rows] = 1.0 / counts.shape[1]
    return out


def _stationary(p: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(p.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()


def _sequence_loglik(labels: np.ndarray, p: np.ndarray, boundary_indices: np.ndarray | None = None) -> float:
    ll = 0.0
    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        for a, b in zip(seg[:-1], seg[1:], strict=False):
            ll += np.log(p[int(a), int(b)] + 1e-12)
    return float(ll)


def _dwell_times(labels: np.ndarray, n_states: int, boundary_indices: np.ndarray | None = None) -> np.ndarray:
    runs: dict[int, list[int]] = {state: [] for state in range(n_states)}
    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        if len(seg) == 0:
            continue
        cur, length = int(seg[0]), 1
        for x in seg[1:]:
            if int(x) == cur:
                length += 1
            else:
                runs[cur].append(length)
                cur, length = int(x), 1
        runs[cur].append(length)
    return np.array([np.mean(runs[state]) if runs[state] else 0.0 for state in range(n_states)])


def dwell_runs(labels: np.ndarray, boundary_indices: np.ndarray | None = None) -> tuple[DwellRun, ...]:
    labels = np.asarray(labels, dtype=int)
    runs: list[DwellRun] = []
    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        if len(seg) == 0:
            continue
        start = slc.start
        cur = int(seg[0])
        run_start = start
        for offset, value in enumerate(seg[1:], start=1):
            if int(value) == cur:
                continue
            stop = start + offset
            runs.append(
                DwellRun(
                    state=cur,
                    start=run_start,
                    stop=stop,
                    duration=stop - run_start,
                    left_censored=run_start == slc.start,
                    right_censored=False,
                )
            )
            cur = int(value)
            run_start = stop
        runs.append(
            DwellRun(
                state=cur,
                start=run_start,
                stop=slc.stop,
                duration=slc.stop - run_start,
                left_censored=run_start == slc.start,
                right_censored=True,
            )
        )
    return tuple(runs)


def fit_transition_model(
    train_labels: np.ndarray,
    n_states: int,
    boundary_indices: np.ndarray | None,
    cfg: TransitionConfig,
    *,
    eligible_pairs: np.ndarray | None = None,
) -> TransitionModel:
    """Fit a frozen transition model from caller-owned training labels."""
    raw_labels = np.asarray(train_labels)
    if raw_labels.ndim != 1 or raw_labels.size == 0:
        raise ValueError("labels must be a non-empty 1D sequence")
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("labels must have an integer dtype")
    labels = raw_labels.astype(int, copy=False)
    if n_states < 1 or np.any(labels < 0) or np.any(labels >= n_states):
        raise ValueError("labels must lie in [0, n_states)")
    if cfg.laplace < 0:
        raise ValueError("laplace must be non-negative")
    eligibility = None
    if eligible_pairs is not None:
        eligibility = np.asarray(eligible_pairs, dtype=bool)
        if eligibility.shape != (max(0, labels.size - 1),):
            raise ValueError("eligible_pairs must align with adjacent label pairs")
    counts = _counts(labels, n_states, boundary_indices, eligibility)
    transition = _row_stochastic(counts, cfg.laplace)
    initial_counts = _initial_counts(labels, n_states, boundary_indices)
    initial = initial_counts + cfg.laplace
    initial /= initial.sum()
    occupancy = np.bincount(labels, minlength=n_states).astype(np.float64)
    row_trials = counts.sum(axis=1)
    exits = row_trials - np.diag(counts)
    segment_lengths = np.array(
        [slc.stop - slc.start for slc in _segment_slices(len(labels), boundary_indices)]
    )
    posterior_mean = transition.copy()
    return TransitionModel(
        transition_matrix=transition,
        stationary=_stationary(transition),
        dwell_times=_dwell_times(labels, n_states, boundary_indices),
        log_likelihood=_sequence_loglik(labels, transition, boundary_indices),
        heldout_transition_log_likelihood=float("nan"),
        heldout_baseline_log_likelihood=float("nan"),
        heldout_delta_log_likelihood=float("nan"),
        heldout_n_transitions=0,
        null_delta_log_likelihoods=np.empty(0, dtype=np.float64),
        p_value=float("nan"),
        boundary_indices=np.array([]) if boundary_indices is None else np.asarray(boundary_indices),
        segment_lengths=segment_lengths,
        counts=counts,
        observed_exits=exits,
        row_trials=row_trials,
        prior_driven_rows=row_trials == 0,
        eligible_pairs=eligibility,
        initial_probs=initial,
        occupancy=occupancy,
        prior_concentration=float(cfg.laplace),
        posterior_mean=posterior_mean,
    )


def score_transition_model(
    model: TransitionModel,
    test_labels: np.ndarray,
    boundary_indices: np.ndarray | None,
    *,
    initial_context: int | None = None,
) -> TransitionScore:
    """Score test transitions without refitting or crossing supplied boundaries."""
    if initial_context is not None and boundary_indices is not None:
        raise ValueError("initial_context is only defined when no explicit test boundary is supplied")
    labels = np.asarray(test_labels, dtype=int)
    scores: list[float] = []
    skipped = 0
    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        if len(seg) == 0:
            continue
        pairs: list[tuple[int, int]] = []
        if initial_context is not None and slc.start == 0:
            pairs.append((int(initial_context), int(seg[0])))
        pairs.extend((int(a), int(b)) for a, b in zip(seg[:-1], seg[1:], strict=False))
        if not pairs:
            skipped += 1
        for a, b in pairs:
            scores.append(float(np.log(model.transition_matrix[a, b] + 1e-12)))
    arr = np.asarray(scores, dtype=np.float64)
    return TransitionScore(
        log_scores=arr,
        n_scored=int(arr.size),
        total_log_score=float(arr.sum()),
        skipped=skipped,
    )


def transition_diagnostics(
    model: TransitionModel,
    labels: np.ndarray,
    boundary_indices: np.ndarray | None,
    *,
    cadence: float = 1.0,
) -> dict[str, object]:
    """Return descriptive transition diagnostics without changing the fitted model."""
    if cadence <= 0:
        raise ValueError("cadence must be positive")
    labels = np.asarray(labels, dtype=int)
    n_states = model.transition_matrix.shape[0]
    two_step_counts = np.zeros((n_states, n_states), dtype=np.float64)
    second_order_scores: list[float] = []
    two_step = model.transition_matrix @ model.transition_matrix
    for slc in _segment_slices(len(labels), boundary_indices):
        segment = labels[slc]
        for first, third in zip(segment[:-2], segment[2:], strict=False):
            two_step_counts[int(first), int(third)] += 1.0
            second_order_scores.append(float(np.log(two_step[int(first), int(third)] + 1e-12)))

    counts = np.asarray(model.counts, dtype=float)
    alpha = counts + model.prior_concentration
    row_intervals = np.empty((n_states, n_states, 2), dtype=float)
    for row in range(n_states):
        total = float(alpha[row].sum())
        for column in range(n_states):
            other = total - alpha[row, column]
            if other <= 0:
                row_intervals[row, column] = (1.0, 1.0)
            else:
                row_intervals[row, column] = beta.ppf(
                    (0.025, 0.975), alpha[row, column], other
                )
    return {
        "counts": model.counts,
        "observed_exits": model.observed_exits,
        "row_trials": model.row_trials,
        "prior_driven_rows": model.prior_driven_rows,
        "initial_probs": model.initial_probs,
        "occupancy": model.occupancy,
        "prior_concentration": model.prior_concentration,
        "dwell_runs": dwell_runs(labels, boundary_indices),
        "dwell_runs_seconds": tuple(
            run.duration * cadence for run in dwell_runs(labels, boundary_indices)
        ),
        "two_step_counts": two_step_counts,
        "second_order_log_scores": np.asarray(second_order_scores),
        "row_intervals": row_intervals,
    }


def _infer_block_length(labels: np.ndarray, n_states: int, boundary_indices: np.ndarray | None) -> int:
    dwell = _dwell_times(labels, n_states, boundary_indices)
    positive = dwell[dwell > 0]
    if positive.size == 0:
        return 2
    return max(2, int(round(float(np.median(positive)))))


def _block_surrogate(segment: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Permute contiguous blocks while preserving their internal label order."""
    if len(segment) <= 1:
        return segment.copy()
    block_length = max(1, min(block_length, len(segment)))
    blocks = [segment[start : start + block_length] for start in range(0, len(segment), block_length)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order])


def _surrogate_labels(
    labels: np.ndarray,
    *,
    n_states: int,
    boundary_indices: np.ndarray | None,
    cfg: TransitionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    surrogates = []
    block_length = cfg.null_block_length or _infer_block_length(labels, n_states, boundary_indices)
    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        if cfg.null_mode == "shuffle":
            surrogates.append(rng.permutation(seg))
        elif cfg.null_mode == "block":
            surrogates.append(_block_surrogate(seg, block_length, rng))
        else:
            raise ValueError(f"unknown null_mode '{cfg.null_mode}'")
    return np.concatenate(surrogates) if surrogates else labels.copy()


def _fit_holdout_chain(
    labels: np.ndarray,
    n_states: int,
    laplace: float,
    boundary_indices: np.ndarray | None,
    heldout_fraction: float,
) -> tuple[float, float, float, int]:
    trans_counts = np.zeros((n_states, n_states), dtype=np.float64)
    occupancy = np.zeros(n_states, dtype=np.float64)
    heldout_transition = 0.0
    heldout_baseline = 0.0
    n_scored = 0
    partitions: list[tuple[np.ndarray, np.ndarray]] = []

    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        if len(seg) < 4:
            continue
        test_len = max(2, int(np.ceil(len(seg) * heldout_fraction)))
        test_len = min(test_len, len(seg) - 2)
        train_stop = len(seg) - test_len
        train_seg = seg[:train_stop]
        test_seg = seg[train_stop - 1 :]
        partitions.append((train_seg, test_seg))

        for a, b in zip(train_seg[:-1], train_seg[1:], strict=False):
            trans_counts[int(a), int(b)] += 1.0
        occupancy += np.bincount(train_seg.astype(int), minlength=n_states)

    if not partitions:
        raise ValueError("held-out transition scoring needs at least one segment of length four")

    p_train = _row_stochastic(trans_counts, laplace)
    baseline = occupancy + laplace
    baseline /= baseline.sum()
    for _, test_seg in partitions:
        for a, b in zip(test_seg[:-1], test_seg[1:], strict=False):
            heldout_transition += np.log(p_train[int(a), int(b)] + 1e-12)
            heldout_baseline += np.log(baseline[int(b)] + 1e-12)
            n_scored += 1

    return (
        float(heldout_transition),
        float(heldout_baseline),
        float(heldout_transition - heldout_baseline),
        n_scored,
    )


def fit_transitions(
    labels: np.ndarray,
    n_states: int,
    cfg: TransitionConfig,
    *,
    window_starts: np.ndarray | None = None,
    boundary_indices: np.ndarray | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TransitionModel:
    """Estimate transition structure with held-out temporal-order scoring."""
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels must be a non-empty 1D sequence")
    if n_states < 1 or np.any(labels < 0) or np.any(labels >= n_states):
        raise ValueError("labels must lie in [0, n_states)")
    if cfg.laplace < 0:
        raise ValueError("laplace must be non-negative")
    if cfg.n_null < 1:
        raise ValueError("n_null must be at least one")
    if not 0.0 < cfg.heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must lie strictly between zero and one")
    if boundary_indices is None and window_starts is not None:
        boundary_indices = infer_boundary_indices(window_starts)
    boundary_indices = None if boundary_indices is None else np.asarray(boundary_indices, dtype=int)

    counts = _counts(labels, n_states, boundary_indices)
    p = _row_stochastic(counts, cfg.laplace)
    obs_ll = _sequence_loglik(labels, p, boundary_indices)
    initial_counts = _initial_counts(labels, n_states, boundary_indices)
    initial = initial_counts + cfg.laplace
    initial /= initial.sum()
    occupancy = np.bincount(labels, minlength=n_states).astype(np.float64)
    heldout_transition, heldout_baseline, heldout_delta, heldout_n = _fit_holdout_chain(
        labels,
        n_states,
        cfg.laplace,
        boundary_indices,
        cfg.heldout_fraction,
    )

    rng = np.random.default_rng(cfg.seed)
    null = np.empty(cfg.n_null, dtype=np.float64)
    for i in range(cfg.n_null):
        surrogate = _surrogate_labels(
            labels,
            n_states=n_states,
            boundary_indices=boundary_indices,
            cfg=cfg,
            rng=rng,
        )
        _, _, null[i], _ = _fit_holdout_chain(
            surrogate,
            n_states,
            cfg.laplace,
            boundary_indices,
            cfg.heldout_fraction,
        )
        if progress_callback is not None:
            progress_callback(i + 1, cfg.n_null)
    p_value = float((1 + np.sum(null >= heldout_delta)) / (cfg.n_null + 1))

    segment_lengths = np.array(
        [slc.stop - slc.start for slc in _segment_slices(len(labels), boundary_indices)]
    )
    logger.info(
        "Transition model: held-out delta=%.2f, null mean=%.2f, p=%.4f, segments=%d",
        heldout_delta,
        null.mean() if null.size else 0.0,
        p_value,
        len(segment_lengths),
    )
    return TransitionModel(
        transition_matrix=p,
        stationary=_stationary(p),
        dwell_times=_dwell_times(labels, n_states, boundary_indices),
        log_likelihood=obs_ll,
        heldout_transition_log_likelihood=heldout_transition,
        heldout_baseline_log_likelihood=heldout_baseline,
        heldout_delta_log_likelihood=heldout_delta,
        heldout_n_transitions=heldout_n,
        null_delta_log_likelihoods=null,
        p_value=p_value,
        boundary_indices=np.array([]) if boundary_indices is None else boundary_indices,
        segment_lengths=segment_lengths,
        counts=counts,
        observed_exits=counts.sum(axis=1) - np.diag(counts),
        row_trials=counts.sum(axis=1),
        prior_driven_rows=counts.sum(axis=1) == 0,
        initial_probs=initial,
        occupancy=occupancy,
        prior_concentration=float(cfg.laplace),
        posterior_mean=p,
    )


__all__ = [
    "DwellRun",
    "TransitionConfig",
    "TransitionModel",
    "TransitionScore",
    "dwell_runs",
    "fit_transition_model",
    "fit_transitions",
    "score_transition_model",
    "transition_diagnostics",
]
