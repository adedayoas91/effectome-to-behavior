"""Boundary-safe Markov transition models for connectivity states."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

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


def _segment_slices(n_items: int, boundary_indices: np.ndarray | None) -> list[slice]:
    boundaries = [0] if boundary_indices is None or len(boundary_indices) == 0 else list(boundary_indices)
    if boundaries[0] != 0:
        boundaries = [0] + boundaries
    boundaries = sorted(set(int(x) for x in boundaries if 0 <= int(x) < n_items))
    boundaries.append(n_items)
    return [slice(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _counts(labels: np.ndarray, n_states: int, boundary_indices: np.ndarray | None = None) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        for a, b in zip(seg[:-1], seg[1:], strict=False):
            counts[int(a), int(b)] += 1.0
    return counts


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
    )


__all__ = ["TransitionConfig", "TransitionModel", "fit_transitions"]
