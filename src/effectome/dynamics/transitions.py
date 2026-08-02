"""Model boundary-safe transitions between connectivity states (Stage 3b).

Given a state-label sequence, estimate the transition probability matrix P, stationary
distribution, dwell-time statistics, and a boundary-aware temporal-structure p-value.
The default null uses within-segment circular block resampling rather than an iid shuffle,
so short-range autocorrelation is preserved approximately while recording boundaries remain
intact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .graph_states import infer_boundary_indices

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitionConfig:
    """Configuration for transition modeling.

    Attributes:
        laplace: Additive smoothing for unseen transitions.
        n_null: Number of surrogate sequences for the structure test.
        null_mode: Null family ("block" or "shuffle").
        null_block_length: Block length for the default circular block bootstrap. If 0, infer
            from observed dwell times.
        seed: Random seed for surrogates.
    """

    laplace: float = 1.0
    n_null: int = 500
    null_mode: str = "block"
    null_block_length: int = 0
    seed: int = 42


@dataclass
class TransitionModel:
    """Estimated Markov transition structure over connectivity states."""

    transition_matrix: np.ndarray
    stationary: np.ndarray
    dwell_times: np.ndarray
    log_likelihood: float
    null_log_likelihoods: np.ndarray
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
    counts = np.zeros((n_states, n_states))
    for slc in _segment_slices(len(labels), boundary_indices):
        seg = labels[slc]
        for a, b in zip(seg[:-1], seg[1:], strict=False):
            counts[a, b] += 1
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
            ll += np.log(p[a, b] + 1e-12)
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
    if len(segment) <= 1:
        return segment.copy()
    block_length = max(1, min(block_length, len(segment)))
    n_blocks = int(np.ceil(len(segment) / block_length))
    starts = rng.integers(0, len(segment), size=n_blocks)
    blocks = []
    for start in starts:
        idx = (np.arange(block_length) + start) % len(segment)
        blocks.append(segment[idx])
    return np.concatenate(blocks)[: len(segment)]


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


def fit_transitions(
    labels: np.ndarray,
    n_states: int,
    cfg: TransitionConfig,
    *,
    window_starts: np.ndarray | None = None,
    boundary_indices: np.ndarray | None = None,
) -> TransitionModel:
    """Estimate transition structure with recording-boundary-aware nulls."""
    labels = np.asarray(labels, dtype=int)
    if boundary_indices is None and window_starts is not None:
        boundary_indices = infer_boundary_indices(window_starts)
    boundary_indices = None if boundary_indices is None else np.asarray(boundary_indices, dtype=int)

    p = _row_stochastic(_counts(labels, n_states, boundary_indices), cfg.laplace)
    obs_ll = _sequence_loglik(labels, p, boundary_indices)

    rng = np.random.default_rng(cfg.seed)
    null = np.empty(cfg.n_null)
    for i in range(cfg.n_null):
        surrogate = _surrogate_labels(
            labels,
            n_states=n_states,
            boundary_indices=boundary_indices,
            cfg=cfg,
            rng=rng,
        )
        p_null = _row_stochastic(_counts(surrogate, n_states, boundary_indices), cfg.laplace)
        null[i] = _sequence_loglik(surrogate, p_null, boundary_indices)
    p_value = float((null >= obs_ll).mean())

    segment_lengths = np.array(
        [slc.stop - slc.start for slc in _segment_slices(len(labels), boundary_indices)]
    )
    logger.info(
        "Transition model: obs LL=%.2f, null mean=%.2f, p=%.4f, segments=%d",
        obs_ll,
        null.mean(),
        p_value,
        len(segment_lengths),
    )
    return TransitionModel(
        transition_matrix=p,
        stationary=_stationary(p),
        dwell_times=_dwell_times(labels, n_states, boundary_indices),
        log_likelihood=obs_ll,
        null_log_likelihoods=null,
        p_value=p_value,
        boundary_indices=np.array([]) if boundary_indices is None else boundary_indices,
        segment_lengths=segment_lengths,
    )
