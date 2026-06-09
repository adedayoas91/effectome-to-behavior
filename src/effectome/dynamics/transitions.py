"""Model transitions between connectivity states as a (semi-)Markov process (Stage 3b).

Given the state-label sequence from `graph_states`, estimate the transition probability matrix
P, the stationary distribution, and dwell-time statistics. A time-shuffled null provides a
significance baseline for whether the observed sequence carries more temporal structure than
chance (tests hypothesis H2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransitionConfig:
    """Configuration for transition modeling.

    Attributes:
        laplace: Additive smoothing for unseen transitions.
        n_null: Number of time-shuffle surrogates for the structure test.
        seed: Random seed for surrogates.
    """

    laplace: float = 1.0
    n_null: int = 500
    seed: int = 42


@dataclass
class TransitionModel:
    """Estimated Markov transition structure over connectivity states.

    Attributes:
        transition_matrix: Row-stochastic P, shape (S, S); P[a, b] = Pr(state b | state a).
        stationary: Stationary distribution over states, shape (S,).
        dwell_times: Mean dwell time (in windows) per state, shape (S,).
        log_likelihood: Log-likelihood of the observed sequence under P.
        null_log_likelihoods: Log-likelihoods of time-shuffled surrogates, shape (n_null,).
        p_value: Fraction of surrogates with log-likelihood >= observed (lower = more structure).
    """

    transition_matrix: np.ndarray
    stationary: np.ndarray
    dwell_times: np.ndarray
    log_likelihood: float
    null_log_likelihoods: np.ndarray
    p_value: float


def _counts(labels: np.ndarray, n_states: int) -> np.ndarray:
    c = np.zeros((n_states, n_states))
    for a, b in zip(labels[:-1], labels[1:], strict=False):
        c[a, b] += 1
    return c


def _row_stochastic(counts: np.ndarray, laplace: float) -> np.ndarray:
    m = counts + laplace
    return m / m.sum(axis=1, keepdims=True)


def _stationary(p: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(p.T)
    idx = np.argmin(np.abs(vals - 1.0))
    pi = np.real(vecs[:, idx])
    pi = np.abs(pi)
    return pi / pi.sum()


def _sequence_loglik(labels: np.ndarray, p: np.ndarray) -> float:
    ll = 0.0
    for a, b in zip(labels[:-1], labels[1:], strict=False):
        ll += np.log(p[a, b] + 1e-12)
    return float(ll)


def _dwell_times(labels: np.ndarray, n_states: int) -> np.ndarray:
    runs = {s: [] for s in range(n_states)}
    cur, length = labels[0], 1
    for x in labels[1:]:
        if x == cur:
            length += 1
        else:
            runs[cur].append(length)
            cur, length = x, 1
    runs[cur].append(length)
    return np.array([np.mean(runs[s]) if runs[s] else 0.0 for s in range(n_states)])


def fit_transitions(labels: np.ndarray, n_states: int, cfg: TransitionConfig) -> TransitionModel:
    """Estimate P, stationary distribution, dwell times, and a temporal-structure p-value."""
    labels = np.asarray(labels, dtype=int)
    p = _row_stochastic(_counts(labels, n_states), cfg.laplace)
    obs_ll = _sequence_loglik(labels, p)

    rng = np.random.default_rng(cfg.seed)
    null = np.empty(cfg.n_null)
    for i in range(cfg.n_null):
        shuffled = rng.permutation(labels)
        p_null = _row_stochastic(_counts(shuffled, n_states), cfg.laplace)
        null[i] = _sequence_loglik(shuffled, p_null)
    p_value = float((null >= obs_ll).mean())

    logger.info("Transition model: obs LL=%.2f, null mean=%.2f, p=%.4f", obs_ll, null.mean(), p_value)
    return TransitionModel(
        transition_matrix=p,
        stationary=_stationary(p),
        dwell_times=_dwell_times(labels, n_states),
        log_likelihood=obs_ll,
        null_log_likelihoods=null,
        p_value=p_value,
    )
