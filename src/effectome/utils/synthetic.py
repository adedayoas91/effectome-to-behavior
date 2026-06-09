"""Synthetic neural data with a *known* causal graph for ground-truth testing.

Generates a multivariate time series from a sparse linear vector-autoregressive (VAR)
process whose coefficient matrix encodes the true directed causal graph. The graph switches
between a few recurring regimes (connectivity states) drawn from a slow Markov chain. Two
behavior variables are produced: a discrete 'motif' tied to the connectivity *regime* (so the
inferred connectivity state carries genuine behavioral information) and a continuous variable
tracking a fixed driver set. This lets us measure how well connectivity estimators recover the
true edges (AUROC), whether connectivity states recover the regimes, and whether the effectome
decodes behavior -- all without the real dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid import cycle at runtime
    from effectome.data_module.schema import NeuralRecording


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for the synthetic generator.

    Attributes:
        n_neurons: Number of variables (neurons) N.
        n_timepoints: Number of time samples T.
        density: Fraction of off-diagonal entries that are true causal edges.
        max_lag: Maximum autoregressive lag of the generating process.
        coupling: Scale of nonzero causal coefficients.
        noise_std: Standard deviation of innovation noise.
        n_states: Number of latent connectivity states (regimes) over time.
        regime_dwell: Expected dwell time (samples) of each regime before switching.
        behavior_drivers: Number of neurons whose activity drives continuous behavior.
        fps: Sampling rate stored in the recording.
        seed: Random seed.
    """

    n_neurons: int = 20
    n_timepoints: int = 3000
    density: float = 0.15
    max_lag: int = 1
    coupling: float = 0.45
    noise_std: float = 0.7
    n_states: int = 3
    regime_dwell: int = 150
    behavior_drivers: int = 4
    fps: float = 10.0
    seed: int = 42


def _random_dag_coeffs(n: int, density: float, coupling: float, rng: np.random.Generator) -> np.ndarray:
    """Build a stable NxN causal graph in source->target convention.

    Returns G where G[i, j] is the influence of source neuron i on target neuron j (so the
    VAR step uses G.T: x_new[target] = sum_source G[source, target] * x_old[source]).
    """
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    signs = rng.choice([-1.0, 1.0], size=(n, n))
    g = coupling * (0.5 + rng.random((n, n))) * signs * mask  # [source, target]
    np.fill_diagonal(g, 0.3)  # mild self-persistence (calcium-like autocorrelation)
    # Rescale to keep the VAR stable (spectral radius < 1).
    radius = np.max(np.abs(np.linalg.eigvals(g)))
    if radius >= 0.9:
        g *= 0.85 / radius
    return g


def _regime_sequence(t: int, n_states: int, dwell: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a piecewise-constant regime label sequence (a slow Markov chain)."""
    seq = np.empty(t, dtype=np.int64)
    state = 0
    switch_p = 1.0 / max(1, dwell)
    for tt in range(t):
        if tt > 0 and rng.random() < switch_p:
            choices = [s for s in range(n_states) if s != state]
            state = int(rng.choice(choices))
        seq[tt] = state
    return seq


def make_synthetic_recording(cfg: SyntheticConfig) -> NeuralRecording:
    """Generate a `NeuralRecording` plus the ground-truth causal graphs per state.

    The returned recording carries the true per-state adjacency matrices in
    `metadata["true_graphs"]` and the regime label per timepoint in
    `metadata["true_states"]`, so tests can score recovery against ground truth.
    """
    from effectome.data_module.schema import NeuralRecording

    rng = np.random.default_rng(cfg.seed)
    n, t = cfg.n_neurons, cfg.n_timepoints

    # One causal graph (source->target convention) per latent regime; regimes recur over time
    # as a slow Markov chain so the transition structure is non-trivial.
    true_graphs = [
        _random_dag_coeffs(n, cfg.density, cfg.coupling, rng) for _ in range(cfg.n_states)
    ]
    state_seq = _regime_sequence(t, cfg.n_states, cfg.regime_dwell, rng)

    x = np.zeros((t, n), dtype=np.float64)
    x[0] = rng.normal(0, cfg.noise_std, size=n)
    for tt in range(1, t):
        g = true_graphs[state_seq[tt]]  # [source, target]
        x[tt] = g.T @ x[tt - 1] + rng.normal(0, cfg.noise_std, size=n)

    # Behavior:
    #   * 'motif' is tied to the connectivity *regime* (a subset of regimes is 'active'),
    #     so the inferred connectivity state carries genuine behavioral information.
    #   * 'continuous' tracks the mean activity of a fixed driver set (fast, activity-linked).
    active_states = set(range(int(np.ceil(cfg.n_states / 2))))
    behavior_motif = np.array([1 if s in active_states else 0 for s in state_seq], dtype=np.int64)
    flip = rng.random(t) < 0.03  # small label noise
    behavior_motif = np.where(flip, 1 - behavior_motif, behavior_motif)

    drivers = rng.choice(n, size=cfg.behavior_drivers, replace=False)
    behavior_continuous = x[:, drivers].mean(axis=1) + rng.normal(0, 0.1, size=t)

    traces = x.T.astype(np.float32)  # N x T
    coords = rng.normal(0, 1, size=(n, 3)).astype(np.float32)
    neuron_ids = np.arange(n)

    return NeuralRecording(
        traces=traces,
        time=np.arange(t) / cfg.fps,
        coords=coords,
        neuron_ids=neuron_ids,
        behavior={"continuous": behavior_continuous, "motif": behavior_motif},
        fps=cfg.fps,
        metadata={
            "source": "synthetic",
            "true_graphs": np.stack(true_graphs),
            "true_states": state_seq,
            "behavior_drivers": drivers,
        },
    )
