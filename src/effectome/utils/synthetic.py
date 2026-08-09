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
    regime_sign_flip_fraction: float = 0.0
    regime_sparsity_jitter: float = 0.0
    instantaneous_density: float = 0.0
    instantaneous_strength: float = 0.0
    n_latent_confounders: int = 0
    latent_strength: float = 0.0
    calcium_decay_range: tuple[float, float] | None = None
    calcium_gain_heterogeneity: float = 0.0
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


def _instantaneous_dag(
    n: int,
    density: float,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an acyclic contemporaneous graph in source->target convention."""
    mask = np.triu(rng.random((n, n)) < density, k=1)
    signs = rng.choice([-1.0, 1.0], size=(n, n))
    return strength * (0.5 + 0.5 * rng.random((n, n))) * signs * mask


def _stabilize_lag_stack(graphs: list[np.ndarray], radius_limit: float = 0.95) -> None:
    """Rescale a VAR(p) coefficient stack using its companion-matrix radius."""
    n = graphs[0].shape[0]
    p = len(graphs)
    for _ in range(12):
        companion = np.zeros((n * p, n * p), dtype=np.float64)
        companion[:n, : n * p] = np.concatenate([graph.T for graph in graphs], axis=1)
        if p > 1:
            companion[n:, :-n] = np.eye(n * (p - 1))
        radius = float(np.max(np.abs(np.linalg.eigvals(companion))))
        if radius < radius_limit:
            return
        scale = min(0.95, radius_limit / max(radius, 1e-12))
        for graph in graphs:
            graph *= scale
    raise RuntimeError("failed to stabilize the synthetic VAR lag stack")


def _validate_config(cfg: SyntheticConfig) -> None:
    if cfg.n_neurons < 2 or cfg.n_timepoints <= cfg.max_lag or cfg.max_lag < 1:
        raise ValueError("synthetic dimensions require N>=2, max_lag>=1, and T>max_lag")
    if not 0 <= cfg.density <= 1:
        raise ValueError("density must be in [0, 1]")
    for name, value in (
        ("regime_sign_flip_fraction", cfg.regime_sign_flip_fraction),
        ("regime_sparsity_jitter", cfg.regime_sparsity_jitter),
        ("instantaneous_density", cfg.instantaneous_density),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if cfg.n_latent_confounders < 0 or cfg.latent_strength < 0:
        raise ValueError("latent-confounder settings must be non-negative")
    if cfg.behavior_drivers < 1 or cfg.behavior_drivers > cfg.n_neurons:
        raise ValueError("behavior_drivers must be in 1..n_neurons")
    if cfg.calcium_decay_range is not None:
        low, high = cfg.calcium_decay_range
        if not 0 <= low <= high < 1:
            raise ValueError("calcium_decay_range must satisfy 0 <= low <= high < 1")


def make_synthetic_recording(cfg: SyntheticConfig) -> NeuralRecording:
    """Generate a `NeuralRecording` plus the ground-truth causal graphs per state.

    The returned recording carries the true per-state adjacency matrices in
    `metadata["true_graphs"]` and the regime label per timepoint in
    `metadata["true_states"]`, so tests can score recovery against ground truth.
    """
    from effectome.data_module.schema import NeuralRecording

    _validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)
    n, t = cfg.n_neurons, cfg.n_timepoints

    # One lag-resolved graph stack per latent regime.  The legacy ``true_graphs``
    # metadata remains the sum across lags so existing edge-recovery consumers keep
    # working while new tests can inspect the exact lag structure.
    lag_graphs: list[list[np.ndarray]] = []
    sign_flip_masks: list[np.ndarray] = []
    for _state in range(cfg.n_states):
        density_scale = 1.0 + rng.uniform(
            -cfg.regime_sparsity_jitter, cfg.regime_sparsity_jitter
        )
        state_density = float(np.clip(cfg.density * density_scale, 0.0, 1.0))
        state_lags = [
            _random_dag_coeffs(n, state_density, cfg.coupling / cfg.max_lag, rng)
            for _ in range(cfg.max_lag)
        ]
        flip_mask = np.zeros((n, n), dtype=bool)
        if cfg.regime_sign_flip_fraction > 0:
            candidates = np.argwhere(np.abs(state_lags[0]) > 1e-12)
            candidates = candidates[candidates[:, 0] != candidates[:, 1]]
            n_flip = int(round(cfg.regime_sign_flip_fraction * len(candidates)))
            if n_flip > 0:
                selected = candidates[rng.choice(len(candidates), size=n_flip, replace=False)]
                flip_mask[selected[:, 0], selected[:, 1]] = True
                for graph in state_lags:
                    graph[flip_mask] *= -1.0
        _stabilize_lag_stack(state_lags)
        lag_graphs.append(state_lags)
        sign_flip_masks.append(flip_mask)
    true_graphs = [np.sum(np.stack(graphs), axis=0) for graphs in lag_graphs]
    instantaneous_graphs = [
        _instantaneous_dag(
            n,
            cfg.instantaneous_density,
            cfg.instantaneous_strength,
            rng,
        )
        for _ in range(cfg.n_states)
    ]
    state_seq = _regime_sequence(t, cfg.n_states, cfg.regime_dwell, rng)

    latent_factors = np.zeros((t, cfg.n_latent_confounders), dtype=np.float64)
    latent_loadings = np.zeros((n, cfg.n_latent_confounders), dtype=np.float64)
    if cfg.n_latent_confounders:
        latent_loadings = rng.normal(
            0.0,
            cfg.latent_strength / np.sqrt(cfg.n_latent_confounders),
            size=(n, cfg.n_latent_confounders),
        )
        latent_factors[0] = rng.normal(size=cfg.n_latent_confounders)
        for tt in range(1, t):
            latent_factors[tt] = 0.8 * latent_factors[tt - 1] + rng.normal(
                0.0, 0.6, size=cfg.n_latent_confounders
            )

    x = np.zeros((t, n), dtype=np.float64)
    x[: cfg.max_lag] = rng.normal(0, cfg.noise_std, size=(cfg.max_lag, n))
    identity = np.eye(n)
    for tt in range(cfg.max_lag, t):
        state = int(state_seq[tt])
        lagged = np.zeros(n, dtype=float)
        for lag, graph in enumerate(lag_graphs[state], start=1):
            lagged += graph.T @ x[tt - lag]
        common = latent_loadings @ latent_factors[tt] if cfg.n_latent_confounders else 0.0
        innovation = lagged + common + rng.normal(0, cfg.noise_std, size=n)
        # For B[source,target], x = B.T x + innovation.  B is a DAG, so
        # (I-B.T) is nonsingular and the contemporaneous solution is exact.
        x[tt] = np.linalg.solve(identity - instantaneous_graphs[state].T, innovation)

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

    observed = x
    calcium_decay = np.zeros(n, dtype=float)
    calcium_gain = np.ones(n, dtype=float)
    if cfg.calcium_decay_range is not None:
        low, high = cfg.calcium_decay_range
        calcium_decay = rng.uniform(low, high, size=n)
        if cfg.calcium_gain_heterogeneity > 0:
            calcium_gain = np.exp(
                rng.normal(0.0, cfg.calcium_gain_heterogeneity, size=n)
            )
        observed = np.zeros_like(x)
        observed[0] = calcium_gain * x[0]
        for tt in range(1, t):
            observed[tt] = calcium_decay * observed[tt - 1] + calcium_gain * x[tt]

    traces = observed.T.astype(np.float32)  # N x T
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
            "true_lag_graphs": np.asarray(lag_graphs),
            "true_instantaneous_graphs": np.stack(instantaneous_graphs),
            "true_states": state_seq,
            "regime_sign_flip_masks": np.stack(sign_flip_masks),
            "latent_factors": latent_factors,
            "latent_loadings": latent_loadings,
            "latent_neural_activity": x,
            "calcium_decay": calcium_decay,
            "calcium_gain": calcium_gain,
            "behavior_drivers": drivers,
            "stress_config": {
                "max_lag": cfg.max_lag,
                "sign_flip_fraction": cfg.regime_sign_flip_fraction,
                "sparsity_jitter": cfg.regime_sparsity_jitter,
                "instantaneous_density": cfg.instantaneous_density,
                "instantaneous_strength": cfg.instantaneous_strength,
                "n_latent_confounders": cfg.n_latent_confounders,
                "latent_strength": cfg.latent_strength,
                "calcium_decay_range": cfg.calcium_decay_range,
                "calcium_gain_heterogeneity": cfg.calcium_gain_heterogeneity,
            },
        },
    )
