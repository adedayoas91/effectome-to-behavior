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
from scipy.linalg import eigvalsh, solve_discrete_lyapunov

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
    graph_seed: int | None = None
    endpoint_seed: int | None = None
    scenario: str = "recurrent_geometric"
    contraction: float = 0.98
    burn_in: int = 1000
    native_time: tuple[float, ...] | None = None
    variance_ratio: float = 1.0
    innovation_correlation: float = 0.0
    measurement_snr: float | None = None
    missing_fraction: float = 0.0
    nonlinear_saturation: bool = False


def _random_var_coeffs(n: int, density: float, coupling: float, rng: np.random.Generator) -> np.ndarray:
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


# Historical private name retained while the generator migrates to explicit VAR terminology.
_random_dag_coeffs = _random_var_coeffs


TEMPORAL_SCENARIOS = (
    "constant", "smooth_drift", "nonrecurring_epochs",
    "recurrent_geometric", "recurrent_non_geometric",
)


def effective_companion(lags: np.ndarray, instantaneous: np.ndarray) -> np.ndarray:
    """Companion of structural [lag, source, target] coefficients after the solve."""
    p, n, _ = lags.shape
    reduced = np.linalg.solve(np.eye(n) - instantaneous.T, np.concatenate(lags.transpose(0, 2, 1), axis=1))
    companion = np.zeros((n * p, n * p))
    companion[:n] = reduced
    companion[n:, :-n] = np.eye(n * (p - 1))
    return companion


def certify_common_metric(companions: np.ndarray, *, rho: float = 0.98,
                          metric: np.ndarray | None = None) -> dict:
    """Certify arbitrary switching in one fixed positive-definite Lyapunov metric.

    A failure means this certificate was not established, not necessarily instability.
    Lyapunov's equation is transposed because SciPy solves A Q A.T - Q = -I.
    """
    matrices = np.asarray(companions, dtype=float)
    if (matrices.ndim != 3 or matrices.shape[0] == 0 or matrices.shape[1] != matrices.shape[2]
            or not np.all(np.isfinite(matrices)) or not 0 < rho < 1):
        raise ValueError("invalid common certificate inputs")
    if metric is None:
        if max(abs(np.linalg.eigvals(matrices[0]))) >= rho:
            raise ValueError("base cannot satisfy target certificate")
        metric = solve_discrete_lyapunov((matrices[0] / rho).T, np.eye(matrices.shape[1]))
    metric = np.asarray(metric, dtype=float)
    if not np.allclose(metric, metric.T) or not np.all(np.isfinite(metric)):
        raise ValueError("certificate metric must be finite and symmetric")
    minimum = float(eigvalsh(metric)[0])
    if minimum <= 0:
        raise ValueError("certificate metric must be positive definite")
    factors = np.array([eigvalsh(c.T @ metric @ c, metric)[-1] for c in matrices])
    worst = float(factors.max())
    if worst > rho ** 2 + 1e-12:
        raise ValueError("common switching certificate failed")
    return {"metric": metric, "rho": rho, "worst_factor": worst,
            "margin": rho ** 2 - worst, "minimum_metric_eigenvalue": minimum}


def _coefficient_family(cfg: SyntheticConfig, rng: np.random.Generator) -> tuple:
    n, p = cfg.n_neurons, cfg.max_lag
    base = np.array([_random_var_coeffs(n, cfg.density, cfg.coupling / p, rng) for _ in range(p)])
    instant = _instantaneous_dag(n, cfg.instantaneous_density, cfg.instantaneous_strength, rng)
    # Scale only the top companion blocks; the multi-lag shift rows stay intact.
    for _ in range(100):
        try:
            certificate = certify_common_metric(np.array([effective_companion(base, instant)]),
                                                rho=cfg.contraction)
            break
        except ValueError:
            base *= 0.8
    else:
        raise ValueError("could not certify base at the fixed target contraction")
    proposals: list[np.ndarray] = []
    flips: list[np.ndarray] = []
    for state in range(cfg.n_states):
        candidate = base.copy()
        mask = (rng.random((n, n)) < cfg.regime_sign_flip_fraction) & np.any(base != 0, axis=0)
        np.fill_diagonal(mask, False)
        candidate[:, mask] *= -1
        support_edit = rng.random(base.shape) < cfg.regime_sparsity_jitter
        candidate[support_edit] = 0
        candidate *= rng.uniform(0.65, 1.35, size=base.shape)
        if state == 0 or cfg.scenario == "constant":
            candidate = base.copy()
            mask[:] = False
        proposals.append(candidate)
        flips.append(mask)
    proposal_array = np.asarray(proposals)
    for shrink in range(100):
        graphs = base + (proposal_array - base) * 0.8 ** shrink
        companions = np.array([effective_companion(g, instant) for g in graphs])
        try:
            certificate = certify_common_metric(companions, rho=cfg.contraction,
                                                metric=certificate["metric"])
            certificate["perturbation_scale"] = 0.8 ** shrink
            return (
                graphs,
                np.repeat(instant[None], cfg.n_states, axis=0),
                companions,
                certificate,
                base,
                flips,
            )
        except ValueError:
            pass
    raise ValueError("could not certify shared-base perturbations")


def _temporal_weights(cfg: SyntheticConfig, length: int, rng: np.random.Generator) -> tuple:
    k = cfg.n_states
    if cfg.scenario == "constant" or k == 1:
        states = np.zeros(length, dtype=int)
    elif cfg.scenario in {"smooth_drift", "nonrecurring_epochs"}:
        states = np.minimum(np.arange(length) * k // length, k - 1)
    elif cfg.scenario == "recurrent_geometric":
        states = _regime_sequence(length, k, cfg.regime_dwell, rng)
    else:
        states = np.zeros(length, dtype=int)
        start, state = 0, 0
        while start < length:
            dwell = max(1, int(round(rng.gamma(4, cfg.regime_dwell / 4))))
            states[start:start + dwell] = state
            start += dwell
            state = int(rng.choice([s for s in range(k) if s != state]))
    weights = np.eye(k)[states]
    if cfg.scenario == "smooth_drift" and k > 1:
        position = np.linspace(0, k - 1, length)
        left = np.minimum(position.astype(int), k - 2)
        fraction = position - left
        weights[:] = 0
        weights[np.arange(length), left] = 1 - fraction
        weights[np.arange(length), left + 1] = fraction
    return states, weights


def _regime_sequence(t: int, n_states: int, dwell: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a piecewise-constant regime label sequence (a slow Markov chain)."""
    if n_states < 1:
        raise ValueError("n_states must be at least one")
    if n_states == 1:
        return np.zeros(t, dtype=np.int64)
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
    if cfg.scenario not in TEMPORAL_SCENARIOS:
        raise ValueError("unknown temporal scenario")
    if not 0 < cfg.contraction < 1 or cfg.burn_in < 0:
        raise ValueError("invalid contraction or burn_in")
    if (not np.isfinite(cfg.variance_ratio) or cfg.variance_ratio <= 0
            or not 0 <= cfg.innovation_correlation < 1 or not 0 <= cfg.missing_fraction < 1):
        raise ValueError("invalid variance, correlation or missingness")
    if cfg.measurement_snr is not None and (not np.isfinite(cfg.measurement_snr) or cfg.measurement_snr <= 0):
        raise ValueError("measurement_snr must be positive")
    if cfg.native_time is not None:
        clock = np.asarray(cfg.native_time)
        if (
            clock.shape != (cfg.n_timepoints,)
            or not np.all(np.isfinite(clock))
            or np.any(np.diff(clock) <= 0)
        ):
            raise ValueError("native_time must be a finite strictly increasing length-T clock")
    if cfg.n_neurons < 2 or cfg.n_timepoints <= cfg.max_lag or cfg.max_lag < 1:
        raise ValueError("synthetic dimensions require N>=2, max_lag>=1, and T>max_lag")
    if cfg.n_states < 1:
        raise ValueError("n_states must be at least one")
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
    trajectory_rng = np.random.default_rng(cfg.seed)
    graph_rng = np.random.default_rng(cfg.seed if cfg.graph_seed is None else cfg.graph_seed)
    endpoint_rng = np.random.default_rng(
        cfg.seed + 1 if cfg.endpoint_seed is None else cfg.endpoint_seed
    )
    n, t = cfg.n_neurons, cfg.n_timepoints
    certified_burn = int(np.ceil(np.log(1e-6) / np.log(cfg.contraction)))
    burn_in = max(1000, cfg.burn_in, certified_burn)
    total = burn_in + t

    (
        lag_graphs,
        instantaneous_graphs,
        companions,
        certificate,
        base_lags,
        sign_flip_masks,
    ) = _coefficient_family(cfg, graph_rng)
    state_seq_full, temporal_weights_full = _temporal_weights(cfg, total, trajectory_rng)

    latent_factors_full = np.zeros((total, cfg.n_latent_confounders), dtype=np.float64)
    latent_loadings = np.zeros((n, cfg.n_latent_confounders), dtype=np.float64)
    if cfg.n_latent_confounders:
        latent_loadings = graph_rng.normal(
            0.0,
            cfg.latent_strength / np.sqrt(cfg.n_latent_confounders),
            size=(n, cfg.n_latent_confounders),
        )
        latent_factors_full[0] = trajectory_rng.normal(size=cfg.n_latent_confounders)
        for tt in range(1, total):
            latent_factors_full[tt] = 0.8 * latent_factors_full[tt - 1] + trajectory_rng.normal(
                0.0, 0.6, size=cfg.n_latent_confounders
            )

    innovation_covariance = cfg.noise_std**2 * (
        (1.0 - cfg.innovation_correlation) * np.eye(n)
        + cfg.innovation_correlation * np.ones((n, n))
    )
    x_full = np.zeros((total, n), dtype=np.float64)
    x_full[: cfg.max_lag] = trajectory_rng.multivariate_normal(
        np.zeros(n), innovation_covariance, size=cfg.max_lag
    )
    identity = np.eye(n)
    for tt in range(cfg.max_lag, total):
        weights = temporal_weights_full[tt]
        effective_lags = np.tensordot(weights, lag_graphs, axes=(0, 0))
        effective_instantaneous = np.tensordot(weights, instantaneous_graphs, axes=(0, 0))
        lagged = np.zeros(n, dtype=float)
        for lag, graph in enumerate(effective_lags, start=1):
            lagged += graph.T @ x_full[tt - lag]
        common = (
            latent_loadings @ latent_factors_full[tt]
            if cfg.n_latent_confounders
            else 0.0
        )
        state_fraction = float(state_seq_full[tt]) / max(1, cfg.n_states - 1)
        variance_scale = np.sqrt(1.0 + state_fraction * (cfg.variance_ratio - 1.0))
        noise = trajectory_rng.multivariate_normal(np.zeros(n), innovation_covariance)
        innovation = lagged + common + variance_scale * noise
        if cfg.nonlinear_saturation:
            innovation = np.tanh(innovation)
        # For B[source,target], x = B.T x + innovation.  B is a DAG, so
        # (I-B.T) is nonsingular and the contemporaneous solution is exact.
        x_full[tt] = np.linalg.solve(identity - effective_instantaneous.T, innovation)

    x = x_full[burn_in:]
    state_seq = state_seq_full[burn_in:]
    temporal_weights = temporal_weights_full[burn_in:]
    latent_factors = latent_factors_full[burn_in:]

    # Behavior:
    #   * 'motif' is tied to the connectivity *regime* (a subset of regimes is 'active'),
    #     so the inferred connectivity state carries genuine behavioral information.
    #   * 'continuous' tracks the mean activity of a fixed driver set (fast, activity-linked).
    active_states = set(range(int(np.ceil(cfg.n_states / 2))))
    behavior_motif = np.array([1 if s in active_states else 0 for s in state_seq], dtype=np.int64)
    flip = endpoint_rng.random(t) < 0.03  # small label noise
    behavior_motif = np.where(flip, 1 - behavior_motif, behavior_motif)

    drivers = endpoint_rng.choice(n, size=cfg.behavior_drivers, replace=False)
    behavior_continuous = x[:, drivers].mean(axis=1) + endpoint_rng.normal(0, 0.1, size=t)

    observed_full = x_full
    calcium_decay = np.zeros(n, dtype=float)
    calcium_gain = np.ones(n, dtype=float)
    if cfg.calcium_decay_range is not None:
        low, high = cfg.calcium_decay_range
        calcium_decay = graph_rng.uniform(low, high, size=n)
        if cfg.calcium_gain_heterogeneity > 0:
            calcium_gain = np.exp(
                graph_rng.normal(0.0, cfg.calcium_gain_heterogeneity, size=n)
            )
        observed_full = np.zeros_like(x_full)
        observed_full[0] = calcium_gain * x_full[0]
        for tt in range(1, total):
            observed_full[tt] = (
                calcium_decay * observed_full[tt - 1] + calcium_gain * x_full[tt]
            )
    observed = observed_full[burn_in:].copy()
    if cfg.measurement_snr is not None:
        signal_variance = np.var(observed, axis=0)
        noise_scale = np.sqrt(signal_variance / cfg.measurement_snr)
        observed += endpoint_rng.normal(size=observed.shape) * noise_scale
    observation_mask = np.ones_like(observed, dtype=bool)
    if cfg.missing_fraction:
        observation_mask = endpoint_rng.random(observed.shape) >= cfg.missing_fraction
        observed[~observation_mask] = np.nan

    traces = observed.T.astype(np.float32)  # N x T
    coords = graph_rng.normal(0, 1, size=(n, 3)).astype(np.float32)
    neuron_ids = np.arange(n)
    clock = np.asarray(cfg.native_time) if cfg.native_time is not None else np.arange(t) / cfg.fps

    return NeuralRecording(
        traces=traces,
        time=clock,
        coords=coords,
        neuron_ids=neuron_ids,
        behavior={"continuous": behavior_continuous, "motif": behavior_motif},
        fps=cfg.fps,
        metadata={
            "source": "synthetic",
            "true_graphs": np.sum(lag_graphs, axis=1),
            "true_lag_graphs": lag_graphs,
            "true_instantaneous_graphs": instantaneous_graphs,
            "true_states": state_seq,
            "temporal_weights": temporal_weights,
            "regime_sign_flip_masks": np.asarray(sign_flip_masks),
            "base_lag_graphs": base_lags,
            "effective_companions": companions,
            "stability_certificate": certificate,
            "burn_in": burn_in,
            "latent_factors": latent_factors,
            "latent_loadings": latent_loadings,
            "latent_neural_activity": x,
            "observation_mask": observation_mask,
            "calcium_decay": calcium_decay,
            "calcium_gain": calcium_gain,
            "behavior_drivers": drivers,
            "truth_contract": {
                "latent_coefficients": "true_lag_graphs",
                "instantaneous_structure": "true_instantaneous_graphs",
                "observed_projection": "unavailable",
            },
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
