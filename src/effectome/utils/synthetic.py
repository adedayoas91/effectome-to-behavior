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

import hashlib
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
    missingness_mode: str = "independent"
    missing_block_length: int = 20
    nonlinear_saturation: bool = False
    observed_projection_replicates: int = 0
    observed_projection_samples: int = 0
    observed_projection_ridge: float = 1.0e-6
    observed_projection_tolerance: float = 0.1
    endpoint_mode: str = "linked"
    conditional_null_m0_variables: tuple[str, ...] = ("activity_mean",)
    conditional_null_noise_std: float = 1.0
    cross_coefficient_range: tuple[float, float] | None = None
    self_lag_coefficients: tuple[float, ...] | None = None
    calcium_kernel_seconds: float | None = None
    calcium_kernel_truncation: float = 5.0
    calcium_kernel_decay_range_seconds: tuple[float, float] | None = None
    measurement_change_ratio: float = 1.0


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
    if cfg.cross_coefficient_range is not None:
        low, high = cfg.cross_coefficient_range
        support = base != 0
        base = rng.uniform(low, high, size=base.shape) * np.sign(base) * support
    if cfg.self_lag_coefficients is not None:
        for lag, value in enumerate(cfg.self_lag_coefficients):
            np.fill_diagonal(base[lag], value)
    instant = _instantaneous_dag(n, cfg.instantaneous_density, cfg.instantaneous_strength, rng)
    # Certify the base against a stricter target so exact support removals and sign
    # changes have room inside the registered family-wide contraction.  Scaling is
    # restricted to lag coefficients; the multi-lag companion shift rows stay intact.
    base_contraction = 0.8 * cfg.contraction
    base_rejections = 0
    for _ in range(100):
        try:
            certificate = certify_common_metric(np.array([effective_companion(base, instant)]),
                                                rho=base_contraction)
            break
        except ValueError:
            base_rejections += 1
            base *= 0.8
    else:
        raise ValueError("could not certify base at the fixed target contraction")
    proposals: list[np.ndarray] = []
    proposed_edits: list[dict[str, np.ndarray]] = []
    off_diagonal = np.ones((p, n, n), dtype=bool)
    for lag in range(p):
        np.fill_diagonal(off_diagonal[lag], False)
    for state in range(cfg.n_states):
        candidate = base.copy()
        base_support = base != 0
        remove = (
            (rng.random(base.shape) < cfg.regime_sparsity_jitter / 2.0)
            & base_support
            & off_diagonal
        )
        add = (
            (rng.random(base.shape) < cfg.regime_sparsity_jitter / 2.0)
            & ~base_support
            & off_diagonal
        )
        candidate[remove] = 0.0
        additions = (
            (cfg.coupling / p)
            * (0.5 + rng.random(base.shape))
            * rng.choice([-1.0, 1.0], size=base.shape)
        )
        candidate[add] = additions[add]
        sign = (
            (rng.random(base.shape) < cfg.regime_sign_flip_fraction)
            & (candidate != 0)
            & off_diagonal
            & ~add
        )
        candidate[sign] *= -1.0
        magnitude = (candidate != 0) & ~sign & ~add
        candidate[magnitude] *= rng.uniform(0.65, 1.35, size=int(magnitude.sum()))
        if state == 0 or cfg.scenario == "constant":
            candidate = base.copy()
            add[:] = False
            remove[:] = False
            sign[:] = False
            magnitude[:] = False
        proposals.append(candidate)
        proposed_edits.append(
            {"support_added": add, "support_removed": remove, "sign": sign, "magnitude": magnitude}
        )
    proposal_array = np.asarray(proposals)
    for family_scale_rejections in range(100):
        family_scale = 0.8**family_scale_rejections
        family_base = base * family_scale
        family_proposals = proposal_array * family_scale
        try:
            base_certificate = certify_common_metric(
                np.array([effective_companion(family_base, instant)]),
                rho=base_contraction,
            )
        except ValueError:
            continue
        for shrink in range(100):
            scale = 0.8**shrink
            graphs = family_base + (family_proposals - family_base) * scale
            for state, edits in enumerate(proposed_edits):
                graphs[state][edits["support_removed"]] = 0.0
                graphs[state][edits["support_added"]] = family_proposals[state][
                    edits["support_added"]
                ] * scale
                sign_mask = edits["sign"]
                graphs[state][sign_mask] = (
                    -np.sign(family_base[sign_mask])
                    * np.abs(family_proposals[state][sign_mask])
                    * scale
                )
            companions = np.array([effective_companion(g, instant) for g in graphs])
            try:
                certificate = certify_common_metric(
                    companions,
                    rho=cfg.contraction,
                    metric=base_certificate["metric"],
                )
                certificate["base_contraction"] = base_contraction
                certificate["family_scale"] = family_scale
                certificate["perturbation_scale"] = scale
                certificate["base_scale_rejections"] = base_rejections
                certificate["family_scale_rejections"] = family_scale_rejections
                certificate["perturbation_rejections"] = shrink
                edit_masks = {
                    "support_added": (graphs != 0) & (family_base[None] == 0),
                    "support_removed": (graphs == 0) & (family_base[None] != 0),
                    "sign": (
                        (graphs != 0)
                        & (family_base[None] != 0)
                        & (np.sign(graphs) != np.sign(family_base[None]))
                    ),
                    "magnitude": (
                        (graphs != 0)
                        & (family_base[None] != 0)
                        & (np.sign(graphs) == np.sign(family_base[None]))
                        & ~np.isclose(
                            np.abs(graphs),
                            np.abs(family_base[None]),
                            rtol=1e-10,
                            atol=1e-12,
                        )
                    ),
                }
                return (
                    graphs,
                    np.repeat(instant[None], cfg.n_states, axis=0),
                    companions,
                    certificate,
                    family_base,
                    edit_masks,
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
    if cfg.missingness_mode not in {"independent", "burst", "gap"}:
        raise ValueError("missingness_mode must be independent, burst, or gap")
    if cfg.missing_block_length < 1:
        raise ValueError("missing_block_length must be positive")
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
    if cfg.cross_coefficient_range is not None:
        low, high = cfg.cross_coefficient_range
        if not 0 < low <= high:
            raise ValueError("cross coefficient range must be positive and ordered")
    if cfg.self_lag_coefficients is not None and len(cfg.self_lag_coefficients) != cfg.max_lag:
        raise ValueError("self lag coefficients must match max_lag")
    if cfg.calcium_kernel_seconds is not None and cfg.calcium_kernel_seconds <= 0:
        raise ValueError("calcium kernel seconds must be positive")
    if cfg.calcium_kernel_truncation <= 0 or cfg.measurement_change_ratio <= 0:
        raise ValueError("kernel truncation and measurement change ratio must be positive")
    if cfg.calcium_kernel_decay_range_seconds is not None:
        low, high = cfg.calcium_kernel_decay_range_seconds
        if not 0 < low <= high:
            raise ValueError("calcium kernel decay range must be positive and ordered")
    if cfg.n_states < 1:
        raise ValueError("n_states must be at least one")
    if cfg.observed_projection_replicates < 0 or cfg.observed_projection_samples < 0:
        raise ValueError("observed projection sizes must be non-negative")
    if (cfg.observed_projection_replicates == 0) != (cfg.observed_projection_samples == 0):
        raise ValueError("observed projection replicates and samples must both be zero or positive")
    if cfg.observed_projection_replicates == 1:
        raise ValueError("observed projection requires at least two independent replicates")
    if cfg.observed_projection_samples and cfg.observed_projection_samples <= cfg.max_lag:
        raise ValueError("observed projection samples must exceed max_lag")
    if cfg.observed_projection_ridge < 0 or cfg.observed_projection_tolerance <= 0:
        raise ValueError("invalid observed projection ridge or tolerance")
    if cfg.endpoint_mode not in {"linked", "conditional_null"}:
        raise ValueError("endpoint_mode must be linked or conditional_null")
    allowed_m0 = {"activity_mean", "activity_variance"}
    if not cfg.conditional_null_m0_variables or not set(cfg.conditional_null_m0_variables) <= allowed_m0:
        raise ValueError("conditional-null M0 variables must be declared supported variables")
    if cfg.conditional_null_noise_std <= 0:
        raise ValueError("conditional_null_noise_std must be positive")
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


def _make_observation_mask(
    shape: tuple[int, int],
    *,
    fraction: float,
    mode: str,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate point, neuron-local burst, or recording-wide gap missingness."""
    timepoints, neurons = shape
    mask = np.ones(shape, dtype=bool)
    if fraction == 0:
        return mask
    if mode == "independent":
        return rng.random(shape) >= fraction
    if mode == "gap":
        target_timepoints = max(1, int(np.ceil(fraction * timepoints)))
        n_blocks = max(1, int(np.ceil(target_timepoints / block_length)))
        for _ in range(n_blocks):
            start = int(rng.integers(0, max(1, timepoints - block_length + 1)))
            mask[start : min(timepoints, start + block_length), :] = False
        return mask
    target_values = max(1, int(np.ceil(fraction * timepoints * neurons)))
    n_blocks = max(1, int(np.ceil(target_values / block_length)))
    for _ in range(n_blocks):
        neuron = int(rng.integers(0, neurons))
        start = int(rng.integers(0, max(1, timepoints - block_length + 1)))
        mask[start : min(timepoints, start + block_length), neuron] = False
    return mask


def _fully_missing_intervals(observation_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return maximal half-open intervals with no observed neuron."""
    fully_missing = ~np.asarray(observation_mask, dtype=bool).any(axis=1)
    padded = np.pad(fully_missing.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def _calcium_observation(
    samples: np.ndarray, cfg: SyntheticConfig, decay: np.ndarray, gain: np.ndarray
) -> np.ndarray:
    if cfg.calcium_kernel_seconds is not None:
        observed = np.empty_like(samples)
        for neuron in range(samples.shape[1]):
            tau_samples = -1.0 / np.log(decay[neuron])
            length = max(1, int(np.ceil(cfg.calcium_kernel_truncation * tau_samples)))
            kernel = decay[neuron] ** np.arange(length)
            kernel /= kernel.sum()
            observed[:, neuron] = np.convolve(samples[:, neuron], gain[neuron] * kernel)[:len(samples)]
        return observed
    if cfg.calcium_decay_range is None:
        return samples.copy()
    observed = np.zeros_like(samples)
    observed[0] = gain * samples[0]
    for tt in range(1, len(samples)):
        observed[tt] = decay * observed[tt - 1] + gain * samples[tt]
    return observed


def _observed_projection_truth(
    cfg: SyntheticConfig,
    *,
    lag_graphs: np.ndarray,
    instantaneous_graphs: np.ndarray,
    innovation_covariance: np.ndarray,
    calcium_decay: np.ndarray,
    calcium_gain: np.ndarray,
    latent_loadings: np.ndarray,
    seeds: list[int],
) -> dict:
    """Estimate state-conditional finite-lag observation projections on fresh trajectories."""
    n = cfg.n_neurons
    p = cfg.max_lag
    burn = max(1000, int(np.ceil(np.log(1e-6) / np.log(cfg.contraction))))
    coefficients = np.empty(
        (len(seeds), cfg.n_states, p, n, n), dtype=np.float64
    )
    innovation_cholesky = np.linalg.cholesky(innovation_covariance)
    for replicate, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        for state in range(cfg.n_states):
            total = burn + cfg.observed_projection_samples + p
            latent = np.zeros((total, n), dtype=np.float64)
            factors = np.zeros((total, cfg.n_latent_confounders), dtype=np.float64)
            loadings = latent_loadings
            for tt in range(p, total):
                if cfg.n_latent_confounders:
                    factors[tt] = 0.8 * factors[tt - 1] + rng.normal(
                        0.0, 0.6, size=cfg.n_latent_confounders
                    )
                lagged = sum(
                    (lag_graphs[state, lag - 1].T @ latent[tt - lag] for lag in range(1, p + 1)),
                    start=np.zeros(n, dtype=float),
                )
                common = loadings @ factors[tt] if cfg.n_latent_confounders else 0.0
                noise = innovation_cholesky @ rng.normal(size=n)
                innovation = lagged + common + noise
                if cfg.nonlinear_saturation:
                    innovation = np.tanh(innovation)
                latent[tt] = (
                    np.linalg.solve(np.eye(n) - instantaneous_graphs[state].T, innovation)
                    if np.any(instantaneous_graphs[state]) else innovation
                )
            observed = _calcium_observation(latent, cfg, calcium_decay, calcium_gain)
            if cfg.measurement_snr is not None:
                variance = np.var(observed[burn:], axis=0)
                observed += rng.normal(size=observed.shape) * np.sqrt(
                    variance / cfg.measurement_snr
                )
            retained = observed[burn:]
            response = retained[p:]
            predictors = np.concatenate(
                [retained[p - lag : -lag] for lag in range(1, p + 1)], axis=1
            )
            gram = predictors.T @ predictors
            ridge = cfg.observed_projection_ridge * np.eye(gram.shape[0])
            fitted = np.linalg.solve(gram + ridge, predictors.T @ response)
            coefficients[replicate, state] = fitted.reshape(p, n, n)
    mean = coefficients.mean(axis=0)
    standard_error = coefficients.std(axis=0, ddof=1) / np.sqrt(len(seeds))
    mcse = float(np.max(standard_error))
    truth_id = "observed-projection-" + hashlib.sha256(
        np.ascontiguousarray(mean).view(np.uint8).tobytes()
    ).hexdigest()[:16]
    return {
        "truth_id": truth_id,
        "coefficients": mean,
        "replicate_coefficients": coefficients,
        "replicate_count": len(seeds),
        "samples_per_replicate": cfg.observed_projection_samples,
        "replicate_design": "independent_innovations",
        "projection_scope": "state_conditional_finite_lag",
        "predictor_definition": "raw_observed_levels_lag_1_to_p",
        "scale_definition": "native_observation_scale",
        "regularization": {"kind": "ridge", "value": cfg.observed_projection_ridge},
        "monte_carlo_standard_error": mcse,
        "convergence_tolerance": cfg.observed_projection_tolerance,
        "converged": bool(mcse <= cfg.observed_projection_tolerance),
        "seed_ids": tuple(f"projection-{seed}" for seed in seeds),
    }


def _finite_rollout_diagnostics(
    history: np.ndarray,
    *,
    lag_graphs: np.ndarray,
    instantaneous_graphs: np.ndarray,
    temporal_weights: np.ndarray,
    nonlinear_saturation: bool,
) -> dict:
    """Check a deterministic, noise-free continuation for non-finite growth."""
    p = lag_graphs.shape[1]
    n = lag_graphs.shape[2]
    horizon = max(100, min(1000, temporal_weights.shape[0]))
    rollout = [np.asarray(row, dtype=float).copy() for row in history[-p:]]
    for step in range(horizon):
        weights = temporal_weights[min(step, temporal_weights.shape[0] - 1)]
        lags = np.tensordot(weights, lag_graphs, axes=(0, 0))
        instantaneous = np.tensordot(weights, instantaneous_graphs, axes=(0, 0))
        innovation = np.zeros(n, dtype=float)
        for lag, graph in enumerate(lags, start=1):
            innovation += graph.T @ rollout[-lag]
        if nonlinear_saturation:
            innovation = np.tanh(innovation)
        rollout.append(np.linalg.solve(np.eye(n) - instantaneous.T, innovation))
    values = np.asarray(rollout[p:])
    return {
        "finite": bool(np.all(np.isfinite(values))),
        "horizon": horizon,
        "max_absolute_value": float(np.nanmax(np.abs(values))) if values.size else 0.0,
        "method": "deterministic_noise_free_continuation",
    }


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
    endpoint_seed = cfg.seed + 1 if cfg.endpoint_seed is None else cfg.endpoint_seed
    endpoint_streams = np.random.SeedSequence(endpoint_seed).spawn(2)
    endpoint_rng = np.random.default_rng(endpoint_streams[0])
    conditional_null_rng = np.random.default_rng(endpoint_streams[1])
    graph_seed = cfg.seed if cfg.graph_seed is None else cfg.graph_seed
    graph_id = f"graph-{graph_seed}"
    trajectory_id = f"trajectory-{cfg.seed}"
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
        edit_masks,
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
    innovation_cholesky = np.linalg.cholesky(innovation_covariance)
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
        state_fraction = (
            float(tt >= burn_in + t // 2)
            if cfg.scenario == "constant"
            else float(state_seq_full[tt]) / max(1, cfg.n_states - 1)
        )
        variance_scale = np.sqrt(1.0 + state_fraction * (cfg.variance_ratio - 1.0))
        noise = innovation_cholesky @ trajectory_rng.normal(size=n)
        innovation = lagged + common + variance_scale * noise
        if cfg.nonlinear_saturation:
            innovation = np.tanh(innovation)
        # For B[source,target], x = B.T x + innovation.  B is a DAG, so
        # (I-B.T) is nonsingular and the contemporaneous solution is exact.
        x_full[tt] = (
            np.linalg.solve(identity - effective_instantaneous.T, innovation)
            if np.any(effective_instantaneous) else innovation
        )

    x = x_full[burn_in:]
    state_seq = state_seq_full[burn_in:]
    temporal_weights = temporal_weights_full[burn_in:]
    latent_factors = latent_factors_full[burn_in:]
    trajectory_diagnostics = {
        "finite": bool(np.all(np.isfinite(x))),
        "max_absolute_value": float(np.nanmax(np.abs(x))),
        "samples": int(x.shape[0]),
    }
    if not trajectory_diagnostics["finite"]:
        raise ValueError("synthetic trajectory contains non-finite values")
    rollout_diagnostics = _finite_rollout_diagnostics(
        x,
        lag_graphs=lag_graphs,
        instantaneous_graphs=instantaneous_graphs,
        temporal_weights=temporal_weights,
        nonlinear_saturation=cfg.nonlinear_saturation,
    )
    if not rollout_diagnostics["finite"]:
        raise ValueError("synthetic deterministic rollout contains non-finite values")

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
    if cfg.calcium_kernel_seconds is not None:
        taus = np.full(n, cfg.calcium_kernel_seconds)
        if cfg.calcium_kernel_decay_range_seconds is not None:
            low, high = cfg.calcium_kernel_decay_range_seconds
            taus = np.exp(graph_rng.uniform(np.log(low), np.log(high), size=n))
        calcium_decay = np.exp(-1.0 / (cfg.fps * taus))
    if cfg.calcium_decay_range is not None:
        low, high = cfg.calcium_decay_range
        calcium_decay = graph_rng.uniform(low, high, size=n)
        if cfg.calcium_gain_heterogeneity > 0:
            calcium_gain = np.exp(
                graph_rng.normal(0.0, cfg.calcium_gain_heterogeneity, size=n)
            )
    observed_full = _calcium_observation(x_full, cfg, calcium_decay, calcium_gain)
    observed = observed_full[burn_in:].copy()
    if cfg.measurement_snr is not None:
        signal_variance = np.var(observed, axis=0)
        noise_scale = np.sqrt(signal_variance / cfg.measurement_snr)
        measurement_noise = endpoint_rng.normal(size=observed.shape) * noise_scale
        measurement_noise[t // 2:] *= np.sqrt(cfg.measurement_change_ratio)
        observed += measurement_noise
    observed_for_endpoint = observed.copy()
    observation_mask = _make_observation_mask(
        (observed.shape[0], observed.shape[1]),
        fraction=cfg.missing_fraction,
        mode=cfg.missingness_mode,
        block_length=cfg.missing_block_length,
        rng=endpoint_rng,
    )
    observed[~observation_mask] = np.nan
    gap_intervals = (
        _fully_missing_intervals(observation_mask)
        if cfg.missingness_mode == "gap"
        else []
    )

    endpoint_contracts: dict[str, dict] = {
        "motif": {
            "kind": "regime_linked",
            "generating_function": "active_regime_indicator_plus_label_noise",
        },
        "continuous": {
            "kind": "activity_linked",
            "generating_function": "driver_mean_plus_fresh_gaussian_noise",
        },
    }
    behavior = {"continuous": behavior_continuous, "motif": behavior_motif}
    if cfg.endpoint_mode == "conditional_null":
        m0_values = {
            "activity_mean": np.mean(observed_for_endpoint, axis=1),
            "activity_variance": np.var(observed_for_endpoint, axis=1),
        }
        design = np.column_stack(
            [m0_values[name] for name in cfg.conditional_null_m0_variables]
        )
        weights = np.linspace(0.5, 1.0, design.shape[1], dtype=float)
        fresh_noise = conditional_null_rng.normal(
            0.0, cfg.conditional_null_noise_std, size=t
        )
        behavior["conditional_null"] = design @ weights + fresh_noise
        endpoint_contracts["conditional_null"] = {
            "kind": "conditional_graph_null",
            "declared_m0_variables": cfg.conditional_null_m0_variables,
            "generating_function": "linear_m0_plus_fresh_gaussian_noise",
            "coefficients": weights,
            "noise_std": cfg.conditional_null_noise_std,
            "noise_stream": f"endpoint-null-{endpoint_seed}",
            "independent_of_graph_and_trajectory_streams": True,
        }

    observed_projection: dict | None = None
    if cfg.observed_projection_replicates:
        projection_sequences = np.random.SeedSequence(
            [endpoint_seed, graph_seed, cfg.seed, 9_103]
        ).spawn(cfg.observed_projection_replicates)
        projection_seeds = [
            int(sequence.generate_state(1, dtype=np.uint32)[0])
            for sequence in projection_sequences
        ]
        observed_projection = _observed_projection_truth(
            cfg,
            lag_graphs=lag_graphs,
            instantaneous_graphs=instantaneous_graphs,
            innovation_covariance=innovation_covariance,
            calcium_decay=calcium_decay,
            calcium_gain=calcium_gain,
            latent_loadings=latent_loadings,
            seeds=projection_seeds,
        )

    traces = observed.T.astype(np.float32)  # N x T
    coords = graph_rng.normal(0, 1, size=(n, 3)).astype(np.float32)
    neuron_ids = np.arange(n)
    clock = np.asarray(cfg.native_time) if cfg.native_time is not None else np.arange(t) / cfg.fps

    return NeuralRecording(
        traces=traces,
        time=clock,
        coords=coords,
        neuron_ids=neuron_ids,
        behavior=behavior,
        fps=cfg.fps,
        metadata={
            "source": "synthetic",
            "graph_id": graph_id,
            "trajectory_id": trajectory_id,
            "true_graphs": np.sum(lag_graphs, axis=1),
            "true_lag_graphs": lag_graphs,
            "true_instantaneous_graphs": instantaneous_graphs,
            "true_states": state_seq,
            "temporal_weights": temporal_weights,
            "regime_sign_flip_masks": edit_masks["sign"],
            "coefficient_edit_masks": edit_masks,
            "base_lag_graphs": base_lags,
            "effective_companions": companions,
            "stability_certificate": certificate,
            "generator_rejections": {
                "base_scale": certificate["base_scale_rejections"],
                "family_scale": certificate["family_scale_rejections"],
                "perturbation_scale": certificate["perturbation_rejections"],
                "total": certificate["base_scale_rejections"]
                + certificate["family_scale_rejections"]
                + certificate["perturbation_rejections"],
            },
            "burn_in": burn_in,
            "trajectory_diagnostics": trajectory_diagnostics,
            "rollout_diagnostics": rollout_diagnostics,
            "latent_factors": latent_factors,
            "latent_loadings": latent_loadings,
            "latent_neural_activity": x,
            "observation_mask": observation_mask,
            "gap_intervals": gap_intervals,
            "missingness": {
                "mode": cfg.missingness_mode,
                "requested_fraction": cfg.missing_fraction,
                "realized_fraction": float(1.0 - observation_mask.mean()),
                "block_length": cfg.missing_block_length,
            },
            "calcium_decay": calcium_decay,
            "calcium_gain": calcium_gain,
            "behavior_drivers": drivers,
            "endpoint_contracts": endpoint_contracts,
            "observed_projection_truth": observed_projection,
            "truth_contract": {
                "latent_coefficients": "true_lag_graphs",
                "instantaneous_structure": "true_instantaneous_graphs",
                "observed_projection": (
                    "unavailable"
                    if observed_projection is None
                    else observed_projection["truth_id"]
                ),
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
                "variance_ratio": cfg.variance_ratio,
                "measurement_change_ratio": cfg.measurement_change_ratio,
                "calcium_kernel_seconds": cfg.calcium_kernel_seconds,
                "calcium_kernel_decay_range_seconds": cfg.calcium_kernel_decay_range_seconds,
            },
        },
        neural_validity=observation_mask.T,
    )
