"""Designed raw-process controls for shared-manifold, cross-method validation.

Truth stays in recording metadata and is never supplied as a behavioral endpoint.
These linear Gaussian examples test inference and visualization contracts; they
are not calcium observation models or biological validation experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from effectome.data_module import PreprocessConfig, WindowConfig, make_windows, preprocess
from effectome.data_module.schema import NeuralRecording, RecordingIdentity, WindowedSegments

TRANSITION_EXAMPLE_SCENARIOS = ("one_to_one", "matched_marginals", "fixed_circuit", "stationary")


@dataclass(frozen=True)
class TransitionExampleConfig:
    """Declared simulation parameters, expressed in samples unless explicitly seconds.

    ``matched_marginals`` uses W = r R and innovation covariance (1-r²) I.
    A stationary N(0,I) initialization therefore gives the same marginal at every
    sample under either circuit, even when the circuit switches. The conditional
    dynamics and temporal correlations still differ.

    ``fixed_circuit`` uses one stable matrix with alternating initial conditions
    across independent trials. Those resets are explicit temporal boundaries.
    """

    scenario: str = "matched_marginals"
    n_samples: int = 1536
    n_neurons: int = 4
    regime_dwell: int = 128
    trial_length: int = 256
    contraction: float = 0.85
    rotation_angle: float = 0.55
    fixed_contraction: float = 0.995
    noise_std: float = 0.2
    fps: float = 10.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.scenario not in TRANSITION_EXAMPLE_SCENARIOS:
            raise ValueError(f"unknown transition example scenario: {self.scenario}")
        if self.n_samples < 3 or self.n_neurons < 2:
            raise ValueError("examples require at least three samples and two neurons")
        if self.regime_dwell < 2 or self.trial_length < 2:
            raise ValueError("regime_dwell and trial_length must be at least two")
        if not 0 < self.contraction < 1 or not 0 < self.fixed_contraction < 1:
            raise ValueError("contractions must lie strictly between zero and one")
        if not np.isfinite(self.rotation_angle) or not 0 < abs(self.rotation_angle) < np.pi:
            raise ValueError("rotation_angle must be finite, nonzero, and smaller than pi in magnitude")
        if not np.isfinite(self.noise_std) or self.noise_std <= 0:
            raise ValueError("noise_std must be finite and positive")
        if not np.isfinite(self.fps) or self.fps <= 0 or self.seed < 0:
            raise ValueError("fps must be finite and positive; seed must be nonnegative")


def _rotation(n_neurons: int, angle: float) -> np.ndarray:
    rotation = np.eye(n_neurons)
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation[:2, :2] = ((cosine, -sine), (sine, cosine))
    return rotation


def make_transition_example(config: TransitionExampleConfig | None = None) -> NeuralRecording:
    """Generate one recording with source-to-target VAR coefficients in metadata.

    Except at declared trial resets, row-vector observations follow
    ``x[t+1] = x[t] @ W[regime[t]] + intercept[regime[t]] + innovation``.
    """
    cfg = config or TransitionExampleConfig()
    rng = np.random.default_rng(cfg.seed)
    n, total = cfg.n_neurons, cfg.n_samples
    if cfg.scenario in {"fixed_circuit", "stationary"}:
        contraction = cfg.fixed_contraction if cfg.scenario == "fixed_circuit" else cfg.contraction
        coefficients = (contraction * _rotation(n, cfg.rotation_angle))[None]
        regimes = np.zeros(total, dtype=int)
        means = np.zeros((1, n))
        innovation_covariances = (cfg.noise_std**2 * np.eye(n))[None]
        ranges = [
            (start, min(total, start + cfg.trial_length))
            for start in range(0, total, cfg.trial_length)
        ] if cfg.scenario == "fixed_circuit" else [(0, total)]
        trial_ids = (
            np.arange(total) // cfg.trial_length if cfg.scenario == "fixed_circuit" else np.zeros(total, int)
        )
        initial_classes = trial_ids % 2
    else:
        coefficients = np.stack([
            cfg.contraction * _rotation(n, cfg.rotation_angle),
            cfg.contraction * _rotation(n, -cfg.rotation_angle),
        ])
        regimes = (np.arange(total) // cfg.regime_dwell) % 2
        means = np.zeros((2, n))
        if cfg.scenario == "one_to_one":
            means[:, 0] = (-2.0, 2.0)
            covariance = cfg.noise_std**2 * np.eye(n)
        else:
            covariance = (1 - cfg.contraction**2) * np.eye(n)
        innovation_covariances = np.repeat(covariance[None], 2, axis=0)
        ranges = [(0, total)]
        trial_ids = np.zeros(total, dtype=int)
        initial_classes = np.zeros(total, dtype=int)
    intercepts = np.einsum("si,sij->sj", means, np.eye(n)[None] - coefficients)
    covariances = np.stack([
        solve_discrete_lyapunov(matrix.T, covariance)
        for matrix, covariance in zip(coefficients, innovation_covariances, strict=True)
    ])
    values = np.empty((total, n), dtype=float)
    for start, stop in ranges:
        if cfg.scenario == "fixed_circuit":
            values[start] = np.zeros(n)
            values[start, 0] = 4.0 if initial_classes[start] == 0 else -4.0
        else:
            state = int(regimes[start])
            values[start] = rng.multivariate_normal(means[state], covariances[state])
        for sample in range(start + 1, stop):
            state = int(regimes[sample - 1])
            innovation = rng.multivariate_normal(np.zeros(n), innovation_covariances[state])
            values[sample] = values[sample - 1] @ coefficients[state] + intercepts[state] + innovation
    recording = NeuralRecording(
        traces=values.T,
        time=np.arange(total, dtype=float) / cfg.fps,
        coords=None,
        neuron_ids=np.arange(n),
        behavior={},
        fps=cfg.fps,
        identity=RecordingIdentity(
            dataset="synthetic_transition_examples",
            dataset_id="synthetic-transition-examples",
            recording_id=f"{cfg.scenario}-seed-{cfg.seed}",
            animal_id=None,
        ),
        metadata={
            "transition_example": asdict(cfg),
            "true_coefficients": coefficients,
            "true_regime": regimes,
            "true_intercepts": intercepts,
            "true_innovation_covariances": innovation_covariances,
            "true_stationary_means": means,
            "true_stationary_covariances": covariances,
            "true_trial_ids": trial_ids,
            "true_initial_condition_class": initial_classes,
            "valid_ranges": ranges,
            "coefficient_convention": "x[t+1] = x[t] @ W[regime[t]] + intercept + innovation",
            "truth_scope": "metadata_only_not_predictors_or_behavior",
            "observation_model": "direct_linear_gaussian_neural_activity_no_calcium_filter",
            "identical_stationary_marginals": cfg.scenario == "matched_marginals",
            "claim_boundary": (
                "Planted examples only. Marginal occupancy equality does not imply equal temporal dynamics; "
                "different fixed-circuit trajectories can be explained by initial activity/history."
            ),
        },
    )
    recording.validate()
    return recording


def make_transition_example_windows(
    recording: NeuralRecording,
    *,
    history_length: int = 64,
    target_length: int = 8,
    stride: int = 16,
) -> WindowedSegments:
    """Create identity-preprocessed anchored windows without leaking truth labels.

    Trial resets are recorded in ``valid_ranges`` by the generator. Windowing
    excludes histories crossing a reset and marks corresponding gap boundaries.
    """
    return make_windows(
        preprocess(recording, PreprocessConfig()),
        WindowConfig(
            mode="temporal",
            history_length=history_length,
            target_length=target_length,
            stride=stride,
            standardize_per_window=False,
        ),
    )
