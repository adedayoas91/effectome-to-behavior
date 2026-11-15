"""Structural driver benchmarks with paired simulator-mechanism interventions.

The bistable population coordinate is a simulator state, not an inferred neural
regime. Neurons 0/1 can drive it, neuron 2 can be a correlated readout, and neurons
3..N-2 observe it. Disconnecting an outgoing influence changes the generating
equation; it does not mask predictor features or censor population measurements.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np

from effectome.data_module.schema import NeuralRecording, RecordingIdentity

DRIVER_BENCHMARK_SCENARIOS = (
    "single_driver",
    "redundant_group",
    "correlated_bystander",
    "null",
    "hidden_confounder",
)


@dataclass(frozen=True)
class DriverBenchmarkConfig:
    """Declared finite-step SCM parameters; no continuous-time accuracy is claimed."""

    scenario: str = "single_driver"
    seed: int = 42
    n_trials: int = 48
    trial_length: int = 128
    n_neurons: int = 6
    fps: float = 10.0
    cue_start: int = 72
    cue_stop: int = 112
    dt: float = 0.06
    coupling: float = 0.9
    driver_ar: float = 0.8
    driver_noise: float = 0.07
    population_noise: float = 0.08
    readout_noise: float = 0.05
    neutral_ar: float = 0.6
    cue_probability: float = 0.6
    cue_amplitude: float = 1.0
    cue_jitter: float = 0.07
    bystander_noise: float = 0.04
    observation_noise: float = 0.0
    calcium_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.scenario not in DRIVER_BENCHMARK_SCENARIOS:
            raise ValueError("unknown driver benchmark scenario")
        integers = (
            self.seed,
            self.n_trials,
            self.trial_length,
            self.n_neurons,
            self.cue_start,
            self.cue_stop,
        )
        if any(type(value) is not int for value in integers):
            raise ValueError("seeds and sample/trial/neuron counts must be integers")
        if self.seed < 0 or self.n_trials < 2 or self.n_neurons < 6:
            raise ValueError("need a nonnegative seed, at least two trials and six neurons")
        if not 0 <= self.cue_start < self.cue_stop < self.trial_length:
            raise ValueError("cue must lie inside each trial")
        for key, value in asdict(self).items():
            if key not in {
                "scenario",
                "seed",
                "n_trials",
                "trial_length",
                "n_neurons",
                "cue_start",
                "cue_stop",
            }:
                if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)):
                    raise ValueError(f"{key} must be finite and numeric")
        if self.fps <= 0 or not 0 < self.dt <= 0.1 or self.coupling <= 0 or self.cue_amplitude <= 0:
            raise ValueError("fps/coupling/amplitude must be positive and dt must lie in (0,0.1]")
        if not 0 <= self.driver_ar < 1 or not 0 <= self.neutral_ar < 1 or not 0 <= self.calcium_decay < 1:
            raise ValueError("AR coefficients and calcium decay must lie in [0,1)")
        if not 0 < self.cue_probability < 1:
            raise ValueError("cue_probability must lie strictly between zero and one")
        if (
            min(
                self.driver_noise,
                self.population_noise,
                self.readout_noise,
                self.cue_jitter,
                self.bystander_noise,
                self.observation_noise,
            )
            < 0
        ):
            raise ValueError("noise scales must be nonnegative")


def _step(
    config: DriverBenchmarkConfig,
    activity: np.ndarray,
    coordinate: np.ndarray,
    initial_states: np.ndarray,
    neural_cue: np.ndarray,
    environment_cue: np.ndarray,
    noise: np.ndarray,
    disconnected: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """Update all structural variables once, using only the previous neural state."""
    direction = 1 - 2 * initial_states
    if config.scenario in {"null", "hidden_confounder"}:
        drive = environment_cue
    elif config.scenario == "redundant_group":
        left = np.zeros(len(activity)) if 0 in disconnected else activity[:, 0]
        right = np.zeros(len(activity)) if 1 in disconnected else activity[:, 1]
        drive = np.maximum(0, np.maximum(left, right))
    else:
        active = initial_states.astype(int)
        drive = activity[np.arange(len(activity)), active].copy()
        drive[np.isin(active, list(disconnected))] = 0
        drive = np.maximum(drive, 0)
    drift = coordinate - coordinate**3 + direction * config.coupling * drive
    next_coordinate = np.clip(
        coordinate + config.dt * drift + np.sqrt(config.dt) * config.population_noise * noise[:, -1],
        -2.0,
        2.0,
    )
    means = np.repeat(neural_cue[:, None], 2, axis=1)
    if config.scenario != "redundant_group":
        means[np.arange(len(activity)), 1 - initial_states] = 0
    output = np.empty_like(activity)
    output[:, :2] = config.driver_ar * activity[:, :2] + (1 - config.driver_ar) * means
    output[:, :2] += config.driver_noise * noise[:, :2]
    if config.scenario in {"correlated_bystander", "hidden_confounder"}:
        output[:, 2] = output[np.arange(len(activity)), initial_states] + config.bystander_noise * noise[:, 2]
    else:
        output[:, 2] = config.neutral_ar * activity[:, 2] + config.driver_noise * noise[:, 2]
    output[:, 3:-1] = next_coordinate[:, None] + config.readout_noise * noise[:, 3:-2]
    output[:, -1] = config.neutral_ar * activity[:, -1] + config.driver_noise * noise[:, -2]
    return output, next_coordinate


def make_driver_benchmark(config: DriverBenchmarkConfig | None = None) -> NeuralRecording:
    """Generate independent trials with structural truth confined to metadata."""
    cfg = config or DriverBenchmarkConfig()
    rng = np.random.default_rng(cfg.seed)
    initial = rng.permutation(np.arange(cfg.n_trials) % 2)
    neural_amplitudes = cfg.cue_amplitude * rng.binomial(1, cfg.cue_probability, cfg.n_trials)
    neural_amplitudes = np.maximum(
        0, neural_amplitudes * (1 + cfg.cue_jitter * rng.normal(size=cfg.n_trials))
    )
    environment_amplitudes = cfg.cue_amplitude * rng.binomial(1, cfg.cue_probability, cfg.n_trials)
    if cfg.scenario == "hidden_confounder":
        environment_amplitudes = neural_amplitudes.copy()
    noise = rng.normal(size=(cfg.n_trials, cfg.trial_length, cfg.n_neurons + 1))
    activity = np.zeros((cfg.n_trials, cfg.trial_length, cfg.n_neurons))
    coordinate = np.empty((cfg.n_trials, cfg.trial_length))
    coordinate[:, 0] = 2 * initial - 1
    activity[:, 0, 3:-1] = coordinate[:, 0, None] + cfg.readout_noise * noise[:, 0, 3:-2]
    activity[:, 0, -1] = cfg.driver_noise * noise[:, 0, -2]
    for sample in range(1, cfg.trial_length):
        enabled = cfg.cue_start <= sample - 1 < cfg.cue_stop
        activity[:, sample], coordinate[:, sample] = _step(
            cfg,
            activity[:, sample - 1],
            coordinate[:, sample - 1],
            initial,
            neural_amplitudes * enabled,
            environment_amplitudes * enabled,
            noise[:, sample],
        )
    observed = activity.copy()
    for sample in range(1, cfg.trial_length):
        observed[:, sample] = (1 - cfg.calcium_decay) * observed[:, sample] + cfg.calcium_decay * observed[
            :, sample - 1
        ]
    measurement_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 29173]))
    observed += cfg.observation_noise * measurement_rng.normal(size=observed.shape)
    total = cfg.n_trials * cfg.trial_length
    nominal: dict[str, list[str]] = {"0->1": ["0"], "1->0": ["1"]}
    if cfg.scenario == "redundant_group":
        nominal = {"0->1": ["0", "1"], "1->0": ["0", "1"]}
    elif cfg.scenario in {"null", "hidden_confounder"}:
        nominal = {"0->1": [], "1->0": []}
    recording = NeuralRecording(
        traces=observed.reshape(total, cfg.n_neurons).T,
        time=np.arange(total) / cfg.fps,
        coords=None,
        neuron_ids=np.arange(cfg.n_neurons),
        behavior={},
        fps=cfg.fps,
        identity=RecordingIdentity(
            dataset="synthetic_driver_benchmark",
            dataset_id="driver-ground-truth",
            recording_id=f"{cfg.scenario}-seed-{cfg.seed}",
        ),
        metadata={
            "driver_benchmark": asdict(cfg),
            "latent_activity": activity.reshape(total, cfg.n_neurons).T,
            "population_coordinate": coordinate.reshape(total),
            "true_state": (coordinate.reshape(total) >= 0).astype(int),
            "true_trial_ids": np.repeat(np.arange(cfg.n_trials), cfg.trial_length),
            "trial_initial_states": initial,
            "trial_neural_amplitudes": neural_amplitudes,
            "trial_environment_amplitudes": environment_amplitudes,
            "valid_ranges": [(i * cfg.trial_length, (i + 1) * cfg.trial_length) for i in range(cfg.n_trials)],
            "nominal_drivers_by_direction": nominal,
            "truth_scope": "metadata_only_not_predictors_or_behavior",
            "intervention_scope": "disconnect_outgoing_population_drive_no_measurement_censoring",
            "outcome_scope": "opposite_latent_well_from_mean_coordinate_over_target_window",
            "mechanism": "bounded_discrete_doublewell_with_source_specific_neuronal_drive",
            "confounding_scope": "hidden_confounder_shares_external_cue_without_observed_neuronal_causation",
        },
    )
    recording.validate()
    return recording


def _sample_indices(values: Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or (array.size and not np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    return array.astype(int)


def driver_intervention_effects(
    recording: NeuralRecording,
    origin_samples: Sequence[int],
    target_samples: Sequence[int],
    interventions: Mapping[str, Sequence[str]],
    *,
    target_length: int = 8,
    n_rollouts: int = 64,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Re-execute future SCMs under paired, newly sampled process innovations.

    The factual origin state, trial context and exogenous cue amplitudes are
    held fixed. Each intervention disconnects outgoing candidate influence at
    the origin; future neuron trajectories are still generated normally. Equal
    random numbers across branches make inert-unit effects exactly zero. These
    are conditional simulator effects, not identified biological interventions.
    """
    cfg = DriverBenchmarkConfig(**recording.metadata["driver_benchmark"])
    for value, name in ((target_length, "target_length"), (n_rollouts, "n_rollouts"), (seed, "seed")):
        if type(value) is not int or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer")
    origins, targets = _sample_indices(origin_samples, "origins"), _sample_indices(target_samples, "targets")
    if len(origins) != len(targets):
        raise ValueError("origin and target samples must align")
    if (
        np.any(origins < 0)
        or np.any(targets >= recording.n_timepoints)
        or np.any(targets - target_length < origins)
    ):
        raise ValueError("targets must have strictly future support within the recording")
    if np.any(origins // cfg.trial_length != targets // cfg.trial_length):
        raise ValueError("intervention futures must not cross trial boundaries")
    if np.any(origins % cfg.trial_length < target_length - 1):
        raise ValueError("source target-length histories must lie inside their trial")
    known = {str(i): i for i in range(cfg.n_neurons)}
    units: dict[str, frozenset[int]] = {}
    for unit, members in interventions.items():
        if unit == "__factual__" or isinstance(members, (str, bytes)):
            raise ValueError("intervention names must be nonreserved and members must be a neuron sequence")
        normalized = [str(member) for member in members]
        if not isinstance(unit, str) or not unit or not normalized or len(set(normalized)) != len(normalized):
            raise ValueError("interventions require named, nonempty unique neuron groups")
        if set(normalized) - set(known):
            raise ValueError("intervention neuron IDs must occur in the recording")
        units[unit] = frozenset(known[member] for member in normalized)
    latent = np.asarray(recording.metadata["latent_activity"])
    coordinate = np.asarray(recording.metadata["population_coordinate"])
    initial_states = np.asarray(recording.metadata["trial_initial_states"])
    neural_amplitudes = np.asarray(recording.metadata["trial_neural_amplitudes"])
    environment_amplitudes = np.asarray(recording.metadata["trial_environment_amplitudes"])
    output = []
    for origin, target in zip(origins, targets, strict=True):
        trial = origin // cfg.trial_length
        horizon = int(target - origin)
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(origin), int(target)]))
        innovations = rng.normal(size=(horizon, n_rollouts, cfg.n_neurons + 1))
        source = int(np.mean(coordinate[origin - target_length + 1 : origin + 1]) >= 0)
        probabilities = {}
        for unit, disconnected in {"__factual__": frozenset(), **units}.items():
            current = np.repeat(latent[:, origin][None], n_rollouts, axis=0)
            z = np.full(n_rollouts, coordinate[origin])
            final_coordinates = []
            for step in range(horizon):
                sample = int(origin % cfg.trial_length) + step
                enabled = cfg.cue_start <= sample < cfg.cue_stop
                current, z = _step(
                    cfg,
                    current,
                    z,
                    np.full(n_rollouts, initial_states[trial], dtype=int),
                    np.full(n_rollouts, neural_amplitudes[trial] * enabled),
                    np.full(n_rollouts, environment_amplitudes[trial] * enabled),
                    innovations[step],
                    disconnected,
                )
                if step >= horizon - target_length:
                    final_coordinates.append(z)
            destinations = np.mean(final_coordinates, axis=0) >= 0
            probabilities[unit] = float(np.mean(destinations != source))
        factual = probabilities["__factual__"]
        for unit in units:
            output.append(
                {
                    "unit": unit,
                    "origin_sample": int(origin),
                    "target_sample": int(target),
                    "source_state": source,
                    "destination_state": 1 - source,
                    "n_rollouts": n_rollouts,
                    "factual_probability": factual,
                    "intervened_probability": probabilities[unit],
                    "probability_reduction": factual - probabilities[unit],
                }
            )
    return output
