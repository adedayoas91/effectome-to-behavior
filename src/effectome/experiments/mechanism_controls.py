"""Fully observed nonlinear controls for the mechanism-readout development ladder.

These model-matched controls isolate a learned readout from support discovery.
They do not replace the completed latent-bistable benchmark. See the assumptions
and verified references in docs/driver_methodology_revision.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real

import numpy as np

from effectome.data_module.schema import NeuralRecording, RecordingIdentity
from effectome.experiments.structural_mechanism import AdditiveMechanism

MECHANISM_SCENARIOS = ("driver", "null")


@dataclass(frozen=True)
class MechanismControlConfig:
    """Pulse strength sets observed initial conditions, never a hidden future cue."""

    scenario: str = "driver"
    seed: int = 20
    n_trials: int = 64
    trial_length: int = 320
    n_neurons: int = 6
    fps: float = 10.0
    driver_ar: float = 0.85
    neutral_ar: float = 0.55
    endpoint_ar: float = 0.75
    coupling: float = 0.8
    bystander_coupling: float = 0.5
    noise_std: float = 0.1
    initial_endpoint: float = 1.0
    pulse_amplitude: float = 1.0
    pulse_probability: float = 0.6

    def __post_init__(self) -> None:
        if self.scenario not in MECHANISM_SCENARIOS:
            raise ValueError("unknown mechanism-control scenario")
        for key in ("seed", "n_trials", "trial_length", "n_neurons"):
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError("seed/counts must be nonnegative/positive integers")
        if self.n_trials < 8 or self.n_neurons < 6 or self.trial_length < 3:
            raise ValueError("mechanism controls require >=8 trials, >=6 neurons and >=3 samples")
        for key, value in asdict(self).items():
            if key == "scenario":
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)):
                raise ValueError(f"{key} must be finite and numeric")
        if any(not 0 <= value < 1 for value in (
            self.driver_ar, self.neutral_ar, self.endpoint_ar,
        )):
            raise ValueError("self-history contractions must lie in [0,1)")
        if min(self.fps, self.coupling, self.noise_std, self.initial_endpoint,
               self.pulse_amplitude) <= 0 or self.bystander_coupling < 0:
            raise ValueError("scales must be positive and bystander coupling nonnegative")
        if not 0 < self.pulse_probability < 1:
            raise ValueError("pulse_probability must lie in (0,1)")


def make_mechanism_control(config: MechanismControlConfig) -> NeuralRecording:
    """Generate independent directly observed Markov trials with additive noise.

    Sources 0/1 have opposite signed outgoing effects on the final coordinate.
    Neuron 2 is a lagged bystander, with no outgoing influence. All other nodes
    are inert controls. In the null only the endpoint's incoming cross terms are
    absent; driver/bystander activity remains. True functions stay in metadata.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_neurons
    self_weights = np.full(n, config.neutral_ar)
    self_weights[:2] = config.driver_ar
    self_weights[-1] = config.endpoint_ar
    cross = np.zeros((n, n))
    cross[:2, 2] = config.bystander_coupling
    if config.scenario == "driver":
        cross[0, -1], cross[1, -1] = config.coupling, -config.coupling
    initial_sides = rng.permutation(np.arange(config.n_trials) % 2)
    pulses = rng.binomial(1, config.pulse_probability, config.n_trials)
    noise = rng.normal(size=(config.n_trials, config.trial_length, n)) * config.noise_std
    values = np.empty_like(noise)
    values[:, 0] = noise[:, 0]
    values[np.arange(config.n_trials), 0, initial_sides] += config.pulse_amplitude * pulses
    values[:, 0, 2] = values[:, 0, :2].sum(axis=1) + noise[:, 0, 2]
    values[:, 0, -1] = (2 * initial_sides - 1) * config.initial_endpoint
    for sample in range(1, config.trial_length):
        previous = values[:, sample - 1]
        values[:, sample] = previous * self_weights + np.tanh(previous) @ cross + noise[:, sample]
    total = config.n_trials * config.trial_length
    recording = NeuralRecording(
        traces=values.reshape(total, n).T,
        time=np.arange(total) / config.fps,
        coords=None,
        neuron_ids=np.arange(n),
        behavior={},
        fps=config.fps,
        identity=RecordingIdentity(
            dataset="fully_observed_mechanism_control",
            dataset_id="mechanism-readout-development",
            recording_id=f"{config.scenario}-seed-{config.seed}",
        ),
        metadata={
            "mechanism_control": asdict(config),
            "true_intercept": np.zeros(n),
            "true_self_weights": self_weights,
            "true_cross_weights": cross,
            "true_noise_scale": np.full(n, config.noise_std),
            "true_support": cross != 0,
            "valid_ranges": [(i * config.trial_length, (i + 1) * config.trial_length)
                             for i in range(config.n_trials)],
            "endpoint_index": n - 1,
            "endpoint_threshold": 0.0,
            "truth_scope": "metadata_only_not_estimator_inputs",
            "observation_scope": "all_structural_state_variables_observed_without_measurement_error",
            "initialization_scope": "observed_initial_condition_pulse_no_hidden_future_input",
            "claim_scope": "model_matched_development_control_not_independent_confirmation",
        },
    )
    recording.validate()
    return recording


def mechanism_control_trials(recording: NeuralRecording) -> np.ndarray:
    """Return the explicit trial axis; never construct cross-trial sample pairs."""
    config = recording.metadata["mechanism_control"]
    return recording.traces.T.reshape(config["n_trials"], config["trial_length"], recording.n_neurons)


def oracle_mechanism(recording: NeuralRecording) -> AdditiveMechanism:
    """Expose generating parameters ONLY for evaluation and the oracle diagnostic."""
    data = recording.metadata
    return AdditiveMechanism(
        intercept=data["true_intercept"],
        self_weights=data["true_self_weights"],
        cross_weights=data["true_cross_weights"],
        noise_scale=data["true_noise_scale"],
        support=data["true_support"],
        metadata={"kind": "simulator_truth_not_a_learned_model"},
    )
