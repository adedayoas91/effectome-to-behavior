"""Fully observed controls with a prespecified distributed continuous behavior.

The structural model stays in the additive-tanh family used by the mechanism
estimator. Behavior is a fixed nonlinear readout, not a fitted principal
component, and nuisance responses have no path into that readout. The null
removes every incoming cross term into the entire behavioral block. See
``docs/behavior_supervised_experiment_plan.md`` for the assumptions and rationale.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np

from effectome.experiments.structural_mechanism import AdditiveMechanism

DISTRIBUTED_SCENARIOS = ("driver", "null")
DRIVER_NODES = (0, 1)
READOUT_NODES = (2, 3, 4, 5)
NUISANCE_NODES = (6, 7, 8, 9)
_READOUT_MATRIX = np.zeros((2, 10))
_READOUT_MATRIX[:, READOUT_NODES] = np.array([[1, 1, 1, 1], [1, -1, 1, -1]]) / 2
_READOUT_MATRIX.setflags(write=False)


@dataclass(frozen=True)
class DistributedBehaviorConfig:
    """Independent observed-state trials; initial pulses are not future inputs."""

    scenario: str = "driver"
    seed: int = 40005
    n_trials: int = 24
    trial_length: int = 1152
    n_neurons: int = 10
    fps: float = 10.0
    driver_ar: float = 0.85
    readout_ar: float = 0.75
    nuisance_ar: float = 0.7
    coupling: float = 0.8
    nuisance_coupling: float = 0.8
    noise_std: float = 0.1
    pulse_amplitude: float = 1.0
    pulse_probability: float = 0.6
    initial_readout: float = 0.8

    def __post_init__(self) -> None:
        if self.scenario not in DISTRIBUTED_SCENARIOS:
            raise ValueError("unknown distributed-behavior scenario")
        for name in ("seed", "n_trials", "trial_length", "n_neurons"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError("seed/counts must be nonnegative/positive integers")
        if self.n_trials < 8 or self.trial_length < 3 or self.n_neurons != 10:
            raise ValueError(
                "the distributed control requires >=8 trials, >=3 samples and exactly 10 neurons"
            )
        for name, value in asdict(self).items():
            if name == "scenario":
                continue
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite and numeric")
        if any(not 0 <= value < 1 for value in (self.driver_ar, self.readout_ar, self.nuisance_ar)):
            raise ValueError("self-history contractions must lie in [0,1)")
        if (
            min(
                self.fps,
                self.coupling,
                self.nuisance_coupling,
                self.noise_std,
                self.pulse_amplitude,
                self.initial_readout,
            )
            <= 0
        ):
            raise ValueError("state, coupling and noise scales must be positive")
        if not 0 < self.pulse_probability < 1:
            raise ValueError("pulse_probability must lie strictly between zero and one")


def distributed_readout_matrix() -> np.ndarray:
    """Export a readonly copy of the fixed behavioral readout, without a fitted map."""
    matrix = _READOUT_MATRIX.copy()
    matrix.setflags(write=False)
    return matrix


def distributed_behavior(states: np.ndarray) -> np.ndarray:
    """Apply the fixed two-output readout to an array ending in ten neural coordinates.

    Each output uses all four declared readout neurons, followed by tanh. Neural
    driver and nuisance coordinates cannot enter this measurement directly.
    """
    values = np.asarray(states, dtype=float)
    if values.ndim < 1 or not values.size or values.shape[-1] != 10 or not np.isfinite(values).all():
        raise ValueError("states must be nonempty and finite with a final ten-neuron axis")
    return np.tanh(values @ _READOUT_MATRIX.T)


def distributed_interventions() -> dict[str, list[int]]:
    """Return predeclared individual and group outgoing disconnections."""
    return {
        **{f"neuron:{index}": [index] for index in range(10)},
        "group:driver_pair": list(DRIVER_NODES),
        "group:nuisance_block": list(NUISANCE_NODES),
    }


@dataclass(frozen=True)
class DistributedBehaviorData:
    """Control trials and generating mechanism, with immutable numerical arrays."""

    neural_trials: np.ndarray
    behavior_trials: np.ndarray
    mechanism: AdditiveMechanism
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        neural = np.asarray(self.neural_trials, dtype=float)
        behavior = np.asarray(self.behavior_trials, dtype=float)
        if (
            neural.ndim != 3
            or neural.shape[-1] != 10
            or min(neural.shape[:2]) < 1
            or not np.isfinite(neural).all()
        ):
            raise ValueError("neural_trials must be finite nonempty trial-by-time-by-ten observations")
        if behavior.shape != (*neural.shape[:2], 2) or not np.isfinite(behavior).all():
            raise ValueError("behavior_trials must match neural trial/time axes and have two outputs")
        if not isinstance(self.mechanism, AdditiveMechanism) or len(self.mechanism.intercept) != 10:
            raise ValueError("the generating mechanism must describe ten observed neural coordinates")
        if not np.array_equal(behavior, distributed_behavior(neural)):
            raise ValueError("behavior_trials must follow the prespecified distributed readout")
        for name, array in (("neural_trials", neural), ("behavior_trials", behavior)):
            frozen = array.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


def _has_path(support: np.ndarray, sources: tuple[int, ...], targets: tuple[int, ...]) -> bool:
    """Check source-to-target cross-neuron reachability for the design audit."""
    pending = list(sources)
    seen: set[int] = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        children = np.flatnonzero(support[source])
        if any(int(child) in targets for child in children):
            return True
        pending.extend(int(child) for child in children if int(child) not in seen)
    return False


def make_distributed_behavior_control(config: DistributedBehaviorConfig) -> DistributedBehaviorData:
    """Generate independent fully observed trials and their fixed behavioral readout.

    No trial's state is carried into another trial. Both drivers have independent
    signed initial pulses; the four readout initial conditions are iid centered
    Gaussians, rather than labels inferred from a fitted chart. Innovation vectors
    are independent across time and trials. The null changes neural cross terms,
    never the definition of behavior or the random-number draw schedule.
    """
    if not isinstance(config, DistributedBehaviorConfig):
        raise ValueError("config must be a DistributedBehaviorConfig")
    rng = np.random.default_rng(config.seed)
    self_weights = np.full(10, config.nuisance_ar)
    self_weights[list(DRIVER_NODES)] = config.driver_ar
    self_weights[list(READOUT_NODES)] = config.readout_ar
    cross = np.zeros((10, 10))
    cross[0, READOUT_NODES] = config.coupling * np.array([1.0, 0.75, 0.5, 0.25])
    cross[1, READOUT_NODES] = config.coupling * np.array([0.75, -1.0, 0.25, -0.5])
    cross[0, 6:8] = config.nuisance_coupling
    cross[1, 6:8] = config.nuisance_coupling * np.array([1.0, -1.0])
    cross[6, 8] = cross[7, 9] = 0.75 * config.nuisance_coupling
    if config.scenario == "null":
        cross[:, READOUT_NODES] = 0.0
    support = cross != 0
    if _has_path(support, NUISANCE_NODES, READOUT_NODES):
        raise ValueError("the nuisance block must have no directed path into behavior")
    mechanism = AdditiveMechanism(
        intercept=np.zeros(10),
        self_weights=self_weights,
        cross_weights=cross,
        noise_scale=np.full(10, config.noise_std),
        support=support,
        metadata={
            "kind": "simulator_truth_not_a_learned_model",
            "family": "additive_tanh_lag_one",
            "noise_source": "simulator_parameters_evaluation_only",
        },
    )
    innovations = rng.normal(size=(config.n_trials, config.trial_length, 10))
    neural = np.empty_like(innovations)
    neural[:, 0] = innovations[:, 0] * config.noise_std
    neural[:, 0, READOUT_NODES] = rng.normal(
        scale=config.initial_readout, size=(config.n_trials, len(READOUT_NODES))
    )
    pulse_present = rng.binomial(1, config.pulse_probability, size=(config.n_trials, 2))
    pulse_sign = rng.choice((-1, 1), size=(config.n_trials, 2))
    neural[:, 0, :2] += config.pulse_amplitude * pulse_present * pulse_sign
    for sample in range(1, config.trial_length):
        neural[:, sample] = mechanism.step(neural[:, sample - 1], innovations[:, sample])
    behavior = distributed_behavior(neural)
    metadata = {
        "distributed_behavior_control": asdict(config),
        "true_intercept": mechanism.intercept.tolist(),
        "true_self_weights": mechanism.self_weights.tolist(),
        "true_cross_weights": mechanism.cross_weights.tolist(),
        "true_noise_scale": mechanism.noise_scale.tolist(),
        "true_support": support.tolist(),
        "roles": {
            "drivers": list(DRIVER_NODES),
            "readout": list(READOUT_NODES),
            "nuisance": list(NUISANCE_NODES),
        },
        "readout_matrix": _READOUT_MATRIX.tolist(),
        "behavior_definition": "tanh(neural_state @ fixed_two_row_readout_matrix.T)",
        "behavior_dimensions": 2,
        "behavior_units": "bounded_dimensionless_tanh_coordinates",
        "neural_state_units": "dimensionless_structural_state",
        "readout_definition_scope": "prespecified_generator_design_not_fitted_or_selected",
        "valid_ranges": [
            (index * config.trial_length, (index + 1) * config.trial_length)
            for index in range(config.n_trials)
        ],
        "independent_trials": True,
        "trial_boundary_state_carry": False,
        "initialization": "independent_signed_driver_pulses_and_centered_gaussian_readout_states",
        "noise_law": "independent_zero_mean_gaussian_innovation_vectors_across_times_and_trials",
        "behavioral_null": config.scenario == "null",
        "null_design": {
            "all_readout_incoming_cross_terms_removed": not bool(support[:, READOUT_NODES].any()),
            "nuisance_has_no_path_to_readout": True,
            "driver_nuisance_path_retained": _has_path(support, DRIVER_NODES, NUISANCE_NODES),
        },
        "observation_scope": "all_structural_neural_state_observed_without_measurement_error",
        "truth_scope": "generating_parameters_metadata_and_oracle_only_not_native_estimator_inputs",
        "claim_scope": "fully_observed_model_matched_continuous_behavior_control",
    }
    return DistributedBehaviorData(neural, behavior, mechanism, metadata)
