"""Recurrent, fully observed mechanisms with prospective behavioral forecast origins."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from effectome.experiments.distributed_behavior_controls import (
    DRIVER_NODES,
    READOUT_NODES,
    distributed_behavior,
)
from effectome.experiments.structural_mechanism import AdditiveMechanism

TRANSITION_SCENARIOS = ("driver_switch", "constant", "nuisance_switch")


@dataclass(frozen=True)
class TransitionControlConfig:
    n_trials: int = 32
    history_lengths: tuple[int, ...] = (256, 1024)
    horizon: int = 8
    target_length: int = 4
    fps: float = 10.0
    washout: int = 64
    origin_jitter: int = 64
    gap_jitter: int = 64

    def __post_init__(self) -> None:
        if type(self.n_trials) is not int or self.n_trials < 2:
            raise ValueError("at least two independent trials are required")
        if (
            not isinstance(self.history_lengths, tuple)
            or not self.history_lengths
            or any(type(length) is not int or length < 3 for length in self.history_lengths)
            or len(set(self.history_lengths)) != len(self.history_lengths)
        ):
            raise ValueError("history_lengths must contain distinct integer lengths of at least three")
        for name in ("horizon", "target_length", "washout", "origin_jitter", "gap_jitter"):
            value = getattr(self, name)
            minimum = 1 if name in {"horizon", "target_length"} else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if self.target_length > self.horizon:
            raise ValueError("target length must not exceed the forecast horizon")
        if isinstance(self.fps, bool) or not isinstance(self.fps, Real) or not np.isfinite(self.fps):
            raise ValueError("fps must be positive and finite")
        if self.fps <= 0:
            raise ValueError("fps must be positive and finite")


@dataclass(frozen=True)
class TransitionControlData:
    """Trial observations, last-observed indices and simulator truths for scoring.

    Regime starts are inclusive. The mechanism at a sample produces that sample
    from the preceding state, including at a switch. Each origin corresponds to
    the regime and regime start in the same column of the trial.
    """

    neural: np.ndarray
    behavior: np.ndarray
    origins: np.ndarray
    regimes: np.ndarray
    mechanisms: tuple[AdditiveMechanism, AdditiveMechanism]
    regime_starts: np.ndarray


def _mechanisms(scenario: str) -> tuple[AdditiveMechanism, AdditiveMechanism]:
    self_weights = np.full(10, 0.7)
    self_weights[list(DRIVER_NODES)] = 0.85
    self_weights[list(READOUT_NODES)] = 0.75
    driver_weights = 0.8 * np.array([1.0, 0.75, 0.5, 0.25])
    common = np.zeros((10, 10))
    common[0, 6:8] = 0.8
    common[1, 6:8] = np.array([0.8, -0.8])

    def make(regime: int) -> AdditiveMechanism:
        cross = common.copy()
        driver = regime if scenario == "driver_switch" else 0
        cross[driver, READOUT_NODES] = driver_weights
        if scenario == "nuisance_switch" and regime == 1:
            cross[6, 9] = cross[7, 8] = 0.6
        else:
            cross[6, 8] = cross[7, 9] = 0.6
        return AdditiveMechanism(
            intercept=np.zeros(10),
            self_weights=self_weights,
            cross_weights=cross,
            noise_scale=np.full(10, 0.1),
            support=cross != 0,
            metadata={
                "kind": "simulator_truth_evaluation_only",
                "family": "additive_tanh_lag_one",
                "noise_source": "simulator_parameters_evaluation_only",
                "scenario": scenario,
                "regime": regime,
            },
        )

    return make(0), make(1)


def make_transition_control(
    config: TransitionControlConfig, *, scenario: str, seed: int
) -> TransitionControlData:
    """Draw independent recurrent trials with shared schedules and noise across scenarios.

    Both drivers have the same autonomous dynamics and either can provide the
    same signed input to the fixed readout block. Switching driver identity
    changes conditional dynamics. Switch times and labels are simulator truth.
    """
    if scenario not in TRANSITION_SCENARIOS:
        raise ValueError(f"unknown transition scenario: {scenario}")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    schedule_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    schedule = np.random.default_rng(schedule_seed)
    trial_ids = np.arange(config.n_trials)
    # Pair randomized orientations so even contiguous trial slices contain both directions.
    first = schedule.integers(0, 2, size=(config.n_trials + 1) // 2).repeat(2)[: config.n_trials]
    first[1::2] = 1 - first[1::2]
    regimes = np.column_stack((first, 1 - first, first))
    origin_offsets = (
        max(config.history_lengths)
        - 1
        + config.washout
        + schedule.integers(0, config.origin_jitter + 1, size=regimes.shape)
    )
    next_offsets = (
        origin_offsets + config.horizon + 1 + schedule.integers(0, config.gap_jitter + 1, size=regimes.shape)
    )
    regime_starts = np.column_stack(
        (np.zeros(config.n_trials, dtype=int), np.cumsum(next_offsets, axis=1)[:, :2])
    )
    origins = regime_starts + origin_offsets

    mechanisms = _mechanisms(scenario)
    # Pad independent trials by continuing their final regime to one common length.
    n_samples = int(np.max(regime_starts[:, -1] + next_offsets[:, -1]))
    innovations = np.random.default_rng(noise_seed).normal(size=(config.n_trials, n_samples, 10))
    neural = np.empty_like(innovations)
    neural[:, 0] = innovations[:, 0] * mechanisms[0].noise_scale
    for sample in range(1, n_samples):
        episode = np.count_nonzero(sample >= regime_starts[:, 1:], axis=1)
        current = regimes[trial_ids, episode]
        for regime, mechanism in enumerate(mechanisms):
            selected = current == regime
            if selected.any():
                neural[selected, sample] = mechanism.step(
                    neural[selected, sample - 1], innovations[selected, sample]
                )
    behavior = distributed_behavior(neural)
    for values in (neural, behavior, origins, regimes, regime_starts):
        values.setflags(write=False)
    return TransitionControlData(neural, behavior, origins, regimes, mechanisms, regime_starts)
