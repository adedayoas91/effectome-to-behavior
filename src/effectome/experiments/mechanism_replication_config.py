"""Disjoint calibration and untouched evaluation for the observed mechanism control."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import numpy as np

from effectome.experiments.mechanism_controls import MechanismControlConfig
from effectome.experiments.mechanism_validation_config import MechanismValidationConfig
from effectome.experiments.structural_mechanism import MechanismFitConfig, MechanismRolloutConfig


def _base_mechanism() -> MechanismValidationConfig:
    return MechanismValidationConfig(
        generator=MechanismControlConfig(n_trials=24, trial_length=1152, n_neurons=6),
        discovery_burn_in=128,
        discovery_history_length=1024,
    )


@dataclass(frozen=True)
class MechanismReplicationConfig:
    profile: str = "replication"
    calibration_seeds: tuple[int, ...] = tuple(range(10000, 10019))
    positive_seeds: tuple[int, ...] = tuple(range(20000, 20020))
    null_seeds: tuple[int, ...] = tuple(range(30000, 30060))
    mechanism: MechanismValidationConfig = field(default_factory=_base_mechanism)
    primary_history: int = 1024
    comparison_history: int = 256
    calibration_alpha: float = 0.05
    nomination_floor: float = 0.1
    confidence_level: float = 0.95
    min_positive_seeds: int = 20
    min_null_seeds: int = 60
    min_topk_recall: float = 0.75
    min_nomination_recall: float = 0.75
    min_nomination_precision: float = 0.75
    max_effect_mae: float = 0.05
    max_null_probability: float = 0.05
    bootstrap_replicates: int = 2000
    analysis_seed: int = 20261117

    def __post_init__(self) -> None:
        if self.profile not in {"pilot", "replication"}:
            raise ValueError("replication profile must be pilot or replication")
        historical = {0, 1, 2, 20, 21, 22, 42, *range(300, 320), *range(400, 460)}
        pools = [self.calibration_seeds, self.positive_seeds, self.null_seeds]
        for pool in pools:
            if (
                not pool
                or any(type(seed) is not int or seed < 0 for seed in pool)
                or len(set(pool)) != len(pool)
            ):
                raise ValueError("seed pools must contain distinct nonnegative integers")
            if set(pool) & historical:
                raise ValueError("historical development/confirmation seeds cannot be reused")
        if len(set().union(*map(set, pools))) != sum(map(len, pools)):
            raise ValueError("calibration, positive and null-test seed pools must be disjoint")
        for name in ("primary_history", "comparison_history"):
            length = getattr(self, name)
            if (
                type(length) is not int
                or length < 256
                or length + self.mechanism.discovery_burn_in > self.mechanism.generator.trial_length
            ):
                raise ValueError("history lengths must be >=256 and stay inside each trial")
        if self.primary_history <= self.comparison_history:
            raise ValueError("primary history must exceed the comparison history")
        for name in (
            "calibration_alpha",
            "confidence_level",
            "min_topk_recall",
            "min_nomination_recall",
            "min_nomination_precision",
            "max_null_probability",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                or not 0 < float(value) < 1
            ):
                raise ValueError(f"{name} must lie strictly between zero and one")
        for name in ("nomination_floor", "max_effect_mae"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"{name} must lie in [0,1]")
        for name in ("min_positive_seeds", "min_null_seeds", "bootstrap_replicates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer >=2")
        if type(self.analysis_seed) is not int or self.analysis_seed < 0:
            raise ValueError("analysis seed must be a nonnegative integer")
        rank = math.ceil((len(self.calibration_seeds) + 1) * (1 - self.calibration_alpha))
        if rank > len(self.calibration_seeds):
            raise ValueError("too few calibration seeds to resolve the requested alpha")
        if self.nomination_floor < self.mechanism.nomination_threshold:
            raise ValueError("replication must retain the existing minimum effect-size nomination cutoff")

    @property
    def planned_cells(self) -> int:
        return len(self.calibration_seeds) + 2 * len(self.positive_seeds) + len(self.null_seeds)


def load_mechanism_replication_profile(profile: str = "replication") -> MechanismReplicationConfig:
    from omegaconf import OmegaConf

    if profile not in {"pilot", "replication"}:
        raise ValueError("unknown mechanism-replication profile")
    path = Path(__file__).resolve().parents[3] / "conf/mechanism_replication" / f"{profile}.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("replication profile must be a mapping")
    values = {str(key): value for key, value in raw.items()}
    for name in ("calibration_seeds", "positive_seeds", "null_seeds"):
        values[name] = tuple(values[name])
    if "mechanism" in values:
        mechanism = dict(values["mechanism"])
        mechanism["seeds"] = tuple(mechanism.get("seeds", (20,)))
        mechanism["generator"] = MechanismControlConfig(**mechanism["generator"])
        if "fit" in mechanism:
            fit = dict(mechanism["fit"])
            fit["ridge_alphas"] = tuple(fit["ridge_alphas"])
            mechanism["fit"] = MechanismFitConfig(**fit)
        if "rollout" in mechanism:
            mechanism["rollout"] = MechanismRolloutConfig(**mechanism["rollout"])
        values["mechanism"] = MechanismValidationConfig(**mechanism)
    return MechanismReplicationConfig(**values)
