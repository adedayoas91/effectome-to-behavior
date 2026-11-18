"""Development-only bridge from full neural mechanisms to learned endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from effectome.experiments.mechanism_controls import MechanismControlConfig
from effectome.experiments.mechanism_validation_config import MechanismValidationConfig
from effectome.experiments.structural_mechanism import MechanismFitConfig, MechanismRolloutConfig


def _mechanism() -> MechanismValidationConfig:
    return MechanismValidationConfig(
        profile="development", seeds=(40000,),
        generator=MechanismControlConfig(n_trials=24, trial_length=1152),
        discovery_history_length=1024, discovery_burn_in=128,
    )


@dataclass(frozen=True)
class ManifoldValidationConfig:
    profile: str = "development"
    seeds: tuple[int, ...] = tuple(range(40000, 40005))
    n_components: int = 2
    decoder_cs: tuple[float, ...] = (0.1, 1.0, 10.0)
    endpoint_seed: int = 42
    mechanism: MechanismValidationConfig = field(default_factory=_mechanism)

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "development"}:
            raise ValueError("only smoke/development learned-endpoint profiles are implemented")
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or not 40000 <= seed < 40020 for seed in self.seeds)):
            raise ValueError("use only reserved manifold development seeds 40000--40019")
        if (type(self.n_components) is not int
                or not 1 <= self.n_components < self.mechanism.generator.n_neurons):
            raise ValueError("n_components must compress the observed neural axis")
        if type(self.endpoint_seed) is not int or self.endpoint_seed < 0:
            raise ValueError("endpoint_seed must be a nonnegative integer")
        if (not self.decoder_cs or len(set(self.decoder_cs)) != len(self.decoder_cs)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not np.isfinite(value) or value <= 0 for value in self.decoder_cs)):
            raise ValueError("decoder_cs must contain distinct positive finite penalties")


def load_manifold_validation_profile(profile: str = "development") -> ManifoldValidationConfig:
    from omegaconf import OmegaConf

    if profile not in {"smoke", "development"}:
        raise ValueError("unknown learned-endpoint profile")
    path = Path(__file__).resolve().parents[3] / "conf/manifold_validation" / f"{profile}.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("learned-endpoint profile must be a mapping")
    values = {str(key): value for key, value in raw.items()}
    values["seeds"] = tuple(values["seeds"])
    values["decoder_cs"] = tuple(values["decoder_cs"])
    mechanism = dict(values["mechanism"])
    mechanism["seeds"] = tuple(values["seeds"])
    mechanism["generator"] = MechanismControlConfig(**mechanism["generator"])
    fit = dict(mechanism["fit"])
    fit["ridge_alphas"] = tuple(fit["ridge_alphas"])
    mechanism["fit"] = MechanismFitConfig(**fit)
    mechanism["rollout"] = MechanismRolloutConfig(**mechanism["rollout"])
    values["mechanism"] = MechanismValidationConfig(**mechanism)
    return ManifoldValidationConfig(**values)
