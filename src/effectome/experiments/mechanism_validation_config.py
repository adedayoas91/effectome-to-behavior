"""Development-only configuration for known-support versus learned-support tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import numpy as np

from effectome.experiments.mechanism_controls import MechanismControlConfig
from effectome.experiments.structural_mechanism import MechanismFitConfig, MechanismRolloutConfig


@dataclass(frozen=True)
class MechanismValidationConfig:
    profile: str = "smoke"
    seeds: tuple[int, ...] = (20,)
    generator: MechanismControlConfig = field(default_factory=MechanismControlConfig)
    fit: MechanismFitConfig = field(default_factory=MechanismFitConfig)
    rollout: MechanismRolloutConfig = field(default_factory=MechanismRolloutConfig)
    evaluation_fraction: float = 0.5
    discovery_history_length: int = 256
    discovery_burn_in: int = 64
    support_selection_fraction: float = 0.5
    effect_threshold: float = 0.1
    nomination_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "development"}:
            raise ValueError("only development/smoke mechanism profiles are implemented")
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or seed < 0 for seed in self.seeds)):
            raise ValueError("seeds must be distinct nonnegative integers")
        if set(self.seeds) & (set(range(300, 320)) | set(range(400, 460))):
            raise ValueError("completed confirmation seeds are reserved, not development seeds")
        if (isinstance(self.evaluation_fraction, bool) or not isinstance(self.evaluation_fraction, Real)
                or not np.isfinite(self.evaluation_fraction) or not 0 < self.evaluation_fraction < 1):
            raise ValueError("evaluation_fraction must lie in (0,1)")
        fit_trials = int(self.generator.n_trials * (1 - self.evaluation_fraction))
        if fit_trials < 4 or self.generator.n_trials - fit_trials < 4:
            raise ValueError("need at least four training and four held-out trials")
        if any(type(value) is not int for value in (self.discovery_history_length, self.discovery_burn_in)):
            raise ValueError("discovery history and burn-in must be integers")
        if self.discovery_history_length < 256 or self.discovery_burn_in < 0:
            raise ValueError("native complete-shift discovery requires >=256 samples and nonnegative burn-in")
        if self.discovery_burn_in + self.discovery_history_length > self.generator.trial_length:
            raise ValueError("discovery history must stay inside each training trial")
        if self.rollout.horizon >= self.generator.trial_length:
            raise ValueError("rollout horizon must stay inside a held-out trial")
        for key in ("support_selection_fraction", "effect_threshold", "nomination_threshold"):
            value = getattr(self, key)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 < value <= 1):
                raise ValueError(f"{key} must lie in (0,1]")


def load_mechanism_validation_profile(profile: str) -> MechanismValidationConfig:
    from omegaconf import OmegaConf

    if profile not in {"smoke", "development"}:
        raise ValueError("unknown mechanism-validation profile")
    root = Path(__file__).resolve().parents[3] / "conf/mechanism_validation"
    raw = OmegaConf.to_container(OmegaConf.load(root / f"{profile}.yaml"), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("mechanism profile must be a mapping")
    values = {str(key): value for key, value in raw.items()}
    values["seeds"] = tuple(values["seeds"])
    values["generator"] = MechanismControlConfig(**values["generator"])
    fit = dict(values["fit"])
    fit["ridge_alphas"] = tuple(fit["ridge_alphas"])
    values["fit"] = MechanismFitConfig(**fit)
    values["rollout"] = MechanismRolloutConfig(**values["rollout"])
    return MechanismValidationConfig(**values)
