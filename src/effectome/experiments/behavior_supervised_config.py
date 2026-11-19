"""Frozen profiles for distributed continuous-behavior representation validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path

import numpy as np

from effectome.experiments.distributed_behavior_controls import DistributedBehaviorConfig
from effectome.experiments.structural_mechanism import MechanismFitConfig, MechanismRolloutConfig


def _positive(value: object, name: str, *, upper: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not np.isfinite(number) or number <= 0 or (upper is not None and number > upper):
        raise ValueError(f"{name} must be finite, positive and within its declared range")


@dataclass(frozen=True)
class BehaviorLearningConfig:
    ridge_alphas: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    dynamics_weights: tuple[float, ...] = (0.05, 0.2)
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 200
    min_epochs: int = 20
    patience: int = 20
    batch_size: int = 128
    max_pairs_per_trial: int = 128
    early_samples: int = 32
    validation_fraction: float = 0.25
    seed: int = 20261118
    device: str = "cpu"
    runtime_python: str | None = None

    def __post_init__(self) -> None:
        for key in ("ridge_alphas", "dynamics_weights"):
            values = getattr(self, key)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{key} must contain distinct candidates")
            for value in values:
                _positive(value, key)
        for key in ("learning_rate", "weight_decay"):
            _positive(getattr(self, key), key)
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in (0,1)")
        for key in (
            "max_epochs",
            "min_epochs",
            "patience",
            "batch_size",
            "max_pairs_per_trial",
            "early_samples",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.min_epochs > self.max_epochs or self.max_pairs_per_trial < 4 or self.max_pairs_per_trial % 2:
            raise ValueError("invalid epoch or sampling budget")
        if type(self.seed) is not int or self.seed < 0 or self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("invalid encoder seed or device")
        if self.runtime_python is not None and not isinstance(self.runtime_python, str):
            raise ValueError("runtime_python must be a path string or None")


@dataclass(frozen=True)
class BehaviorDecisionConfig:
    """Prospective behavioral-unit budgets; these are not the old probability/nat cutoffs."""

    gold_effect_floor: float = 0.1
    nomination_floor: float = 0.1
    max_behavioral_mae: float = 0.05
    max_positive_unit_mae: float = 0.05
    max_semantic_mae: float = 0.05
    max_origin_decoding_mae: float = 0.1
    min_nomination_recall: float = 0.75
    min_nomination_precision: float = 0.75
    max_null_probability: float = 0.05
    calibration_alpha: float = 0.05
    confidence_level: float = 0.95
    min_positive_seeds: int = 20
    min_null_seeds: int = 60
    primary_readout: str = "bunddle_behavior_2d"

    def __post_init__(self) -> None:
        for key in (
            "gold_effect_floor",
            "nomination_floor",
            "max_behavioral_mae",
            "max_positive_unit_mae",
            "max_semantic_mae",
            "max_origin_decoding_mae",
            "min_nomination_recall",
            "min_nomination_precision",
            "max_null_probability",
            "calibration_alpha",
            "confidence_level",
        ):
            _positive(getattr(self, key), key, upper=1)
        if self.calibration_alpha >= 1 or self.confidence_level >= 1:
            raise ValueError("calibration/confidence probabilities must be below one")
        for key in ("min_positive_seeds", "min_null_seeds"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 2:
                raise ValueError("independent seed minima must be integers >=2")
        if self.primary_readout != "bunddle_behavior_2d":
            raise ValueError("the primary readout is frozen as the two-dimensional joint encoder")


@dataclass(frozen=True)
class BehaviorSupervisedConfig:
    profile: str = "development"
    development_seeds: tuple[int, ...] = tuple(range(40005, 40010))
    calibration_seeds: tuple[int, ...] = tuple(range(50000, 50019))
    positive_seeds: tuple[int, ...] = tuple(range(60000, 60020))
    null_seeds: tuple[int, ...] = tuple(range(70000, 70060))
    generator: DistributedBehaviorConfig = field(default_factory=DistributedBehaviorConfig)
    fit: MechanismFitConfig = field(default_factory=MechanismFitConfig)
    rollout: MechanismRolloutConfig = field(default_factory=MechanismRolloutConfig)
    learning: BehaviorLearningConfig = field(default_factory=BehaviorLearningConfig)
    decisions: BehaviorDecisionConfig = field(default_factory=BehaviorDecisionConfig)
    evaluation_fraction: float = 0.5
    discovery_history_length: int = 1024
    discovery_burn_in: int = 128
    support_selection_fraction: float = 0.5
    retain_rollout_samples: bool = False

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "development", "confirmation"}:
            raise ValueError("unknown behavior-supervised profile")
        for key, lower, upper in (
            ("development_seeds", 40005, 40020),
            ("calibration_seeds", 50000, 50019),
            ("positive_seeds", 60000, 60020),
            ("null_seeds", 70000, 70060),
        ):
            values = getattr(self, key)
            if (
                not values
                or len(set(values)) != len(values)
                or any(type(seed) is not int or not lower <= seed < upper for seed in values)
            ):
                raise ValueError(f"{key} must use distinct seeds from the reserved uninspected pool")
        if not 0 < self.evaluation_fraction < 1:
            raise ValueError("evaluation_fraction must lie in (0,1)")
        n_train = int(self.generator.n_trials * (1 - self.evaluation_fraction))
        if n_train < 4 or self.generator.n_trials - n_train < 4:
            raise ValueError("need at least four training and evaluation trials")
        if (
            type(self.discovery_history_length) is not int
            or self.discovery_history_length < 256
            or type(self.discovery_burn_in) is not int
            or self.discovery_burn_in < 0
            or self.discovery_burn_in + self.discovery_history_length > self.generator.trial_length
        ):
            raise ValueError("native discovery histories must stay within trials and contain >=256 samples")
        if self.rollout.horizon >= self.generator.trial_length:
            raise ValueError("rollout support must stay within the evaluation trial")
        _positive(self.support_selection_fraction, "support_selection_fraction", upper=1)
        if type(self.retain_rollout_samples) is not bool:
            raise ValueError("retain_rollout_samples must be boolean")
        if self.profile == "confirmation":
            rank = int(np.ceil((len(self.calibration_seeds) + 1) * (1 - self.decisions.calibration_alpha)))
            if rank > len(self.calibration_seeds):
                raise ValueError("not enough calibration seeds for the requested finite-sample rank")

    @property
    def planned_cells(self) -> int:
        if self.profile == "confirmation":
            return len(self.calibration_seeds) + len(self.positive_seeds) + len(self.null_seeds)
        return 2 * len(self.development_seeds)


def load_behavior_supervised_profile(profile: str = "development") -> BehaviorSupervisedConfig:
    from omegaconf import OmegaConf

    if profile not in {"smoke", "development", "confirmation"}:
        raise ValueError("unknown behavior-supervised profile")
    path = Path(__file__).resolve().parents[3] / "conf/behavior_supervised" / f"{profile}.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("behavior profile must be a mapping")
    values = {str(key): value for key, value in raw.items()}
    for key in ("development_seeds", "calibration_seeds", "positive_seeds", "null_seeds"):
        values[key] = tuple(values[key])
    values["generator"] = DistributedBehaviorConfig(**values["generator"])
    fit = dict(values["fit"])
    fit["ridge_alphas"] = tuple(fit["ridge_alphas"])
    values["fit"] = MechanismFitConfig(**fit)
    values["rollout"] = MechanismRolloutConfig(**values["rollout"])
    learning = dict(values["learning"])
    learning["ridge_alphas"] = tuple(learning["ridge_alphas"])
    learning["dynamics_weights"] = tuple(learning["dynamics_weights"])
    values["learning"] = BehaviorLearningConfig(**learning)
    values["decisions"] = BehaviorDecisionConfig(**values["decisions"])
    return BehaviorSupervisedConfig(**values)
