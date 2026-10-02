"""Coupling-aware surrogate modeling and predictor-feature sensitivity screening."""

from .surrogate import (
    DoseResponsePoint,
    LinearSurrogateModel,
    PerturbationResult,
    SurrogateValidation,
    fit_linear_surrogate,
    run_predictor_feature_sensitivity,
    run_virtual_perturbation,
)

__all__ = [
    "DoseResponsePoint",
    "LinearSurrogateModel",
    "PerturbationResult",
    "SurrogateValidation",
    "fit_linear_surrogate",
    "run_predictor_feature_sensitivity",
    "run_virtual_perturbation",
]
