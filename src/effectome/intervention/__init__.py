"""Coupling-aware surrogate modeling and virtual perturbations."""

from .surrogate import (
    LinearSurrogateModel,
    PerturbationResult,
    SurrogateValidation,
    fit_linear_surrogate,
    run_virtual_perturbation,
)

__all__ = [
    "LinearSurrogateModel",
    "PerturbationResult",
    "SurrogateValidation",
    "fit_linear_surrogate",
    "run_virtual_perturbation",
]
