"""Small predictive-model contracts for neural dynamics estimators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class UnsupportedForecastError(NotImplementedError):
    """Raised when a graph estimator has no neural forecast semantics."""


@dataclass(frozen=True)
class PredictiveCapabilities:
    independent_window: bool = False
    stateful_online: bool = False
    retrospective_sequence: bool = False
    neural_forecast: bool = False
    normalized_density: bool = False


@dataclass(frozen=True)
class FilterState:
    model_hash: str
    history: np.ndarray
    sample_index: int | None = None
    recording_identity: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NeuralForecast:
    mean: np.ndarray
    horizon_samples: int
    origin_sample: int | None
    mask: np.ndarray | None = None
    log_density_nats: float | None = None
    failure_reason: str | None = None
    model_hash: str | None = None
    state_hash: str | None = None
    raw_support: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEstimate:
    matrix: np.ndarray
    lagged_coefficients: np.ndarray
    coefficient_semantics: str
    model_hash: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


class FittedDynamicsModel(Protocol):
    capabilities: PredictiveCapabilities
    model_hash: str

    def initial_state(self, *, recording_identity: Any = None) -> FilterState:
        """Create an empty prospective state for a recording."""

    def observe(
        self,
        state: FilterState,
        samples: np.ndarray,
        *,
        sample_indices: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        available_inputs: np.ndarray | None = None,
    ) -> FilterState:
        """Update the prospective state from newly observed samples."""

    def forecast(
        self,
        state: FilterState,
        *,
        horizon_samples: int,
        known_future_inputs: np.ndarray | None = None,
    ) -> NeuralForecast:
        """Forecast future neural means from the prospective state."""

    def graph_at(self, state: FilterState | None = None) -> GraphEstimate:
        """Return a graph snapshot at the provided prospective state."""
