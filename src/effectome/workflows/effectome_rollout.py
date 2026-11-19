"""Forecast behavior by propagating a trailing-history neural mechanism."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from effectome.connectivity import ConnectivityConfig
from effectome.data_module.schema import Window
from effectome.experiments.behavior_transition_graphs import (
    discover_transition_supports,
    fit_transition_mechanism,
)
from effectome.experiments.continuous_behavior_rollouts import (
    ContinuousBehaviorReadout,
    paired_behavior_effects,
)
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig
from effectome.workflows.resume import content_fingerprint


@dataclass(frozen=True)
class EffectomeForecast:
    """Behavioral target means and paired factual-minus-disconnected effects."""

    factual: np.ndarray
    effects: dict[str, np.ndarray]


def _validate_origins(origin_ids: np.ndarray, count: int) -> None:
    if (
        origin_ids.shape != (count,)
        or not np.issubdtype(origin_ids.dtype, np.integer)
        or np.any(origin_ids < 0)
        or len(np.unique(origin_ids)) != count
    ):
        raise ValueError("every origin needs a distinct nonnegative integer identifier")


def forecast_mechanisms(
    mechanisms: Sequence[AdditiveMechanism],
    states: np.ndarray,
    origin_ids: np.ndarray,
    readout: ContinuousBehaviorReadout,
    interventions: Mapping[str, Sequence[int]],
    config: MechanismRolloutConfig,
) -> EffectomeForecast:
    """Decode every future sample using one frozen, training-owned readout."""
    if states.ndim != 2 or not len(states) or len(mechanisms) != len(states):
        raise ValueError("every origin needs one observed state and one fitted mechanism")
    _validate_origins(origin_ids, len(states))
    groups: dict[int, list[int]] = {}
    for index, mechanism in enumerate(mechanisms):
        groups.setdefault(id(mechanism), []).append(index)

    factual = np.empty((len(states), 2))
    effects = {name: np.empty_like(factual) for name in interventions}
    for indices in groups.values():
        result = paired_behavior_effects(
            mechanisms[indices[0]],
            states[indices],
            interventions,
            readouts={"behavior": readout},
            reference="behavior",
            config=config,
            origin_ids=origin_ids[indices],
        )["readouts"]["behavior"]
        factual[indices] = result["factual_mean"]
        for name, values in result["effect_vectors"].items():
            effects[name][indices] = values
    return EffectomeForecast(factual, effects)


def run_effectome_rollout(
    histories: np.ndarray,
    windows: list[Window],
    origin_ids: np.ndarray,
    *,
    method: ConnectivityConfig,
    fps: float,
    evidence: Path,
    signature: str,
    readout: ContinuousBehaviorReadout,
    interventions: Mapping[str, Sequence[int]],
    config: MechanismRolloutConfig,
    ridge_alpha: float = 1.0,
) -> tuple[EffectomeForecast, tuple[AdditiveMechanism, ...]]:
    """Fit signed local mechanisms from histories ending at the forecast origin."""
    if histories.ndim != 3 or not histories.size or histories.shape[1] < 3:
        raise ValueError("expected finite origin-by-time-by-neuron histories")
    if not np.isfinite(histories).all() or not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("histories must be finite and the ridge penalty must be positive")
    if len(windows) != len(histories) or any(w.length != histories.shape[1] or w.start < 0 for w in windows):
        raise ValueError("raw dependencies must describe the supplied trailing histories")
    _validate_origins(origin_ids, len(histories))
    # Include the ordered raw inputs in the evidence identity even when the caller
    # deliberately reuses a protocol label on a new set of forecast origins.
    identity = content_fingerprint((signature, histories, windows, method, fps, origin_ids))
    masks = discover_transition_supports(histories, windows, method, fps, evidence, identity)
    mechanisms = tuple(
        fit_transition_mechanism(history[None], support, ridge_alpha)
        for history, support in zip(histories, masks, strict=True)
    )
    return (
        forecast_mechanisms(mechanisms, histories[:, -1], origin_ids, readout, interventions, config),
        mechanisms,
    )
