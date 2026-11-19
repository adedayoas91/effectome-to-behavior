"""Training-owned transition nominations and paired intervention evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .behavior_readout import FrozenBehaviorMap, fixed_behavior_readouts
from .behavior_transition_forecast import TransitionForecast
from .continuous_behavior_rollouts import paired_behavior_effects
from .structural_mechanism import AdditiveMechanism, MechanismRolloutConfig

UNITS = {
    **{f"neuron:{node}": (node,) for node in range(10)},
    **{f"group:{node},{node + 1}": (node, node + 1) for node in range(0, 10, 2)},
}
CONTEXTS = ("decrease", "increase")


def forecast_context(prediction: np.ndarray) -> np.ndarray:
    """Assign contexts using only the first coordinate of a frozen forecast."""
    return np.where(prediction[:, 0] >= 0, CONTEXTS[1], CONTEXTS[0])


def nominate_transitions(
    predictor: TransitionForecast,
    features: np.ndarray,
    states: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Freeze nominations on trials reserved from encoder/forecast fitting."""
    full = predictor.predict(features, states, weights)
    contexts = forecast_context(full)
    full_error = np.abs(full - target).mean(axis=1)
    losses, matching = {}, {}
    for unit, members in UNITS.items():
        modified = weights.copy()
        modified[:, members, :] = 0.0
        omitted = predictor.predict(features, states, modified)
        losses[unit] = np.abs(omitted - target).mean(axis=1) - full_error
        activity = np.sqrt(np.mean(states[:, members] ** 2, axis=1))
        degree = np.count_nonzero(weights[:, members, :], axis=2).mean(axis=1)
        matching[unit] = np.column_stack((activity, degree))
    result: dict[str, Any] = {}
    for context in CONTEXTS:
        rows = contexts == context
        if not rows.any():
            result[context] = {"status": "unavailable", "reason": "no nomination events", "events": 0}
            continue
        scores = {unit: float(values[rows].mean()) for unit, values in losses.items()}
        selected: dict[str, dict[str, str | float | None]] = {}
        for kind in ("neuron", "group"):
            units = [unit for unit in UNITS if unit.startswith(f"{kind}:")]
            candidate = max(units, key=scores.__getitem__)
            if scores[candidate] <= 0:
                selected[kind] = {"candidate": None, "matched_control": None}
                continue
            descriptors = np.stack([matching[unit][rows].mean(axis=0) for unit in units])
            scale = descriptors.std(axis=0)
            scale[scale == 0] = 1
            distance = np.linalg.norm((descriptors - descriptors[units.index(candidate)]) / scale, axis=1)
            distance[units.index(candidate)] = np.inf
            nearest = int(np.argmin(distance))
            selected[kind] = {
                "candidate": candidate,
                "matched_control": units[nearest],
                "matching_distance": float(distance[nearest]),
            }
        result[context] = {
            "status": "available",
            "events": int(rows.sum()),
            "omission_mae_increase": scores,
            "selections": selected,
        }
    return result


def paired_transition_outcomes(
    models: Sequence[AdditiveMechanism],
    states: np.ndarray,
    origins: np.ndarray,
    config: MechanismRolloutConfig,
    readout: FrozenBehaviorMap | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return factual forecasts and paired effects with common origin innovations."""
    if len(models) != len(states) or len(origins) != len(states) or not len(states):
        raise ValueError("every forecast origin must have one state and one mechanism")
    gold = fixed_behavior_readouts()["behavior_reference"]
    readouts = {"behavior_reference": gold}
    arm = "behavior_reference"
    if readout is not None:
        arm = readout.name
        readouts[arm] = readout
    grouped: dict[int, list[int]] = {}
    for index, model in enumerate(models):
        grouped.setdefault(id(model), []).append(index)
    factual = np.empty((len(states), 2))
    vectors = {name: np.empty((len(states), 2)) for name in UNITS}
    for indices in grouped.values():
        outcomes = paired_behavior_effects(
            models[indices[0]],
            states[indices],
            UNITS,
            readouts=readouts,
            config=config,
            origin_ids=origins[indices],
        )
        summary = outcomes["readouts"][arm]
        factual[indices] = summary["factual_mean"]
        for name, values in summary["effect_vectors"].items():
            vectors[name][indices] = values
    return factual, vectors


def summarize_interventions(
    prediction: np.ndarray,
    regimes: np.ndarray,
    truth: dict[str, np.ndarray],
    estimated: dict[str, np.ndarray],
    static: dict[str, np.ndarray],
    nominations: dict[str, Any],
    active_floor: float,
) -> dict[str, Any]:
    contexts = forecast_context(prediction)
    results: dict[str, Any] = {"directions": {}, "oracle_regimes": {}}
    gold = np.stack([truth[unit] for unit in UNITS], axis=1)
    active = np.linalg.norm(gold, axis=2) / np.sqrt(2) >= active_floor
    for name, values in (("dynamic", estimated), ("static", static)):
        error = np.abs(np.stack([values[unit] for unit in UNITS], axis=1) - gold).mean(axis=2)
        results[f"{name}_all_effect_mae"] = float(error.mean())
        results[f"{name}_active_effect_mae"] = float(error[active].mean()) if active.any() else None
    for regime in np.unique(regimes):
        rows = regimes == regime
        results["oracle_regimes"][str(regime)] = {
            "events": int(rows.sum()),
            "mean_truth_magnitude": {
                unit: float(np.linalg.norm(values[rows], axis=1).mean() / np.sqrt(2))
                for unit, values in truth.items()
            },
        }
    for context, nomination in nominations.items():
        rows = contexts == context
        result: dict[str, Any] = {"events": int(rows.sum()), "nominations": {}}
        results["directions"][context] = result
        if not rows.any():
            continue
        for kind, selection in nomination.get("selections", {}).items():
            metrics = {}
            for role in ("candidate", "matched_control"):
                unit = selection[role]
                if unit is None:
                    continue
                gold = truth[unit][rows]
                effect = estimated[unit][rows]
                magnitude = np.linalg.norm(gold, axis=1) / np.sqrt(2)
                active = magnitude >= active_floor
                metrics[role] = {
                    "unit": unit,
                    "active_events": int(active.sum()),
                    "gold_mean_magnitude": float(magnitude.mean()),
                    "dynamic_mae": float(np.abs(effect - gold).mean()),
                    "static_mae": float(np.abs(static[unit][rows] - gold).mean()),
                    "active_direction_agreement": (
                        float((np.sum(effect[active] * gold[active], axis=1) > 0).mean())
                        if active.any()
                        else None
                    ),
                }
            result["nominations"][kind] = metrics
    return results
