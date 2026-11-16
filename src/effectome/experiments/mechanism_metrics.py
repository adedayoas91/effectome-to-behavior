"""Development summaries in probability-effect units, separate from omission nats."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.stats import beta


def _tie_aware_recall(scores: Mapping[str, float], positive: set[str]) -> float | None:
    """Expected top-k recall under random ordering of ties, including empty graphs."""
    if not positive:
        return None
    cutoff = sorted(scores.values(), reverse=True)[len(positive) - 1]
    above = {name for name, value in scores.items() if value > cutoff + 1e-12}
    tied = {name for name, value in scores.items() if abs(value - cutoff) <= 1e-12}
    remaining = len(positive) - len(above)
    recovered = len(positive & above) + len(positive & tied) * remaining / len(tied)
    return recovered / len(positive)


def mechanism_recovery_metrics(
    truth: dict[str, Any],
    estimated: dict[str, Any],
    observed_transitions: np.ndarray,
    *,
    effect_threshold: float,
    nomination_threshold: float,
) -> dict[str, Any]:
    """Keep directional ranking, intervention agreement and prediction distinct."""
    sources = np.asarray(truth["source_states"])
    factual = np.asarray(estimated["factual_transition_probability"])
    observed = np.asarray(observed_transitions, dtype=bool)
    errors = [np.abs(np.asarray(estimated["probability_reductions"][name]) - values)
              for name, values in truth["probability_reductions"].items()]
    directions: list[dict[str, Any]] = []
    for source in (0, 1):
        selected = sources == source
        if not selected.any():
            directions.append({"source_state": source, "status": "unsupported", "n_origins": 0})
            continue
        gold = {name: float(np.asarray(values)[selected].mean())
                for name, values in truth["probability_reductions"].items()}
        scores = {name: float(np.asarray(values)[selected].mean())
                  for name, values in estimated["probability_reductions"].items()}
        positive = {name for name, value in gold.items()
                    if name.startswith("neuron:") and value >= effect_threshold}
        nominees = {name for name, value in scores.items() if value > nomination_threshold}
        false = nominees - {name for name, value in gold.items() if value >= effect_threshold}
        neuron_scores = {name: value for name, value in scores.items() if name.startswith("neuron:")}
        directions.append({
            "source_state": source,
            "destination_state": 1 - source,
            "status": "available",
            "n_origins": int(selected.sum()),
            "n_events": int(observed[selected].sum()),
            "true_probability_effects": gold,
            "estimated_probability_effects": scores,
            "intervention_positive_neurons": sorted(positive),
            "nominated_units": sorted(nominees),
            "false_nominated_units": sorted(false),
            "driver_nomination_recall": len(positive & nominees) / len(positive) if positive else None,
            "topk_recall": _tie_aware_recall(neuron_scores, positive),
            "mean_absolute_effect_error": float(np.mean([abs(scores[name] - gold[name]) for name in gold])),
        })
    clipped = np.clip(factual, 1e-12, 1 - 1e-12)
    return {
        "mean_absolute_effect_error": float(np.mean(errors)),
        "factual_brier": float(np.mean((factual - observed) ** 2)),
        "factual_log_loss": float(-np.mean(np.where(observed, np.log(clipped), np.log1p(-clipped)))),
        "directions": directions,
        "any_false_nomination": any(row.get("false_nominated_units") for row in directions),
        "effect_units": "factual_minus_outgoing_disconnected_transition_probability",
        "nomination_scope": "development_cutoff_not_calibrated_causal_significance",
    }


def summarize_mechanism_cells(
    cells: list[dict[str, Any]], planned_methods: list[str], planned_seeds: tuple[int, ...]
) -> dict[str, Any]:
    """Use raw-process seeds for repetition; retain unavailable cells explicitly."""
    summaries = {}
    for method in planned_methods:
        available = [cell for cell in cells if cell["methods"][method]["status"] == "available"]
        positive = [cell["methods"][method]["metrics"] for cell in available
                    if cell["scenario"] == "driver"]
        nulls = [cell["methods"][method]["metrics"] for cell in available if cell["scenario"] == "null"]
        recalls = [float(np.mean([direction["topk_recall"] for direction in result["directions"]
                                  if direction.get("topk_recall") is not None]))
                   for result in positive
                   if all(direction.get("topk_recall") is not None for direction in result["directions"])]
        errors = [result["mean_absolute_effect_error"] for result in positive]
        failures = sum(bool(result["any_false_nomination"]) for result in nulls)
        complete_null = len(nulls) == len(planned_seeds)
        upper = (float(beta.ppf(0.95, failures + 1, len(nulls) - failures))
                 if failures < len(nulls) else 1.0) if complete_null else None
        summaries[method] = {
            "available_cells": len(available),
            "planned_cells": 2 * len(planned_seeds),
            "positive_seeds_evaluated": len(positive),
            "positive_topk_eligible_seeds": len(recalls),
            "positive_topk_planned_seeds": len(planned_seeds),
            "positive_topk_eligible_seed_set_complete": len(recalls) == len(planned_seeds),
            "positive_mean_absolute_effect_error": float(np.mean(errors)) if errors else None,
            "positive_mean_topk_recall": float(np.mean(recalls))
            if len(recalls) == len(planned_seeds) else None,
            "null_seeds_evaluated": len(nulls),
            "null_seeds_with_any_false_nomination": failures,
            "null_one_sided_95_upper_bound": upper,
            "null_bound_scope": "per_method_independent_seed_event_not_simultaneous_across_methods",
            "scientific_validation_passed": False,
        }
    return summaries
