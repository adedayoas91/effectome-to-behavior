"""Measure frozen behavioral endpoints on shared neural counterfactual futures.

The causal model remains in observed neural space. Endpoints receive the same
factual origins and the same generated futures, including off-support samples.
See ``docs/learned_manifold_validation_plan.md`` for the event-error bound and
the distinction between within-chart agreement and behavioral validity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from effectome.experiments.learned_endpoints import Endpoint as BinaryEndpoint
from effectome.experiments.structural_mechanism import (
    AdditiveMechanism,
    MechanismRolloutConfig,
    _disconnected,
)
from effectome.workflows.resume import content_fingerprint


def _binary(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must be binary with shape {shape}")
    return array.astype(bool)


def paired_endpoint_probabilities(
    model: AdditiveMechanism,
    initial_states: np.ndarray,
    interventions: Mapping[str, Sequence[int]],
    *,
    endpoints: Mapping[str, BinaryEndpoint],
    config: MechanismRolloutConfig | None = None,
    origin_ids: np.ndarray | None = None,
    training_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Reuse each neural branch for every endpoint, with paired innovations.

    ``target_states`` receives ``(target_length, origins, rollouts, nodes)``.
    Each endpoint's factual source is classified once, before any intervention.
    Events include disagreement with that source; source-label mistakes are not
    removed when comparing an endpoint with the independent behavior reference.
    """
    if config is None:
        config = MechanismRolloutConfig()
    elif not isinstance(config, MechanismRolloutConfig):
        raise ValueError("config must be a MechanismRolloutConfig")
    initial = np.array(initial_states, dtype=float, copy=True)
    if (initial.ndim != 2 or not initial.size or initial.shape[1] != len(model.intercept)
            or not np.isfinite(initial).all()):
        raise ValueError("initial_states must be finite nonempty origin-by-node observations")
    initial.setflags(write=False)
    n_origins, n_nodes = initial.shape
    ids = np.arange(n_origins) if origin_ids is None else np.asarray(origin_ids)
    if (ids.shape != (n_origins,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or len(np.unique(ids)) != n_origins):
        raise ValueError("origin_ids must be unique nonnegative integer origin identifiers")
    if not isinstance(interventions, Mapping) or any(
        not isinstance(name, str) or not name or name == "__factual__" for name in interventions
    ):
        raise ValueError("interventions need nonempty unit names; __factual__ is reserved")
    units = {name: _disconnected(members, n_nodes) for name, members in interventions.items()}
    if any(not members for members in units.values()):
        raise ValueError("each intervention must disconnect at least one source")
    if not isinstance(endpoints, Mapping) or not endpoints or any(
        not isinstance(name, str) or not name for name in endpoints
    ):
        raise ValueError("endpoints must be a nonempty mapping with nonempty string names")
    for endpoint in endpoints.values():
        if any(not callable(getattr(endpoint, method, None))
               for method in ("source_states", "target_states", "to_dict")):
            raise ValueError("each endpoint must implement source_states, target_states and to_dict")
    bounds = None
    if training_bounds is not None:
        if not isinstance(training_bounds, tuple) or len(training_bounds) != 2:
            raise ValueError("training_bounds must be a (lower, upper) tuple")
        lower, upper = (np.asarray(item, dtype=float) for item in training_bounds)
        if (lower.shape != (n_nodes,) or upper.shape != (n_nodes,)
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any(lower > upper)):
            raise ValueError("training bounds must be finite ordered node vectors")
        bounds = (lower, upper)

    sources = {name: _binary(endpoint.source_states(initial), (n_origins,), f"{name} sources")
               for name, endpoint in endpoints.items()}
    artifacts = {name: endpoint.to_dict() for name, endpoint in endpoints.items()}
    innovations = np.stack([
        np.random.default_rng(np.random.SeedSequence([config.seed, int(origin)])).normal(
            size=(config.horizon, config.n_rollouts, n_nodes)
        ) for origin in ids
    ], axis=1)
    events: dict[str, dict[str, np.ndarray]] = {name: {} for name in endpoints}
    support_diagnostics: dict[str, Any] = {}
    if bounds is not None:
        outside_initial = np.any((initial < bounds[0]) | (initial > bounds[1]), axis=-1)
        support_diagnostics = {
            "scope": "raw_neural_training_minmax_envelope_no_filtering",
            "initial_outside_by_origin": outside_initial.tolist(),
            "initial_fraction_outside": float(outside_initial.mean()),
            "branches": {},
        }
    for branch, members in {"__factual__": (), **units}.items():
        states = np.repeat(initial[:, None, :], config.n_rollouts, axis=1)
        target = np.empty((config.target_length, n_origins, config.n_rollouts, n_nodes))
        for step in range(config.horizon):
            states = model.step(states, innovations[step], members)
            if step >= config.horizon - config.target_length:
                target[step - config.horizon + config.target_length] = states
        target.setflags(write=False)
        for name, endpoint in endpoints.items():
            destinations = _binary(endpoint.target_states(target), (n_origins, config.n_rollouts),
                                   f"{name} destinations")
            events[name][branch] = destinations != sources[name][:, None]
        if bounds is not None:
            outside = np.any((target < bounds[0]) | (target > bounds[1]), axis=-1)
            support_diagnostics["branches"][branch] = {
                "fraction_outside": float(outside.mean()),
                "per_origin_fraction": outside.mean(axis=(0, 2)).tolist(),
                "target_sample_count": int(outside.size),
            }
    metadata = {
        "origin_ids": ids.tolist(), "horizon": config.horizon, "target_length": config.target_length,
        "initial_state_fingerprint": content_fingerprint(initial),
        "n_rollouts": config.n_rollouts, "seed": config.seed, "paired_innovations": True,
        "shared_neural_futures_across_endpoints": True,
        "noise_source": model.metadata.get("noise_source", "caller_supplied_model"),
        "intervention": "disconnect_outgoing_cross_terms_retain_self_and_source_evolution",
        "event": "future_endpoint_state_differs_from_its_frozen_factual_origin_state",
        "conditioning": "identical_factual_origins_for_all_endpoints_no_source_filtering",
    }
    summaries = {}
    for name in endpoints:
        factual = events[name]["__factual__"].mean(axis=1)
        disconnected = {unit: events[name][unit].mean(axis=1) for unit in units}
        summaries[name] = {
            "source_states": sources[name].astype(int),
            "factual_transition_probability": factual,
            "disconnected_transition_probabilities": disconnected,
            "probability_reductions": {unit: factual - value for unit, value in disconnected.items()},
            "metadata": {**metadata, "endpoint_artifact": artifacts[name]},
        }
    return {"endpoints": summaries, "events": events, "metadata": metadata,
            "support_diagnostics": support_diagnostics}


def _checked_events(run: Mapping[str, Any]) -> tuple[list[str], list[str], tuple[int, int]]:
    """Check event/probability coherence before using the empirical bound."""
    arms = list(run["endpoints"])
    if not arms or set(arms) != set(run["events"]):
        raise ValueError("endpoint summaries and events must identify the same nonempty arm set")
    first = np.asarray(run["events"][arms[0]]["__factual__"])
    if first.ndim != 2 or not first.size:
        raise ValueError("events must have nonempty origin-by-rollout axes")
    shape = (first.shape[0], first.shape[1])
    units = list(run["endpoints"][arms[0]]["probability_reductions"])
    if not units:
        raise ValueError("effect errors require at least one intervention")
    branches = {"__factual__", *units}
    for arm in arms:
        summary = run["endpoints"][arm]
        if (set(run["events"][arm]) != branches
                or set(summary["probability_reductions"]) != set(units)
                or set(summary["disconnected_transition_probabilities"]) != set(units)):
            raise ValueError("all endpoints must share the same intervention branches")
        _binary(summary["source_states"], (shape[0],), f"{arm} sources")
        probabilities = {}
        for branch in branches:
            event = _binary(run["events"][arm][branch], shape, f"{arm}/{branch} events")
            probabilities[branch] = event.mean(axis=1)
        expected = {"__factual__": summary["factual_transition_probability"],
                    **summary["disconnected_transition_probabilities"]}
        for branch in branches:
            values = np.asarray(expected[branch], dtype=float)
            if (values.shape != (shape[0],) or not np.isfinite(values).all()
                    or not np.allclose(values, probabilities[branch], rtol=0, atol=1e-12)):
                raise ValueError("event samples and transition probabilities are inconsistent")
        for unit in units:
            delta = np.asarray(summary["probability_reductions"][unit], dtype=float)
            if (delta.shape != (shape[0],) or not np.isfinite(delta).all()
                    or not np.allclose(delta, probabilities["__factual__"] - probabilities[unit],
                                       rtol=0, atol=1e-12)):
                raise ValueError("event samples and probability reductions are inconsistent")
    return arms, units, shape


def semantics_errors(
    truth_multi: Mapping[str, Any],
    estimated_multi: Mapping[str, Any],
    *,
    reference: str = "behavior_reference",
) -> dict[str, dict[str, Any]]:
    """Separate mechanism and behavioral measurement errors on common origins.

    This is an empirical, coupled-sample triangle inequality, not a population
    confidence bound. Absolute errors are taken per origin and intervention
    before averaging, so opposite errors cannot cancel. The source label is part
    of each event, including when learned and behavioral sources disagree.
    """
    arms, units, shape = _checked_events(truth_multi)
    estimate_arms, estimate_units, estimate_shape = _checked_events(estimated_multi)
    if (set(arms) != set(estimate_arms) or set(units) != set(estimate_units)
            or shape != estimate_shape or reference not in arms):
        raise ValueError("oracle and estimated effects need matching arms, units, samples and reference")
    for key in ("origin_ids", "initial_state_fingerprint", "horizon", "target_length", "n_rollouts", "seed"):
        if truth_multi["metadata"][key] != estimated_multi["metadata"][key]:
            raise ValueError("oracle and estimated effects must share origins, initial states, "
                             "innovations and outcome support")
    reference_summary = truth_multi["endpoints"][reference]
    reference_events = truth_multi["events"][reference]
    results = {}
    for arm in arms:
        true_summary = truth_multi["endpoints"][arm]
        estimate_summary = estimated_multi["endpoints"][arm]
        if true_summary["metadata"]["endpoint_artifact"] != estimate_summary["metadata"]["endpoint_artifact"]:
            raise ValueError("oracle and estimated effects must use the same frozen endpoint artifact")
        if not np.array_equal(true_summary["source_states"], estimate_summary["source_states"]):
            raise ValueError("factual source labels must be frozen across neural models")
        factual_error = np.mean(
            np.asarray(truth_multi["events"][arm]["__factual__"])
            != np.asarray(reference_events["__factual__"]), axis=1,
        )
        per_unit = {}
        end_to_end, within, semantic, bounds, disagreements = [], [], [], [], []
        violations = 0
        for unit in units:
            gold = np.asarray(reference_summary["probability_reductions"][unit], dtype=float)
            true_chart = np.asarray(true_summary["probability_reductions"][unit], dtype=float)
            estimated = np.asarray(estimate_summary["probability_reductions"][unit], dtype=float)
            total = np.abs(estimated - gold)
            mechanism = np.abs(estimated - true_chart)
            measurement = np.abs(true_chart - gold)
            intervened_error = np.mean(
                np.asarray(truth_multi["events"][arm][unit]) != np.asarray(reference_events[unit]), axis=1,
            )
            bound = mechanism + factual_error + intervened_error
            violation_count = int(np.count_nonzero(total > bound + 1e-12))
            violations += violation_count
            end_to_end.extend(total.tolist())
            within.extend(mechanism.tolist())
            semantic.extend(measurement.tolist())
            bounds.extend(bound.tolist())
            disagreements.extend(intervened_error.tolist())
            per_unit[unit] = {
                "end_to_end_absolute_error": total.tolist(),
                "within_chart_absolute_error": mechanism.tolist(),
                "semantic_absolute_error": measurement.tolist(),
                "intervened_event_disagreement": intervened_error.tolist(),
                "triangle_bound": bound.tolist(), "bound_violation_count": violation_count,
            }
        results[arm] = {
            "end_to_end_mae": float(np.mean(end_to_end)),
            "within_chart_mae": float(np.mean(within)),
            "semantic_mae": float(np.mean(semantic)),
            "mean_triangle_bound": float(np.mean(bounds)),
            "bound_violation_count": violations,
            "factual_event_disagreement": factual_error.tolist(),
            "mean_factual_event_disagreement": float(factual_error.mean()),
            "mean_intervened_event_disagreement": float(np.mean(disagreements)),
            "source_agreement": float(np.mean(np.asarray(true_summary["source_states"])
                                               == np.asarray(reference_summary["source_states"]))),
            "per_unit": per_unit, "origin_count": shape[0], "intervention_count": len(units),
            "averaging": "per_origin_candidate_absolute_errors_then_mean_no_source_filtering",
        }
    return results
