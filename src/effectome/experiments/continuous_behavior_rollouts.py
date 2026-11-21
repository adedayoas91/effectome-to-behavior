"""Continuous behavioral measurements on shared neural intervention futures.

Readouts predict each neural sample before target-window averaging. Neural
simulation remains in the full observed state; this module assumes no closed
latent dynamics. See ``docs/behavior_supervised_experiment_plan.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from effectome.experiments.structural_mechanism import (
    AdditiveMechanism,
    MechanismRolloutConfig,
    _disconnected,
)
from effectome.workflows.resume import content_fingerprint

_FACTUAL = "__factual__"
_OUTPUT_DIMENSION = 2


class ContinuousBehaviorReadout(Protocol):
    """Frozen pointwise prediction ending in two behavioral coordinates."""

    def predict(self, states: np.ndarray) -> np.ndarray: ...

    def to_dict(self) -> dict[str, Any]: ...


def _finite(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _artifact(readout: ContinuousBehaviorReadout) -> dict[str, Any]:
    try:
        artifact = json.loads(json.dumps(readout.to_dict(), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("readout artifacts must be finite JSON-safe mappings") from error
    if not isinstance(artifact, dict):
        raise ValueError("readout artifacts must be finite JSON-safe mappings")
    return artifact


def _predict(
    readout: ContinuousBehaviorReadout,
    states: np.ndarray,
    name: str,
    *,
    output_width: int = _OUTPUT_DIMENSION,
) -> np.ndarray:
    shape = (*states.shape[:-1], output_width)
    # A copy prevents a readout's scratch buffer from changing earlier summaries.
    return _finite(readout.predict(states), shape, f"{name} prediction").copy()


def paired_behavior_effects(
    model: AdditiveMechanism,
    initial_states: np.ndarray,
    interventions: Mapping[str, Sequence[int]],
    *,
    readouts: Mapping[str, ContinuousBehaviorReadout],
    config: MechanismRolloutConfig | None = None,
    origin_ids: np.ndarray | None = None,
    training_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    retain_samples: bool = False,
    reference: str = "behavior_reference",
) -> dict[str, Any]:
    """Measure all readouts on the same paired, sample-wise neural futures.

    Effects are factual minus disconnected expected future behavior, with each
    target decoded pointwise and then averaged over the last ``target_length``
    future samples. All readouts use identical origins without selection.
    Discrepancies retain the pathwise information needed for an empirical
    triangle bound, even when individual Monte Carlo samples are not retained.
    """
    if config is None:
        config = MechanismRolloutConfig()
    elif not isinstance(config, MechanismRolloutConfig):
        raise ValueError("config must be a MechanismRolloutConfig")
    if type(retain_samples) is not bool:
        raise ValueError("retain_samples must be a boolean")
    initial = np.array(initial_states, dtype=float, copy=True)
    if (
        initial.ndim != 2
        or not initial.size
        or initial.shape[1] != len(model.intercept)
        or not np.isfinite(initial).all()
    ):
        raise ValueError("initial_states must be finite nonempty origin-by-node observations")
    initial.setflags(write=False)
    n_origins, n_nodes = initial.shape
    ids = np.arange(n_origins) if origin_ids is None else np.asarray(origin_ids)
    if (
        ids.shape != (n_origins,)
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 0)
        or len(np.unique(ids)) != n_origins
    ):
        raise ValueError("origin_ids must be unique nonnegative integer origin identifiers")
    if (
        not isinstance(interventions, Mapping)
        or not interventions
        or any(not isinstance(name, str) or not name or name == _FACTUAL for name in interventions)
    ):
        raise ValueError("interventions need nonempty unit names; __factual__ is reserved")
    units = {name: tuple(sorted(_disconnected(members, n_nodes))) for name, members in interventions.items()}
    if any(not members for members in units.values()):
        raise ValueError("each intervention must disconnect at least one source")
    if (
        not isinstance(readouts, Mapping)
        or not readouts
        or any(not isinstance(name, str) or not name for name in readouts)
    ):
        raise ValueError("readouts must be a nonempty mapping with nonempty string names")
    if not isinstance(reference, str) or reference not in readouts:
        raise ValueError("the behavioral discrepancy reference must identify a supplied readout")
    for readout in readouts.values():
        if any(not callable(getattr(readout, method, None)) for method in ("predict", "to_dict")):
            raise ValueError("each readout must implement predict and to_dict")
    artifacts = {name: _artifact(readout) for name, readout in readouts.items()}
    initial_behavior = {name: _predict(readout, initial, name) for name, readout in readouts.items()}
    bounds = None
    if training_bounds is not None:
        if not isinstance(training_bounds, tuple) or len(training_bounds) != 2:
            raise ValueError("training_bounds must be a (lower, upper) tuple")
        lower = _finite(training_bounds[0], (n_nodes,), "lower training bounds")
        upper = _finite(training_bounds[1], (n_nodes,), "upper training bounds")
        if np.any(lower > upper):
            raise ValueError("training bounds must be ordered node vectors")
        bounds = (lower, upper)
    innovations = np.stack(
        [
            np.random.default_rng(np.random.SeedSequence([config.seed, int(origin)])).normal(
                size=(config.horizon, config.n_rollouts, n_nodes)
            )
            for origin in ids
        ],
        axis=1,
    )
    means: dict[str, dict[str, np.ndarray]] = {name: {} for name in readouts}
    discrepancies: dict[str, dict[str, np.ndarray]] = {name: {} for name in readouts}
    retained: dict[str, dict[str, np.ndarray]] = {name: {} for name in readouts}
    diagnostics: dict[str, dict[str, Any]] = {
        name: {"scope": "target_window_mean_behavior_across_origins_and_rollouts", "branches": {}}
        for name in readouts
    }
    moments = {name: (np.zeros(_OUTPUT_DIMENSION), np.zeros(_OUTPUT_DIMENSION)) for name in readouts}
    support: dict[str, Any] = {}
    if bounds is not None:
        outside = np.any((initial < bounds[0]) | (initial > bounds[1]), axis=-1)
        support = {
            "scope": "raw_neural_training_minmax_envelope_no_filtering",
            "initial_outside_by_origin": outside.tolist(),
            "initial_fraction_outside": float(outside.mean()),
            "branches": {},
        }
    count = 0
    for branch, members in {_FACTUAL: (), **units}.items():
        states = np.repeat(initial[:, None, :], config.n_rollouts, axis=1)
        target_sum = {name: np.zeros((n_origins, config.n_rollouts, _OUTPUT_DIMENSION)) for name in readouts}
        outside_count = np.zeros(n_origins, dtype=np.int64)
        for step in range(config.horizon):
            states = model.step(states, innovations[step], members)
            if step < config.horizon - config.target_length:
                continue
            states.setflags(write=False)
            for name, readout in readouts.items():
                # Pointwise decoding precedes both time and rollout averaging.
                target_sum[name] += _predict(readout, states, name)
            if bounds is not None:
                outside_count += np.any((states < bounds[0]) | (states > bounds[1]), axis=-1).sum(axis=1)
        targets = {name: values / config.target_length for name, values in target_sum.items()}
        true_behavior = targets[reference]
        for name, target in targets.items():
            target.setflags(write=False)
            means[name][branch] = target.mean(axis=1)
            discrepancies[name][branch] = np.mean(np.abs(target - true_behavior), axis=(1, 2))
            std = target.std(axis=(0, 1))
            diagnostics[name]["branches"][branch] = {
                "prediction_std_by_output": std.tolist(),
                "sample_count": int(n_origins * config.n_rollouts),
                "collapsed": bool(np.max(std) <= 1e-12),
            }
            first, second = moments[name]
            first += target.sum(axis=(0, 1))
            second += np.square(target).sum(axis=(0, 1))
            if retain_samples:
                retained[name][branch] = target
        count += n_origins * config.n_rollouts
        if bounds is not None:
            denominator = config.target_length * config.n_rollouts
            fraction = outside_count / denominator
            support["branches"][branch] = {
                "fraction_outside": float(fraction.mean()),
                "per_origin_fraction": fraction.tolist(),
                "target_sample_count": int(denominator * n_origins),
                "observations_removed": 0,
            }
    for name, readout in readouts.items():
        if _artifact(readout) != artifacts[name]:
            raise ValueError("readout artifacts changed during inference; fitting/refitting is forbidden")
        first, second = moments[name]
        variance = np.maximum(0.0, second / count - np.square(first / count))
        std = np.sqrt(variance)
        diagnostics[name].update(
            {
                "prediction_std_by_output": std.tolist(),
                "collapsed": bool(np.max(std) <= 1e-12),
                "collapse_tolerance": 1e-12,
            }
        )
    metadata = {
        "origin_ids": ids.tolist(),
        "initial_state_fingerprint": content_fingerprint(initial),
        "horizon": config.horizon,
        "target_length": config.target_length,
        "n_rollouts": config.n_rollouts,
        "seed": config.seed,
        "output_dimension": _OUTPUT_DIMENSION,
        "interventions": {name: list(members) for name, members in units.items()},
        "paired_innovations": True,
        "shared_neural_futures_across_readouts": True,
        "noise_source": model.metadata.get("noise_source", "caller_supplied_model"),
        "intervention": "disconnect_outgoing_cross_terms_retain_self_and_source_evolution",
        "effect_definition": "factual_expected_future_behavior_minus_disconnected_expected_future_behavior",
        "aggregation": "pointwise_decode_then_target_time_mean_then_rollout_mean",
        "magnitude_definition": "per_origin_L2_effect_divided_by_sqrt_output_dimension",
        "conditioning": "identical_factual_origins_for_all_readouts_no_filtering",
        "discrepancy_reference": reference,
        "retained_samples": retain_samples,
    }
    summaries = {}
    for name in readouts:
        factual = means[name][_FACTUAL]
        disconnected = {unit: means[name][unit] for unit in units}
        effects = {unit: factual - value for unit, value in disconnected.items()}
        summaries[name] = {
            "initial_behavior": initial_behavior[name],
            "factual_mean": factual,
            "disconnected_means": disconnected,
            "effect_vectors": effects,
            "effect_magnitudes": {
                unit: np.linalg.norm(value, axis=1) / np.sqrt(_OUTPUT_DIMENSION)
                for unit, value in effects.items()
            },
            "metadata": {**metadata, "readout_artifact": artifacts[name]},
        }
    result = {
        "readouts": summaries,
        "readout_discrepancies": discrepancies,
        "metadata": metadata,
        "support_diagnostics": support,
        "readout_diagnostics": diagnostics,
    }
    if retain_samples:
        result["target_behavior_means"] = retained
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _checked_run(run: Mapping[str, Any], reference: str) -> tuple[list[str], list[str], int]:
    """Check saved summary algebra and retained samples before comparing effects."""
    readouts = _mapping(run.get("readouts"), "readouts")
    metadata = _mapping(run.get("metadata"), "metadata")
    if not readouts or reference not in readouts or metadata.get("discrepancy_reference") != reference:
        raise ValueError("comparison reference must match the run's behavioral discrepancy reference")
    ids = np.asarray(metadata.get("origin_ids"))
    if (
        ids.ndim != 1
        or not ids.size
        or not np.issubdtype(ids.dtype, np.integer)
        or np.any(ids < 0)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("saved origin_ids must be unique nonnegative integer identifiers")
    n_origins = len(ids)
    if metadata.get("output_dimension") != _OUTPUT_DIMENSION:
        raise ValueError("continuous behavior must have two declared output coordinates")
    if type(metadata.get("n_rollouts")) is not int or metadata["n_rollouts"] < 1:
        raise ValueError("saved n_rollouts must be a positive integer")
    units = list(_mapping(metadata.get("interventions"), "interventions"))
    if not units:
        raise ValueError("effect comparisons require at least one intervention")
    branches = {_FACTUAL, *units}
    discrepancies = _mapping(run.get("readout_discrepancies"), "readout discrepancies")
    if set(readouts) != set(discrepancies):
        raise ValueError("readouts and discrepancies must identify the same arms")
    saved_samples = run.get("target_behavior_means")
    if saved_samples is not None:
        saved_samples = _mapping(saved_samples, "target behavior samples")
        if set(saved_samples) != set(readouts):
            raise ValueError("retained samples must identify every readout")
    branch_means: dict[str, dict[str, np.ndarray]] = {}
    for arm, raw_summary in readouts.items():
        summary = _mapping(raw_summary, f"{arm} summary")
        _finite(summary.get("initial_behavior"), (n_origins, _OUTPUT_DIMENSION), f"{arm} initial behavior")
        factual = _finite(summary.get("factual_mean"), (n_origins, _OUTPUT_DIMENSION), f"{arm} factual mean")
        disconnected = _mapping(summary.get("disconnected_means"), f"{arm} disconnected means")
        effects = _mapping(summary.get("effect_vectors"), f"{arm} effects")
        magnitudes = _mapping(summary.get("effect_magnitudes"), f"{arm} magnitudes")
        arm_discrepancies = _mapping(discrepancies[arm], f"{arm} discrepancies")
        if (
            any(set(values) != set(units) for values in (disconnected, effects, magnitudes))
            or set(arm_discrepancies) != branches
        ):
            raise ValueError("all readouts must share exactly the declared intervention branches")
        branch_means[arm] = {_FACTUAL: factual}
        for unit in units:
            value = _finite(disconnected[unit], (n_origins, _OUTPUT_DIMENSION), f"{arm}/{unit} mean")
            delta = _finite(effects[unit], (n_origins, _OUTPUT_DIMENSION), f"{arm}/{unit} effect")
            magnitude = _finite(magnitudes[unit], (n_origins,), f"{arm}/{unit} magnitude")
            if not np.allclose(delta, factual - value, atol=1e-12, rtol=0) or not np.allclose(
                magnitude, np.linalg.norm(delta, axis=1) / np.sqrt(_OUTPUT_DIMENSION), atol=1e-12, rtol=0
            ):
                raise ValueError("saved means, effects and magnitudes are inconsistent")
            branch_means[arm][unit] = value
        if saved_samples is not None:
            samples = _mapping(saved_samples[arm], f"{arm} retained samples")
            if set(samples) != branches:
                raise ValueError("retained samples must share all intervention branches")
            for branch in branches:
                sample = _finite(
                    samples[branch],
                    (n_origins, metadata["n_rollouts"], _OUTPUT_DIMENSION),
                    f"{arm}/{branch} retained sample",
                )
                if not np.allclose(sample.mean(axis=1), branch_means[arm][branch], atol=1e-12, rtol=0):
                    raise ValueError("retained samples and saved means are inconsistent")
    for arm in readouts:
        for branch in branches:
            values = _finite(discrepancies[arm][branch], (n_origins,), f"{arm}/{branch} discrepancy")
            jensen_lower = np.mean(
                np.abs(branch_means[arm][branch] - branch_means[reference][branch]), axis=1
            )
            if np.any(values < 0) or np.any(values + 1e-12 < jensen_lower):
                raise ValueError("saved pathwise behavioral discrepancies are inconsistent with means")
            if arm == reference and np.any(np.abs(values) > 1e-12):
                raise ValueError("reference discrepancy must be zero")
            if saved_samples is not None:
                sample_error = np.mean(
                    np.abs(
                        np.asarray(saved_samples[arm][branch], dtype=float)
                        - np.asarray(saved_samples[reference][branch], dtype=float)
                    ),
                    axis=(1, 2),
                )
                if not np.allclose(values, sample_error, atol=1e-12, rtol=0):
                    raise ValueError("retained samples and pathwise discrepancies are inconsistent")
    return list(readouts), units, n_origins


def continuous_semantics_errors(
    truth: Mapping[str, Any],
    estimated: Mapping[str, Any],
    *,
    reference: str = "behavior_reference",
) -> dict[str, dict[str, Any]]:
    """Separate neural-model and behavioral-readout error using paired oracle paths.

    The empirical L1 triangle bound is checked per origin and intervention before
    averaging. It is a coupled finite-sample arithmetic bound, not a population
    confidence guarantee. Source histories and intervention definitions must
    match; no chart-dependent selection or refitting is permitted.
    """
    arms, units, n_origins = _checked_run(truth, reference)
    estimate_arms, estimate_units, estimate_origins = _checked_run(estimated, reference)
    if set(arms) != set(estimate_arms) or set(units) != set(estimate_units) or n_origins != estimate_origins:
        raise ValueError("oracle and estimated effects require matching readouts, interventions and origins")
    for key in (
        "origin_ids",
        "initial_state_fingerprint",
        "seed",
        "horizon",
        "target_length",
        "n_rollouts",
        "output_dimension",
        "interventions",
        "aggregation",
        "effect_definition",
    ):
        if truth["metadata"].get(key) != estimated["metadata"].get(key):
            raise ValueError(
                "oracle and estimated effects must share origins, initial states, innovations "
                "and intervention/outcome definitions"
            )
    gold_summary = truth["readouts"][reference]
    results = {}
    for arm in arms:
        oracle = truth["readouts"][arm]
        fitted = estimated["readouts"][arm]
        if oracle["metadata"]["readout_artifact"] != fitted["metadata"]["readout_artifact"]:
            raise ValueError("oracle and estimated effects must use the same frozen readout artifact")
        if not np.array_equal(oracle["initial_behavior"], fitted["initial_behavior"]):
            raise ValueError("the initial behavioral readout must remain frozen across neural models")
        factual = np.asarray(truth["readout_discrepancies"][arm][_FACTUAL], dtype=float)
        per_unit = {}
        total_errors, within_errors, semantic_errors, bounds, intervened_errors = [], [], [], [], []
        violations = 0
        for unit in units:
            gold = np.asarray(gold_summary["effect_vectors"][unit], dtype=float)
            oracle_effect = np.asarray(oracle["effect_vectors"][unit], dtype=float)
            fitted_effect = np.asarray(fitted["effect_vectors"][unit], dtype=float)
            total = np.mean(np.abs(fitted_effect - gold), axis=1)
            within = np.mean(np.abs(fitted_effect - oracle_effect), axis=1)
            semantic = np.mean(np.abs(oracle_effect - gold), axis=1)
            intervened = np.asarray(truth["readout_discrepancies"][arm][unit], dtype=float)
            bound = within + factual + intervened
            violation_count = int(np.count_nonzero(total > bound + 1e-12))
            violations += violation_count
            total_errors.extend(total.tolist())
            within_errors.extend(within.tolist())
            semantic_errors.extend(semantic.tolist())
            bounds.extend(bound.tolist())
            intervened_errors.extend(intervened.tolist())
            per_unit[unit] = {
                "end_to_end_absolute_error": total.tolist(),
                "within_chart_absolute_error": within.tolist(),
                "semantic_absolute_error": semantic.tolist(),
                "triangle_bound": bound.tolist(),
                "intervened_behavior_disagreement": intervened.tolist(),
                "gold_effect_vectors": gold.tolist(),
                "true_chart_effect_vectors": oracle_effect.tolist(),
                "estimated_effect_vectors": fitted_effect.tolist(),
                "gold_effect_magnitudes": np.asarray(gold_summary["effect_magnitudes"][unit]).tolist(),
                "estimated_effect_magnitudes": np.asarray(fitted["effect_magnitudes"][unit]).tolist(),
                "bound_violation_count": violation_count,
            }
        results[arm] = {
            "end_to_end_mae": float(np.mean(total_errors)),
            "within_chart_mae": float(np.mean(within_errors)),
            "semantic_mae": float(np.mean(semantic_errors)),
            "mean_triangle_bound": float(np.mean(bounds)),
            "bound_violation_count": violations,
            "factual_behavior_disagreement": factual.tolist(),
            "mean_factual_behavior_disagreement": float(factual.mean()),
            "mean_intervened_behavior_disagreement": float(np.mean(intervened_errors)),
            "per_unit": per_unit,
            "origin_count": n_origins,
            "intervention_count": len(units),
            "output_dimension": _OUTPUT_DIMENSION,
            "averaging": "per_origin_candidate_L1_divided_by_output_dimension_then_mean_no_filtering",
            "units": "declared_continuous_behavior_units_not_probability_effects",
        }
    return results
