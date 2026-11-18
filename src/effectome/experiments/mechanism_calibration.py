"""Separate-null calibration of the maximum directional intervention score.

One raw-process seed contributes one maximum over all candidates and both
directions. The finite-sample order statistic controls the marginal *global-null*
event under exchangeable calibration/test seeds and an unchanged fitting/scoring
procedure. It does not give strong control of false nominations under non-null
mechanisms, simultaneous control across methods, or biological identification.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from decimal import ROUND_CEILING, Decimal
from numbers import Real
from typing import Any

from effectome.workflows.resume import content_fingerprint


def _finite_real(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _effect_scores(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must contain nonempty candidate scores")
    result = {}
    for unit, score in value.items():
        if not isinstance(unit, str) or ":" not in unit:
            raise ValueError(f"{label} has an invalid candidate label")
        kind, identifier = unit.split(":", 1)
        if kind not in {"neuron", "group"} or not identifier:
            raise ValueError(f"{label} has an invalid candidate label: {unit}")
        if kind == "neuron" and (not identifier.isdecimal() or str(int(identifier)) != identifier):
            raise ValueError(f"{label} has an invalid neuron label: {unit}")
        result[unit] = _finite_real(score, f"{label}/{unit}")
    if not any(name.startswith("neuron:") for name in result):
        raise ValueError(f"{label} must include neuron candidates")
    return result


def _directions(metrics: Any) -> list[dict[str, Any]]:
    if not isinstance(metrics, Mapping) or not isinstance(metrics.get("directions"), list):
        raise ValueError("metrics must contain the two source-state directions")
    directions = metrics["directions"]
    sources = [row.get("source_state") if isinstance(row, dict) else None for row in directions]
    if len(sources) != 2 or any(type(source) is not int for source in sources) or set(sources) != {0, 1}:
        raise ValueError("directions must have exactly one source_state 0 and one source_state 1")
    return sorted(directions, key=lambda row: row["source_state"])


def calibrate_seed_maxima(
    cells: list[dict[str, Any]],
    method: str,
    planned_seeds: tuple[int, ...],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Freeze a strict-exceedance cutoff using only the declared null seed pool.

    Non-null cells are ignored without inspecting their scores or gold effects.
    Missing/unavailable cells and unsupported directions block calibration rather
    than shrinking its denominator. Malformed evidence, duplicate/unplanned null
    seeds and an under-resolved alpha raise ``ValueError``. Pool disjointness
    against development and test seeds must be enforced by the study orchestrator.
    """
    level = _finite_real(alpha, "alpha")
    if not 0 < level < 1:
        raise ValueError("alpha must lie in (0,1)")
    if not isinstance(method, str) or not method:
        raise ValueError("method must be a nonempty name")
    if (not planned_seeds or any(type(seed) is not int or seed < 0 for seed in planned_seeds)
            or len(set(planned_seeds)) != len(planned_seeds)):
        raise ValueError("planned seeds must be distinct nonnegative integers")
    planned = sorted(planned_seeds)
    n = len(planned)
    # Decimal avoids ceil overshooting an integer rank due to binary roundoff.
    rank = int((Decimal(n + 1) * (1 - Decimal(str(level)))).to_integral_value(rounding=ROUND_CEILING))
    if rank > n:
        raise ValueError("alpha is under-resolved by the planned calibration seed count")

    maxima: list[dict[str, Any]] = []
    score_evidence: list[dict[str, Any]] = []
    reasons: list[dict[str, Any]] = []
    observed: set[int] = set()
    candidates: set[str] | None = None
    ignored = 0
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("each calibration cell must be a mapping")
        if cell.get("scenario") != "null":
            ignored += 1
            continue
        seed = cell.get("seed")
        if type(seed) is not int or seed < 0 or seed not in planned:
            raise ValueError("null cell seed must belong to the declared calibration pool")
        if seed in observed:
            raise ValueError("duplicate null calibration seed")
        observed.add(seed)
        methods = cell.get("methods", {})
        if not isinstance(methods, Mapping):
            raise ValueError("cell methods must be a mapping")
        result = methods.get(method)
        if result is not None and not isinstance(result, Mapping):
            raise ValueError("method result must be a mapping")
        if result is None or result.get("status") != "available":
            reasons.append({"seed": seed, "reason": "method missing or unavailable"})
            continue
        directional_scores = []
        supported = True
        local_candidates: set[str] | None = None
        for direction in _directions(result.get("metrics")):
            source = direction["source_state"]
            if direction.get("status") != "available":
                reasons.append({"seed": seed, "reason": f"source {source} direction unsupported"})
                supported = False
                continue
            origins = direction.get("n_origins")
            if type(origins) is not int or origins <= 0:
                reasons.append({"seed": seed, "reason": f"source {source} has no positive origin count"})
                supported = False
                continue
            scores = _effect_scores(direction.get("estimated_probability_effects"), "estimated effects")
            if local_candidates is not None and set(scores) != local_candidates:
                raise ValueError("candidate set differs across calibration directions")
            local_candidates = set(scores)
            directional_scores.append({"source_state": source, "scores": scores})
        if not supported:
            continue
        if candidates is not None and local_candidates != candidates:
            raise ValueError("candidate set differs across calibration seeds")
        candidates = local_candidates
        maximum = max(score for row in directional_scores for score in row["scores"].values())
        maxima.append({"seed": seed, "maximum_score": maximum})
        score_evidence.append({"seed": seed, "directions": directional_scores})

    available = sorted(row["seed"] for row in maxima)
    missing = sorted(set(planned) - observed)
    unavailable = sorted(observed - set(available))
    complete = len(available) == n
    threshold = max(0.0, sorted(row["maximum_score"] for row in maxima)[rank - 1]) if complete else None
    identity = {
        "method": method, "alpha": level, "rank": rank, "planned_seeds": planned,
        "score_evidence": sorted(score_evidence, key=lambda row: row["seed"]),
        "missing_seeds": missing, "unavailable_seeds": unavailable,
    }
    return {
        "status": "complete" if complete else "incomplete", "complete": complete,
        "method": method, "alpha": level, "rank": rank, "threshold": threshold,
        "planned_seed_count": n, "available_seed_count": len(available),
        "planned_seeds": planned, "available_seeds": available,
        "missing_seeds": missing, "unavailable_seeds": unavailable,
        "seed_maxima": sorted(maxima, key=lambda row: row["seed"]),
        "candidates": sorted(candidates) if candidates else [], "reasons": reasons,
        "ignored_non_null_cells": ignored,
        "nomination_comparison": "strictly_greater_than",
        "finite_sample_tail_probability_bound": (n + 1 - rank) / (n + 1) if complete else None,
        "scope": "per_method_whole_seed_global_null_not_strong_nonnull_or_simultaneous_control",
        "assumptions": "independent exchangeable null seeds; unchanged fitting/scoring/candidate procedure",
        "fingerprint": content_fingerprint(identity),
    }


def apply_nomination_threshold(
    metrics: dict[str, Any], threshold: float, effect_threshold: float = 0.1,
) -> dict[str, Any]:
    """Apply a frozen cutoff without refitting or altering scores and ranking.

    Stored oracle effects are used only to evaluate nominations. They play no part
    in score thresholding. The caller supplies the matching gold-effect threshold
    and must establish the cutoff's separate-calibration provenance.
    """
    cutoff = _finite_real(threshold, "threshold")
    gold_cutoff = _finite_real(effect_threshold, "effect_threshold")
    if cutoff < 0:
        raise ValueError("threshold must be nonnegative")
    if not 0 < gold_cutoff <= 1:
        raise ValueError("effect_threshold must lie in (0,1]")
    result = deepcopy(metrics)
    directions = _directions(result)
    complete = True
    for direction in directions:
        if direction.get("status") != "available":
            complete = False
            continue
        origins = direction.get("n_origins")
        if type(origins) is not int or origins <= 0:
            raise ValueError("available nomination direction requires positive origins")
        scores = _effect_scores(direction.get("estimated_probability_effects"), "estimated effects")
        gold = _effect_scores(direction.get("true_probability_effects"), "true effects")
        if set(scores) != set(gold):
            raise ValueError("estimated and true candidate sets differ")
        nominees = {name for name, value in scores.items() if value > cutoff}
        gold_units = {name for name, value in gold.items() if value >= gold_cutoff}
        positive = {name for name in gold_units if name.startswith("neuron:")}
        direction["nominated_units"] = sorted(nominees)
        direction["false_nominated_units"] = sorted(nominees - gold_units)
        direction["driver_nomination_recall"] = len(positive & nominees) / len(positive) if positive else None
    result["any_false_nomination"] = any(
        direction.get("false_nominated_units") for direction in directions
        if direction.get("status") == "available"
    )
    result["nomination_threshold"] = cutoff
    result["nomination_effect_threshold"] = gold_cutoff
    result["nomination_complete"] = complete
    result["nomination_scope"] = "calibration_seed_maximum_global_null_per_method_not_strong_nonnull_control"
    return result
