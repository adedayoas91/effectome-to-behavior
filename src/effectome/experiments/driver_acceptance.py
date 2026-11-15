"""Proposed, versioned gates for simulator-grounded transition-driver recovery.

Predictive nominations are compared with same-horizon simulator interventions.
They are not causal significance tests. Directions are averaged within each raw
simulation seed; missing fits or support never become successful replicates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

POSITIVE_SCENARIOS = ("single_driver", "redundant_group", "correlated_bystander")
REQUIRED_SCENARIOS = (*POSITIVE_SCENARIOS, "null")
SCENARIOS = (*REQUIRED_SCENARIOS, "hidden_confounder")
DIRECTIONS = ((0, 1), (1, 0))


@dataclass(frozen=True)
class DriverAcceptanceCriteria:
    version: str = "proposed-driver-validation-v1"
    min_positive_seeds: int = 20
    min_null_seeds: int = 60
    min_origins: int = 20
    min_events: int = 5
    min_alignment_accuracy: float = 0.8
    min_intervention_effect: float = 0.10
    nomination_threshold: float = 0.01
    min_availability: float = 0.9
    min_topk_recovery: float = 0.75
    min_group_win_fraction: float = 0.75
    min_mean_log_loss_gain: float = 0.0
    max_null_false_nomination_probability: float = 0.05
    confidence_level: float = 0.95
    bootstrap_replicates: int = 2000
    analysis_seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "min_positive_seeds",
            "min_null_seeds",
            "min_origins",
            "min_events",
            "bootstrap_replicates",
            "analysis_seed",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "analysis_seed" else 1):
                raise ValueError(f"{name} must be a nonnegative seed or positive integer count")
        if not self.version or self.bootstrap_replicates < 100:
            raise ValueError("a version and at least 100 bootstrap replicates are required")
        for name in (
            "min_alignment_accuracy",
            "min_availability",
            "min_topk_recovery",
            "min_group_win_fraction",
            "max_null_false_nomination_probability",
            "confidence_level",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must lie in (0,1]")
        if self.confidence_level == 1 or self.max_null_false_nomination_probability == 1:
            raise ValueError("confidence and null bounds must be strictly smaller than one")
        for name in ("min_intervention_effect", "nomination_threshold", "min_mean_log_loss_gain"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.min_intervention_effect <= 1 or self.nomination_threshold < 0:
            raise ValueError(
                "intervention effect must lie in (0,1]; nomination threshold must be nonnegative"
            )


def load_driver_acceptance_criteria(path: str | Path | None = None) -> DriverAcceptanceCriteria:
    from omegaconf import OmegaConf

    source = (
        Path(path) if path else Path(__file__).resolve().parents[3] / "conf/driver_validation/acceptance.yaml"
    )
    raw = OmegaConf.to_container(OmegaConf.load(source), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("driver acceptance criteria must be a mapping")
    return DriverAcceptanceCriteria(**{str(key): value for key, value in raw.items()})


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (float, int, np.number))
        and not isinstance(value, bool)
        and bool(np.isfinite(value))
    )


def _ineligible(row: Mapping[str, Any], criteria: DriverAcceptanceCriteria) -> list[str]:
    if row.get("status") != "available":
        return [str(row.get("reason") or "method_unavailable")]
    reasons = []
    for key, minimum in (
        ("n_origins", criteria.min_origins),
        ("n_events", criteria.min_events),
        ("alignment_accuracy", criteria.min_alignment_accuracy),
    ):
        if not _finite(row.get(key)) or row[key] < minimum:
            reasons.append(f"insufficient_{key}")
    if not _finite(row.get("log_loss_gain")):
        reasons.append("nonfinite_log_loss_gain")
    if _finite(row.get("alignment_accuracy")) and row["alignment_accuracy"] > 1:
        reasons.append("invalid_alignment_accuracy")
    if any(type(row.get(key)) is not int for key in ("n_origins", "n_events")):
        reasons.append("event_counts_must_be_integers")
    elif row["n_events"] > row["n_origins"]:
        reasons.append("event_count_exceeds_source_risk_set")
    scores, effects = row.get("driver_scores", {}), row.get("intervention_effects", {})
    if not scores or set(scores) != set(effects):
        reasons.append("incomplete_paired_unit_scores_and_interventions")
    elif any(not _finite(x) for x in (*scores.values(), *effects.values())):
        reasons.append("nonfinite_unit_score_or_intervention")
    elif any(abs(x) > 1 for x in effects.values()):
        reasons.append("intervention_probability_reduction_outside_bounds")
    return reasons


def _direction_metrics(row: Mapping[str, Any], criteria: DriverAcceptanceCriteria) -> dict[str, Any]:
    scores, effects = row["driver_scores"], row["intervention_effects"]
    gold = {
        u
        for u, effect in effects.items()
        if u.startswith("neuron:") and effect >= criteria.min_intervention_effect
    }
    ordered = sorted((u for u in scores if u.startswith("neuron:")), key=lambda u: (-scores[u], u))
    selected = set(ordered[: len(gold)]) if gold else set()
    recovery = len(selected & gold) / len(gold) if gold else None
    group_effect = effects.get("group:driver_pair")
    advantage = (
        scores["group:driver_pair"] - scores["group:control_pair"]
        if {"group:driver_pair", "group:control_pair"} <= scores.keys()
        else None
    )
    return {
        "log_loss_gain": float(row["log_loss_gain"]),
        "topk_precision": recovery,
        "topk_recall": recovery,
        "n_intervention_positive_neurons": len(gold),
        "oracle_positive_group": group_effect is not None
        and group_effect >= criteria.min_intervention_effect,
        "group_advantage": advantage,
        "group_win": float(advantage > 0) if advantage is not None else None,
        "false_nomination": any(
            score > criteria.nomination_threshold and effects[unit] < criteria.min_intervention_effect
            for unit, score in scores.items()
        ),
        "oracle_null_integrity": all(abs(effect) <= 1e-12 for effect in effects.values()),
    }


def _summary(
    values: list[float], complete: bool, criteria: DriverAcceptanceCriteria, rng: np.random.Generator
) -> dict[str, Any]:
    data = np.asarray(values, dtype=float)
    interval = None
    if complete and len(data) >= 3:
        means = data[rng.integers(len(data), size=(criteria.bootstrap_replicates, len(data)))].mean(1)
        tail = (1 - criteria.confidence_level) / 2
        interval = np.quantile(means, [tail, 1 - tail]).tolist()
    return {
        "n_seeds": len(data),
        "mean": float(data.mean()) if len(data) else None,
        "sd": float(data.std(ddof=1)) if len(data) > 1 else None,
        "mean_interval": interval,
        "seed_values": values,
        "interval_scope": "descriptive_percentile_bootstrap_independent_seeds",
    }


def _scenario_result(
    scenario: str,
    rows: list[Mapping[str, Any]],
    seeds: Sequence[int],
    criteria: DriverAcceptanceCriteria,
    rng: np.random.Generator,
) -> dict[str, Any]:
    expected = len(seeds) * len(DIRECTIONS)
    lookup = {(row["seed"], row["source_state"], row["destination_state"]): row for row in rows}
    unavailable, seed_metrics = [], []
    available = sum(row["status"] == "available" for row in rows)
    eligible_directions = 0
    for seed in seeds:
        metrics = []
        for source, target in DIRECTIONS:
            row = lookup.get((seed, source, target))
            reasons = ["not_executed"] if row is None else _ineligible(row, criteria)
            if not reasons:
                current = _direction_metrics(row, criteria)  # type: ignore[arg-type]
                if scenario in {"single_driver", "correlated_bystander"} and current["topk_recall"] is None:
                    reasons.append("no_supported_intervention_positive_neuron")
                if scenario == "redundant_group" and (
                    not current["oracle_positive_group"] or current["group_advantage"] is None
                ):
                    reasons.append("unsupported_joint_driver_or_missing_control_score")
            if reasons:
                unavailable.append(
                    {"seed": seed, "source_state": source, "destination_state": target, "reasons": reasons}
                )
                continue
            eligible_directions += 1
            metrics.append(current)
        if len(metrics) == len(DIRECTIONS):
            aggregate: dict[str, Any] = {"seed": seed}
            for key in ("log_loss_gain", "topk_precision", "topk_recall", "group_advantage", "group_win"):
                values = [m[key] for m in metrics]
                aggregate[key] = float(np.mean(values)) if all(v is not None for v in values) else None
            aggregate["false_nomination"] = any(m["false_nomination"] for m in metrics)
            aggregate["oracle_null_integrity"] = all(m["oracle_null_integrity"] for m in metrics)
            seed_metrics.append(aggregate)
    complete = bool(seeds) and len(seed_metrics) == len(seeds)
    summaries = {}
    for key in ("log_loss_gain", "topk_precision", "topk_recall", "group_advantage", "group_win"):
        values = [m[key] for m in seed_metrics if m[key] is not None]
        summaries[key] = _summary(values, complete and len(values) == len(seeds), criteria, rng)
    n_null, failures = len(seed_metrics), sum(m["false_nomination"] for m in seed_metrics)
    upper = (
        float(beta.ppf(criteria.confidence_level, failures + 1, n_null - failures))
        if complete and failures < n_null
        else 1.0
        if complete
        else None
    )
    availability = available / expected if expected else 0.0
    minimum = criteria.min_null_seeds if scenario == "null" else criteria.min_positive_seeds
    mature = len(seeds) >= minimum and len(rows) == expected
    ready = mature and complete
    gates = {"availability": availability >= criteria.min_availability}
    if scenario == "null":
        gates.update(
            oracle_null_integrity=all(m["oracle_null_integrity"] for m in seed_metrics),
            null_false_nomination_bound=upper is not None
            and upper <= criteria.max_null_false_nomination_probability,
        )
    elif scenario in POSITIVE_SCENARIOS:
        gates["predictive_gain"] = (
            summaries["log_loss_gain"]["mean"] is not None
            and summaries["log_loss_gain"]["mean"] >= criteria.min_mean_log_loss_gain
        )
        if scenario == "redundant_group":
            gates["joint_group_recovery"] = (
                summaries["group_win"]["mean"] is not None
                and summaries["group_win"]["mean"] >= criteria.min_group_win_fraction
                and summaries["group_advantage"]["mean"] > 0
            )
        else:
            gates["topk_driver_recovery"] = (
                summaries["topk_recall"]["mean"] is not None
                and summaries["topk_recall"]["mean"] >= criteria.min_topk_recovery
            )
    status = "pass" if ready and all(gates.values()) else "fail" if ready else "inconclusive"
    if mature and not gates["availability"]:
        status = "fail"
    if scenario == "hidden_confounder":
        status = "diagnostic"
    return {
        "status": status,
        "n_planned_seeds": len(seeds),
        "minimum_required_seeds": minimum,
        "n_attempted_directions": len(rows),
        "n_planned_directions": expected,
        "n_available_directions": available,
        "availability_fraction": availability,
        "n_eligible_directions": eligible_directions,
        "n_complete_eligible_seeds": len(seed_metrics),
        "complete_planned_eligible_seed_set": complete,
        "ineligible_directions": unavailable,
        "gates": gates,
        "summaries": summaries,
        "seed_metrics": seed_metrics,
        "null_nomination": {
            "n_independent_seeds": n_null,
            "seeds_with_any_false_nominee": failures,
            "fraction": failures / n_null if n_null else None,
            "one_sided_exact_upper_bound": upper,
            "confidence_level": criteria.confidence_level,
            "coverage_scope": "per_method_seed_event_not_simultaneous_across_methods",
        },
        "scope": "nonidentifiability_challenge_not_required_pass"
        if scenario == "hidden_confounder"
        else "proposed_benchmark_acceptance",
    }


def evaluate_driver_acceptance(
    seed_rows: Sequence[Mapping[str, Any]],
    criteria: DriverAcceptanceCriteria,
    planned_seeds_by_scenario: Mapping[str, Sequence[int]],
    planned_methods: Sequence[str],
) -> dict[str, Any]:
    """Score proposed gates without substituting successful methods or directions.

    Top-k size is the number of intervention-positive neurons in that direction,
    a benchmark recovery metric rather than an operational candidate selector.
    Null calibration instead tests any fixed-threshold false nomination per seed.
    Exact binomial bounds concern that seed-level event, not a neuron-level FDR.
    """
    if not planned_methods or len(set(planned_methods)) != len(planned_methods):
        raise ValueError("planned methods must be nonempty and unique")
    for scenario, seeds in planned_seeds_by_scenario.items():
        if (
            scenario not in SCENARIOS
            or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s < 0 for s in seeds)
        ):
            raise ValueError("planned scenarios and independent seeds must be known, unique and nonnegative")
    seen = set()
    for row in seed_rows:
        identity = (
            row["scenario"],
            row["method"],
            row["seed"],
            row["source_state"],
            row["destination_state"],
        )
        if identity in seen:
            raise ValueError("duplicate seed/method/scenario/semantic direction")
        seen.add(identity)
        if (
            row["scenario"] not in planned_seeds_by_scenario
            or row["method"] not in planned_methods
            or row["seed"] not in planned_seeds_by_scenario[row["scenario"]]
            or (row["source_state"], row["destination_state"]) not in DIRECTIONS
            or row["status"] not in {"available", "unavailable"}
        ):
            raise ValueError("evidence row is outside the declared protocol")
    rng = np.random.default_rng(criteria.analysis_seed)
    methods = {}
    scenarios = list(dict.fromkeys([*REQUIRED_SCENARIOS, *planned_seeds_by_scenario]))
    for method in planned_methods:
        results = {
            scenario: _scenario_result(
                scenario,
                [r for r in seed_rows if r["scenario"] == scenario and r["method"] == method],
                planned_seeds_by_scenario.get(scenario, ()),
                criteria,
                rng,
            )
            for scenario in scenarios
        }
        statuses = [results[scenario]["status"] for scenario in REQUIRED_SCENARIOS]
        status = (
            "pass" if all(s == "pass" for s in statuses) else "fail" if "fail" in statuses else "inconclusive"
        )
        methods[method] = {"status": status, "scenarios": results}
    statuses = [result["status"] for result in methods.values()]
    overall = (
        "pass"
        if all(s == "pass" for s in statuses)
        else "fail"
        if all(s == "fail" for s in statuses)
        else "inconclusive"
    )
    return {
        "schema_version": "driver_acceptance_v1",
        "criteria": asdict(criteria),
        "protocol_status": "proposed_not_preregistered",
        "overall_status": overall,
        "methods": methods,
        "supported_methods": [method for method, result in methods.items() if result["status"] == "pass"],
        "planned_methods": list(planned_methods),
        "planned_seeds_by_scenario": dict(planned_seeds_by_scenario),
        "unit_of_replication": "independent_raw_process_seed",
        "directions_are_replicates": False,
        "nomination_scope": "fixed_predictive_score_cutoff_not_causal_significance_or_neuron_FDR",
        "claim_eligibility": "synthetic_only",
        "biological_necessity_validated": False,
        "confounding_identification_established": False,
    }
