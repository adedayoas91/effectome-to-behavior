"""Seed-level recovery, paired history effects and untouched null acceptance."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest

from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.mechanism_calibration import apply_nomination_threshold
from effectome.experiments.mechanism_replication_config import MechanismReplicationConfig
from effectome.workflows.resume import content_fingerprint


def _describe(
    values: list[float | None], planned: int, config: MechanismReplicationConfig, label: str
) -> dict[str, Any]:
    valid = np.asarray([v for v in values if v is not None], dtype=float)
    complete = len(valid) == planned
    interval = None
    if complete and len(valid) >= 2:
        seed = np.random.SeedSequence([config.analysis_seed, int(content_fingerprint(label)[:8], 16)])
        rng = np.random.default_rng(seed)
        means = rng.choice(valid, (config.bootstrap_replicates, len(valid))).mean(axis=1)
        interval = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "n_seeds": len(valid),
        "planned_seeds": planned,
        "complete": complete,
        "mean": float(valid.mean()) if complete and len(valid) else None,
        "available_mean_descriptive": float(valid.mean()) if len(valid) else None,
        "sd": float(valid.std(ddof=1)) if complete and len(valid) >= 2 else None,
        "mean_95ci": interval,
        "interval_scope": "descriptive_percentile_bootstrap_raw_seed_means",
    }


def _seed_row(
    cell: dict, method: str, threshold: float | None, config: MechanismReplicationConfig
) -> dict[str, Any]:
    result = cell["methods"][method]
    row = {
        "phase": cell["phase"],
        "history_length": cell["history_length"],
        "scenario": cell["scenario"],
        "seed": cell["seed"],
        "method": method,
        "status": result["status"],
        "reason": result.get("reason"),
    }
    if result["status"] != "available":
        return row
    metrics = result["metrics"]
    if threshold is not None:
        metrics = apply_nomination_threshold(metrics, threshold, config.mechanism.effect_threshold)
    directions = metrics["directions"]
    eligible = len(directions) == 2 and all(d.get("topk_recall") is not None for d in directions)
    recall = [d.get("driver_nomination_recall") for d in directions]
    nomination_eligible = len(directions) == 2 and all(value is not None for value in recall)
    precisions = [
        (
            len(set(d.get("nominated_units", [])) - set(d.get("false_nominated_units", [])))
            / len(d["nominated_units"])
        )
        if d.get("nominated_units")
        else 0.0
        for d in directions
    ]
    row.update(
        {
            "effect_mae": metrics["mean_absolute_effect_error"],
            "factual_brier": metrics["factual_brier"],
            "factual_log_loss": metrics["factual_log_loss"],
            "topk_recall": float(np.mean([d["topk_recall"] for d in directions])) if eligible else None,
            "nomination_recall": float(np.mean(recall))
            if nomination_eligible and threshold is not None
            else None,
            "nomination_precision": float(np.mean(precisions))
            if nomination_eligible and threshold is not None
            else None,
            "any_false_nomination": metrics["any_false_nomination"]
            if threshold is not None and metrics.get("nomination_complete", False)
            else None,
            "nomination_complete": metrics.get("nomination_complete", False),
            "nomination_threshold": threshold,
            "support_recall": result["support_recovery"]["recall"],
            "empty_effects_zero": all(
                np.all(np.asarray(values) == 0)
                for values in result["rollout"]["probability_reductions"].values()
            ),
            "any_nomination": any(d.get("nominated_units") for d in directions),
        }
    )
    return row


def summarize_replication(
    cells: list[dict[str, Any]], methods: list[str], config: MechanismReplicationConfig, calibration: dict
) -> dict[str, Any]:
    rows = []
    for cell in cells:
        if cell["phase"] == "calibration":
            continue
        for method in methods:
            frozen = calibration.get(method, {})
            cutoff = frozen.get("operational_threshold") if cell["phase"] != "comparison" else None
            rows.append(_seed_row(cell, method, cutoff, config))
    summaries: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    all_pvalues: dict[str, float] = {}
    for method in methods:
        available = [r for r in rows if r["method"] == method and r["status"] == "available"]
        positive = [r for r in available if r["phase"] == "positive"]
        null = [r for r in available if r["phase"] == "null-test"]
        comparison = [r for r in available if r["phase"] == "comparison"]
        primary = {
            metric: _describe(
                [r.get(metric) for r in positive],
                len(config.positive_seeds),
                config,
                f"{method}/primary/{metric}",
            )
            for metric in (
                "topk_recall",
                "nomination_recall",
                "nomination_precision",
                "effect_mae",
                "factual_brier",
            )
        }
        complete_null = {r["seed"] for r in null} == set(config.null_seeds) and all(
            r.get("any_false_nomination") is not None for r in null
        )
        failures = sum(bool(r.get("any_false_nomination")) for r in null)
        upper = (
            (
                float(beta.ppf(config.confidence_level, failures + 1, len(null) - failures))
                if failures < len(null)
                else 1.0
            )
            if complete_null
            else None
        )
        gates = {
            "replication_profile": config.profile == "replication",
            "complete_positive_seeds": all(value["complete"] for value in primary.values()),
            "positive_seed_count": len(positive) >= config.min_positive_seeds,
            "null_seed_count": complete_null and len(null) >= config.min_null_seeds,
            "topk_recovery": primary["topk_recall"]["mean"] is not None
            and primary["topk_recall"]["mean"] >= config.min_topk_recall,
            "nomination_recall": primary["nomination_recall"]["mean"] is not None
            and primary["nomination_recall"]["mean"] >= config.min_nomination_recall,
            "nomination_precision": primary["nomination_precision"]["mean"] is not None
            and primary["nomination_precision"]["mean"] >= config.min_nomination_precision,
            "effect_agreement": primary["effect_mae"]["mean"] is not None
            and primary["effect_mae"]["mean"] <= config.max_effect_mae,
            "null_bound": upper is not None and upper <= config.max_null_probability,
            "calibration_complete": calibration.get(method, {}).get("complete", False),
        }
        summaries[method] = {
            "positive": primary,
            "positive_available_seeds": len(positive),
            "null_available_seeds": len(null),
            "null_planned_seeds": len(config.null_seeds),
            "null_false_nomination_seeds": failures,
            "null_complete": complete_null,
            "null_one_sided_95_upper_bound": upper,
            "null_bound_scope": "per_method_whole_seed_event_not_simultaneous_or_nonnull_FDR",
            "gates": gates,
            "raw_gates_passed": all(gates.values()),
            "comparison": {
                metric: _describe(
                    [r.get(metric) for r in comparison],
                    len(config.positive_seeds),
                    config,
                    f"{method}/comparison/{metric}",
                )
                for metric in ("topk_recall", "effect_mae", "factual_brier")
            },
        }
        pairs = []
        for first in positive:
            short = next((r for r in comparison if r["seed"] == first["seed"]), None)
            if short is None:
                continue
            long_cell = next(c for c in cells if c["phase"] == "positive" and c["seed"] == first["seed"])
            short_cell = next(c for c in cells if c["phase"] == "comparison" and c["seed"] == first["seed"])
            identity = ("training_input_fingerprint", "evaluation_initial_state_fingerprint", "oracle")
            if content_fingerprint([long_cell[k] for k in identity]) != content_fingerprint(
                [short_cell[k] for k in identity]
            ):
                raise ValueError("paired history comparison changed training inputs/origins/oracle effects")
            pairs.append(
                {
                    "seed": first["seed"],
                    "delta_effect_mae": first["effect_mae"] - short["effect_mae"],
                    "delta_brier": first["factual_brier"] - short["factual_brier"],
                    "delta_topk": (first["topk_recall"] - short["topk_recall"])
                    if first["topk_recall"] is not None and short["topk_recall"] is not None
                    else None,
                }
            )
        complete_pairs = {r["seed"] for r in pairs} == set(config.positive_seeds)
        changes = [r["delta_effect_mae"] for r in pairs]
        nonzero = [v for v in changes if v != 0]
        pvalue = (
            float(binomtest(sum(v < 0 for v in nonzero), len(nonzero)).pvalue)
            if complete_pairs and nonzero
            else None
        )
        if method in CSL_METHODS:
            all_pvalues[method] = pvalue if pvalue is not None else 1.0
        paired[method] = {
            "seed_changes": pairs,
            "complete": complete_pairs,
            "n_paired_seeds": len(pairs),
            "delta_effect_mae": _describe(changes, len(config.positive_seeds), config, f"{method}/delta_mae"),
            "delta_topk": _describe(
                [r["delta_topk"] for r in pairs], len(config.positive_seeds), config, f"{method}/delta_topk"
            ),
            "raw_paired_mae_sign_test_p": pvalue,
            "paired_inputs_verified": complete_pairs,
        }
    # Preserve four planned native contrasts even when a method is omitted/unavailable.
    ordered = sorted(CSL_METHODS, key=lambda name: all_pvalues.get(name, 1.0))
    previous = 0.0
    for index, name in enumerate(ordered):
        previous = max(previous, min(1.0, (4 - index) * all_pvalues.get(name, 1.0)))
        if name in paired:
            paired[name]["holm_paired_mae_p"] = previous
    known_pass = summaries.get("known_support", {}).get("raw_gates_passed", False)
    empty_rows = [r for r in rows if r["method"] == "empty_support" and r["phase"] != "comparison"]
    expected_empty = len(config.positive_seeds) + len(config.null_seeds)
    empty_pass = len(empty_rows) == expected_empty and all(
        r["status"] == "available" and r.get("empty_effects_zero") and not r.get("any_nomination")
        for r in empty_rows
    )
    supported = [
        name
        for name in CSL_METHODS
        if summaries.get(name, {}).get("raw_gates_passed") and known_pass and empty_pass
    ]
    return {
        "summary": summaries,
        "seed_metrics": rows,
        "paired_history": paired,
        "calibration_complete": bool(calibration) and all(c.get("complete") for c in calibration.values()),
        "known_support_control_passed": known_pass,
        "empty_support_control_passed": empty_pass,
        "supported_native_methods": supported,
        "bounded_control_replication_passed": len(supported) == len(CSL_METHODS),
        "decision_scope": "fixed_gates_for_this_fully_observed_control_not_general_or_biological_validation",
    }


def render_mechanism_replication_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fresh-seed mechanism replication",
        "",
        f"Stage: {report['stage']}. Completed cells: {report['completed_cells']}/{report['planned_cells']}. "
        f"Available lanes: {report['available_method_runs']}/{report['planned_method_runs']}.",
        "",
        "Calibration uses separate global-null seed maxima across all candidates and both directions. "
        "The strict-exceedance cutoff is the maximum of the finite-sample order statistic and the "
        "existing 0.10 minimum effect size. Calibration is frozen before test jobs are submitted. "
        "Protection is marginal under exchangeability, per method under the global null; it is not "
        "strong non-null FDR control or simultaneous method coverage.",
        "",
        "| Method | Cutoff | Driver top-k | Nomination recall | Nomination precision | Effect MAE "
        "| Null false seeds | Null one-sided 95% upper | Gates |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def fmt(value: Any) -> str:
        return "NA" if value is None else f"{value:.6f}"

    for name, summary in report["summary"].items():
        positive = summary["positive"]
        lines.append(
            f"| {name} | {fmt(report['calibration'].get(name, {}).get('operational_threshold'))} "
            f"| {fmt(positive['topk_recall']['mean'])} | {fmt(positive['nomination_recall']['mean'])} "
            f"| {fmt(positive['nomination_precision']['mean'])} | {fmt(positive['effect_mae']['mean'])} "
            f"| {summary['null_false_nomination_seeds']}/{summary['null_available_seeds']} "
            f"| {fmt(summary['null_one_sided_95_upper_bound'])} "
            f"| {'pass' if summary['raw_gates_passed'] else 'not passed'} |"
        )
    lines += [
        "",
        f"Supported native methods (including control requirements): {report['supported_native_methods']}.",
        f"All-four bounded-control replication passed: {report['bounded_control_replication_passed']}.",
        "",
        "Acceptance requires complete calibration, at least 20 positive and 60 separate null seeds, "
        "mean top-k/nomination recall/nomination precision >=0.75, mean effect MAE <=0.05, "
        "and a one-sided 95% null-event upper bound <=0.05. Empty positive nominations have precision "
        "zero; missing or ineligible seed metrics suppress complete-set aggregates. Pilot runs cannot "
        "satisfy the full seed-count gates.",
        "",
        "## Paired history comparison",
        "",
        "| Method | Paired seeds | Mean MAE change (1024 - 256) | Mean top-k change | Holm MAE sign-test p |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, paired in report["paired_history"].items():
        lines.append(
            f"| {name} | {paired['n_paired_seeds']} | {fmt(paired['delta_effect_mae']['mean'])} "
            f"| {fmt(paired['delta_topk']['mean'])} | {fmt(paired.get('holm_paired_mae_p'))} |"
        )
    lines += [
        "",
        "History 256 uses the same full training/evaluation inputs as history 1024 for each driver "
        "seed. It is a paired sensitivity diagnostic; it has no separately calibrated nomination or "
        "null claim. Negative MAE changes favor the longer history. The four native paired sign tests "
        "form one fixed Holm family; seed-bootstrap intervals in JSON are descriptive.",
        "",
        "## Limits",
        "",
        report["conventional_randomness"],
        "",
        "This is a favorable fully observed, model-matched control with a predeclared observed endpoint. "
        "It does not overturn the old latent-bistable benchmark, validate learned manifold alignment, "
        "or establish biological necessity. No graph-similarity extension is included. Protocol and "
        "citations: docs/mechanism_replication_protocol.md and its reference/audit files.",
    ]
    path = destination / "analysis-report.md"
    path.write_text("\n".join(lines) + "\n")
    rows = report["seed_metrics"]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (destination / "seed-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "statistics.json").write_text(
        json.dumps(
            {
                "summary": report["summary"],
                "paired_history": report["paired_history"],
                "supported_native_methods": report["supported_native_methods"],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return path
