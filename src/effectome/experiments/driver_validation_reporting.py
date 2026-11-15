"""Descriptive figures and explicit acceptance reporting for driver benchmarks."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from effectome.experiments.driver_recovery_statistics import render_recovery_statistics
from effectome.experiments.driver_validation import DRIVER_SCENARIOS


def _number(value: Any) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _heatmap(axis: Any, values: np.ndarray, labels: list[str], units: list[str], limit: float) -> Any:
    import matplotlib

    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="0.85")
    mesh = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(range(len(labels)), labels, fontsize=9)
    axis.set_xticks(range(len(units)), units, rotation=45, ha="right", fontsize=9)
    for (i, j), value in np.ndenumerate(values):
        axis.text(
            j,
            i,
            "NA" if not np.isfinite(value) else f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white" if abs(value) > 0.65 * limit else "black",
        )
    return mesh


def _figures(report: dict, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, methods = report["seed_rows"], report["planned_methods"]
    scenarios = [s for s in DRIVER_SCENARIOS if s in report["config"]["scenarios"]]
    all_units = sorted({unit for row in rows for unit in row.get("driver_scores", {})})
    units = [u for u in all_units if u.startswith("neuron:")] + [
        u for u in all_units if u.startswith("group:")
    ]
    if not units:
        units = [f"neuron:{i}" for i in range(report["config"]["generator"]["n_neurons"])]
        units += ["group:driver_pair", "group:control_pair"]
    labels = [
        u.replace("neuron:", "N")
        .replace("group:driver_pair", "Driver pair")
        .replace("group:control_pair", "Control pair")
        for u in units
    ]
    # Oracle rollouts are shared across methods. Verify duplicates, then retain
    # one copy per seed/direction; estimators never become oracle replicates.
    oracle: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in rows:
        if "intervention_effects" not in row:
            continue
        key = (row["scenario"], row["seed"], row["source_state"])
        if key in oracle and oracle[key] != row["intervention_effects"]:
            raise ValueError("methods disagree about the shared simulator intervention truth")
        oracle[key] = row["intervention_effects"]
    arrays = []
    for source in (0, 1):
        arrays.append(
            np.asarray(
                [
                    [
                        np.mean(values)
                        if (
                            values := [
                                effects[unit]
                                for (s, _, d), effects in oracle.items()
                                if s == scenario and d == source and unit in effects
                            ]
                        )
                        else np.nan
                        for unit in units
                    ]
                    for scenario in scenarios
                ]
            )
        )
    finite = [abs(x) for array in arrays for x in array.flat if np.isfinite(x)]
    limit = max(finite, default=1.0) or 1.0
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for source, axis in enumerate(axes):
        counts = [
            sum(bool(effects) for (s, _, d), effects in oracle.items() if s == scenario and d == source)
            for scenario in scenarios
        ]
        case_labels = [f"{s}\n(n={n})" for s, n in zip(scenarios, counts, strict=True)]
        mesh = _heatmap(axis, arrays[source], case_labels, labels, limit)
        axis.set_title(f"Semantic well {source} → {1 - source}")
    figure.colorbar(mesh, ax=axes.tolist(), label="Simulator transition-probability reduction", shrink=0.8)
    for extension in ("pdf", "png"):
        figure.savefig(destination / f"figure-01-intervention-truth.{extension}", dpi=220)
    plt.close(figure)

    arrays = []
    for method in methods:
        arrays.append(
            np.asarray(
                [
                    [
                        np.mean(values)
                        if (
                            values := [
                                r["driver_scores"][unit]
                                for r in rows
                                if r["method"] == method
                                and r["scenario"] == scenario
                                and r["source_state"] == 0
                                and unit in r.get("driver_scores", {})
                            ]
                        )
                        else np.nan
                        for unit in units
                    ]
                    for scenario in scenarios
                ]
            )
        )
    finite = [abs(x) for array in arrays for x in array.flat if np.isfinite(x)]
    limit = max(finite, default=1.0) or 1.0
    n_rows = (len(methods) + 1) // 2
    figure, axes = plt.subplots(n_rows, 2, figsize=(10, 3 * n_rows), layout="constrained", squeeze=False)
    for axis, method, plot_values in zip(axes.ravel(), methods, arrays, strict=False):
        counts = [
            len(
                {
                    r["seed"]
                    for r in rows
                    if r["method"] == method
                    and r["scenario"] == scenario
                    and r["source_state"] == 0
                    and r.get("driver_scores")
                }
            )
            for scenario in scenarios
        ]
        case_labels = [f"{s}\n(n={n})" for s, n in zip(scenarios, counts, strict=True)]
        mesh = _heatmap(axis, plot_values, case_labels, labels, limit)
        axis.set_title(method)
    for axis in axes.ravel()[len(methods) :]:
        axis.axis("off")
    figure.colorbar(mesh, ax=axes.ravel().tolist(), label="Omitted − full binary log loss (nats)", shrink=0.7)
    for extension in ("pdf", "png"):
        figure.savefig(destination / f"figure-02-predictive-omissions.{extension}", dpi=220)
    plt.close(figure)


def render_driver_validation_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Expose raw descriptive scores separately from support/acceptance gates."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    acceptance = report["acceptance"]
    (destination / "acceptance.json").write_text(json.dumps(acceptance, indent=2, allow_nan=False) + "\n")
    summaries = []
    table = [
        "| Scenario | Method | Available seed pairs/planned | Mean gain (nats) | SD | "
        "Eligible directions | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, result in acceptance["methods"].items():
        for scenario, details in result["scenarios"].items():
            scores: dict[int, list[float]] = defaultdict(list)
            for row in report["seed_rows"]:
                if (
                    row["method"] == method
                    and row["scenario"] == scenario
                    and row.get("log_loss_gain") is not None
                ):
                    scores[row["seed"]].append(row["log_loss_gain"])
            paired = [float(np.mean(gains)) for gains in scores.values() if len(gains) == 2]
            summary = {
                "scenario": scenario,
                "method": method,
                "available_seed_pairs": len(paired),
                "planned_seeds": details["n_planned_seeds"],
                "mean_gain_nats": float(np.mean(paired)) if paired else None,
                "sd_gain_nats": float(np.std(paired, ddof=1)) if len(paired) > 1 else None,
                "eligible_directions": details["n_eligible_directions"],
                "planned_directions": details["n_planned_directions"],
                "status": details["status"],
            }
            summaries.append(summary)
            table.append(
                f"| {scenario} | {method} | {len(paired)}/{details['n_planned_seeds']} | "
                f"{_number(summary['mean_gain_nats'])} | {_number(summary['sd_gain_nats'])} | "
                f"{details['n_eligible_directions']}/{details['n_planned_directions']} | "
                f"{details['status']} |"
            )
    (destination / "numeric-summary.md").write_text("\n".join(table) + "\n")
    with (destination / "numeric-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summaries[0]) if summaries else ["scenario", "method"]
        )
        writer.writeheader()
        writer.writerows(summaries)
    _figures(report, destination)
    render_recovery_statistics(report, destination)
    (destination / "analysis-report.md").write_text(
        "# Simulator-grounded driver recovery\n\n"
        f"Profile: {report['profile']}. Available method runs: {report['available_method_runs']}/"
        f"{report['attempted_method_runs']}. Acceptance status: **{acceptance['overall_status']}**.\n\n"
        "This is a proposed protocol. Graph predictions use observed raw histories; "
        "truth enters subsequent interventions and training-owned semantic correspondence. "
        "The oracle disconnects outgoing influence and reruns the future mechanism with paired innovations. "
        "It does not mask predictor features or censor measured readouts.\n\n"
        "The numeric table averages directions within each independent seed before descriptive means/SD. "
        "It includes all available pairs, even below acceptance thresholds; the eligible "
        "direction count and acceptance artifact make this explicit. No pilot p-values are reported.\n\n"
        "Figure01 shows mean oracle risk reduction per semantic direction, deduplicating methods. "
        "Figure02 shows predictive omission gain for well 0→1; both directions remain in tables "
        "and raw evidence. Positive and negative entries are retained; gray entries are unavailable. "
        "Figures show means over available independent seeds, without uncertainty bars or causal "
        "significance claims. Sparse source counts and alignment failures limit interpretation.\n\n"
        "Proposed gates require 20 positive-control seeds and 60 null seeds, plus per-direction "
        "support and held-out semantic alignment. The null upper bound concerns any false nomination in one "
        "method/seed; coverage is not simultaneous across methods. Hidden confounding is an explicit "
        "nonidentifiability challenge. Simulator recovery does not validate biological "
        "necessity, and this report does not automatically pass the scientific study.\n"
    )
    return {
        "summaries": summaries,
        "acceptance": acceptance["overall_status"],
        "manifest_fingerprint": report["manifest_fingerprint"],
        "scientific_validation_passed": False,
    }
