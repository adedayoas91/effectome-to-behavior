"""Descriptive figures and complete seed tables for the four-CSL sanity check."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from .csl_sanity import CSL_METHODS

METHOD_LABELS = ("c-GC", "c-GC*", "fastC-GC", "fastC-GC*")


def _formatted(rows: list[dict[str, Any]], key: str) -> str:
    data = np.asarray([row[key] for row in rows], dtype=float)
    if not len(data):
        return "unavailable"
    sd = f"{np.std(data, ddof=1):.2f}" if len(data) > 1 else "undefined"
    return f"{np.mean(data):.2f} ± {sd}"


def render_csl_sanity_report(report: dict[str, Any], output_dir: str | Path) -> None:
    """Keep null graphs and failures visible; never infer driver recovery here."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output_dir)
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    all_rows = report["seed_rows"]
    available = [row for row in all_rows if row["status"] == "available"]
    fields = ["method", "scenario", "seed", "status", "tp", "fp", "fn", "tn", "n_selected",
              "precision", "recall", "f1", "false_positive_rate", "exact_support", "reason"]
    with (destination / "seed-metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    with (destination / "pairwise-agreement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "first", "second", "scenario", "seed", "status", "intersection", "union",
            "jaccard", "both_empty", "exact_support",
        ])
        writer.writeheader()
        writer.writerows(report["pairwise_rows"])
    summary = [
        "# Four imported CSL estimators: native graph sanity check", "",
        f"Available fits: {report['available_runs']}/{report['attempted_runs']}. "
        f"Independent data seeds per condition: {len(report['config']['seeds'])}.", "",
        "| Method | Condition | Available/planned | TP mean ± SD | FP mean ± SD | "
        "Recall mean | Exact graphs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    appendix = [
        "# Statistical scope", "",
        "The independent unit is a simulation seed. Conditions share innovations within seed; "
        "all methods receive the identical recording. Neurons, directed edges and method variants "
        "are not independent replicates. Means and sample SD describe seed-level metrics.", "",
        "This bounded pilot is descriptive. No significance test, ranked winner or population "
        "recovery guarantee is justified. No multiple-comparison claims are made. Null intervals "
        "below are two-sided exact 95% binomial intervals for ANY false edge in a null graph, "
        "separately per method, without simultaneous coverage across methods.", "",
    ]
    for method, label in zip(CSL_METHODS, METHOD_LABELS, strict=True):
        for scenario in ("chain", "null"):
            selected = [r for r in available if r["method"] == method and r["scenario"] == scenario]
            recalls = [r["recall"] for r in selected if r["recall"] is not None]
            recall = f"{np.mean(recalls):.3f}" if recalls else "undefined"
            exact = sum(r["exact_support"] for r in selected)
            summary.append(f"| {label} | {scenario} | {len(selected)}/{len(report['config']['seeds'])} | "
                           f"{_formatted(selected, 'tp')} | {_formatted(selected, 'fp')} | "
                           f"{recall} | {exact}/{len(selected)} |")
            if scenario == "null" and selected:
                failures = sum(r["fp"] > 0 for r in selected)
                total = len(selected)
                low = float(beta.ppf(.025, failures, total - failures + 1)) if failures else 0.
                high = float(beta.ppf(.975, failures + 1, total - failures)) if failures < total else 1.
                appendix.append(f"- {label}: null graphs with any false edge {failures}/{total}; "
                                f"exact 95% interval [{low:.3f}, {high:.3f}].")
    summary.extend([
        "", "A correct chain graph contains 0→1 and 1→2; neuron 3 is disconnected. "
        "Negative coupling on 1→2 is evaluated as support because these effectomes are unsigned.", "",
        "Pairwise Jaccard scores are in `pairwise-agreement.csv`. Two empty graphs have "
        "undefined Jaccard and never count as nontrivial agreement. Compare agreement with "
        "truth: methods can share the same false edges.", "",
        "These four methods form one discovery family. This study checks native graph recovery "
        "on directly observed linear Gaussian data; it does not establish manifold transitions, "
        "transition-driver recovery, calcium robustness or biological necessity. The upstream "
        "conventional CPU permutation RNG is not fully controlled by its random_state argument. "
        "Saved artifacts preserve the actual realization; reruns can differ.",
    ])
    (destination / "analysis-report.md").write_text("\n".join(summary) + "\n")
    (destination / "numeric-summary.md").write_text("\n".join(summary[:15]) + "\n")
    (destination / "stats-appendix.md").write_text("\n".join(appendix) + "\n")

    # All seeds enter recovery plots; the first predeclared seed illustrates support.
    fig = Figure(figsize=(12, 2.8), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 5)
    first_seed = report["config"]["seeds"][0]
    displays = [np.asarray(report["truth"]["chain"], float)]
    for method in CSL_METHODS:
        example = next((r for r in available if r["method"] == method and r["scenario"] == "chain"
                        and r["seed"] == first_seed), None)
        displays.append(np.asarray(example["support"], float) if example else np.full((4, 4), np.nan))
    for axis, matrix, label in zip(axes, displays, ("Planted truth", *METHOD_LABELS), strict=True):
        axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set(title=label, xlabel="Target", xticks=range(4), yticks=range(4))
        if np.isnan(matrix).all():
            axis.text(1.5, 1.5, "unavailable", ha="center", va="center")
    axes[0].set_ylabel("Source")
    fig.suptitle(f"Positive-lag directed support · first declared seed {first_seed}")
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"figure-01-native-support.{extension}", dpi=180, bbox_inches="tight")
    fig.clear()

    fig = Figure(figsize=(11, 3.4), layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    for axis, scenario, metric, title in zip(
        axes, ("chain", "chain", "null"), ("recall", "fp", "fp"),
        ("Planted-edge recall", "False edges: chain", "False edges: null"), strict=True,
    ):
        for index, method in enumerate(CSL_METHODS):
            data = [float(r[metric]) for r in available
                    if r["method"] == method and r["scenario"] == scenario]
            if data:
                axis.scatter(index + np.linspace(-.08, .08, len(data)), data, alpha=.7, s=32)
                axis.plot([index - .18, index + .18], [np.mean(data)] * 2, color="black", linewidth=2)
            else:
                axis.text(index, .05, "unavailable", rotation=90, ha="center")
        axis.set(title=title, xticks=range(4), xticklabels=METHOD_LABELS)
        axis.tick_params(axis="x", rotation=30)
        axis.set_ylim(-.1, 1.1 if metric == "recall" else 12.5)
        axis.grid(axis="y", alpha=.25)
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"figure-02-seed-recovery.{extension}", dpi=180, bbox_inches="tight")
    fig.clear()
    (destination / "figure-catalog.md").write_text(
        "# Sanity figures\n\n"
        "- Figure 01: directed truth and native binary decisions on the first declared chain seed. "
        "Rows are sources; columns are targets. This is an illustration, not an average. "
        "Look for the two planted links and isolated neuron; this determines whether agreement "
        "is meaningful.\n"
        "- Figure 02: every available independent seed's recall and false-link counts; black bars are "
        "means, points are seeds and no error bars are implied. Chain and null results are separate. "
        "Look for missed links or false links despite method agreement; the outcome determines the "
        "next stress tests, not driver-validation acceptance.\n\n"
        "Source: ../sanity.json. Seed-level metrics and undefined/empty agreement are retained in CSV. "
        "Both figures are descriptive pilots; see stats-appendix.md for uncertainty and RNG limits.\n"
    )
