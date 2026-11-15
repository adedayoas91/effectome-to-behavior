"""Recovery uncertainty and nomination agreement; methods are never replicates."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
from typing import Any


def render_recovery_statistics(report: dict[str, Any], destination: Path) -> None:
    import matplotlib.pyplot as plt

    acceptance = report["acceptance"]
    table = ["| Method | Scenario | Eligible seeds/planned | Top-k recall [95% CI] | "
             "Group wins [95% CI] | Null upper bound | Decision |", "|---|---|---:|---|---|---|---|"]
    for method, result in acceptance["methods"].items():
        for scenario, row in result["scenarios"].items():
            def format_summary(key: str, data: dict = row) -> str:
                summary = data["summaries"][key]
                value, interval = summary["mean"], summary["mean_interval"]
                if value is None:
                    return "unavailable"
                if interval:
                    return f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"
                return f"{value:.3f} [NA]"
            bound = row["null_nomination"]["one_sided_exact_upper_bound"] if scenario == "null" else None
            table.append(f"| {method} | {scenario} | {row['n_complete_eligible_seeds']}/"
                         f"{row['n_planned_seeds']} | {format_summary('topk_recall')} | "
                         f"{format_summary('group_win')} | "
                         f"{f'{bound:.3f}' if bound is not None else 'NA'} | {row['status']} |")
    (destination / "recovery-summary.md").write_text("\n".join(table) + "\n")
    methods = report["planned_methods"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    labels = [name.replace("conventional_", "").replace("_", " ") for name in methods]
    for index, method in enumerate(methods):
        scenarios = acceptance["methods"][method]["scenarios"]
        for offset, scenario, color in ((-.1, "single_driver", "#0072B2"),
                                         (.1, "correlated_bystander", "#D55E00")):
            summary = scenarios[scenario]["summaries"]["topk_recall"]
            if summary["mean"] is not None:
                axes[0].plot(summary["mean"], index + offset, "o", color=color,
                             label=scenario if index == 0 else None)
                if summary["mean_interval"]:
                    axes[0].plot(summary["mean_interval"], [index + offset] * 2, color=color)
        summary = scenarios["redundant_group"]["summaries"]["group_win"]
        if summary["mean"] is not None:
            axes[1].plot(summary["mean"], index, "o", color="#009E73")
            if summary["mean_interval"]:
                axes[1].plot(summary["mean_interval"], [index] * 2, color="#009E73")
        bound = scenarios["null"]["null_nomination"]["one_sided_exact_upper_bound"]
        if bound is not None:
            axes[2].plot(bound, index, "o", color="#CC79A7")
    criteria = acceptance["criteria"]
    titles = ("Intervention-positive neuron recovery", "Redundant pair exceeds control",
              "Any false nomination: null upper bound")
    gates = (criteria["min_topk_recovery"], criteria["min_group_win_fraction"],
             criteria["max_null_false_nomination_probability"])
    for axis, title, gate in zip(axes, titles, gates, strict=True):
        axis.axvline(gate, color="0.4", linestyle="--", linewidth=1)
        axis.set(yticks=range(len(methods)), yticklabels=labels, xlim=(-.03, 1.03), title=title)
        axis.grid(axis="x", alpha=.2)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(destination / f"figure-03-recovery-and-null.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    threshold = acceptance["criteria"]["nomination_threshold"]
    lookup = {(row["scenario"], row["seed"], row["source_state"], row["method"]): row
              for row in report["seed_rows"] if row["status"] == "available" and row.get("driver_scores")}
    pairs = []
    contexts = sorted({key[:3] for key in lookup})
    for first, second in combinations(methods, 2):
        for scenario, seed, source in contexts:
            a, b = lookup.get((scenario, seed, source, first)), lookup.get((scenario, seed, source, second))
            if a is None or b is None:
                continue
            selected = [{unit for unit, score in row["driver_scores"].items() if score > threshold}
                        for row in (a, b)]
            union = selected[0] | selected[1]
            pairs.append({"first": first, "second": second, "scenario": scenario, "seed": seed,
                          "source_state": source, "both_empty": not union,
                          "jaccard": len(selected[0] & selected[1]) / len(union) if union else None})
    with (destination / "nomination-agreement.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["first", "second", "scenario", "seed",
                                                    "source_state", "both_empty", "jaccard"])
        writer.writeheader()
        writer.writerows(pairs)
    (destination / "stats-appendix.md").write_text(
        "# Recovery analysis\n\nIndependent units are raw-process simulation seeds; directions are averaged "
        "within seed. Percentile bootstrap 95% intervals require all planned eligible seeds and at least "
        "three seeds; incomplete sets receive no interval. Top-k uses intervention-defined gold-set size, "
        "not an operational nomination threshold. Null nomination uses the fixed 0.01-nat cutoff and any "
        "false candidate within method/seed; its exact one-sided 95% bound is per method, "
        "without simultaneous "
        "coverage. No paired superiority p-values or independent-method confirmation is claimed. "
        "All four methods belong to one family. Empty nomination Jaccard is undefined.\n"
    )
    (destination / "figure-catalog.md").write_text(
        "# Driver validation figures\n\nFigure01: actual outgoing-disconnection risk reduction for both "
        "semantic directions; one oracle realization per seed, without method replication. Figure02: "
        "predictive feature omissions for direction0→1, with negative values and available counts. "
        "These first two figures show descriptive means, not causal confidence intervals.\n\n"
        "Figure03: seed-level recovery and null calibration. Points are eligible means or null upper "
        "bounds; horizontal bars are descriptive percentile-bootstrap 95% intervals when complete. "
        "Dashed lines are the frozen acceptance cutoffs; missing points are unavailable. Inspect "
        "recovery-summary.md for denominators and decisions. A positive heatmap cannot override failed "
        "support, alignment, recovery or null gates. Source: ../driver-validation.json.\n"
    )
