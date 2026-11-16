"""Export descriptive mechanism diagnostics without promotion to confirmation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _formatted(summary: dict[str, Any], key: str) -> str:
    value = summary[key]
    return "NA" if value is None else f"{value:.4f}"


def render_mechanism_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    lines = [
        "# Mechanism-readout development diagnostic", "",
        f"Execution complete: {report['execution_complete']}. "
        f"Available method/recording runs: {report['available_method_runs']}/"
        f"{report['attempted_method_runs']} attempted.", "",
        "This is a model-matched, fully observed positive/null control. Known support is an oracle "
        "diagnostic; estimated support comes from the imported CSL packages. It does not replace "
        "the completed failed latent-bistable driver study or establish biological necessity.", "",
        "| Lane | Available/planned | Driver effect MAE | Driver top-k recall | Null false-nomination seeds "
        "| "
        "Null 95% upper bound |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in report["summary"].items():
        lines.append(
            f"| {name} | {summary['available_cells']}/{summary['planned_cells']} | "
            f"{_formatted(summary, 'positive_mean_absolute_effect_error')} | "
            f"{_formatted(summary, 'positive_mean_topk_recall')} "
            f"({summary['positive_topk_eligible_seeds']}/{summary['positive_topk_planned_seeds']}) | "
            f"{summary['null_seeds_with_any_false_nomination']}/{summary['null_seeds_evaluated']} | "
            f"{_formatted(summary, 'null_one_sided_95_upper_bound')} |"
        )
    lines.extend([
        "", "Effect MAE averages per-origin absolute errors in transition-probability units. "
        "Top-k recall uses the number of "
        "intervention-positive neurons and averages ties fairly. Each direction is first summarized "
        "within its raw-process seed. The null bound concerns any false nominee across units and "
        "directions per seed/method, with no simultaneous coverage across methods.", "",
        "Ranking entries show eligible/planned independent-seed counts. Aggregate top-k recall "
        "is suppressed if any planned seed lacks both eligible directions.", "",
        "The probability-effect nomination cutoff is a provisional development setting, not the "
        "old omission-nat cutoff or an independently calibrated significance rule. Few null seeds "
        "cannot establish a 5% false-nomination guarantee. Native CPU permutation draws are not "
        "fully reproducible upstream; saved graphs/evidence are retained.", "",
        "Protocol, mechanisms, noise estimates, paired per-origin probabilities and unavailable "
        "reasons are in mechanism-validation.json and cells/. Rationale and verified references: "
        "docs/driver_methodology_revision.md and docs/driver_methodology_citation_audit.md.",
    ])
    for cell in report["cells"]:
        for name, result in cell["methods"].items():
            row = {"scenario": cell["scenario"], "seed": cell["seed"], "lane": name,
                   "status": result["status"], "reason": result.get("reason", "")}
            if result["status"] == "available":
                row.update({key: result["metrics"][key] for key in (
                    "mean_absolute_effect_error", "factual_brier", "factual_log_loss", "any_false_nomination",
                )})
                row["one_step_heldout_mse"] = result["one_step_heldout_mse"]
                row["support_recall"] = result["support_recovery"]["recall"]
                row["support_false_positives"] = result["support_recovery"]["fp"]
            rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (destination / "numeric-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path = destination / "analysis-report.md"
    path.write_text("\n".join(lines) + "\n")
    return path
