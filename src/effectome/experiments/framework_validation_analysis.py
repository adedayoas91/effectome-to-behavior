"""Seed-level statistics, scientific figures and auditable manuscript exports."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from effectome.workflows.cross_method import _json_safe


def _holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.ones(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[int(index)]))
        adjusted[index] = previous
    return adjusted.tolist()


def analyze_framework_validation(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Analyze paired gains using independent simulation seeds, never anchor counts.

    The primary sign test concerns Pr(seed-level gain > 0)=1/2 after zero ties
    are removed. No normality assumption or data-driven test switching is used.
    Failed or unexecuted contrasts remain in the planned Holm family with p=1;
    incomplete seed sets receive no inferential claim.
    """
    if bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for row in report["rows"]:
        identity = (row["spec_id"], row["method"], row["seed"])
        if identity in seen:
            raise ValueError("duplicate independent seed in one specification/method contrast")
        seen.add(identity)
        groups[(row["spec_id"], row["method"])].append(row)
    planned = {
        (cell["spec_id"], method) for cell in report["planned_cells"] for method in report["planned_methods"]
    }
    expected_seeds = set(report["config"]["seeds"])
    rng = np.random.default_rng(seed)
    summaries = []
    for spec_id, method in sorted(planned):
        rows = groups[(spec_id, method)]
        good = [
            row
            for row in rows
            if row["status"] == "available"
            and row.get("log_loss_gain") is not None
            and np.isfinite(row["log_loss_gain"])
        ]
        gains = np.asarray([row["log_loss_gain"] for row in good], dtype=float)
        complete = {row["seed"] for row in good} == expected_seeds
        nonzero = gains[gains != 0]
        p_value = (
            float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue)
            if (complete and len(nonzero) >= 3)
            else None
        )
        interval = None
        if complete and len(gains) >= 3:
            bootstrap = gains[rng.integers(0, len(gains), size=(bootstrap_replicates, len(gains)))].mean(1)
            interval = np.quantile(bootstrap, [0.025, 0.975]).tolist()
        source: Mapping[str, Any] = next(
            (cell for cell in report["planned_cells"] if cell["spec_id"] == spec_id), {}
        )
        summaries.append(
            {
                "spec_id": spec_id,
                "scenario": source.get("scenario"),
                "method": method,
                "n_planned_seeds": len(expected_seeds),
                "n_attempted_seeds": len(rows),
                "n_available_seeds": len(gains),
                "n_failed_or_unexecuted_seeds": len(expected_seeds) - len(gains),
                "mean_gain_nats": float(gains.mean()) if len(gains) else None,
                "sd_gain_nats": float(gains.std(ddof=1)) if len(gains) >= 2 else None,
                "median_gain_nats": float(np.median(gains)) if len(gains) else None,
                "mean_gain_95ci": interval,
                "ci_method": "percentile_bootstrap_independent_seeds",
                "ci_limitation": "small seed counts have unstable coverage" if len(gains) < 20 else None,
                "positive_seed_fraction": float(np.mean(gains > 0)) if len(gains) else None,
                "raw_sign_test_p": p_value,
                "inferentially_complete": complete,
                "inference_reason": "exact_sign_test"
                if p_value is not None
                else "incomplete or fewer than three nonzero independent seed gains",
                "independent_unit": "raw_process_seed",
                "seed_gains": gains.tolist(),
                "available_seeds": [row["seed"] for row in good],
            }
        )
    adjusted = _holm(
        [row["raw_sign_test_p"] if row["raw_sign_test_p"] is not None else 1.0 for row in summaries]
    )
    for row, p_value in zip(summaries, adjusted, strict=True):
        row["holm_p"] = p_value if row["raw_sign_test_p"] is not None else None
    summary = {
        "profile": report["profile"],
        "manifest_fingerprint": report["manifest_fingerprint"],
        "planned_contrasts": len(planned),
        "summaries": summaries,
        "publication_suite_complete": report["publication_suite_complete"],
        "scientific_validation_passed": False,
        "multiple_comparison_method": "Holm full planned family",
        "bootstrap_replicates": bootstrap_replicates,
        "analysis_seed": seed,
    }
    (destination / "statistics.json").write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False) + "\n"
    )
    fields = [
        "spec_id",
        "scenario",
        "method",
        "n_planned_seeds",
        "n_available_seeds",
        "mean_gain_nats",
        "sd_gain_nats",
        "median_gain_nats",
        "positive_seed_fraction",
        "raw_sign_test_p",
        "holm_p",
    ]
    with (destination / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    table = [
        "| Scenario/spec | Method | Seeds available/planned | Mean gain (nats) | SD | 95% CI | Holm p |",
        "|---|---|---|---|---|---|---|",
    ]
    latex = [
        r"\begin{longtable}{llllll}",
        r"Scenario/spec & Method & Seeds & Gain (nats) & SD & 95\% CI \\",
        r"\hline",
    ]

    def number(value: Any) -> str:
        return "unavailable" if value is None else f"{value:.5g}"

    for row in summaries:
        ci = row["mean_gain_95ci"]
        bounds = "unavailable" if ci is None else f"[{ci[0]:.4g}, {ci[1]:.4g}]"
        table.append(
            f"| {row['scenario']}/{row['spec_id'][:6]} | {row['method']} | "
            f"{row['n_available_seeds']}/{row['n_planned_seeds']} | {number(row['mean_gain_nats'])} | "
            f"{number(row['sd_gain_nats'])} | {bounds} | {number(row['holm_p'])} |"
        )
        label = f"{row['scenario']}/{row['spec_id'][:6]}".replace("_", r"\_")
        method = row["method"].replace("_", r"\_")
        latex.append(
            f"{label} & {method} & {row['n_available_seeds']}/{row['n_planned_seeds']} & "
            f"{number(row['mean_gain_nats'])} & {number(row['sd_gain_nats'])} & {bounds} " + r"\\"
        )
    latex.append(r"\end{longtable}")
    (destination / "numeric-summary.md").write_text("\n".join(table) + "\n")
    (destination / "summary.tex").write_text("\n".join(latex) + "\n")
    _figures(summaries, report, destination)
    completed = report["publication_suite_complete"]
    boundary = (
        "The declared manuscript matrix was executed completely; scientific acceptance still requires "
        "review of recovery, controls and failure criteria."
        if completed
        else "This is a bounded development run. The declared manuscript matrix has not been completed."
    )
    (destination / "analysis-report.md").write_text(
        "# Shared-transition framework validation\n\n" + boundary + "\n\n"
        "Primary question: does each graph representation improve held-out transition log loss over the same "
        "activity/history baseline? Positive gain favors graph augmentation.\n\n"
        f"Attempted method runs: {report['attempted_methods']}; available: {report['available_methods']}. "
        f"Planned contrasts: {len(planned)}. All seeds, failures and unavailable contrasts are retained.\n\n"
        "The numerical table is `numeric-summary.md`. Figures show the observed seed-level gains and "
        "method availability. Snapshot correspondence and graph-code diversity do not establish circuit "
        "equivalence. No real animals or physical interventions were evaluated.\n"
    )
    (destination / "stats-appendix.md").write_text(
        "# Statistical specification\n\nIndependent unit: a newly simulated raw-process seed, "
        "paired across methods. "
        "Specifications remain separate; overlapping anchors and methods are never replicates.\n\n"
        "Manifold and native state codes are fitted separately for each seed. Transition codes in the "
        "attribution export are local to that seed and must not be pooled by numeric label across seeds.\n\n"
        "Report mean, sample SD, median, raw nats gain and positive-seed fraction. Percentile bootstrap "
        "95% intervals resample independent seeds within a specification, "
        "with at least three available seeds and the complete planned seed set. "
        "Intervals at small n are exploratory and coverage is not established.\n\n"
        "The exact two-sided sign test tests equal probability of positive and negative seed-level gain, "
        "excluding exact zero ties. It requires independent seeds; it does not assume Gaussian gains. "
        "No parametric test or normality-based test selection was used. Missing planned seeds suppress "
        "inference for that contrast. Holm correction uses the entire planned contrast family; unavailable "
        "contrasts enter as p=1. No best-seed selection or neuron-level significance is reported.\n\n"
        "Graph recovery is scored only on unmixed generating histories and in common scaled units. "
        "Under calcium/noise observation changes, latent structural coefficients are not automatically "
        "the observed-process estimand. Those recovery entries remain unavailable.\n"
    )
    (destination / "figure-catalog.md").write_text(
        "# Figure catalog\n\n## figure-01-predictive-gain.pdf / .png\n\n"
        "Purpose: assess whether graph augmentation predicts the same future outcomes better than M0. "
        "Points are mean seed-level nats gains; error bars are 95% seed-bootstrap intervals when available. "
        "For compactness the first declared specification per scenario is shown; full numeric tables cover "
        "every specification. Notice negative as well as positive gains and intervals spanning zero. "
        "This identifies follow-up comparisons, not neuronal necessity.\n\n"
        "## figure-02-availability.pdf / .png\n\nPurpose: expose numerical/eligibility failures rather than "
        "selecting successful methods. Bars show available/attempted method runs across all executed cells. "
        "These are exact counts, with no sampling error bars. Low availability limits scientific comparison; "
        "an unavailable method is not evidence of poor biological validity.\n"
    )
    (destination / "manuscript_results.tex").write_text(
        "% Generated from validation.json; do not edit numbers by hand.\n"
        r"\paragraph{Executed framework validation.}" + "\n" + boundary.replace("_", r"\_") + " "
        f"The artifact records {report['attempted_methods']} attempted method runs, "
        f"of which {report['available_methods']} were available. "
        "Independent simulation seeds, rather than overlapping windows, define replication. "
        "The complete contrast table and figure catalog retain negative and unavailable results. "
        "These observations do not establish biological circuit degeneracy or intervention effects.\n"
    )
    return summary


def _figures(summaries: list[dict[str, Any]], report: Mapping[str, Any], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = destination / "figures"
    figures.mkdir(exist_ok=True)
    scenarios = list(dict.fromkeys(cell["scenario"] for cell in report["planned_cells"]))
    figure, axes = plt.subplots(len(scenarios), 1, figsize=(8, 2.8 * len(scenarios)), squeeze=False)
    for axis, scenario in zip(axes[:, 0], scenarios, strict=True):
        spec = next(cell["spec_id"] for cell in report["planned_cells"] if cell["scenario"] == scenario)
        selected = [row for row in summaries if row["spec_id"] == spec]
        for index, row in enumerate(selected):
            if row["mean_gain_nats"] is None:
                axis.text(index, 0, "NA", ha="center")
                continue
            interval = row["mean_gain_95ci"]
            axis.plot(index, row["mean_gain_nats"], "o", color="#0072B2")
            if interval:
                axis.plot([index, index], interval, color="#0072B2")
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xticks(range(len(selected)), [row["method"] for row in selected], rotation=35, ha="right")
        axis.set_ylabel(f"{scenario}\nGain (nats)")
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(figures / f"figure-01-predictive-gain.{extension}", dpi=300)
    plt.close(figure)

    methods = report["planned_methods"]
    attempted = [sum(row["method"] == method for row in report["rows"]) for method in methods]
    available = [
        sum(row["method"] == method and row["status"] == "available" for row in report["rows"])
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.bar(range(len(methods)), attempted, color="white", edgecolor="black", label="Attempted", hatch="//")
    axis.bar(range(len(methods)), available, color="#56B4E9", edgecolor="black", label="Available")
    axis.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
    axis.set_ylabel("Method runs")
    axis.legend()
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(figures / f"figure-02-availability.{extension}", dpi=300)
    plt.close(figure)


def export_manuscript_validation_bundle(analysis_dir: str | Path, manuscript_dir: str | Path) -> Path:
    """Copy only generated, provenance-linked artifacts into the active manuscript."""
    source = Path(analysis_dir)
    destination = Path(manuscript_dir) / "validation_artifacts"
    statistics = json.loads((source / "statistics.json").read_text())
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("manuscript_results.tex", "summary.tex"):
        shutil.copy2(source / name, destination / name)
    for name in ("figure-01-predictive-gain.pdf", "figure-02-availability.pdf"):
        shutil.copy2(source / "figures" / name, destination / name)
    provenance = {
        "source_analysis_directory": str(source.resolve()),
        "manifest_fingerprint": statistics["manifest_fingerprint"],
        "profile": statistics["profile"],
        "publication_suite_complete": statistics["publication_suite_complete"],
        "scientific_validation_passed": False,
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return destination
