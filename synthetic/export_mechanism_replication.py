"""Export completed replication figures without refitting or changing acceptance."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "effectome-replication-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = [
    "known_support",
    "empty_support",
    "conventional_cgc",
    "conventional_cgc_star",
    "fast_cgc",
    "fast_cgc_star",
]
LABELS = ["Known\nsupport", "Empty\nsupport", "c-GC", "c-GC*", "fastC-GC", "fastC-GC*"]


def export_replication_figures(root: Path) -> list[Path]:
    report = json.loads((root / "replication-report.json").read_text())
    if not report["execution_complete"]:
        raise ValueError("figures require the complete declared study; partial results are not promoted")
    destination = root / "analysis/figures"
    destination.mkdir(parents=True, exist_ok=True)
    methods = [name for name in METHODS if name in report["summary"]]
    labels = [LABELS[METHODS.index(name)] for name in methods]
    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    saved = []

    def save(fig: Any, name: str) -> None:
        for extension in ("pdf", "png"):
            path = destination / f"{name}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            saved.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, metric, title in zip(
        axes, ("topk_recall", "effect_mae"), ("Driver top-k recall", "Intervention-effect MAE"), strict=True
    ):
        for index, method in enumerate(methods):
            summary = report["summary"][method]
            for offset, scope, color in ((-0.15, "comparison", "#697d9b"), (0.15, "positive", "#cb7e35")):
                statistic = summary[scope][metric]
                mean = statistic["mean"] if statistic["complete"] else statistic["available_mean_descriptive"]
                interval = statistic["mean_95ci"]
                if mean is None:
                    ax.text(index + offset, 0.02, "NA", rotation=90, ha="center", fontsize=7)
                    continue
                ax.errorbar(
                    index + offset,
                    mean,
                    yerr=[[max(0, mean - interval[0])], [max(0, interval[1] - mean)]]
                    if interval is not None
                    else None,
                    fmt="o",
                    capsize=3,
                    color=color,
                    markerfacecolor=color if statistic["complete"] else "none",
                    label=("256" if scope == "comparison" else "1024") if index == 0 else None,
                )
        ax.set(xticks=range(len(methods)), xticklabels=labels, title=title)
        ax.axhline(
            0.75 if metric == "topk_recall" else 0.05, color="firebrick", linestyle="--", linewidth=0.8
        )
        if metric == "topk_recall":
            ax.set_ylim(-0.025, 1.15)
        ax.legend(title="Discovery history", fontsize=8)
    fig.suptitle(
        "Planned n=20 fresh driver seeds; hollow points describe eligible subsets\n"
        "Bars: descriptive seed-bootstrap 95% intervals for complete sets only"
    )
    save(fig, "01-fresh-seed-history")

    native = [name for name in methods if name not in {"known_support", "empty_support"}]
    native_labels = [LABELS[METHODS.index(name)] for name in native]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for index, method in enumerate(native):
        summary = report["summary"][method]
        for offset, metric, color, label in (
            (-0.15, "nomination_recall", "#697d9b", "Recall"),
            (0.15, "nomination_precision", "#cb7e35", "Precision"),
        ):
            statistic = summary["positive"][metric]
            mean = statistic["mean"] if statistic["complete"] else statistic["available_mean_descriptive"]
            if mean is not None:
                interval = statistic["mean_95ci"]
                axes[0].errorbar(
                    index + offset,
                    mean,
                    yerr=[[max(0, mean - interval[0])], [max(0, interval[1] - mean)]] if interval else None,
                    fmt="o",
                    capsize=3,
                    color=color,
                    markerfacecolor=color if statistic["complete"] else "none",
                    label=label if index == 0 else None,
                )
        upper = summary["null_one_sided_95_upper_bound"]
        n = summary["null_available_seeds"]
        failures = summary["null_false_nomination_seeds"]
        if upper is not None and n:
            rate = failures / n
            axes[1].plot([index, index], [rate, upper], color="#466a81")
            axes[1].scatter(index, rate, color="#466a81")
            axes[1].scatter(index, upper, color="#466a81", marker="_")
            axes[1].annotate(
                f"{failures}/{n}",
                (index, upper),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        else:
            axes[1].text(index, 0.02, "NA", ha="center")
    for ax in axes:
        ax.set(xticks=range(len(native)), xticklabels=native_labels)
    axes[0].set(title="Calibrated primary nomination", ylim=(-0.025, 1.15))
    axes[0].axhline(0.75, color="firebrick", linestyle="--", linewidth=0.8)
    axes[0].legend(loc="center right", fontsize=8)
    axes[1].set(title="Untouched null-test seeds: any false nominee")
    axes[1].axhline(0.05, color="firebrick", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(bottom=-0.005)
    fig.suptitle(
        "Hollow positive points are eligible-subset descriptions, not full acceptance\n"
        "Null caps: per-method one-sided exact 95% upper bounds"
    )
    save(fig, "02-nomination-and-null")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for index, method in enumerate(native):
        paired = report["paired_history"][method]
        statistic = paired["delta_effect_mae"]
        values = [row["delta_effect_mae"] for row in paired["seed_changes"]]
        ax.scatter(
            values,
            np.full(len(values), index) + np.linspace(-0.12, 0.12, len(values)),
            marker="x",
            alpha=0.5,
            color="#697d9b",
            s=20,
        )
        if statistic["mean"] is not None:
            mean, interval = statistic["mean"], statistic["mean_95ci"]
            ax.errorbar(
                mean,
                index,
                xerr=[[max(0, mean - interval[0])], [max(0, interval[1] - mean)]] if interval else None,
                fmt="o",
                color="#cb7e35",
                capsize=3,
            )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        yticks=range(len(native)),
        yticklabels=native_labels,
        xlabel="MAE at history 1024 minus MAE at history 256 (probability units)",
        title="Paired fresh-seed effects; negative changes favor longer histories",
    )
    save(fig, "03-paired-mae-changes")

    catalog = """# Replication figure catalog

All files have PDF and PNG versions. Source: the completed replication-report.json,
its seed_metrics, summary and paired_history records. No new fitting, threshold
selection or acceptance recomputation occurs during export. Missing complete-set
statistics remain NA in the frozen acceptance tables. Hollow points additionally
show the already-recorded available_mean_descriptive on eligible seeds, without
an interval or a complete-set acceptance claim. Actual eligible/planned counts
remain explicit in the JSON and results discussion. Related methods and
directions are not independent seed replicates.

## 01-fresh-seed-history

Purpose: compare the known/empty-support controls and all native methods at paired
256/1024 histories. Filled dots are complete-set seed means, hollow dots are
eligible-subset descriptions; bars are the runner's 2,000-resample
descriptive seed-bootstrap 95% intervals, with the actual complete counts in JSON.
Dashed lines show the frozen 0.75 ranking and 0.05 MAE gates. Notice whether longer
histories approach the known-support control consistently across fresh seeds.
The secondary history has no calibrated nomination/null claim. The two histories
share inputs within seed; their means are not independent groups.

## 02-nomination-and-null

Purpose: separate positive nomination performance from specificity. Positive
points show recall/precision with descriptive seed-bootstrap 95% intervals. Null
points show any-false-nominee fractions in untouched seeds; caps are one-sided
Clopper–Pearson 95% upper bounds, per method. Dashed thresholds are 0.75 and0.05.
Read actual counts and all gate decisions from analysis-report.md. Calibration
is global-null marginal protection and does not imply strong non-null FDR control.

## 03-paired-mae-changes

Purpose: expose every paired raw-seed MAE change rather than a selected example.
Crosses are independent process-seed differences, dots their mean and bars the
runner's descriptive seed-bootstrap 95% interval. Negative changes favor history
1024. The four planned native paired sign tests use one Holm family in JSON.
Neither a significant history contrast nor a favorable mean overrides a failed
primary nomination, null or control gate. This observed model-matched generator
does not establish robustness to hidden state or biological intervention validity.
"""
    (root / "analysis/figure-catalog.md").write_text(catalog)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/driver-mechanism-replication/replication/mechanism-replication-01"),
    )
    args = parser.parse_args()
    for path in export_replication_figures(args.output_root):
        print(path)


if __name__ == "__main__":
    main()
