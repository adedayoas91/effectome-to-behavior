"""Rebuild descriptive tables and figures from the shipped, audited run summaries.

Run with the project's existing Python environment. This script never fits a model,
changes a notebook, or treats directions/windows as independent process seeds.
Original reports and per-file hashes are frozen separately in inputs.zip/provenance.json.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "effectome-collation-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binomtest

ROOT = Path(__file__).resolve().parent
METHODS = ["conventional_cgc", "conventional_cgc_star", "fast_cgc", "fast_cgc_star"]
LABELS = ["cGC", "cGC*", "fast cGC", "fast cGC*"]
SCENARIOS = ["one_to_one", "matched_marginals", "fixed_circuit", "stationary"]


def read(name: str) -> dict:
    return json.loads((ROOT / "audits" / f"{name}_review.json").read_text())


def write_csv(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / "tables" / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def markdown_table(headers: list[str], rows: list[list]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(str(v) for v in row) + " |" for row in rows],
        ]
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for folder in ("tables", "figures"):
        (ROOT / folder).mkdir(exist_ok=True)
    framework, driver, mechanism = read("framework"), read("driver"), read("mechanism")
    checks: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            raise AssertionError(label)
        checks.append(label)

    sanity = framework["native_sanity"]["records"]
    contrasts = framework["framework_smoke"]["existing_analysis"]["summaries"]
    shared = framework["shared_comparison"]["records"]
    mechanism_rows = mechanism["numeric_summary_rows"]
    check(
        len(sanity) == 24 and all(r["status"] == "available" and r["exact_support"] for r in sanity),
        "24 native sanity fits have exact support",
    )
    check(len(contrasts) == 16, "all 16 planned framework contrasts retained")
    check(len(framework["framework_smoke"]["records"]) == 48, "all 48 framework method runs retained")
    check(driver["completion"]["available_method_runs"] == 560, "560 driver method runs available")
    check(
        len(mechanism_rows) == 36 and all(r["status"] == "available" for r in mechanism_rows),
        "all 36 mechanism lanes available",
    )
    check(mechanism["all_validation_checks_passed"], "upstream mechanism input/metric audit passed")

    # Recompute the already-specified two-sided sign tests and full-family Holm.
    raw_p = []
    for contrast in contrasts:
        gains = np.asarray(contrast["seed_gains"], dtype=float)
        check(len(gains) == 3, f"three raw seeds: {contrast['scenario']}/{contrast['method']}")
        check(
            np.isclose(gains.mean(), contrast["mean_gain_nats"], atol=1e-14, rtol=0),
            "framework mean agrees with seed values",
        )
        check(
            np.isclose(gains.std(ddof=1), contrast["sd_gain_nats"], atol=1e-14, rtol=0),
            "framework sample SD agrees with seed values",
        )
        nonzero = gains[gains != 0]
        pvalue = binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue if len(nonzero) >= 3 else 1.0
        # The stored pipeline declares an untestable contrast as p=1 in Holm.
        stored = contrast["raw_sign_test_p"]
        check(stored is None or np.isclose(pvalue, stored), "framework exact sign test agrees")
        raw_p.append(pvalue)
    order = np.argsort(raw_p, kind="stable")
    adjusted = np.empty(len(order))
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate([(len(order) - i) * raw_p[k] for i, k in enumerate(order)])
    )
    check(np.allclose(adjusted, [c["holm_p"] for c in contrasts]), "full 16-contrast Holm adjustment agrees")

    driver_summaries: list[dict] = []
    driver_seeds: list[dict] = []
    null_rows: list[dict] = []
    prior_rows: list[dict] = []
    for method, evidence in driver["methods"].items():
        for scenario, result in evidence["scenarios"].items():
            seed_rows = result["independent_seed_metrics"]
            driver_seeds.extend({"method": method, "scenario": scenario, **row} for row in seed_rows)
            for metric, summary in result["summaries"].items():
                values = [row[metric] for row in seed_rows if row.get(metric) is not None]
                check(
                    len(values) == summary["n_seeds"],
                    f"driver independent seed count: {method}/{scenario}/{metric}",
                )
                if values:
                    check(
                        np.isclose(np.mean(values), summary["mean"], atol=1e-14),
                        "driver mean agrees with seed values",
                    )
                    check(
                        np.isclose(np.std(values, ddof=1), summary["sd"], atol=1e-14),
                        "driver sample SD agrees with seed values",
                    )
                driver_summaries.append(
                    {
                        "method": method,
                        "scenario": scenario,
                        "metric": metric,
                        "decision": result["status"],
                        **summary,
                    }
                )
            if scenario == "null":
                row = result["false_nomination_seed_event"]
                k, n = row["seeds_with_any_false_nominee"], row["n_independent_seeds"]
                upper = float(beta.ppf(0.95, k + 1, n - k)) if k < n else 1.0
                check(
                    np.isclose(upper, row["one_sided_exact_upper_bound"], atol=1e-12),
                    "driver one-sided exact null bound agrees",
                )
                null_rows.append(
                    {
                        "method": method,
                        **row,
                        "maximum_allowed_upper_bound": 0.05,
                        "decision": result["status"],
                    }
                )
        for scenario, comparison in driver["prior_comparison"]["methods"][method]["scenarios"].items():
            prior_rows.append({"method": method, "scenario": scenario, **comparison})
    for effect in mechanism["paired_history_effect_sizes"]:
        if effect["scenario"] == "driver" and effect["method"] in METHODS:
            check(
                np.isclose(effect["delta_topk_1024_minus_256"], 5 / 6),
                "paired native top-k improvement is 5/6",
            )
            check(
                effect["paired_independent_seed_count"] == 1,
                "paired mechanism process seed count remains one",
            )

    availability = [
        {
            "stage": "01 native sanity",
            "planned_method_entries": 24,
            "available_method_entries": 24,
            "raw_seed_count_per_scenario": 3,
            "scope": "chain/null technical control",
        },
        {
            "stage": "02 shared transition example",
            "planned_method_entries": 4,
            "available_method_entries": 4,
            "raw_seed_count_per_scenario": 1,
            "scope": "development example",
        },
        {
            "stage": "04 framework smoke",
            "planned_method_entries": 48,
            "available_method_entries": 48,
            "raw_seed_count_per_scenario": 3,
            "scope": "four scenarios; publication matrix not executed",
        },
        {
            "stage": "05 driver recovery",
            "planned_method_entries": 560,
            "available_method_entries": 560,
            "raw_seed_count_per_scenario": "20 positive/diagnostic; 60 null",
            "scope": "frozen criteria; all methods fail",
        },
        *[
            {
                "stage": f"06 mechanism {name}",
                "planned_method_entries": 12,
                "available_method_entries": 12,
                "raw_seed_count_per_scenario": 1,
                "scope": "development seed 20 reused; includes four oracle/control entries",
            }
            for name in ["smoke", "history-256", "history-1024"]
        ],
    ]
    shared_rows = [
        {
            "method": row["method"],
            "status": row["status"],
            "raw_seed": 42,
            "n_training_origins": row["n_training_events"],
            "n_evaluation_origins": row["n_evaluation_events"],
            "baseline_log_loss": row["baseline_scores"]["log_loss"],
            "augmented_log_loss": row["augmented_scores"]["log_loss"],
            "gain_nats": row["log_loss_gain_nats"],
            "requested_graph_states": row["native_graph_states"]["requested_n_states"],
            "occupied_heldout_graph_states": sum(
                np.sum(row["native_graph_states"]["correspondence"]["heldout_contingency"], axis=1) > 0
            ),
            "heldout_H_effectome_given_manifold_nats": row["native_graph_states"]["correspondence"][
                "heldout_entropy_effectome_given_manifold_nats"
            ],
        }
        for row in shared
    ]
    for name, rows in [
        ("availability", availability),
        ("native-sanity", sanity),
        ("shared-prediction", shared_rows),
        ("framework-contrasts", contrasts),
        ("framework-seeds", framework["framework_smoke"]["records"]),
        ("driver-summary", driver_summaries),
        ("driver-seeds", driver_seeds),
        ("driver-null", null_rows),
        ("driver-prior-comparison", prior_rows),
        ("mechanism-lanes", mechanism_rows),
        ("mechanism-paired-effects", mechanism["paired_history_effect_sizes"]),
    ]:
        write_csv(name, rows)

    text = [
        "# Numeric summary: completed synthetic notebooks, 17 November 2026\n",
        "Run label `faster-machine-03`. Values below are descriptive unless the test/interval"
        " is explicitly named. CSV tables retain full floating-point precision. Method "
        "entries, directions, trials and overlapping windows are not independent seed "
        "replicates.\n",
        "## Completion\n",
        markdown_table(
            ["Stage", "Available/planned", "Raw seeds per scenario", "Scope"],
            [
                [
                    r["stage"],
                    f"{r['available_method_entries']}/{r['planned_method_entries']}",
                    r["raw_seed_count_per_scenario"],
                    r["scope"],
                ]
                for r in availability
            ],
        ),
        "\n## Shared transition example (one seed)\n",
        "Gain = activity-only baseline log loss minus graph-augmented log loss, in nats; larger is better.\n",
        markdown_table(
            ["Method", "Baseline", "Augmented", "Gain", "Occupied held-out graph states"],
            [
                [
                    r["method"],
                    f"{r['baseline_log_loss']:.6f}",
                    f"{r['augmented_log_loss']:.6f}",
                    f"{r['gain_nats']:+.6f}",
                    r["occupied_heldout_graph_states"],
                ]
                for r in shared_rows
            ],
        ),
        "\n## Framework smoke: all 16 contrasts\n",
        "Independent n=3 raw process seeds per row. SD is the sample SD. Intervals are the "
        "existing 2,000-resample percentile bootstrap of seed means: coverage is unstable at "
        "n=3. All Holm-adjusted p=1.000; no corrected evidence for improved prediction.\n",
        markdown_table(
            ["Scenario", "Method", "Mean gain ± SD (nats)", "Descriptive 95% bootstrap CI", "Holm p"],
            [
                [
                    c["scenario"],
                    c["method"],
                    f"{c['mean_gain_nats']:+.6f} ± {c['sd_gain_nats']:.6f}",
                    f"[{c['mean_gain_95ci'][0]:+.6f}, {c['mean_gain_95ci'][1]:+.6f}]",
                    f"{c['holm_p']:.3f}",
                ]
                for c in contrasts
            ],
        ),
        "\nRounded zeros include floating-point residuals; exact original values and sign-test"
        " decisions are retained in CSV.\n",
        "## Frozen driver-recovery criteria\n",
        "Positive scenarios have n=20 independent process seeds, averaging the two directions"
        " within each seed. Minimum recovery/group-win gate=0.75; null upper-bound gate≤0.05."
        " Hidden confounding is diagnostic.\n",
    ]
    driver_gate_rows = []
    for method in METHODS:
        scenarios = driver["methods"][method]["scenarios"]
        driver_gate_rows.append(
            [
                method,
                f"{scenarios['single_driver']['summaries']['topk_recall']['mean']:.3f}",
                f"{scenarios['correlated_bystander']['summaries']['topk_recall']['mean']:.3f}",
                f"{scenarios['redundant_group']['summaries']['group_win']['mean']:.3f}",
                "fail",
            ]
        )
    text += [
        markdown_table(
            ["Method", "Single top-k recall", "Bystander top-k recall", "Redundant group win", "Overall"],
            driver_gate_rows,
        ),
        "\nIndividual top-k is undefined for the redundant pair under the frozen intervention "
        "threshold; the group-win metric is its prespecified criterion. All required positive"
        " scenarios also have negative mean predictive gain. Full means, SDs and descriptive "
        "intervals are in `tables/driver-summary.csv`.\n",
        markdown_table(
            ["Method", "Null seeds with any false nominee", "Rate", "One-sided 95% exact upper bound"],
            [
                [
                    r["method"],
                    f"{r['seeds_with_any_false_nominee']}/{r['n_independent_seeds']}",
                    f"{r['fraction']:.6f}",
                    f"{r['one_sided_exact_upper_bound']:.6f}",
                ]
                for r in null_rows
            ],
        ),
        "\n## Model-matched mechanism development\n",
        "One existing independent raw seed (20) per scenario, reused in all panels; six lanes"
        " include known- and empty-support controls. Top-k recall is tie-aware and averaged "
        "over two directions, not two independent replicates. Effect MAE measures error in "
        "oracle probability reductions; factual Brier scores predicted probabilities against "
        "observed binary transitions. Lower is better for both.\n",
        markdown_table(
            ["Panel", "Method", "Driver top-k", "Effect MAE", "Factual Brier", "Support recall"],
            [
                [
                    r["report"],
                    r["method"],
                    f"{r['topk_recall']:.6f}",
                    f"{r['effect_MAE']:.9f}",
                    f"{r['factual_brier']:.9f}",
                    f"{r['support_recall']:.3f}",
                ]
                for r in mechanism_rows
                if r["scenario"] == "driver"
            ],
        ),
        "\nFor each native method, paired history 1,024 vs 256 gives Δtop-k=+0.833333, Δeffect"
        " MAE=−0.138224284 (98.6926% reduction) and ΔBrier=−0.629734039. The paired panels "
        "use the same longer recordings, origins and rollout noise; smoke uses a different "
        "recording length and is not a controlled history contrast. All null lanes have zero "
        "false nominees in one seed; their one-sided 95% exact upper bound is 0.95. These "
        "data cannot establish a 5% false-nomination guarantee.\n",
        "See [stats-appendix.md](stats-appendix.md) for tests and uncertainty limits and "
        "[analysis-report.md](analysis-report.md) for interpretation.\n",
    ]
    (ROOT / "numeric-summary.md").write_text("\n".join(text))

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    )
    colors = [plt.get_cmap("tab10")(index) for index in range(4)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, constrained_layout=True)
    for ax, scenario in zip(axes.flat, SCENARIOS, strict=True):
        for index, method in enumerate(METHODS):
            contrast = next(c for c in contrasts if c["scenario"] == scenario and c["method"] == method)
            mean = contrast["mean_gain_nats"]
            low, high = contrast["mean_gain_95ci"]
            ax.errorbar(
                mean,
                index,
                xerr=[[max(0, mean - low)], [max(0, high - mean)]],
                fmt="o",
                color=colors[index],
                capsize=3,
            )
            ax.scatter(
                contrast["seed_gains"],
                np.array([index - 0.12, index, index + 0.12]),
                marker="x",
                color=colors[index],
                s=24,
                alpha=0.75,
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set(
            yticks=range(4),
            yticklabels=LABELS,
            title=scenario.replace("_", " "),
            xlabel="Log-loss gain (nats; higher is better)",
        )
        ax.invert_yaxis()
    fig.suptitle(
        "Framework smoke: 3 seeds per contrast; all 16 Holm p = 1\nDots: means; bars: "
        "descriptive bootstrap 95% CI; crosses: raw seeds"
    )
    save_figure(fig, "01-framework-gains")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    gates = [
        ("single_driver", "topk_recall", "Single-driver top-k recall"),
        ("correlated_bystander", "topk_recall", "Correlated-bystander top-k recall"),
        ("redundant_group", "group_win", "Redundant-pair group-win fraction"),
    ]
    for ax, (scenario, metric, title) in zip(list(axes.flat)[:3], gates, strict=True):
        for index, method in enumerate(METHODS):
            summary = driver["methods"][method]["scenarios"][scenario]["summaries"][metric]
            mean, (low, high) = summary["mean"], summary["mean_interval"]
            ax.errorbar(
                index,
                mean,
                yerr=[[max(0, mean - low)], [max(0, high - mean)]],
                fmt="o",
                color=colors[index],
                capsize=4,
            )
        ax.axhline(0.75, color="firebrick", linestyle="--", label="Acceptance minimum 0.75")
        ax.set(xticks=range(4), xticklabels=LABELS, ylim=(-0.025, 1.025), title=f"{title} (n=20)")
        ax.legend(fontsize=8)
    ax = axes.flat[3]
    for index, row in enumerate(null_rows):
        ax.plot([index, index], [row["fraction"], row["one_sided_exact_upper_bound"]], color=colors[index])
        ax.scatter(index, row["fraction"], color=colors[index], marker="o")
        ax.scatter(index, row["one_sided_exact_upper_bound"], color=colors[index], marker="_")
        ax.annotate(
            f"{row['seeds_with_any_false_nominee']}/60",
            (index, row["fraction"]),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.axhline(0.05, color="firebrick", linestyle="--", label="Upper-bound maximum 0.05")
    ax.set(xticks=range(4), xticklabels=LABELS, ylim=(-0.025, 1.025), title="Null: any false nominee (n=60)")
    ax.legend(fontsize=8)
    fig.suptitle(
        "Frozen driver criteria: no method passes\nRecovery bars: descriptive seed-bootstrap "
        "95% CI; null caps: one-sided exact 95% upper bounds"
    )
    save_figure(fig, "02-driver-gates")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    lanes = ["known_support", "empty_support", *METHODS]
    for ax, (metric, title) in zip(
        axes,
        [
            ("topk_recall", "Driver top-k recall (higher is better)"),
            ("effect_MAE", "Intervention-effect MAE (lower is better)"),
        ],
        strict=True,
    ):
        for index, (history, color) in enumerate([("history-256", "#697d9b"), ("history-1024", "#cb7e35")]):
            values = [
                next(
                    r[metric]
                    for r in mechanism_rows
                    if r["report"] == history and r["scenario"] == "driver" and r["method"] == method
                )
                for method in lanes
            ]
            ax.bar(np.arange(6) + (index - 0.5) * 0.35, values, width=0.35, label=history, color=color)
        ax.set(
            xticks=range(6),
            xticklabels=["Known\nsupport", "Empty\nsupport", "cGC", "cGC*", "fast\ncGC", "fast\ncGC*"],
            title=title,
            ylim=(0, 1.25 if metric == "topk_recall" else 0.18),
        )
        ax.legend(fontsize=8)
    fig.suptitle(
        "Paired history diagnostic: one existing process seed (20)\nSame recordings and "
        "rollout noise; no process-level confidence interval"
    )
    save_figure(fig, "03-mechanism-history")
    (ROOT / "verification.json").write_text(
        json.dumps(
            {
                "all_checks_passed": True,
                "check_count": len(checks),
                "checks": checks,
                "fits_rerun": False,
                "independent_new_seeds": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"Verified {len(checks)} checks; wrote 11 CSV tables, numeric summary, "
        "and 3 figures (PDF + PNG). No fits run."
    )


if __name__ == "__main__":
    main()
