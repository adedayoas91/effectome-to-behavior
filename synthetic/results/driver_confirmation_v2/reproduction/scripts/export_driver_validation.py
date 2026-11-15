"""Export the actual four-CSL driver study, retaining partial and negative outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.csl_sanity_reporting import METHOD_LABELS
from effectome.experiments.driver_validation_reporting import render_driver_validation_report


def export_driver_validation(validation_dir: str | Path, manuscript_dir: str | Path) -> Path:
    source = Path(validation_dir) / "driver-validation.json"
    source_bytes = source.read_bytes()
    report = json.loads(source_bytes)
    if set(report["planned_methods"]) != set(CSL_METHODS):
        raise ValueError("manuscript export requires exactly the four native CSL methods")
    analysis = source.parent / "analysis"
    render_driver_validation_report(report, analysis)
    destination = Path(manuscript_dir) / "driver_validation_artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    planned = len(report["planned_cells"])
    table = [r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
             r"Method & Available/planned & Attempted & Eligible directions & Decision \\", r"\midrule"]
    labels = dict(zip(CSL_METHODS, METHOD_LABELS, strict=True))
    config = report["config"]
    generator = config["generator"]
    origin = config.get("trial_origin_offset")
    if origin is None:
        raise ValueError("current manuscript driver export requires the scheduled-origin protocol")
    horizon = config["stride"] / generator["fps"]
    positive = config["positive_seeds"]
    null_seeds = config["null_seeds"]
    for method, result in report["acceptance"]["methods"].items():
        attempted = sum(r["method"] == method and r["source_state"] == 0 for r in report["seed_rows"])
        available = sum(r["method"] == method and r["source_state"] == 0
                        and r["status"] == "available" for r in report["seed_rows"])
        eligible = sum(s["n_eligible_directions"] for s in result["scenarios"].values())
        directions = sum(s["n_planned_directions"] for s in result["scenarios"].values())
        table.append(f"{labels[method]} & {available}/{planned} & {attempted} & "
                     f"{eligible}/{directions} & {result['status']} " + r"\\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (destination / "availability.tex").write_text("\n".join(table) + "\n")
    recovery = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
                r"Method & Single-driver recall & Bystander recall & Joint-group wins & Null upper bound \\",
                r"\midrule"]
    for method, result in report["acceptance"]["methods"].items():
        scenarios = result["scenarios"]
        values = []
        for scenario, key in (("single_driver", "topk_recall"), ("correlated_bystander", "topk_recall"),
                              ("redundant_group", "group_win")):
            summary = scenarios[scenario]["summaries"][key]
            count = scenarios[scenario]["n_complete_eligible_seeds"]
            planned_count = scenarios[scenario]["n_planned_seeds"]
            mean = summary["mean"]
            values.append(("NA" if mean is None else f"{mean:.3f}") + f" ({count}/{planned_count})")
        null = scenarios["null"]["null_nomination"]
        bound = null["one_sided_exact_upper_bound"]
        values.append(("NA" if bound is None else f"{bound:.3f}")
                      + f" ({null['n_independent_seeds']}/{len(null_seeds)})")
        recovery.append(labels[method] + " & " + " & ".join(values) + r" \\")
    recovery.extend([r"\bottomrule", r"\end{tabular}"])
    (destination / "recovery.tex").write_text("\n".join(recovery) + "\n")
    state = "complete" if report["execution_complete"] else "incomplete"
    text = (
        "% Generated from actual four-CSL driver-validation.json.\n"
        "\\paragraph{Scheduled-origin four-CSL validation.}\n"
        f"The revised protocol fixes one origin at sample {origin} of each "
        f"{generator['trial_length']}-sample trial, using {config['history_length']} history samples "
        f"and a strictly future {config['target_length']}-sample target ending at "
        f"{origin + config['stride']} ({horizon:.1f} seconds at {generator['fps']:g} Hz). "
        f"Each recording contains {generator['n_trials']} independent trials; the chronological "
        f"evaluation fraction is {config['evaluation_fraction']:.2f}. A training-fitted "
        f"{config['manifold_dims']}-dimensional PCA chart and "
        "two-region vocabulary are common to all methods. Graphs are estimated only at origins, "
        "without inventing future graph states. Timing and chart choices were developed on seeds 10/11. "
        f"The executed profile declares positive seeds {min(positive)}--{max(positive)} "
        f"and null seeds {min(null_seeds)}--{max(null_seeds)}. "
        "Recovery criteria were not relaxed. The protocol is frozen in the repository, "
        "not externally preregistered.\n\n"
        f"Execution is \\textbf{{{state}}}: {report['attempted_method_runs']}/{4 * planned} "
        f"planned method/recording runs have finished, and {report['available_method_runs']} "
        f"produced usable predictions. There are {planned} declared independent-seed recordings and "
        f"{4 * planned * generator['n_trials']:,} native window fits in the full panel. "
        f"The current overall decision is \\textit{{{report['acceptance']['overall_status']}}}. "
        "Incomplete seed sets, ineligible directions and failed fits remain in the denominators; "
        "execution and graph agreement do not establish reliable driver recovery.\n\n"
        "\\begin{table}[htbp]\n\\caption{Four-CSL driver-study numerical availability and eligibility. "
        "The planned recording denominator is retained even for unexecuted methods. Eligibility "
        "requires direction-specific event support, semantic agreement and paired unit scores/truth.}"
        "\n\\label{tab:driverstudyavailability}\n\\centering\\small\n"
        "\\input{driver_validation_artifacts/availability}\n\\end{table}\n"
        "\\begin{table}[htbp]\n\\caption{Intervention-defined recovery and null calibration. "
        "Values are eligible-seed means, with eligible/planned counts in parentheses. Top-k size is "
        "the intervention-positive set size; group wins compare the driver pair with its control. "
        "The null entry is the exact one-sided 95\\% upper bound for any false nominee per method/seed. "
        "NA indicates unsupported evidence. Seed intervals and full negative results are retained "
        "in the numerical artifact.}\n\\label{tab:driverstudyrecovery}\n\\centering\\small\n"
        "\\resizebox{\\linewidth}{!}{\\input{driver_validation_artifacts/recovery}}\n\\end{table}\n"
        "Predictive feature omissions are evaluated against actual paired simulator outgoing-disconnection "
        "effects. These observed traces include a noisy latent population coordinate, so success would "
        "be specific to this observation model, not proof of generic CSL identifiability. Hidden "
        "confounding is a diagnostic challenge. The four estimators belong to one family and are "
        "not independent methodological replications. Biological necessity remains unvalidated.\n"
    )
    for stem, caption, label in (
        ("figure-01-intervention-truth", "Actual simulator outgoing-disconnection probability reductions "
         "for both semantic directions. Means use independent seeds and one shared oracle per seed; "
         "negative effects and unavailable entries remain visible.", "drivertruth"),
        ("figure-02-predictive-omissions", "Transition-specific predictive omission scores for well 0 to 1 "
         "from the four native CSL methods. These are descriptive seed means, not intervention effects.",
         "driveromissions"),
        ("figure-03-recovery-and-null", "Independent-seed recovery and null calibration. Horizontal bars "
         "are descriptive bootstrap 95\\% intervals only for complete eligible seed sets; null points "
         "are exact one-sided 95\\% upper bounds. Dashed lines show frozen gates; missing points are "
         "unsupported. Incomplete or failed validation cannot be replaced by method agreement.",
         "drivergates"),
    ):
        text += ("\\begin{figure}[htbp]\n\\begin{widepage}\n\\centering\n"
                 f"\\includegraphics[width=\\linewidth]{{driver_validation_artifacts/{stem}.pdf}}\n"
                 f"\\caption{{{caption}}}\n\\label{{fig:{label}}}\n\\end{{widepage}}\n\\end{{figure}}\n")
        shutil.copy2(analysis / (stem + ".pdf"), destination / (stem + ".pdf"))
    (destination / "results.tex").write_text(text)
    (destination / "driver-validation.json").write_bytes(source_bytes)
    for name in ("acceptance.json", "numeric-summary.csv", "recovery-summary.md", "stats-appendix.md",
                 "nomination-agreement.csv", "figure-catalog.md"):
        shutil.copy2(analysis / name, destination / name)
    provenance = {"source_manifest_fingerprint": report["manifest_fingerprint"],
                  "source_validation_sha256": hashlib.sha256(source_bytes).hexdigest(),
                  "execution_complete": report["execution_complete"],
                  "attempted_method_runs": report["attempted_method_runs"],
                  "acceptance": report["acceptance"]["overall_status"],
                  "criteria": report["acceptance"]["criteria"],
                  "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in destination.iterdir() if p.is_file() and p.name != "provenance.json"}}
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path,
                        default=Path("outputs/driver-validation/csl_sanity/scheduled-v2/confirmatory"))
    parser.add_argument("--manuscript-dir", type=Path, default=Path("../manuscript-260926"))
    args = parser.parse_args()
    print(export_driver_validation(args.validation_dir, args.manuscript_dir))
