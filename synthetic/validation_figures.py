"""Render the four-CSL validation from frozen evidence without refitting methods.

Detailed worked panels require the comparison saved by the original run. Missing
artifacts remain unavailable, including when stochastic estimators could be rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from effectome.experiments.csl_sanity import CSL_METHODS  # noqa: E402
from effectome.experiments.transition_examples import (  # noqa: E402
    TransitionExampleConfig,
    make_transition_example,
)
from effectome.workflows.resume import content_fingerprint  # noqa: E402
from synthetic._validation_plots import (  # noqa: E402
    METHOD_LABELS,
    SCENARIO_LABELS,
    _attribution,
    _designed_controls,
    _predictive_overview,
    _shared_transitions,
)

CONTROL_DESIGN = {
    "one_to_one": ("Two circuits; separated means", "Favorable graph/activity correspondence"),
    "matched_marginals": ("Two circuits; equal snapshot law", "Separate occupancy from temporal dynamics"),
    "fixed_circuit": ("One circuit; different trial starts", "Separate trajectory diversity from switching"),
    "stationary": ("One circuit; no trial resets", "Expose estimation-induced graph multiplicity"),
}


def _recording(report: dict[str, Any], cell: dict[str, Any]) -> Any:
    config = {
        **report["resolved_protocol"]["generator_defaults"],
        "scenario": cell["scenario"],
        "seed": cell["seed"],
        "n_samples": report["config"]["n_samples"],
        "n_neurons": cell["n_neurons"],
        "regime_dwell": cell["regime_dwell"],
        "trial_length": max(256, 2 * cell["history_length"]),
    }
    return make_transition_example(TransitionExampleConfig(**config))


def _worked_case(report: dict[str, Any], source: Path) -> tuple[dict | None, dict, dict]:
    """Read the first declared worked cell; never substitute seeds or refit graphs."""
    cells = report["planned_cells"]
    scenario = (
        "matched_marginals"
        if any(c["scenario"] == "matched_marginals" for c in cells)
        else cells[0]["scenario"]
    )
    cell = next(c for c in cells if c["scenario"] == scenario)
    status = {"status": "unavailable",
              "selection_rule": "first declared matched_marginals cell or first cell"}
    if not any(row["cell_id"] == cell["cell_id"] for row in report["rows"]):
        return None, cell, {**status, "reason": "first declared worked case was not executed"}
    path = source / "cells" / f"{cell['cell_id']}.json"
    if not path.is_file():
        return None, cell, {**status, "reason": "original detailed cell artifact is missing"}
    saved = json.loads(path.read_text())
    if saved.get("signature") != report["manifest_fingerprint"]:
        raise ValueError("worked-case cell artifact does not match the frozen run signature")
    comparison = saved.get("comparison")
    if comparison is None:
        return None, cell, {**status, "reason": "original cell did not save its detailed comparison"}
    if set(comparison.get("methods", {})) != set(CSL_METHODS):
        raise ValueError("frozen worked comparison must contain exactly the four native CSL methods")
    return comparison, cell, {
        "status": "available",
        "source_artifact": str(path.relative_to(source)),
        "source_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "estimator_refit": False,
    }


def _tables(report: dict, statistics: dict, destination: Path) -> None:
    scenarios = list(dict.fromkeys(c["scenario"] for c in report["planned_cells"]))
    specs = {s: next(c["spec_id"] for c in report["planned_cells"] if c["scenario"] == s) for s in scenarios}
    lookup = {(r["spec_id"], r["method"]): r for r in statistics["summaries"]}
    markdown = [
        "| Method | " + " | ".join(SCENARIO_LABELS.get(s, s) for s in scenarios) + " |",
        "|---|" + "---|" * len(scenarios),
    ]
    latex = [
        r"\begin{tabular}{@{}l" + "r" * len(scenarios) + r"@{}}",
        r"\toprule",
        "Method & "
        + " & ".join(r"\shortstack{" + SCENARIO_LABELS.get(s, s).replace(" ", r"\\") + "}" for s in scenarios)
        + r" \\",
        r"\midrule",
    ]
    incomplete = []
    for method in report["planned_methods"]:
        md, tex = [], []
        for scenario in scenarios:
            row = lookup[(specs[scenario], method)]
            mean, sd = row["mean_gain_nats"], row["sd_gain_nats"]
            counts = f"{row['n_available_seeds']}/{row['n_planned_seeds']}"
            md.append(
                "Unavailable (" + counts + ")"
                if mean is None
                else f"{mean:.3f} ± {sd:.3f} ({counts})"
                if sd is not None
                else f"{mean:.3f} (SD unavailable; {counts})"
            )
            value = "NA" if mean is None else f"${mean:.3f}" + (rf"\pm {sd:.3f}$" if sd is not None else "$")
            if row["n_available_seeds"] != row["n_planned_seeds"]:
                value += r"$^{\dagger}$"
                incomplete.append(
                    f"{METHOD_LABELS.get(method, method)} / "
                    f"{SCENARIO_LABELS.get(scenario, scenario)}: {counts}"
                )
            tex.append(value)
        label = METHOD_LABELS.get(method, method).replace("_", r"\_")
        markdown.append(f"| {label} | " + " | ".join(md) + " |")
        latex.append(label + " & " + " & ".join(tex) + r" \\")
    latex.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip\footnotesize Mean $\pm$ sample SD; gains in nats. "
            + f"{len(report['config']['seeds'])} planned independent seeds per cell. "
            + (r"$\dagger$ " + "; ".join(incomplete) + "." if incomplete else "All planned seeds available."),
        ]
    )
    (destination / "table-01-predictive-summary.md").write_text("\n".join(markdown) + "\n")
    (destination / "table-01-predictive-summary.tex").write_text("\n".join(latex) + "\n")
    design_md = ["| Control | Generating truth | Validation question |", "|---|---|---|"]
    design_tex = [
        r"\begin{tabular}{@{}p{0.19\linewidth}p{0.34\linewidth}p{0.40\linewidth}@{}}",
        r"\toprule",
        r"Control & Generating truth & Validation question \\",
        r"\midrule",
    ]
    for scenario in scenarios:
        truth, question = CONTROL_DESIGN[scenario]
        label = SCENARIO_LABELS[scenario]
        design_md.append(f"| {label} | {truth} | {question} |")
        design_tex.append(f"{label} & {truth} & {question}" + r" \\")
    design_tex.extend([r"\bottomrule", r"\end{tabular}"])
    (destination / "table-02-control-design.md").write_text("\n".join(design_md) + "\n")
    (destination / "table-02-control-design.tex").write_text("\n".join(design_tex) + "\n")


def render_validation_visuals(
    validation_dir: str | Path,
    output_dir: str | Path,
    manuscript_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Render true controls, one reproducible worked case, and exact saved summaries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, destination = Path(validation_dir), Path(output_dir)
    report = json.loads((source / "validation.json").read_text())
    statistics = json.loads((source / "analysis/statistics.json").read_text())
    if report["manifest_fingerprint"] != statistics["manifest_fingerprint"]:
        raise ValueError("simulation and statistical provenance disagree")
    if set(report["planned_methods"]) != set(CSL_METHODS):
        raise ValueError(
            "simulation figures require exactly the four native CSL methods; legacy panel excluded"
        )
    destination.mkdir(parents=True, exist_ok=True)
    renderer_hash = content_fingerprint(
        {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("validation_figures.py", "_validation_plots.py", "validation_figure_catalog.md")
        }
    )
    comparison, cell, worked_status = _worked_case(report, source)
    with plt.rc_context(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42}
    ):
        _predictive_overview(report, statistics, destination)
        _designed_controls(report, destination, _recording)
        if comparison is not None:
            _shared_transitions(comparison, cell, destination)
            _attribution(comparison, destination)
        else:
            for stem in ("figure-04-shared-transitions", "figure-05-transition-attribution"):
                for suffix in (".pdf", ".png"):
                    (destination / (stem + suffix)).unlink(missing_ok=True)
    _tables(report, statistics, destination)
    catalog = Path(__file__).with_name("validation_figure_catalog.md").read_text()
    (destination / "figure-catalog.md").write_text(catalog)
    files = sorted(p.name for p in destination.glob("figure-*.*") if p.suffix in {".pdf", ".png"})
    files += sorted(p.name for p in destination.glob("table-*.*"))
    provenance = {
        "source_manifest_fingerprint": report["manifest_fingerprint"],
        "source_validation_sha256": hashlib.sha256((source / "validation.json").read_bytes()).hexdigest(),
        "source_statistics_sha256": hashlib.sha256(
            (source / "analysis/statistics.json").read_bytes()
        ).hexdigest(),
        "renderer_sha256": renderer_hash,
        "worked_case": cell,
        "selection_rule": (
            "first declared matched_marginals specification and first declared seed; no score selection"
        ),
        "worked_case_status": worked_status,
        "worked_case_read_from_original_run": comparison is not None,
        "source_profile": report["profile"],
        "publication_suite_complete": report["publication_suite_complete"],
        "scientific_validation_passed": False,
        "files": files,
    }
    (destination / "visual-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if manuscript_dir is not None:
        exported = Path(manuscript_dir) / "validation_artifacts"
        exported.mkdir(parents=True, exist_ok=True)
        for name in files + ["visual-provenance.json"]:
            if name.endswith((".pdf", ".tex", ".json")):
                shutil.copy2(destination / name, exported / name)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path,
                        default=ROOT / "outputs/framework-validation/csl_sanity/smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manuscript-dir", type=Path)
    arguments = parser.parse_args()
    manifest = render_validation_visuals(
        arguments.validation_dir,
        arguments.output_dir or arguments.validation_dir / "visuals",
        arguments.manuscript_dir,
    )
    print(
        json.dumps({"source_profile": manifest["source_profile"], "artifacts": manifest["files"]}, indent=2)
    )
