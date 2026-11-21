"""Build the run-enabled framework-validation notebook."""

# ruff: noqa: E501 -- Markdown paragraphs are kept intact in notebook templates.

from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell, write_notebook


def main() -> None:
    cells = [
        markdown_cell(
            "# Framework validation: protocol, independent repeats and manuscript exports",
            "",
            "This is the canonical synthetic validation entrypoint. The preceding notebooks inspect generating controls and native discovery; this notebook executes a frozen experiment matrix with independent seeds and explicit failure denominators.",
            "Smoke checks software. Development uses a larger sensitivity matrix with engineering estimator settings. The manuscript profile proposes the larger production-configuration study; completing a profile does not automatically validate a scientific claim.",
            "The current protocol uses conventional c-GC/c-GC* and fastC-GC/fastC-GC* as four related CSL methods. Native lag support has no VAR coefficient readout. Histories are at least 256 samples so the fast complete-shift test can attain beta=0.005. This revised protocol supersedes the earlier eight-method pilot and writes to its own output directory.",
            "Run All fits the declared smoke matrix by default. Change `PROFILE` only when you intend to run a different matrix. Results use a separate resumable run-label directory; set `RUN_VALIDATION=False` for saved-evidence inspection.",
        ),
        code_cell('''
import json

from IPython.display import Markdown, display

from effectome.experiments.framework_validation import (
    framework_validation_cells,
    load_framework_validation_profile,
    run_framework_validation,
)
from effectome.experiments.framework_validation_analysis import analyze_framework_validation
from effectome.workflows.cross_method import load_cross_method_profile
from notebooks._shared import PROJECT_ROOT, require_conventional_runtime

PROFILE = "smoke"  # smoke | development | manuscript
RUN_VALIDATION = True  # Run All fits the declared profile.
RUN_LABEL = "faster-machine-03"  # Reuse unchanged settings to resume; change after protocol/code edits.
RENDER_VISUALS = False  # Explicitly regenerate the manuscript visual bundle.
MAX_CELLS = None  # A cap produces a partial report and cannot complete the publication suite.
validation_config = load_framework_validation_profile(PROFILE)
validation_cells = framework_validation_cells(validation_config)
_, validation_methods = load_cross_method_profile(validation_config.estimator_profile)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "framework-validation"
    / validation_config.estimator_profile / PROFILE / "reruns" / RUN_LABEL
)
print("Independent seeds:", validation_config.seeds)
print("Planned raw-process cells:", len(validation_cells))
print("Planned method runs:", len(validation_methods) * len(validation_cells))
print("Methods:", tuple(validation_methods))
print("Estimation profile:", validation_config.estimator_profile)
print("Fitting enabled:", RUN_VALIDATION)
print("Outputs:", OUTPUT_DIRECTORY)
'''),
        markdown_cell(
            "## Execute or inspect the fixed validation matrix",
            "",
            "Positive log-loss gain means graph augmentation improves on the same activity/history baseline. All methods use the same held-out event origins. Histories, dwell times, horizons and observation stresses are independent declared factors. A stationary no-switching case tests estimation-induced graph-code multiplicity; this quantity is not a calibrated false-discovery rate.",
        ),
        code_cell('''
validation_report = None
if RUN_VALIDATION:
    require_conventional_runtime(validation_methods)
    validation_report = run_framework_validation(
        validation_config, OUTPUT_DIRECTORY, max_cells=MAX_CELLS,
    )
elif (OUTPUT_DIRECTORY / "validation.json").is_file():
    validation_report = json.loads((OUTPUT_DIRECTORY / "validation.json").read_text())
else:
    print("No results exist. Inspect the profile, set RUN_VALIDATION=True and run this cell.")

if validation_report is not None:
    validation_statistics = analyze_framework_validation(
        validation_report, OUTPUT_DIRECTORY / "analysis",
    )
    print("Method runs available/attempted:", validation_report["available_methods"],
          validation_report["attempted_methods"])
    print("Publication matrix complete:", validation_report["publication_suite_complete"])
    display(Markdown((OUTPUT_DIRECTORY / "analysis/analysis-report.md").read_text()))
    display(Markdown((OUTPUT_DIRECTORY / "analysis/numeric-summary.md").read_text()))
'''),
        markdown_cell(
            "## Manuscript-ready artifacts and interpretation gates",
            "",
            "The analysis directory contains exact numeric tables, seed-bootstrap 95% intervals, exact sign tests with full planned-family Holm correction, vector figures, a statistical appendix and generated LaTeX. Small seed counts and unavailable runs remain visible. The manuscript protocol must distinguish the executed bounded experiment from the proposed larger study.",
            "Predicted transition probabilities and omission scores validate predictive operations, not biological necessity. The matched-marginal control concerns equal snapshot distributions; it does not imply equal history distributions. Independent-animal replication and intervention experiments remain separate gates.",
        ),
        code_cell('''
if validation_report is not None:
    from IPython.display import Image

    for filename in ("figure-01-predictive-gain.png", "figure-02-availability.png"):
        display(Image(filename=str(OUTPUT_DIRECTORY / "analysis/figures" / filename)))
    print("LaTeX table:", OUTPUT_DIRECTORY / "analysis/summary.tex")
    print("Executed-run text:", OUTPUT_DIRECTORY / "analysis/manuscript_results.tex")
    print("Statistical details:", OUTPUT_DIRECTORY / "analysis/stats-appendix.md")
'''),
        markdown_cell(
            "## See the controls, transition predictions and neuron/group contributions",
            "",
            "The visual bundle makes three different questions visible. Figure 03 shows the generating circuits, raw two-neuron occupancy and lagged dependence for each designed control. Its activity panels are raw projections, not an inferred manifold. Figure 04 shows a shared held-out manifold example beside observed and method-predicted transition matrices. Figure 05 shows transition-specific neuron/group omission scores, including negative scores.",
            "The worked comparison uses the first declared matched-marginal specification and seed 0; it is not selected for a favorable result. State codes are local to this one fitted example. Do not pool numeric state labels across seeds. The numeric tables retain the repeated-study means, sample standard deviations, seed counts and unavailable results.",
            "The same `visuals/` directory also contains compact manuscript versions of Figure 01 (four predictive-gain forest panels) and Figure 02 (available/attempted counts). Together the bundle provides five figures and two tables. The earlier cells display the original analysis versions of Figures 01–02.",
            "Set `RENDER_VISUALS=True` to regenerate these presentation artifacts from the frozen validation results. This does not execute the full validation matrix. Existing visuals are displayed when present. The visual provenance distinguishes the frozen experiment from its presentation code.",
        ),
        code_cell('''
VISUAL_DIRECTORY = OUTPUT_DIRECTORY / "visuals"
if RENDER_VISUALS:
    from synthetic.validation_figures import render_validation_visuals

    visual_manifest = render_validation_visuals(OUTPUT_DIRECTORY, VISUAL_DIRECTORY)
    print("Visual manifest:", visual_manifest)

if (VISUAL_DIRECTORY / "visual-provenance.json").is_file():
    from IPython.display import Image

    for filename in (
        "figure-03-designed-controls.png",
        "figure-04-shared-transitions.png",
        "figure-05-transition-attribution.png",
    ):
        path = VISUAL_DIRECTORY / filename
        if path.is_file():
            display(Image(filename=str(path)))
    for filename in ("table-01-predictive-summary.md", "table-02-control-design.md"):
        path = VISUAL_DIRECTORY / filename
        if path.is_file():
            display(Markdown(path.read_text()))
    display(Markdown((VISUAL_DIRECTORY / "figure-catalog.md").read_text()))
    print("Visual provenance:", VISUAL_DIRECTORY / "visual-provenance.json")
else:
    print("No visual bundle exists. After validation, set RENDER_VISUALS=True and run this cell.")
'''),
    ]
    write_notebook(Path(__file__).resolve().parent / "archive" / "04_framework_validation.ipynb", cells)


if __name__ == "__main__":
    main()
