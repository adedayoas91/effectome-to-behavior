"""Build the run-enabled intervention-defined driver-recovery notebook."""

# ruff: noqa: E501 -- Markdown paragraphs remain intact in notebook templates.

from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell, write_notebook


def main() -> None:
    cells = [
        markdown_cell(
            "# Driver recovery: simulator interventions and acceptance criteria",
            "",
            "The earlier circuit/activity controls show that the framework can make its intended comparisons. This notebook asks the additional question: do its transition-specific neuron/group scores recover contributors verified by interventions in the generating process?",
            "The five conditions are single driver, redundant group, correlated bystander, no-driver null, and hidden-confounder challenge. Truth and replay states stay in recording metadata; inference receives observed traces. The hidden-confounder challenge is diagnostic and cannot certify causal identification.",
            "The frozen confirmation is complete: all 560 runs were available and all four methods failed the recovery gates. Its versioned evidence remains preserved; notebook 06 evaluates the separate mechanism revision.",
            "Run All executes the full 560-method-run confirmatory-profile reproduction by default, using four workers and a separate resumable run-label directory. This reruns the existing seeds and is not fresh independent confirmation. Set `RUN_DRIVER_VALIDATION=False` for saved-evidence inspection. Read `../docs/driver_ground_truth_protocol.md` before changing settings.",
            "The current protocol uses four imported CSL methods and one fixed origin per trial. A 256-sample history ends at303 and a strictly future target ends at335 (3.2seconds). The training-owned one-dimensional chart and held-out semantic audit were developed on seeds10/11. The frozen confirmation used seeds300–319/400–459. Thresholds are unchanged; the hidden-confounder condition is diagnostic.",
        ),
        code_cell('''
import json
from dataclasses import asdict

import pandas as pd
from IPython.display import Markdown, display

from effectome.experiments.driver_study import run_driver_study
from effectome.experiments.driver_validation import (
    driver_validation_cells,
    load_driver_validation_profile,
)
from effectome.experiments.driver_validation_reporting import render_driver_validation_report
from effectome.workflows.cross_method import load_cross_method_profile
from notebooks._shared import PROJECT_ROOT, require_conventional_runtime

PROFILE = "confirmatory"  # smoke | development | confirmatory
WORKERS = 4
RUN_DRIVER_VALIDATION = True  # Run All executes the full declared 560-method-run profile.
RUN_LABEL = "faster-machine-03"  # Reuse unchanged settings to resume; change after protocol/code edits.
RENDER_REPORT = False  # Recreate presentation artifacts from an existing report.
MAX_CELLS = None  # A cap records a partial run; it cannot satisfy confirmation.
validation_config = load_driver_validation_profile(PROFILE)
validation_cells = driver_validation_cells(validation_config)
_, validation_methods = load_cross_method_profile(validation_config.estimator_profile)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "outputs" / "driver-validation"
    / validation_config.estimator_profile / "scheduled-v2" / PROFILE / "reruns" / RUN_LABEL
)
ANALYSIS_DIRECTORY = OUTPUT_DIRECTORY / "analysis"
print("Resolved validation configuration:")
print(json.dumps(asdict(validation_config), indent=2, default=str))
print("Planned simulation cells:", len(validation_cells))
print("Planned method runs:", len(validation_methods) * len(validation_cells))
print("Methods:", tuple(validation_methods))
print("Fitting enabled:", RUN_DRIVER_VALIDATION, "| workers:", WORKERS)
print("Outputs:", OUTPUT_DIRECTORY)
display(pd.DataFrame(validation_cells).head(15))
'''),
        markdown_cell(
            "## Run a declared pilot or inspect the saved evidence",
            "",
            "The factual and disconnected futures are generated from the same origin with paired random numbers. The intervention disables a neuron's or group's outgoing influence in the simulator; it is not predictor-column deletion or measured-readout censoring. Its probability reduction supplies the recovery target at the same physical forecast horizon.",
            "One training-owned PCA chart, event table and activity/history baseline serve all methods. A training-only assignment aligns geometric region labels with semantic simulator wells; held-out agreement is audited, and poor alignment blocks a recovery claim. Numeric state codes are never pooled across seeds.",
        ),
        code_cell('''
driver_report = None
report_path = OUTPUT_DIRECTORY / "driver-validation.json"
evidence_directory = OUTPUT_DIRECTORY
frozen_directory = PROJECT_ROOT / "synthetic/results/driver_confirmation_v2"
if RUN_DRIVER_VALIDATION:
    require_conventional_runtime(validation_methods)
    driver_report = run_driver_study(
        validation_config, OUTPUT_DIRECTORY, workers=WORKERS, max_cells=MAX_CELLS,
    )
elif report_path.is_file():
    driver_report = json.loads(report_path.read_text())
elif PROFILE == "confirmatory" and (frozen_directory / "driver-validation.json").is_file():
    evidence_directory = frozen_directory
    driver_report = json.loads((frozen_directory / "driver-validation.json").read_text())
    print("Displaying frozen completed confirmation, not a new run:", evidence_directory)
else:
    print("No saved driver-recovery run exists. Inspect the protocol, then enable "
          "RUN_DRIVER_VALIDATION for a bounded pilot.")

if driver_report is not None:
    print("Evidence source:", evidence_directory)
    print("Execution complete:", driver_report["execution_complete"])
    print("Scientific validation passed:", driver_report["scientific_validation_passed"])
    print("Acceptance decisions:")
    print(json.dumps(driver_report["acceptance"], indent=2, default=str))
    rows = pd.DataFrame(driver_report["seed_rows"])
    columns = [name for name in (
        "scenario", "seed", "method", "source_state", "destination_state", "status",
        "n_origins", "n_events", "alignment_accuracy", "log_loss_gain", "reason",
    ) if name in rows.columns]
    display(rows[columns])
'''),
        markdown_cell(
            "## Compare simulator effects with predictive omission scores",
            "",
            "The intervention-truth figure shows actual simulator probability reductions, while the omission figure shows how refitting without a graph-feature block changes held-out predictive loss. Positive predictive contribution does not by itself establish a simulator effect or biological necessity. Negative values, control groups, sparse event counts and unavailable methods remain visible.",
            "A small pilot remains inconclusive. Proposed confirmation requires at least 20 independent positive-control seeds and 60 independent null seeds, adequate direction-specific support, and the configured recovery, availability and false-nomination gates. The independent unit is a raw-process simulation seed, not neurons, directions, overlapping windows, or methods. All numeric gates and their scope are versioned in the acceptance configuration.",
        ),
        code_cell('''
from IPython.display import Image

if driver_report is not None and (RUN_DRIVER_VALIDATION or RENDER_REPORT):
    render_driver_validation_report(driver_report, ANALYSIS_DIRECTORY)

evidence_analysis = evidence_directory / "analysis"
for filename in ("analysis-report.md", "numeric-summary.md", "recovery-summary.md", "stats-appendix.md"):
    path = evidence_analysis / filename
    if path.is_file():
        display(Markdown(path.read_text()))

for filename in (
    "figure-01-intervention-truth.png", "figure-02-predictive-omissions.png",
    "figure-03-recovery-and-null.png",
):
    path = evidence_analysis / filename
    if path.is_file():
        display(Image(filename=str(path)))

acceptance_path = evidence_analysis / "acceptance.json"
if acceptance_path.is_file():
    print("Saved acceptance artifact:", acceptance_path)
print("Protocol:", PROJECT_ROOT / "docs" / "driver_ground_truth_protocol.md")
visual_notes = PROJECT_ROOT / "synthetic" / "driver_visual_notes.md"
if visual_notes.is_file():
    display(Markdown(visual_notes.read_text()))
'''),
    ]
    write_notebook(Path(__file__).resolve().parent / "archive" / "05_driver_recovery.ipynb", cells)


if __name__ == "__main__":
    main()
