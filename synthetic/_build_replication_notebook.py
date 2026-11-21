"""Build the ready-to-run fresh-seed mechanism replication notebook."""

# ruff: noqa: E501 -- notebook prose and embedded cells are kept readable.

from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell, write_notebook


def main() -> None:
    cells = [
        markdown_cell(
            "# Fresh-seed mechanism replication and nomination calibration",
            "",
            "This is the next experiment after the completed synthetic collation. It tests the fully observed, model-matched mechanism on fresh process seeds, using separate null-calibration, positive-test and null-test pools. **Run All starts the complete replication by default**, with all four native CSL methods and the known/empty-support controls. It does not rerun the older latent-bistable benchmark or merge graph-similarity work.",
            "The primary discovery history is 1,024 samples. Its nomination threshold is the larger of the prespecified minimum probability-effect cutoff and an order statistic of separate calibration-null whole-seed maxima across candidate units and directions, then frozen before test evaluation. A 256-sample history is paired on the same positive-test seeds for the history sensitivity comparison; that secondary panel does not receive an independent calibrated-null claim.",
            "Read the [completed results and discussion](result_collation.pdf), [methodology rationale](../docs/driver_methodology_revision.md), and [verified citations](result_collation-citations.md). Completion, recovery, calibrated nomination and biological necessity remain separate outcomes. The pilot profile is a smaller execution check and cannot establish the full null-rate acceptance gate.",
        ),
        code_cell('''
import json
import subprocess
from dataclasses import asdict

import pandas as pd
from IPython.display import Markdown, display

from effectome.experiments.mechanism_replication_config import load_mechanism_replication_profile
from effectome.workflows.cross_method import load_cross_method_profile
from notebooks._shared import PROJECT_ROOT, require_conventional_runtime
from synthetic.mechanism_replication import replication_inventory

ROOT = PROJECT_ROOT
PROFILE = "replication"  # replication | pilot
RUN = True
WORKERS = 4
METHODS = None  # None = all four native methods; subsets are explicitly incomplete panels.
RUN_LABEL = "mechanism-replication-01"  # New label after changing protocol or implementation.

if not RUN_LABEL or Path(RUN_LABEL).name != RUN_LABEL or RUN_LABEL in {".", ".."}:
    raise ValueError("RUN_LABEL must be one nonempty directory name")
if type(WORKERS) is not int or WORKERS < 1:
    raise ValueError("WORKERS must be a positive integer")
config = load_mechanism_replication_profile(PROFILE)
_, native_methods = load_cross_method_profile("csl_sanity")
selected_names = tuple(native_methods if METHODS is None else dict.fromkeys(METHODS))
if not selected_names or set(selected_names) - set(native_methods):
    raise ValueError("METHODS must be a nonempty subset of the four native CSL methods")
selected_methods = {name: native_methods[name] for name in selected_names}
OUTPUT = ROOT / "outputs/driver-mechanism-replication" / PROFILE / RUN_LABEL
inventory = replication_inventory(config, selected_methods)
inventory["fitting_enabled"] = RUN
print("Declared configuration:", json.dumps(asdict(config), indent=2))
print("Inventory:", json.dumps(inventory, indent=2))
print("Output:", OUTPUT)
print("Fitting enabled:", RUN, "| workers:", WORKERS)
print("Calibration, positive-test and null-test seed pools are disjoint.")
print("Primary nominations use the calibrated cutoff, retaining the fixed effect-size floor.")
print("The full four-method replication has 119 cells, 714 lanes and 4,284 native window fits.")
print("Resume unchanged code/settings/label; keep first completed evidence without favorable reruns.")
'''),
        markdown_cell(
            "## Run with visible progress or inspect current evidence",
            "",
            "The runner freezes the study protocol, fits calibration-null cases first, freezes the maximum of each calibrated threshold and the prespecified effect-size floor, and only then evaluates the untouched test cases. Progress is streamed below and appended to `run.log`; `replication-report.json` records the latest completed work. Identical protocol reruns resume saved first-completed evidence. Use a new label after a code/protocol change, and keep failures visible.",
            "Both pinned runtimes are required for the full panel. Conventional CPU permutation randomness is part of the fixed estimator procedure and is not completely controlled by the requested NumPy seed. Saved native graphs and their provenance are retained; do not rerun a result until it improves.",
        ),
        code_cell('''
def run_logged(arguments, log_path):
    """Stream the runner and keep the same log outside notebook output."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Executing:", arguments, "\\nLog:", log_path, flush=True)
    with log_path.open("a") as log:
        process = subprocess.Popen(
            arguments, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdout.close()
    if return_code:
        raise RuntimeError(f"Runner exited with status {return_code}; inspect {log_path}")


if RUN:
    require_conventional_runtime(selected_methods, project_root=ROOT)
    command = [sys.executable, "-u", "-m", "synthetic.mechanism_replication", "--profile", PROFILE,
               "--workers", str(WORKERS), "--run", "--output-root", str(OUTPUT)]
    if METHODS is not None:
        for name in selected_names:
            command.extend(["--method", name])
    run_logged(command, OUTPUT / "run.log")

report_path = OUTPUT / "replication-report.json"
report = json.loads(report_path.read_text()) if report_path.is_file() else None
if report is None:
    print("No saved report for this label. The inventory above describes planned work, not completed fits.")
else:
    print("Saved evidence:", report_path)
    print("Execution complete:", report.get("execution_complete", False))
    print("Claim scope:", report.get("claim_scope", "See the saved protocol"))
    if report.get("progress") is not None:
        display(report["progress"])
    if report.get("summary"):
        display(pd.DataFrame.from_dict(report["summary"], orient="index"))
    analysis_path = OUTPUT / "analysis/analysis-report.md"
    if analysis_path.is_file():
        display(Markdown(analysis_path.read_text()))
'''),
        markdown_cell(
            "## Inspect progress from another terminal",
            "",
            "Read-only status does not start fits. From the repository root, use the same profile and exact output directory:",
            "```sh",
            ".venv/bin/python -m synthetic.mechanism_replication --status --profile replication --output-root outputs/driver-mechanism-replication/replication/mechanism-replication-01",
            "```",
            "You can also follow `run.log` while the notebook runs. An absent report means no completed checkpoint has yet been written; it is not a scientific result. The initial conventional import/JIT and native lanes can produce gaps between recording-level checkpoints.",
        ),
        code_cell('''
from synthetic.mechanism_replication import inspect_replication_status

status = inspect_replication_status(OUTPUT)
print("Status:", status["status"], "| report:", status["report"])
if "saved" in status:
    saved = status["saved"]
    print("Latest completion flag:", saved.get("execution_complete", False))
    if saved.get("progress") is not None:
        display(saved["progress"])
for name in ("calibration.json", "study-protocol.json"):
    path = OUTPUT / name
    if path.is_file():
        print("Recorded artifact:", path)
'''),
        markdown_cell(
            "## Interpret the independent-seed results",
            "",
            "Compare primary and short-history top-k recall and intervention-effect error at the process-seed level. Trials, directions, candidates and four related method variants do not multiply independent replication. Keep known support, empty support and unavailable lanes visible. Primary nominations use the frozen calibration-only threshold; the null-test whole-seed event counts any false candidate/group across supported directions. Small null-test counts can have zero observed failures and still fail the 5% exact-upper-bound gate.",
            "The full run tests 20 fresh positive seeds and 60 untouched null-test seeds after 19 separate calibration-null seeds. Success would establish a bounded result in a fully observed, model-matched control. It would not overwrite the failed 560-run latent benchmark, identify biological necessity, or validate the separate graph-similarity extension. All planned failures and short-history outcomes remain part of the report.",
        ),
    ]
    write_notebook(Path(__file__).resolve().parent / "archive" / "07_mechanism_replication.ipynb", cells)


if __name__ == "__main__":
    main()
