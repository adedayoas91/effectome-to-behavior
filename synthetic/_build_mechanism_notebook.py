"""Build the mechanism notebook with smoke and paired-history fitting enabled."""

# ruff: noqa: E501 -- notebook prose and embedded cells are kept readable.

from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell, write_notebook


def main() -> None:
    cells = [
        markdown_cell(
            "# Mechanism-based driver recovery: rerun and compare",
            "",
            "The completed four-CSL latent-bistable study produced 560/560 available runs and failed all frozen recovery gates. This notebook evaluates the separate fully observed, model-matched mechanism revision. It does not replace the old benchmark or establish biological necessity.",
            "Run this notebook independently in the project Python 3.14 kernel. **Run All fits the smoke panel and the paired 256/1024-history comparison by default**, using all four native methods plus the known/empty-support controls. `PROFILE='development', ORACLE_ONLY=True` selects the three-seed readout diagnostic instead of the regular smoke panel. The paired-history switch is independent; disable it if you only want that oracle diagnostic. Keep `METHODS=None` for complete native panels. Set both run switches to `False` for inspection only.",
            "The cited rationale, assumptions and limits are in [the methodology revision](../docs/driver_methodology_revision.md), [citation audit](../docs/driver_methodology_citation_audit.md), and [measured findings](../docs/mechanism_development_findings.md). These are development experiments; fresh confirmation and nomination calibration remain outstanding.",
        ),
        code_cell('''
import json
import math
import subprocess
from dataclasses import asdict

import pandas as pd
from IPython.display import Markdown, display

from effectome.experiments.mechanism_validation_config import load_mechanism_validation_profile
from effectome.workflows.cross_method import load_cross_method_profile
from notebooks._shared import PROJECT_ROOT, require_conventional_runtime

ROOT = PROJECT_ROOT
PROFILE = "smoke"  # smoke | development
RUN = True
ORACLE_ONLY = False  # True tests learned readout without native graph discovery.
METHODS = None  # None = all four; e.g. ["fast_cgc"] is explicitly a subset.
RUN_HISTORY_COMPARISON = True  # Paired 256/1024 histories; independent of RUN/PROFILE.
RUN_LABEL = "faster-machine-03"  # Change for a new protocol/code version; same label resumes.

if not RUN_LABEL or Path(RUN_LABEL).name != RUN_LABEL or RUN_LABEL in {".", ".."}:
    raise ValueError("RUN_LABEL must be one nonempty directory name")
if ORACLE_ONLY and METHODS is not None:
    raise ValueError("METHODS applies to native discovery; use METHODS=None with ORACLE_ONLY=True")
config = load_mechanism_validation_profile(PROFILE)
_, native_methods = load_cross_method_profile("csl_sanity")
selected_names = tuple(native_methods if METHODS is None else dict.fromkeys(METHODS))
if not selected_names or set(selected_names) - set(native_methods):
    raise ValueError("METHODS must be a nonempty subset of the four native CSL methods")
lanes = ("known_support", "empty_support", *(() if ORACLE_ONLY else selected_names))
panel = "oracle-only" if ORACLE_ONLY else "-".join(selected_names)
OUTPUT_ROOT = ROOT / "outputs/driver-mechanism-validation/reruns" / RUN_LABEL
OUTPUT = OUTPUT_ROOT / f"{PROFILE}-{panel}"
HISTORY_OUTPUT = OUTPUT_ROOT / "history-comparison"
FROZEN = ROOT / "synthetic/results/mechanism_development_2026-11-15"
fit_trials = int(config.generator.n_trials * (1 - config.evaluation_fraction))
discovery_trials = fit_trials - max(1, math.ceil(config.fit.validation_fraction * fit_trials))
print("Requested configuration:", json.dumps(asdict(config), indent=2))
print("Lanes:", lanes)
print("Planned method runs:", 2 * len(config.seeds) * len(lanes))
print("Native history fits:",
      0 if ORACLE_ONLY else 2 * len(config.seeds) * len(selected_names) * discovery_trials)
print("Current-run output:", OUTPUT)
print("Paired-history output:", HISTORY_OUTPUT)
print("Versioned historical evidence:", FROZEN)
print("Fitting enabled:", RUN, "| paired-history fitting enabled:", RUN_HISTORY_COMPARISON)
print("Resume with unchanged code/settings/label. Use a new label after code/settings change.")
'''),
        markdown_cell(
            "## Run with visible progress, or inspect saved evidence",
            "",
            "The existing CLI runners emit scenario, seed and lane starts. Output is streamed below and appended to `run.log`. Per-lane results are atomically saved in each `cells/` directory; `mechanism-validation.json` updates after each recording. A slow native lane may take time between updates. Re-execution with identical code and settings reuses completed lanes; an incompatible frozen protocol raises an error instead of silently mixing runs.",
            "Native inference requires both the pinned main environment and `tools/conventional-cgc/.venv/bin/python`. The preflight checks this path before a run. Conventional upstream CPU permutation randomness is not completely seeded; compare actual outputs and retain their native evidence. The three-seed oracle-only run does not require the conventional runtime.",
        ),
        code_cell('''
def run_logged(arguments, log_path):
    """Stream an existing runner to this notebook and retain the same progress log."""
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


def require_native_runtime(names):
    require_conventional_runtime({name: native_methods[name] for name in names}, project_root=ROOT)


if RUN:
    if not ORACLE_ONLY:
        require_native_runtime(selected_names)
    command = [sys.executable, "-u", "pipeline/15_mechanism_validation.py", "--profile", PROFILE,
               "--run", "--output-dir", str(OUTPUT)]
    if ORACLE_ONLY:
        command.append("--oracle-only")
    elif METHODS is not None:
        for name in selected_names:
            command.extend(["--method", name])
    run_logged(command, OUTPUT / "run.log")

report = None
report_path = OUTPUT / "mechanism-validation.json"
evidence_kind = "current rerun"
if not report_path.is_file() and not RUN and METHODS is None:
    archived_panel = "smoke" if PROFILE == "smoke" and not ORACLE_ONLY else (
        "known-support-three-seed" if PROFILE == "development" and ORACLE_ONLY else None
    )
    if archived_panel is not None:
        report_path = FROZEN / archived_panel / "mechanism-validation.json"
        evidence_kind = "versioned historical result (not a new run)"
if report_path.is_file():
    report = json.loads(report_path.read_text())
    print("Evidence source:", evidence_kind, report_path)
    print("Saved configuration:", report["protocol"]["config"])
    print("Execution complete:", report["execution_complete"])
    print("Available / attempted:", report["available_method_runs"], "/", report["attempted_method_runs"])
    print("Claim scope:", report["claim_scope"])
    display(pd.DataFrame.from_dict(report["summary"], orient="index"))
    analysis = report_path.parent / "analysis/analysis-report.md"
    if analysis.is_file():
        display(Markdown(analysis.read_text()))
else:
    print("No matching saved evidence. Inspect the inventory, then set RUN=True to execute.")
'''),
        markdown_cell(
            "## Inspect direction-level effects and unavailable lanes",
            "",
            "All structural state variables are observed. The final coordinate and zero threshold define the endpoint. Disconnection removes outgoing cross-neuron terms while retaining self-history and source evolution. Native discovery sees inner-training histories; inner-validation trials select ridge/noise settings; final mechanism fits use all outer training trials. Held-out future observations enter evaluation only. Known support is an explicit oracle control, never an estimated-support input.",
        ),
        code_cell('''
if report is not None:
    rows = []
    for case in report["cells"]:
        for method, result in case["methods"].items():
            row = {"scenario": case["scenario"], "seed": case["seed"], "method": method}
            if result["status"] != "available":
                rows.append({**row, "status": result["status"], "reason": result["reason"]})
            else:
                rows.extend({**row, **direction} for direction in result["metrics"]["directions"])
    display(pd.DataFrame(rows))
'''),
        markdown_cell(
            "## Paired history-length diagnostic: same recordings, different discovery context",
            "",
            "Set `RUN_HISTORY_COMPARISON=True` in the parameter cell to execute both histories using the existing diagnostic runner. This remains independent of `RUN`, `PROFILE` and `ORACLE_ONLY`. `METHODS=None` includes all four native methods. Each history has 12 method runs and 72 native history fits in the complete panel. The same longer recording arrays, mechanism training data, held-out origins, endpoint, horizon, alpha, beta and quorum are retained. More samples can improve discovery of the fixed mechanism; they would mix regimes if the mechanism changed within a window.",
            "Both the short-history failure and longer-history operating case remain visible. This is one paired development seed, not two independent confirmations. A single null seed has an exact 95% upper false-nomination bound of 0.95 even when no false nominees are observed.",
        ),
        code_cell('''
if RUN_HISTORY_COMPARISON:
    require_native_runtime(selected_names)
    command = [sys.executable, "-u", "-m", "synthetic.mechanism_history_diagnostic", "--run",
               "--output-root", str(HISTORY_OUTPUT)]
    if METHODS is not None:
        for name in selected_names:
            command.extend(["--method", name])
    run_logged(command, HISTORY_OUTPUT / "run.log")

comparison_path = HISTORY_OUTPUT / "history-comparison.json"
comparison_kind = "current rerun"
if not comparison_path.is_file() and not RUN_HISTORY_COMPARISON and METHODS is None:
    comparison_path = FROZEN / "history-comparison.json"
    comparison_kind = "versioned historical result (not a new run)"
if comparison_path.is_file():
    comparison = json.loads(comparison_path.read_text())
    print("Evidence source:", comparison_kind, comparison_path)
    print("Identical paired inputs verified:", comparison["paired_inputs_verified"])
    rows = []
    for setting in comparison["runs"]:
        for lane, summary in setting["summary"].items():
            rows.append({"history": setting["history_length"], "lane": lane, **summary})
    display(pd.DataFrame(rows))
else:
    print("No saved paired comparison. Set RUN_HISTORY_COMPARISON=True to execute.")
'''),
        markdown_cell(
            "## Interpret the comparison",
            "",
            "Probability-effect error, factual prediction, driver ranking and false nomination answer different questions. Top-k handles ties without favoring low-numbered neurons. The nomination threshold is provisional, not the old omission-nat gate or a calibrated significance test. Keep the frozen 560-run failure, the new 256-sample failure and the limited 1024-sample positive case together; the old and new studies have different processes and endpoints, so their scores are not a matched treatment comparison. Redundancy, observation stresses, mechanism misspecification, independent calibration and fresh confirmation remain future work.",
        ),
    ]
    write_notebook(Path(__file__).resolve().parent / "archive" / "06_mechanism_driver_validation.ipynb", cells)


if __name__ == "__main__":
    main()
