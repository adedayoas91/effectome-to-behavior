"""Build the independent learned-endpoint development notebook without changing earlier runs."""

# ruff: noqa: E501 -- prose and embedded notebook cells.

from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell, write_notebook


def main() -> None:
    cells = [
        markdown_cell(
            "# Learned behavioral endpoint: frozen PCA bridge",
            "",
            "Run this notebook independently after notebook 07. **Run All fits the development panel by default:** five new process seeds, driver and endpoint-null scenarios, true/known/empty neural mechanisms and all four imported native support estimates. Each neural fit and its futures are reused across all seven frozen endpoint arms. It does not rerun or revise the completed replication.",
            "The behavioral reference is defined by the generator before manifold learning. We report behavioral effect error, error within the learned chart, and the chart's semantic discrepancy separately. Constant, source-proxy and neural-nuisance endpoints expose misleading chart agreement. This first bridge uses linear PCA; it does not validate a nonlinear manifold, hidden-state robustness or biological necessity.",
            "Read the [method plan and justifications](../docs/learned_manifold_validation_plan.md), [verified references](../docs/learned_manifold_validation_references.bib), and [citation scope audit](../docs/learned_manifold_validation_citation_audit.md). No confirmation or calibration seeds are run here.",
        ),
        code_cell('''
import json
import subprocess

import pandas as pd
from IPython.display import display

from effectome.experiments.manifold_validation import manifold_validation_inventory
from effectome.experiments.manifold_validation_config import load_manifold_validation_profile
from effectome.workflows.cross_method import load_cross_method_profile
from notebooks._shared import PROJECT_ROOT, require_conventional_runtime

ROOT = PROJECT_ROOT
PROFILE = "development"  # smoke = one seed; development = five seeds
RUN = True
WORKERS = 4
METHODS = None  # None requests all four native methods
ORACLE_ONLY = False
RUN_LABEL = "endpoint-bridge-03"  # 01 is the archived first development implementation.

if not RUN_LABEL or Path(RUN_LABEL).name != RUN_LABEL or RUN_LABEL in {".", ".."}:
    raise ValueError("RUN_LABEL must be one nonempty directory name")
if type(WORKERS) is not int or WORKERS < 1:
    raise ValueError("WORKERS must be a positive integer")
config = load_manifold_validation_profile(PROFILE)
_, native_methods = load_cross_method_profile("csl_sanity")
selected_names = tuple(native_methods if METHODS is None else dict.fromkeys(METHODS))
if not selected_names or set(selected_names) - set(native_methods):
    raise ValueError("METHODS must be a nonempty native-method subset")
selected_methods = {} if ORACLE_ONLY else {name: native_methods[name] for name in selected_names}
OUTPUT = ROOT / "outputs/learned-manifold-validation" / PROFILE / RUN_LABEL
print(json.dumps(manifold_validation_inventory(config, selected_methods), indent=2))
print("Fitting:", RUN, "| output:", OUTPUT)
print("Identical origins, paired futures and frozen endpoints; no held-out label alignment.")
'''),
        markdown_cell(
            "## Fit or inspect with saved progress",
            "",
            "The log below streams each neural-dynamics lane. Cell checkpoints retain first-completed native evidence, endpoint artifacts and per-rollout binary events; fingerprints reject changed code or settings under the same label. `manifold-validation.json` is updated after each completed cell. `--status` also counts partially saved cell files. A lane marked unavailable remains visible and is not silently replaced by a control.",
        ),
        code_cell('''
if RUN:
    require_conventional_runtime(selected_methods, project_root=ROOT)
    command = [sys.executable, "-u", "-m", "synthetic.manifold_validation", "--run",
               "--profile", PROFILE, "--workers", str(WORKERS), "--output-root", str(OUTPUT)]
    if ORACLE_ONLY:
        command.append("--oracle-only")
    elif METHODS is not None:
        for name in selected_names:
            command.extend(["--method", name])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "run.log").open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
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
            raise RuntimeError(f"Runner failed ({return_code}); inspect {OUTPUT / 'run.log'}")

report_path = OUTPUT / "manifold-validation.json"
report = json.loads(report_path.read_text()) if report_path.is_file() else None
if report is None:
    print("No saved evidence for this label; the inventory is planned work only.")
else:
    display({key: report[key] for key in (
        "execution_complete", "completed_cells", "planned_cells", "available_method_runs",
        "attempted_method_runs", "all_requested_lanes_available", "claim_scope",
    )})
metrics_path = OUTPUT / "analysis/seed-metrics.csv"
if metrics_path.is_file():
    metrics = pd.read_csv(metrics_path)
    display(metrics.groupby(["scenario", "method", "endpoint"])[
        ["end_to_end_mae", "within_chart_mae", "semantic_mae", "mean_factual_event_disagreement"]
    ].mean())
    print("Full diagnostic discussion:", OUTPUT / "analysis/analysis-report.md")
'''),
        markdown_cell(
            "## Interpret the endpoint, then the mechanism",
            "",
            "Low within-chart error only shows agreement about that chart. Require small behavioral error and inspect semantic event disagreements on both factual and intervened simulator futures. A constant chart has zero chart effect by construction; natural occupancy and independently defined behavior expose that failure. The empirical triangle bound uses the same origins and paired samples, including source-label mistakes, without filtering out difficult interventions.",
            "Each raw process seed is one replicate. Keep the two scenarios, endpoint arms and methods as paired diagnostic conditions. Training-envelope excursions are reported and never removed. The source is instantaneous and the target averages the last four future samples, so no temporal encoder is allowed to borrow future context. Actual nonlinear BunDLe/CEBRA validation, a distributed behavioral readout, and fresh calibration/confirmation remain subsequent experiments described in the plan.",
        ),
    ]
    write_notebook(Path(__file__).resolve().parent / "08_learned_manifold_validation.ipynb", cells)


if __name__ == "__main__":
    main()
