"""Build notebook 09 for the distributed behavior study."""

from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell, write_notebook


def main() -> None:
    cells = [
        markdown_cell(
            "# Behavior-supervised compression and continuous intervention effects",
            "",
            "Run All fits the five-seed development experiment independently. "
            "The fixed two-dimensional behavior mixes four observed neural coordinates, "
            "and the null preserves responsive nuisance dynamics without a path to behavior.",
            "",
            "Install the isolated encoder runtime from the repository root with "
            "`uv sync --project tools/behavior-encoder --python 3.12 --locked`. "
            "The full panel also uses the pinned conventional c-GC runtime.",
            "",
            "The [study plan](../docs/behavior_supervised_experiment_plan.md) documents the "
            "design, the BunDLe-inspired continuous encoder, and the citation audit. "
            "Causal simulation stays in the full observed neural state.",
        ),
        code_cell('''
import json
import subprocess

import pandas as pd
from IPython.display import display

from effectome.experiments.behavior_supervised import REPORT, behavior_supervised_inventory
from effectome.experiments.behavior_supervised_config import load_behavior_supervised_profile
from effectome.workflows.cross_method import load_cross_method_profile
from notebooks._shared import PROJECT_ROOT

ROOT = PROJECT_ROOT
PROFILE = "development"  # smoke, development, or confirmation
RUN = True
RUN_LABEL = "distributed-behavior-01"
WORKERS = 4
METHODS = None  # None selects all four native estimators.
ORACLE_ONLY = False
ENCODER_PYTHON = None  # Default: tools/behavior-encoder/.venv/bin/python
DEVELOPMENT_REPORT = (
    ROOT / "outputs/behavior-supervised/development/distributed-behavior-01" / REPORT
)

if not RUN_LABEL or Path(RUN_LABEL).name != RUN_LABEL or RUN_LABEL in {".", ".."}:
    raise ValueError("RUN_LABEL must be one directory name")
if type(WORKERS) is not int or WORKERS < 1:
    raise ValueError("WORKERS must be a positive integer")
if ORACLE_ONLY and METHODS is not None:
    raise ValueError("Choose ORACLE_ONLY or METHODS")
config = load_behavior_supervised_profile(PROFILE)
_, native = load_cross_method_profile("csl_sanity")
selected = {} if ORACLE_ONLY else (
    native if METHODS is None else {name: native[name] for name in METHODS}
)
OUTPUT = ROOT / "outputs/behavior-supervised" / PROFILE / RUN_LABEL
inventory = behavior_supervised_inventory(config, selected)
display({**inventory, "fitting_enabled": RUN, "output": str(OUTPUT)})
'''),
        markdown_cell(
            "### Fit and monitor",
            "",
            "Progress is streamed below and appended to `run.log`. The JSON report records completed cases. "
            "An unchanged study label resumes recorded checkpoints. The confirmation profile freezes "
            "independent null calibration before running the positive and null test pools.",
        ),
        code_cell('''
if RUN:
    command = [
        sys.executable, "-u", "-m", "synthetic.behavior_supervised", "--run",
        "--profile", PROFILE, "--workers", str(WORKERS), "--output-root", str(OUTPUT),
    ]
    if PROFILE == "confirmation":
        command.extend(["--development-report", str(DEVELOPMENT_REPORT)])
    if ENCODER_PYTHON is not None:
        command.extend(["--encoder-python", str(ENCODER_PYTHON)])
    if ORACLE_ONLY:
        command.append("--oracle-only")
    elif METHODS is not None:
        for name in selected:
            command.extend(["--method", name])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "run.log").open("a") as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
            if return_code:
                raise RuntimeError(f"Study exited with status {return_code}. See {OUTPUT / 'run.log'}.")

report_path = OUTPUT / REPORT
if report_path.is_file():
    report = json.loads(report_path.read_text())
    display({key: report[key] for key in (
        "stage", "completed_cells", "planned_cells", "available_method_runs",
        "attempted_method_runs", "nomination_calibrated", "scientific_validation_passed",
    )})
else:
    print("This label has no saved report.")
table = OUTPUT / "analysis/seed-metrics.csv"
if table.is_file():
    metrics = pd.read_csv(table, keep_default_na=False, na_values=[""])
    display(metrics.groupby(["phase", "scenario", "dynamics", "readout"])[
        ["end_to_end_mae", "positive_unit_mae", "within_chart_mae", "semantic_mae", "origin_mae"]
    ].mean())
    print("Analysis:", OUTPUT / "analysis/analysis-report.md")
'''),
        markdown_cell(
            "### Interpretation",
            "",
            "Behavioral error measures agreement with the generator-defined outcome. "
            "Within-chart error compares dynamics using the same frozen measurement. "
            "Semantic error isolates the measurement by using true neural futures. "
            "The positive-unit column prevents many inactive interventions from masking driver errors.",
            "",
            "Each seed is one independent unit. Development uses provisional magnitude cutoffs, "
            "and confirmation uses its separately calibrated cutoffs.",
        ),
    ]
    write_notebook(Path(__file__).resolve().parent / "09_behavior_supervised_validation.ipynb", cells)


if __name__ == "__main__":
    main()
