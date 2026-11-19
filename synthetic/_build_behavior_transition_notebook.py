"""Build notebook 10 for the dynamic-effectome forecasting experiment."""

from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell, write_notebook


def main() -> None:
    cells = [
        markdown_cell(
            "# Do dynamic effectomes anticipate behavioral change?",
            "",
            "Run All starts the independent development panel. Each forecast uses only a trailing history. "
            "A common two-dimensional behavior-only encoder supports a comparison of recent history, "
            "a pooled static graph, an adaptive dynamic graph and simulator-known dynamic couplings.",
            "",
            "The [experiment plan](../docs/behavior_supervised_transition_plan.md) documents the controls, "
            "analysis decisions and reference sources. Install the isolated encoder runtime with "
            "`uv sync --project tools/behavior-encoder --python 3.12 --locked` from the repository root.",
        ),
        code_cell('''
import json
import subprocess

import pandas as pd
from IPython.display import display

from notebooks._shared import PROJECT_ROOT
from synthetic.behavior_transition import REPORT

ROOT = PROJECT_ROOT
PROFILE = "development"  # smoke, development, confirmation
RUN = True
RUN_LABEL = "recurrent-coupling-01"
WORKERS = 3
METHODS = ["fast_cgc"]  # Optional: fast_cgc_star, conventional_cgc, conventional_cgc_star.
ENCODER_PYTHON = None
DEVELOPMENT_REPORT = (
    ROOT / "outputs/behavior-transition/development/recurrent-coupling-01" / REPORT
)

if not RUN_LABEL or Path(RUN_LABEL).name != RUN_LABEL or RUN_LABEL in {".", ".."}:
    raise ValueError("Choose one directory name for RUN_LABEL")
OUTPUT = ROOT / "outputs/behavior-transition" / PROFILE / RUN_LABEL
command = [
    sys.executable, "-u", "-m", "synthetic.behavior_transition", "--profile", PROFILE,
    "--workers", str(WORKERS), "--output-root", str(OUTPUT),
]
for method in METHODS:
    command.extend(["--method", method])
if not METHODS:
    command.append("--oracle-only")
if ENCODER_PYTHON:
    command.extend(["--encoder-python", ENCODER_PYTHON])
if PROFILE == "confirmation":
    command.extend(["--development-report", str(DEVELOPMENT_REPORT)])
subprocess.run(command, cwd=ROOT, check=True)
'''),
        markdown_cell(
            "## Fit and monitor",
            "",
            "The runner saves each case and streams progress to `run.log`. The same label resumes a matching "
            "source, runtime and protocol. Open `seed-metrics.csv` for paired forecast errors "
            "per process seed.",
        ),
        code_cell('''
if RUN:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "run.log").open("a") as log:
        with subprocess.Popen(
            [*command, "--run"], cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        ) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                result = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
            if result:
                raise RuntimeError(f"Study exited with status {result}. See {OUTPUT / 'run.log'}.")

report = OUTPUT / REPORT
if report.is_file():
    display(json.loads(report.read_text()))
metrics = OUTPUT / "seed-metrics.csv"
if metrics.is_file():
    results = pd.read_csv(metrics)
    columns = [
        "history_mae", "oracle_mae", "static_mae", "dynamic_mae", "gain_over_history", "gain_over_static",
        "oracle_rollout_mae", "static_rollout_mae", "dynamic_rollout_mae",
    ]
    display(results.groupby(["scenario", "history_samples", "method", "status"])[columns].mean())
'''),
        markdown_cell(
            "## Read the comparisons",
            "",
            "Paired gains are positive when the dynamic effectome reduces future behavioral error. "
            "The constant-coupling and nuisance-switch controls check whether any improvement is specific "
            "to coupling changes with behavioral consequences. Candidate effects and matched controls are "
            "in the per-case records under `cases/`. The forecast origin follows a full "
            "within-regime history.",
        ),
    ]
    write_notebook(Path(__file__).resolve().parent / "10_behavior_transition_experiment.ipynb", cells)


if __name__ == "__main__":
    main()
