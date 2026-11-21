"""Generate the numbered synthetic rollout and validation notebooks."""

import json
from pathlib import Path

from notebooks._build_celegans_rollout import BOOTSTRAP
from notebooks._build_notebooks import code_cell, markdown_cell

BOOTSTRAP_SUPPORT = (
    BOOTSTRAP.replace("from IPython.display import display\n\n", "")
    + """
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
PROJECT_ROOT = REPO
"""
)

RUN_SETTINGS = """
from pathlib import Path

ROOT = PROJECT_ROOT
PROFILE = "confirmation"  # Set "smoke" here and in the other run notebook for a short check.
RUN_ID = "support-01"

if PROFILE not in {"smoke", "confirmation"}:
    raise ValueError("PROFILE must be smoke or confirmation.")
if not RUN_ID or Path(RUN_ID).name != RUN_ID or RUN_ID in {".", ".."}:
    raise ValueError("Choose a single directory name for RUN_ID.")
RUN_ROOT = ROOT / "outputs" / "effectome-rollout-support" / PROFILE / RUN_ID
print("Run directory", RUN_ROOT)
"""


def write_notebook(path: Path, cells: list[dict]) -> None:
    """Write stable notebook cells and retain outputs for identical executed code."""
    cells = [cells[0], code_cell(BOOTSTRAP_SUPPORT), *cells[1:]]
    previous = json.loads(path.read_text()) if path.exists() else {}
    old_cells = {
        tuple(cell["source"]): cell for cell in previous.get("cells", []) if cell["cell_type"] == "code"
    }
    for index, cell in enumerate(cells):
        old = old_cells.get(tuple(cell["source"]))
        if old is not None and cell["cell_type"] == "code":
            cell.update({key: old[key] for key in ("outputs", "execution_count", "metadata")})
        cell["id"] = f"support-{index}"
    payload = {
        "cells": cells,
        "metadata": previous.get(
            "metadata",
            {
                "kernelspec": {
                    "display_name": "Python 3.14 (effectome .venv)",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.14"},
            },
        ),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parent
    write_notebook(
        root / "00_simulator_and_sanity.ipynb",
        [
            markdown_cell(
                "# 00 · Inspect the simulator and check the runtime",
                "",
                "The current benchmark forecasts two continuous behavioral coordinates from a known "
                "ten-neuron circuit. It holds the tuned selector fixed and compares learned, dense "
                "and shuffled support plus direct neural history. A separate preflight seed checks the "
                "planted changes, one native fastC-GC fit, and the isolated behavior encoder runtime.",
                "",
                "Install the application Python 3.14 environment with "
                "`uv sync --python 3.14 --locked --extra notebooks`. From the same repository run "
                "`uv sync --project tools/behavior-encoder --python 3.12 --locked` for encoder training. "
                "See the [experiment protocol](../docs/effectome_rollout_support.md).",
            ),
            code_cell("""
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from effectome.experiments.behavior_transition_controls import TRANSITION_SCENARIOS, make_transition_control
from effectome.experiments.behavior_transition_graphs import discover_transition_supports, trailing_histories
from effectome.manifold.behavior_dynamics import behavior_runtime_info
from effectome.workflows.cross_method import load_cross_method_profile
from effectome.workflows.resume import content_fingerprint
from synthetic.behavior_rollout_online import GENERATOR
from synthetic.effectome_rollout import READOUT_TRIALS

RUN_SANITY = True
PREFLIGHT_SEED = 90998
print("fastC-GC package", version("fastcgc"))
display(behavior_runtime_info())

conditions = {
    scenario: make_transition_control(GENERATOR, scenario=scenario, seed=PREFLIGHT_SEED)
    for scenario in TRANSITION_SCENARIOS
}
constant, nuisance = conditions["constant"], conditions["nuisance_switch"]
driver = conditions["driver_switch"]
assert all(data.neural.shape[-1] == 10 and data.behavior.shape[-1] == 2 for data in conditions.values())
np.testing.assert_array_equal(constant.mechanisms[0].support, constant.mechanisms[1].support)
assert not np.array_equal(nuisance.mechanisms[0].support, nuisance.mechanisms[1].support)
np.testing.assert_array_equal(constant.behavior, nuisance.behavior)
assert not np.array_equal(driver.behavior, constant.behavior)

fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
for column, (scenario, data) in enumerate(conditions.items()):
    for regime, model in enumerate(data.mechanisms):
        ax = axes[regime, column]
        ax.imshow(model.support, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{scenario.replace('_', ' ').capitalize()} · regime {regime}")
        ax.set(xlabel="Target neuron", ylabel="Source neuron")
plt.show()

if RUN_SANITY:
    _, methods = load_cross_method_profile("csl_sanity", config_root=PROJECT_ROOT / "conf")
    method = methods["fast_cgc"]
    first_training = slice(READOUT_TRIALS, READOUT_TRIALS + 1)
    histories, windows = trailing_histories(
        driver.neural[first_training], driver.origins[first_training], max(GENERATOR.history_lengths),
    )
    signature = content_fingerprint((PREFLIGHT_SEED, histories, windows, method))
    evidence = PROJECT_ROOT / "outputs/effectome-rollout-support/preflight" / f"native-{signature}.json"
    support = discover_transition_supports(histories, windows, method, GENERATOR.fps, evidence, signature)
    assert support.shape == (len(windows), 10, 10)
    display(pd.DataFrame({
        "training_window": np.arange(len(windows)), "estimated_edges": support.sum((1, 2)),
    }))
    print("Native evidence", evidence)
"""),
            markdown_cell(
                "The planted masks describe the simulator. The native fit checks the executable "
                "fastC-GC path on three training histories. Continue to notebook 01 for the comparison.",
            ),
        ],
    )

    write_notebook(
        root / "01_run_support_benchmark.ipynb",
        [
            markdown_cell(
                "# 01 · Run the frozen support benchmark",
                "",
                "Run All starts the 20-seed confirmation by default. Each seed runs driver-switch, "
                "constant and nuisance-switch cases. The learned, dense and shuffled arms share a frozen "
                "readout, origin grid and tuned selector. Direct history supplies a forecast baseline.",
                "",
                'For a short check set `PROFILE = "smoke"` in this notebook and notebook 02. '
                "Adjust `WORKERS` for the machine. Reuse the same run ID to resume a matching run.",
            ),
            code_cell(RUN_SETTINGS),
            code_cell("""
import os
import signal
import subprocess
import sys

RUN = True
WORKERS = 3

command = [
    sys.executable, "-u", "-m", "synthetic.behavior_rollout_support", "--profile", PROFILE,
    "--workers", str(WORKERS), "--output-root", str(RUN_ROOT),
]
if RUN:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with (RUN_ROOT / "run.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(
            [*command, "--run"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=(os.name == "posix"),
        ) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                if process.wait() != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
            except KeyboardInterrupt:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                raise
else:
    status = subprocess.check_output([*command, "--status"], cwd=ROOT, text=True)
    print(status)
"""),
            markdown_cell(
                "Watch `support-report.json` or `run.log` in the printed run directory for progress. "
                "Continue to notebook 02 once the report records `execution_complete: true`.",
            ),
        ],
    )

    write_notebook(
        root / "02_validate_and_visualize.ipynb",
        [
            markdown_cell(
                "# 02 · Validate results and compare disconnection effects",
                "",
                "Use the same profile and run ID as notebook 01. This stage checks case provenance, "
                "replays forecast selection and seed-level metrics, and regenerates the joint decisions "
                "from the completed run. It exports figures and a standalone interactive view.",
            ),
            code_cell(RUN_SETTINGS),
            code_cell("""
import os

import pandas as pd
from IPython.display import FileLink, Image, display

from synthetic.support_validation import validate_and_visualize

result = validate_and_visualize(RUN_ROOT)
display({key: result[key] for key in ("profile", "case_count", "seed_count", "support", "learned_selector")})
metrics = pd.read_csv(result["seed_metrics"])
columns = ["forecast_mae", "early_forecast_mae", "active_effect_mae"]
display(metrics.groupby(["scenario", "arm"])[columns].mean().round(5))
for title, files in result["figures"].items():
    print(title)
    display(Image(filename=files["png"]))
    display(FileLink(os.path.relpath(files["pdf"], Path.cwd())))
print("Open the paired intervention viewer in a browser", result["interactive_view"])
display(FileLink(os.path.relpath(result["interactive_view"], Path.cwd())))
"""),
            markdown_cell(
                "The browser view lets you choose a condition, seed, fitted arm, held-out origin, and "
                "saved neuron or group. It compares signed fitted effects with the simulator on both "
                "behavioral coordinates. A smoke run reports descriptive diagnostics on one seed.",
            ),
        ],
    )


if __name__ == "__main__":
    main()
