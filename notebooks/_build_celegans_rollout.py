"""Generate the three notebooks for the within-worm motif rollout."""

from __future__ import annotations

import json
from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell

ROOT = Path(__file__).resolve().parent / "c_elegans"

BOOTSTRAP = """\
import sys
from pathlib import Path

from IPython.display import display

if sys.version_info[:2] != (3, 14):
    raise RuntimeError("Select the project Python 3.14 .venv kernel.")

working_directory = Path.cwd().resolve()
try:
    REPO = next(
        path
        for path in (working_directory, *working_directory.parents)
        if (path / "pyproject.toml").is_file() and (path / "src" / "effectome").is_dir()
    )
except StopIteration as error:
    raise RuntimeError("Open this notebook from inside the effectome application repository.") from error

project_src = str(REPO / "src")
if project_src not in sys.path:
    sys.path.insert(0, project_src)

print("Notebook kernel", sys.executable)
"""

RUN_SETTINGS = """\
RUN_ID = "motif-rollout-01"
OUTPUT_ROOT = REPO / "outputs" / "celegans-rollout" / RUN_ID
print("Run directory", OUTPUT_ROOT)
"""


def write_notebook(name: str, cells: list[dict]) -> None:
    """Write one clean notebook with stable cell identifiers."""
    payload = {
        "cells": [{"id": f"worm-rollout-{index}", **cell} for index, cell in enumerate(cells)],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.14 (effectome .venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.14"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    write_notebook(
        "00_data_and_timing.ipynb",
        [
            markdown_cell(
                "# 00 · Data and timing",
                "",
                "Start here. Check the pinned BunDLe-Net NoStim files and freeze the worm list,",
                "chronological splits, and forecast timing for this run.",
                "",
                "Select the project Python 3.14 `.venv` kernel. The default processes all five",
                "worms separately. Use `WORM_INDICES = (0,)` for a quick pilot.",
                "Set a new `RUN_ID` when changing the worm list or input settings, then use the same",
                "ID in notebooks 01 and 02.",
                "[Environment and data setup](../README.md#c-elegans-motif-rollout)",
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                RUN_SETTINGS
                + """\

# None uses data/c_elegans/bundle_net_raw in the application repository.
DATA_ROOT = None
WORM_INDICES = (0, 1, 2, 3, 4)
"""
            ),
            code_cell(
                """\
import pandas as pd

from effectome.workflows.celegans_rollout import preflight

plans = preflight(REPO, OUTPUT_ROOT, worm_indices=WORM_INDICES, data_root=DATA_ROOT)
overview_columns = ["recording_id", "neurons", "frames", "fps", "absent_training_motifs"]
display(pd.DataFrame(plans).reindex(columns=overview_columns))
for worm_plan in plans:
    display({
        "recording_id": worm_plan["recording_id"],
        "timing": worm_plan["timing"],
        "fictive_motif_counts": worm_plan["fictive_motif_counts"],
    })
print("Saved", OUTPUT_ROOT / "preflight.json")
"""
            ),
            markdown_cell(
                "The target is the supplied eight-class fictive motor motif annotation.",
                "The preflight resolves timing from each worm's observed sampling rate.",
                "Continue to [01 · Fit and forecast](01_fit_and_forecast.ipynb).",
            ),
        ],
    )

    write_notebook(
        "01_fit_and_forecast.ipynb",
        [
            markdown_cell(
                "# 01 · Fit and forecast",
                "",
                "Use the run prepared by [00 · Data and timing](00_data_and_timing.ipynb).",
                "The early block fits the motif probability readout. The next block selects",
                "fastC-GC support and fits the pooled dynamics. Forecasts use frozen support and",
                "trailing local refits in the held-out block.",
                "",
                "The selector switches between paired pooled and local forecasts using past",
                "forecasts whose targets have already been observed. Each worm keeps its own fit.",
            ),
            code_cell(BOOTSTRAP),
            code_cell(RUN_SETTINGS),
            code_cell(
                """\
import pandas as pd

from effectome.workflows.celegans_rollout import run_forecasts

display(pd.DataFrame(run_forecasts(REPO, OUTPUT_ROOT)))
"""
            ),
            markdown_cell("Continue to [02 · Score and disconnections](02_score_and_disconnections.ipynb)."),
        ],
    )

    write_notebook(
        "02_score_and_disconnections.ipynb",
        [
            markdown_cell(
                "# 02 · Score forecasts and inspect disconnections",
                "",
                "Use the same run ID as [01 · Fit and forecast](01_fit_and_forecast.ipynb).",
                "Compare held-out multiclass Brier scores for the selector, pooled and local",
                "fits, decoded persistence, and training prevalence. Lower scores are better.",
                "",
                "Leave `DISCONNECT_ROIS` empty for forecast evaluation. To inspect a paired",
                "model-based disconnection, provide a tuple of original recording-local ROI IDs.",
                "For ROI-specific disconnections, use a run for the worm of interest.",
                "The reported shift describes motif probabilities under the fitted model.",
            ),
            code_cell(BOOTSTRAP),
            code_cell(
                RUN_SETTINGS
                + """\

DISCONNECT_ROIS = ()
"""
            ),
            code_cell(
                """\
import matplotlib.pyplot as plt
import pandas as pd

from effectome.workflows.celegans_rollout import summarize

evaluations = summarize(REPO, OUTPUT_ROOT, disconnect_rois=DISCONNECT_ROIS)
brier = pd.DataFrame(
    {entry["recording_id"]: entry["brier"] for entry in evaluations}
).T
display(brier.rename_axis("Recording"))
axes = brier.plot.bar(figsize=(10, 4), rot=15)
axes.set(xlabel="Recording", ylabel="Held-out Brier score (lower is better)")
axes.legend(title="Forecast", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
print("Saved", OUTPUT_ROOT / "summary.json")
for evaluation in evaluations:
    display({key: value for key, value in evaluation.items() if key != "brier"})
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()
