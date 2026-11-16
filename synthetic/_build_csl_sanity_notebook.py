"""Build an inspectable upstream-method sanity study without rerunning discovery."""

from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell, write_notebook


def main() -> None:
    cells = [
        markdown_cell(
            "# Four imported CSL methods: graph-recovery sanity check",
            "",
            "Run this prerequisite before the larger transition studies. Conventional c-GC, c-GC*, "
            "fastC-GC and fastC-GC* receive identical four-neuron recordings. A directed chain has "
            "links 0→1 and 1→2 (the second is negative); neuron 3 is isolated. The null has no links. "
            "Truth is used only for scoring. Native outputs become unsigned support, "
            "without a fitted VAR readout.",
            "The linear lagged data generator supplies known circuit truth; it is not a fitted VAR method. "
            "The four estimators share one family, and empty-graph agreement is uninformative. "
            "Successful graph recovery remains separate from transition-driver validation.",
            "Install the official conventional package using `../tools/conventional-cgc/README.md`. "
            "This notebook uses the main project kernel; its adapter invokes the pinned CPU environment.",
        ),
        code_cell('''
import json
from dataclasses import asdict

import pandas as pd
from IPython.display import Image, Markdown, display

from effectome.experiments.csl_sanity import CSLSanityConfig, run_csl_sanity
from effectome.experiments.csl_sanity_reporting import render_csl_sanity_report
from effectome.workflows.cross_method import load_cross_method_profile
from notebooks._shared import PROJECT_ROOT

RUN_SANITY = False  # Read the saved realization unless explicitly rerunning.
config = CSLSanityConfig(seeds=(0, 1, 2), n_samples=1024)
_, methods = load_cross_method_profile("csl_sanity")
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "csl-sanity" / "smoke"
FROZEN_DIRECTORY = PROJECT_ROOT / "synthetic" / "results" / "four_csl_sanity_2026-11-15"
print(asdict(config))
print("Imported method panel:", list(methods))
print("Planned fits:", 2 * len(config.seeds) * len(methods))
print("Conventional CPU permutation seeds are not fully controlled upstream.")
'''),
        markdown_cell(
            "## Recovery and nontrivial agreement",
            "",
            "All independent data seeds enter the summary. The first declared seed illustrates "
            "the recovered circuit. Compare truth, recall and false edges before interpreting "
            "agreement; methods can agree on mistakes. No discovery thresholds are selected "
            "using these results.",
        ),
        code_cell('''
report = None
evidence_directory = OUTPUT_DIRECTORY
path = OUTPUT_DIRECTORY / "sanity.json"
if RUN_SANITY:
    report = run_csl_sanity(config, OUTPUT_DIRECTORY, methods=methods)
    render_csl_sanity_report(report, OUTPUT_DIRECTORY / "analysis")
elif path.is_file():
    report = json.loads(path.read_text())
elif (FROZEN_DIRECTORY / "sanity.json").is_file():
    evidence_directory = FROZEN_DIRECTORY
    report = json.loads((FROZEN_DIRECTORY / "sanity.json").read_text())

if report is not None:
    print("Saved configuration:", report["config"])
    if report["config"] != json.loads(json.dumps(asdict(config))):
        print("Saved results use a different configuration; enable RUN_SANITY to fit the requested one.")
    print("Evidence:", evidence_directory)
    print("Available / attempted:", report["available_runs"], "/", report["attempted_runs"])
    display(pd.DataFrame(report["seed_rows"]).drop(columns="support", errors="ignore"))
    display(pd.DataFrame(report["pairwise_rows"]))
else:
    print("No saved study. Inspect the method settings, then enable RUN_SANITY.")
'''),
        code_cell('''
for name in ("analysis-report.md", "stats-appendix.md", "figure-catalog.md"):
    path = evidence_directory / "analysis" / name
    if path.is_file():
        display(Markdown(path.read_text()))
for name in ("figure-01-native-support.png", "figure-02-seed-recovery.png"):
    path = evidence_directory / "analysis" / "figures" / name
    if path.is_file():
        display(Image(filename=str(path)))

if report is not None:
    for method, result in report["diagnostics"].items():
        print(method, result["status"])
        if result["status"] != "available":
            print(result["reason"])
print("Next: 02 shared transitions; 04 framework validation; 05 driver recovery; 06 mechanism revision.")
'''),
    ]
    write_notebook(Path(__file__).resolve().parent / "01_four_csl_sanity.ipynb", cells)


if __name__ == "__main__":
    main()
