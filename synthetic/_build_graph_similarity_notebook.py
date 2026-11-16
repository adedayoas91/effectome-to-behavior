"""Build the explicitly executed graph-identity/topology comparison notebook."""

# ruff: noqa: E501 -- Markdown and code templates retain notebook layout.

from pathlib import Path

from notebooks._build_notebooks import code_cell, markdown_cell, write_notebook


def main() -> None:
    cells = [
        markdown_cell(
            "# Graph similarity: neuron-specific states and recurring topology",
            "",
            "This is an independently runnable, bounded structural diagnostic. It compares directed lag-aware edge Jaccard with a direction/lag-aware Weisfeiler–Lehman kernel. The first preserves neuron identity; the second compares recurring circuit topology under neuron relabelling. Grouping uses medoids fitted only on training graphs, then frozen for held-out assignment.",
            "**Run All inventories and reads saved results by default.** Set `RUN=True` below to run the small planted-graph comparison. No native CSL estimator is fitted here. Run notebook 06 to evaluate the separate mechanism-readout revision, including native discovery history. This structural test cannot recover missing neural edges or establish causal drivers.",
            "The rationale, exact graph/label contracts and references are in [the graph-similarity methodology](../docs/graph_similarity_methodology.md). Directed WL adapts the neighbourhood comparison of [Shervashidze et al. (2011)](https://jmlr.org/papers/v12/shervashidze11a.html); equal features are not proof of isomorphism.",
        ),
        code_cell('''
import json
import logging
from dataclasses import asdict

import pandas as pd
from IPython.display import Image, Markdown, display

from effectome.experiments.graph_similarity_validation import (
    graph_similarity_inventory,
    load_graph_similarity_profile,
    render_graph_similarity_report,
    run_graph_similarity_validation,
)
from notebooks._shared import PROJECT_ROOT

PROFILE = "smoke"  # smoke: three graph-sampling seeds | development: ten
RUN = False
RUN_LABEL = "smoke-01"  # Change after a code/protocol change.
if not RUN_LABEL or Path(RUN_LABEL).name != RUN_LABEL or RUN_LABEL in {".", ".."}:
    raise ValueError("RUN_LABEL must be one nonempty directory name")
config = load_graph_similarity_profile(PROFILE)
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs/graph-similarity" / PROFILE / RUN_LABEL
FROZEN_DIRECTORY = PROJECT_ROOT / "synthetic/results/graph_similarity_2026-11-16"
print("Configuration:", json.dumps(asdict(config), indent=2))
print("Inventory:", json.dumps(graph_similarity_inventory(config), indent=2))
print("Current run:", OUTPUT_DIRECTORY)
print("Default execution performs no fits.")
'''),
        markdown_cell(
            "## Equal density, different identity and architecture",
            "",
            "The four generating archetypes are chain A, a neuron-disjoint chain B, an outward star and an inward star. Every graph has six neurons and exactly two off-diagonal directed edges. Chain A/B share topology but involve different neurons. One-edge-replacement noise at fixed probability preserves density, with paired label ordering and candidate rewirings across noise settings. Training and held-out graphs are sampled separately.",
            "Fixed K=2,3,4 comparisons use labels only for evaluation: four neuron-specific archetypes versus three topology classes. K is not selected to maximize held-out adjusted Rand index. Clean WL has only three distinguishable classes; an unavailable K=4 fit is retained with its reason rather than supplied with fabricated extra states.",
        ),
        code_cell('''
report = None
report_path = OUTPUT_DIRECTORY / "graph-similarity.json"
evidence_kind = "new run" if RUN else "saved local run (not a new run)"
if RUN:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("effectome")
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    handler = logging.FileHandler(OUTPUT_DIRECTORY / "run.log")
    logger.addHandler(handler)
    try:
        report = run_graph_similarity_validation(config, OUTPUT_DIRECTORY)
        render_graph_similarity_report(report, OUTPUT_DIRECTORY / "analysis")
    finally:
        logger.removeHandler(handler)
        handler.close()
elif report_path.is_file():
    report = json.loads(report_path.read_text())
elif PROFILE == "smoke" and (FROZEN_DIRECTORY / "graph-similarity.json").is_file():
    report_path = FROZEN_DIRECTORY / "graph-similarity.json"
    evidence_kind = "versioned structural diagnostic (not a new run)"
    report = json.loads(report_path.read_text())

if report is None:
    print("No matching saved diagnostic. Inspect the declared workload, then set RUN=True.")
else:
    print("Evidence source:", evidence_kind, report_path)
    stored_config = report.get("protocol", {}).get("config")
    print("Saved configuration:", json.dumps(stored_config, indent=2))
    if evidence_kind != "versioned structural diagnostic (not a new run)":
        requested = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
        saved = json.dumps(stored_config, sort_keys=True, separators=(",", ":"))
        if saved != requested:
            raise ValueError(
                "Saved diagnostic configuration does not match requested profile/settings. "
                "Choose a different RUN_LABEL or explicitly run the requested configuration."
            )
    print("Execution complete:", report["execution_complete"])
    print("Available / attempted fits:", report["available_fits"], "/", report["attempted_fits"])
    print("Scope:", report["scope"])
    table = pd.DataFrame(report["rows"])
    display(table.drop(
        columns=["fit_indices", "training_assignments", "heldout_assignments"], errors="ignore"
    ))
    display(pd.DataFrame(report["probes"]))
    display(pd.DataFrame(report["characteristics"]))
'''),
        markdown_cell(
            "## Held-out grouping and known counterexamples",
            "",
            "Identity-sensitive and topology-sensitive assignments are scored against different targets. A topology grouping can intentionally merge the two chains while losing which neurons initiate them. Directed/lag labels remain important: inward and outward stars differ, and identical edges at different physical lags remain distinguishable. A six-cycle and two disjoint three-cycles provide an explicit non-isomorphic pair with identical WL features. Exact isomorphism checks that diagnostic; empty-graph agreement is not positive recovery evidence.",
            "Each black dot is a graph-sampling seed; bars are descriptive means. These are not independent neural recordings or biological replicates. Keep unavailable partitions and noisy failures visible. The benchmark supplies planted graph inputs and therefore does not measure uncertainty in native CSL edge estimation.",
        ),
        code_cell('''
if report is not None:
    analysis = report_path.parent / "analysis"
    for filename in ("figure-01-heldout-grouping.png", "figure-02-distance-probes.png"):
        path = analysis / filename
        if path.is_file():
            display(Image(filename=str(path)))
    if (analysis / "analysis-report.md").is_file():
        display(Markdown((analysis / "analysis-report.md").read_text()))
    print("Numeric rows, unavailable reasons, PDF/PNG figures:", analysis)
    print("Protocol and implementation fingerprint:", report["protocol_fingerprint"])
'''),
        markdown_cell(
            "## Candidate optimizer comparison",
            "",
            "The [e-means assessment](../docs/graph_similarity_methodology.md#fit-of-e-means-to-this-methodology) describes a possible future optimizer comparison for fixed vector representations. E-means is not implemented or evaluated by this notebook; no speedup or driver-recovery result is claimed for it. Direct Jaccard graph medoids remain a separate geometry from Euclidean centroid optimization.",
        ),
    ]
    write_notebook(Path(__file__).resolve().parent / "07_graph_similarity.ipynb", cells)


if __name__ == "__main__":
    main()
