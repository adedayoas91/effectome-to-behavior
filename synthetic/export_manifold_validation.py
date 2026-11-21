"""Archive a completed learned-endpoint diagnostic without rerunning any fits."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from effectome.experiments.manifold_validation import render_manifold_validation_report
from effectome.workflows.resume import implementation_fingerprint


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_run(run_root: Path, output: Path) -> Path:
    """Freeze reports, binary events, native evidence, source and diagnostic figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    repository = Path(__file__).resolve().parents[1]
    report_path = run_root / "manifold-validation.json"
    report = json.loads(report_path.read_text())
    if not report["execution_complete"] or not report["all_requested_lanes_available"]:
        raise ValueError("archive requires a complete diagnostic with every requested lane available")
    n_seeds = len(report["development_seeds"])
    if report["protocol"]["implementation_fingerprint"] != implementation_fingerprint(repository):
        raise ValueError("archive executed source before changing implementation; source fingerprint differs")
    if output.exists() and any(output.iterdir()):
        raise ValueError("curated output already exists; preserve it and choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(report_path, output / "manifold-validation.json")
    shutil.copyfile(run_root / "study-protocol.json", output / "study-protocol.json")
    render_manifold_validation_report(report, output / "analysis")
    with zipfile.ZipFile(output / "run-evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_root.rglob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                archive.write(path, path.relative_to(run_root))
    source = [*repository.glob("src/**/*.py"), *repository.glob("pipeline/**/*.py"),
              *repository.glob("conf/**/*.yaml"), *repository.glob("tests/test_*manifold*.py"),
              repository / "tests/test_learned_endpoints.py", repository / "notebooks/_shared.py",
              repository / "notebooks/_build_notebooks.py", repository / "pyproject.toml",
              repository / "uv.lock", repository / "synthetic/manifold_validation.py",
              repository / "synthetic/_build_manifold_notebook.py", Path(__file__).resolve(),
              repository / "synthetic/archive/08_learned_manifold_validation.ipynb"]
    with zipfile.ZipFile(output / "implementation-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(source)):
            archive.write(path, path.relative_to(repository))
    for name in ("learned_manifold_validation_plan.md", "learned_manifold_validation_references.bib",
                 "learned_manifold_validation_citation_audit.md"):
        shutil.copyfile(repository / "docs" / name, output / name)
    arms = ("behavior_reference", "pca_geometry", "pca_behavior", "full_neural_behavior",
            "constant", "proxy", "nuisance")
    colors = ("#234e70", "#368f8b", "#b96d40", "#888888", "#af4248", "#8265a7", "#4d7963")
    figures = output / "analysis/figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = []
    for cell in report["cells"]:
        for method, result in cell["methods"].items():
            if result["status"] == "available":
                for arm, values in result["metrics"].items():
                    data.append({"scenario": cell["scenario"], "seed": cell["seed"], "method": method,
                                 "endpoint": arm, **values})
    for metric, name, title, methods in (
        ("semantic_mae", "01-endpoint-semantic-error", "Endpoint semantic effect error", ("oracle",)),
        ("end_to_end_mae", "02-behavioral-effect-error", "Behavioral intervention-effect error",
         ("known_support", "conventional_cgc", "conventional_cgc_star", "fast_cgc", "fast_cgc_star")),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
        for ax, scenario in zip(axes, ("driver", "null"), strict=True):
            width = 0.75 / len(methods)
            for method_index, method in enumerate(methods):
                x = np.arange(len(arms)) + (method_index - (len(methods) - 1) / 2) * width
                samples = [[row[metric] for row in data if row["scenario"] == scenario
                            and row["method"] == method and row["endpoint"] == arm] for arm in arms]
                if any(len(values) != len(report["development_seeds"]) for values in samples):
                    raise ValueError("every plotted bar requires the complete declared seed set")
                ax.bar(x, [np.mean(values) for values in samples], width=width,
                       color=colors[method_index] if len(methods) > 1 else colors,
                       label=method, alpha=0.8)
                for position, values in zip(x, samples, strict=True):
                    ax.scatter(np.full(len(values), position), values, s=12, color="black", zorder=3)
            ax.set_xticks(np.arange(len(arms)), [arm.replace("_", "\n") for arm in arms], fontsize=8)
            ax.set_title(scenario)
            ax.set_ylabel("Mean absolute probability-effect error")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.2)
        if len(methods) > 1:
            axes[1].legend(fontsize=7, loc="upper right")
        fig.suptitle(title + "; development seeds only, dots = process seeds")
        fig.savefig(figures / f"{name}.pdf")
        fig.savefig(figures / f"{name}.png", dpi=160)
        plt.close(fig)
    (output / "analysis/figure-catalog.md").write_text(
        "# Diagnostic figure catalog\n\n"
        "`01-endpoint-semantic-error`: true simulator effects through each endpoint versus "
        "independently defined behavior, by scenario. `02-behavioral-effect-error`: fitted "
        "mechanism effects through each endpoint versus true behavioral effects.\n\n"
        f"Bars average {n_seeds} process seeds, and each dot is one process seed. Methods, "
        "trials, candidates and endpoint arms are paired conditions, not extra replicates. "
        "No confidence intervals or confirmatory significance are asserted. Values come "
        "from the frozen report and `seed-metrics.csv`; all methods and endpoints retain "
        "their declared identity.\n"
    )
    (output / "README.md").write_text(
        "# Learned behavioral endpoint: initial development bridge\n\n"
        f"Development seeds {report['development_seeds']}, driver and endpoint-null scenarios, "
        "seven neural-dynamics lanes and seven endpoint arms. This is a linear PCA "
        "bridge, not nonlinear-manifold confirmation.\n\n"
        "Start with `analysis/analysis-report.md`, `analysis/seed-metrics.csv` and the "
        "copied method plan. `run-evidence.zip` contains first-completed native support, "
        "endpoint artifacts and binary rollout events. `implementation-source.zip` "
        "preserves the executed implementation. No native estimator was rerun to export.\n"
    )
    manifest = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size,
                 "sha256": _hash(path)} for path in sorted(output.rglob("*")) if path.is_file()]
    (output / "manifest.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export_run(args.run_root, args.output_dir)


if __name__ == "__main__":
    main()
