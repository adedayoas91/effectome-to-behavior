"""Verify and plot a completed continuous-behavior experiment from saved evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from effectome.experiments.behavior_supervised import REPORT
from effectome.experiments.csl_sanity import CSL_METHODS


def audit_results(root: Path) -> dict:
    report = json.loads((root / REPORT).read_text())
    if not report["execution_complete"] or not report["all_requested_lanes_available"]:
        raise ValueError("a complete run with all requested lanes is required")
    hashes = rows = comparisons = null_cells = exact_supports = native = 0
    tolerance = 1e-12
    for entry in report["cells"]:
        path = root / entry["path"]
        cell = json.loads(path.read_text())
        assert cell["signature"] == report["signature"]
        files = dict(cell["readout_artifacts"])
        oracle = cell["methods"]["oracle"]
        if cell["scenario"] == "null":
            for unit in oracle["metrics"]["behavior_reference"]["per_unit"].values():
                assert not np.any(np.asarray(unit["gold_effect_vectors"]))
            null_cells += 1
        for method, outcome in cell["methods"].items():
            files.update(outcome.get("artifacts", {}))
            files.update(outcome["support_metadata"].get("artifacts", {}))
            if method in CSL_METHODS:
                native += 1
                exact_supports += np.array_equal(
                    outcome["model"]["support"], cell["generator"]["true_support"]
                )
            for arm, result in outcome["metrics"].items():
                paths = result["per_unit"]
                error_rows = []
                positive_errors = []
                for unit, data in paths.items():
                    gold = np.asarray(data["gold_effect_vectors"])
                    true = np.asarray(data["true_chart_effect_vectors"])
                    estimated = np.asarray(data["estimated_effect_vectors"])
                    np.testing.assert_array_equal(
                        gold, oracle["metrics"][arm]["per_unit"][unit]["gold_effect_vectors"]
                    )
                    np.testing.assert_array_equal(
                        true, oracle["metrics"][arm]["per_unit"][unit]["true_chart_effect_vectors"]
                    )
                    total = np.mean(np.abs(estimated - gold), axis=1)
                    within = np.mean(np.abs(estimated - true), axis=1)
                    semantic = np.mean(np.abs(true - gold), axis=1)
                    gold_magnitude = np.linalg.norm(gold, axis=1) / np.sqrt(2)
                    estimated_magnitude = np.linalg.norm(estimated, axis=1) / np.sqrt(2)
                    bound = (
                        within
                        + np.asarray(result["factual_behavior_disagreement"])
                        + data["intervened_behavior_disagreement"]
                    )
                    for key, expected in (
                        ("end_to_end_absolute_error", total),
                        ("within_chart_absolute_error", within),
                        ("semantic_absolute_error", semantic),
                        ("triangle_bound", bound),
                    ):
                        np.testing.assert_allclose(data[key], expected, rtol=0, atol=tolerance)
                    assert np.all(total <= bound + tolerance)
                    np.testing.assert_allclose(
                        result["candidate_scores"][unit], estimated_magnitude.mean(), rtol=0, atol=tolerance
                    )
                    np.testing.assert_allclose(
                        result["gold_scores"][unit], gold_magnitude.mean(), rtol=0, atol=tolerance
                    )
                    if unit in result["eligible_units"]:
                        positive_errors.extend(total.tolist())
                    if method == "empty_support":
                        np.testing.assert_array_equal(estimated, np.zeros_like(estimated))
                    comparisons += len(total)
                    error_rows.append([total.mean(), within.mean(), semantic.mean(), bound.mean()])
                np.testing.assert_allclose(
                    np.mean(error_rows, axis=0),
                    [
                        result[key]
                        for key in (
                            "end_to_end_mae",
                            "within_chart_mae",
                            "semantic_mae",
                            "mean_triangle_bound",
                        )
                    ],
                    rtol=0,
                    atol=tolerance,
                )
                assert result["bound_violation_count"] == 0
                if positive_errors:
                    np.testing.assert_allclose(
                        result["positive_unit_mae"], np.mean(positive_errors), rtol=0, atol=tolerance
                    )
                else:
                    assert result["positive_unit_mae"] is None
                rows += 1
        for name, expected in files.items():
            assert Path(name).name == name
            assert hashlib.sha256(path.with_name(name).read_bytes()).hexdigest() == expected
            hashes += 1
    return {
        "passed": True,
        "profile": report["profile"],
        "completed_cells": report["completed_cells"],
        "available_method_runs": report["available_method_runs"],
        "native_window_fits": report["native_window_fits"],
        "exact_native_consensus_supports": int(exact_supports),
        "tested_native_consensus_supports": native,
        "exact_oracle_null_cells": null_cells,
        "verified_artifact_hashes": hashes,
        "seed_dynamics_readout_rows": rows,
        "origin_candidate_effects_recomputed": comparisons,
        "bound_violations": 0,
        "tolerance": tolerance,
        "scope": "numerical_and_artifact_checks_of_saved_evidence",
    }


def plot_results(root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.read_csv(root / "analysis/seed-metrics.csv", keep_default_na=False, na_values=[""])
    arms = [
        "pca_2d",
        "pca_4d",
        "pls_2d",
        "full_linear",
        "behavior_only_2d",
        "bunddle_behavior_2d",
        "bunddle_behavior_4d",
        "full_nonlinear",
    ]
    names = [
        "PCA 2D",
        "PCA 4D",
        "PLS 2D",
        "Full linear",
        "Behavior-only 2D",
        "Joint 2D",
        "Joint 4D",
        "Full nonlinear",
    ]
    driver = data[(data.scenario == "driver") & (data.dynamics == "known_support")]
    columns = ("end_to_end_mae", "positive_unit_mae")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for ax, key, title in zip(
        axes, columns, ("All interventions", "Oracle-positive interventions"), strict=True
    ):
        samples = [driver.loc[driver.readout == arm, key].dropna().to_numpy() for arm in arms]
        ax.bar(
            range(len(arms)),
            [array.mean() if len(array) else np.nan for array in samples],
            color=["#56889e"] * 4 + ["#c78346"] * 4,
        )
        for index, values in enumerate(samples):
            ax.scatter(np.full(len(values), index), values, color="#222222", s=16, zorder=3)
        ax.set_xticks(range(len(arms)), names, rotation=35, ha="right")
        ax.set(title=title, ylabel="Behavioral effect MAE", ylim=(0, None))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Distributed behavior, fitted known support (dots are process seeds)")
    for extension in ("pdf", "png"):
        fig.savefig(root / f"analysis/development-effect-error.{extension}", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = audit_results(args.run_root)
    (args.run_root / "analysis/independent-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.plot:
        plot_results(args.run_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
