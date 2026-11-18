"""Independently reconstruct learned-endpoint errors from saved binary events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit_run(run_root: Path) -> dict:
    report = json.loads((run_root / "manifold-validation.json").read_text())
    if not report["execution_complete"] or not report["all_requested_lanes_available"]:
        raise ValueError("audit requires complete, available evidence")
    artifact_count = 0
    metric_rows = 0
    pair_count = 0
    native_windows = 0
    for cell in report["cells"]:
        folder = run_root / "cells" / cell["cell_id"]
        artifacts = dict(cell["artifacts"])
        for result in cell["methods"].values():
            artifacts.update(result["artifacts"])
            artifacts.update(result["support_metadata"].get("artifacts", {}))
            native_windows += result["support_metadata"]["native_window_fits"]
        for name, digest in artifacts.items():
            assert Path(name).name == name
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name
            artifact_count += 1
        oracle = np.load(folder / "oracle-events.npz")
        candidates = cell["interventions"]
        for method, result in cell["methods"].items():
            with np.load(folder / f"{method}-events.npz") as fitted:
                for arm, stored in result["metrics"].items():
                    q0 = np.asarray(oracle[f"{arm}|__factual__"], dtype=float)
                    b0 = np.asarray(oracle["behavior_reference|__factual__"], dtype=float)
                    h0 = np.asarray(fitted[f"{arm}|__factual__"], dtype=float)
                    assert np.isin(q0, (0, 1)).all() and q0.shape == b0.shape == h0.shape
                    factual_disagreement = np.abs(q0 - b0).mean(axis=1)
                    np.testing.assert_allclose(factual_disagreement, stored["factual_event_disagreement"],
                                               atol=1e-12, rtol=0)
                    total: list[float] = []
                    within: list[float] = []
                    semantic: list[float] = []
                    bounds: list[float] = []
                    mismatches: list[float] = []
                    for candidate in candidates:
                        qs = np.asarray(oracle[f"{arm}|{candidate}"], dtype=float)
                        bs = np.asarray(oracle[f"behavior_reference|{candidate}"], dtype=float)
                        hs = np.asarray(fitted[f"{arm}|{candidate}"], dtype=float)
                        gold = (b0 - bs).mean(axis=1)
                        true_chart = (q0 - qs).mean(axis=1)
                        fitted_chart = (h0 - hs).mean(axis=1)
                        absolute_error = np.abs(fitted_chart - gold)
                        mechanism_error = np.abs(fitted_chart - true_chart)
                        semantic_error = np.abs(true_chart - gold)
                        mismatch = np.abs(qs - bs).mean(axis=1)
                        bound = mechanism_error + factual_disagreement + mismatch
                        assert np.all(absolute_error <= bound + 1e-12)
                        saved = stored["per_unit"][candidate]
                        for key, values in (
                            ("end_to_end_absolute_error", absolute_error),
                            ("within_chart_absolute_error", mechanism_error),
                            ("semantic_absolute_error", semantic_error), ("triangle_bound", bound),
                            ("intervened_event_disagreement", mismatch),
                        ):
                            np.testing.assert_allclose(saved[key], values, atol=1e-12, rtol=0)
                        total.extend(absolute_error)
                        within.extend(mechanism_error)
                        semantic.extend(semantic_error)
                        bounds.extend(bound)
                        mismatches.extend(mismatch)
                        pair_count += len(gold)
                    for key, expected in (
                        ("end_to_end_mae", np.mean(total)), ("within_chart_mae", np.mean(within)),
                        ("semantic_mae", np.mean(semantic)), ("mean_triangle_bound", np.mean(bounds)),
                        ("mean_factual_event_disagreement", factual_disagreement.mean()),
                        ("mean_intervened_event_disagreement", np.mean(mismatches)),
                    ):
                        np.testing.assert_allclose(stored[key], expected, atol=1e-12, rtol=0)
                    assert stored["bound_violation_count"] == 0
                    metric_rows += 1
        oracle.close()
    return {"passed": True, "completed_cells": len(report["cells"]),
            "artifact_hashes_verified": artifact_count, "native_window_fits": native_windows,
            "endpoint_method_seed_rows_reconstructed": metric_rows,
            "origin_candidate_comparisons": pair_count, "empirical_bound_violations": 0,
            "tolerance": 1e-12, "source": "independent_numpy_reconstruction_from_binary_event_NPZs",
            "population_guarantee": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_run(args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
