"""Independent arithmetic and source checks for the archived compression probe."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from effectome.experiments.mechanism_controls import (
    MechanismControlConfig,
    make_mechanism_control,
    mechanism_control_trials,
)
from effectome.workflows.resume import content_fingerprint


def main() -> None:
    root = Path(__file__).resolve().parent
    parent_path = root / "parent-manifold-validation.json"
    parent = json.loads(parent_path.read_text())
    protocol = json.loads((root / "probe-protocol.json").read_text())
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == protocol["baseline_report_sha256"]
    assert (
        hashlib.sha256((root / "probe_behavioral_compression.py").read_bytes()).hexdigest()
        == protocol["script_sha256"]
    )
    natural_count = event_count = rows = comparisons = violations = 0
    with zipfile.ZipFile(root / "probe-evidence.zip") as archive:
        for original in parent["cells"]:
            cell_id = original["cell_id"]
            cell = json.loads(archive.read(f"{cell_id}.json"))
            artifacts = json.loads(archive.read(f"{cell_id}-endpoints.json"))
            full = artifacts["full_neural_behavior"]
            score = artifacts["behavior_score_1d"]
            weights = np.asarray(full["readout_weights"])
            expected_weight = weights @ np.asarray(full["encoder_weights"])
            expected_bias = float(weights @ np.asarray(full["encoder_bias"]) + full["readout_intercept"])
            np.testing.assert_array_equal(np.asarray(score["encoder_weights"])[0], expected_weight)
            np.testing.assert_array_equal(np.asarray(score["encoder_bias"]), [expected_bias])
            parameters = dict(parent["protocol"]["config"]["mechanism"]["generator"])
            parameters.update(scenario=original["scenario"], seed=original["seed"])
            trials = mechanism_control_trials(make_mechanism_control(MechanismControlConfig(**parameters)))
            assert (
                content_fingerprint(trials[original["training_trial_ids"]])
                == original["training_fingerprint"]
            )
            assert (
                content_fingerprint(trials[original["evaluation_trial_ids"]])
                == original["evaluation_fingerprint"]
            )
            full_score = (
                trials @ np.asarray(full["encoder_weights"]).T + full["encoder_bias"]
            ) @ weights + full["readout_intercept"]
            small_score = trials @ expected_weight + expected_bias
            np.testing.assert_array_equal(full_score >= 0, small_score >= 0)
            natural_count += full_score.size
            with np.load(io.BytesIO(archive.read(f"{cell_id}-oracle-events.npz"))) as oracle:
                gold0 = np.asarray(oracle["behavior_reference|__factual__"], dtype=float)
                for method, result in cell["results"].items():
                    raw = archive.read(f"{cell_id}-{method}-events.npz")
                    assert hashlib.sha256(raw).hexdigest() == result["events_sha256"]
                    with np.load(io.BytesIO(raw)) as fitted:
                        for branch in ("__factual__", *original["interventions"]):
                            left = fitted[f"behavior_score_1d|{branch}"]
                            right = fitted[f"full_neural_behavior|{branch}"]
                            np.testing.assert_array_equal(left, right)
                            event_count += left.size
                        for arm, metrics in result["metrics"].items():
                            h0 = np.asarray(fitted[f"{arm}|__factual__"], dtype=float)
                            q0 = np.asarray(oracle[f"{arm}|__factual__"], dtype=float)
                            factual_disagreement = np.abs(q0 - gold0).mean(axis=1)
                            total, within, semantic = [], [], []
                            for candidate in original["interventions"]:
                                gold = (gold0 - oracle[f"behavior_reference|{candidate}"]).mean(axis=1)
                                true_chart = (q0 - oracle[f"{arm}|{candidate}"]).mean(axis=1)
                                fitted_chart = (h0 - fitted[f"{arm}|{candidate}"]).mean(axis=1)
                                absolute_error = np.abs(fitted_chart - gold)
                                mechanism_error = np.abs(fitted_chart - true_chart)
                                semantic_error = np.abs(true_chart - gold)
                                mismatch = np.abs(
                                    np.asarray(oracle[f"{arm}|{candidate}"], dtype=float)
                                    - oracle[f"behavior_reference|{candidate}"]
                                ).mean(axis=1)
                                bound = mechanism_error + factual_disagreement + mismatch
                                violations += int(np.count_nonzero(absolute_error > bound + 1e-12))
                                for key, values in (
                                    ("end_to_end_absolute_error", absolute_error),
                                    ("within_chart_absolute_error", mechanism_error),
                                    ("semantic_absolute_error", semantic_error),
                                ):
                                    np.testing.assert_allclose(
                                        metrics["per_unit"][candidate][key], values, atol=1e-12, rtol=0
                                    )
                                total.extend(absolute_error)
                                within.extend(mechanism_error)
                                semantic.extend(semantic_error)
                                comparisons += len(gold)
                            for key, values in (
                                ("end_to_end_mae", total),
                                ("within_chart_mae", within),
                                ("semantic_mae", semantic),
                            ):
                                np.testing.assert_allclose(metrics[key], np.mean(values), atol=1e-12, rtol=0)
                            rows += 1
    assert natural_count == 276480 and event_count == 1935360
    assert rows == 560 and comparisons == 53760 and violations == 0
    result = {
        "passed": True,
        "parent_report_hash_verified": True,
        "probe_source_hash_verified": True,
        "natural_classifications_identical": natural_count,
        "rollout_events_identical": event_count,
        "independent_metric_rows_checked": rows,
        "origin_candidate_errors_checked": comparisons,
        "triangle_bound_violations": violations,
        "new_native_fits": 0,
        "new_mechanism_fits": 0,
        "independent_development_seeds": 5,
        "new_independent_confirmatory_seeds": 0,
    }
    (root / "independent-verification.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
