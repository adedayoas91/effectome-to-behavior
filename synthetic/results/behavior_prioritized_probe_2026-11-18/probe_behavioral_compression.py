"""Exploratory, training-owned compression probes using cached neural mechanisms.

No causal discovery or mechanism refitting occurs. The scalar discriminant is an
algebraic factorization of the already trained full-neural decoder, not an
independent successful algorithm. PCA dimension sweeps are development probes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from effectome.experiments.learned_endpoints import FrozenEndpoint, endpoint_from_dict, fit_endpoint_arms
from effectome.experiments.manifold_endpoint_rollouts import paired_endpoint_probabilities, semantics_errors
from effectome.experiments.mechanism_controls import (
    MechanismControlConfig,
    make_mechanism_control,
    mechanism_control_trials,
)
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig
from effectome.workflows.cross_method import _json_safe
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

logger = logging.getLogger(__name__)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output = Path(__file__).resolve().parent
    repository = output.parents[2]
    base = repository / "outputs/learned-manifold-validation/development/endpoint-bridge-03"
    saved = json.loads((base / "manifold-validation.json").read_text())
    if not saved["execution_complete"] or not saved["all_requested_lanes_available"]:
        raise ValueError("the original complete evidence is required")
    protocol = {
        "status": "exploratory_investigation_on_inspected_development_seeds",
        "baseline_report_sha256": digest(base / "manifold-validation.json"),
        "implementation_fingerprint": implementation_fingerprint(repository),
        "script_sha256": digest(Path(__file__)),
        "native_fits": 0,
        "mechanism_fits": 0,
        "arms": [
            "behavior_reference",
            "full_neural_behavior",
            "behavior_score_1d",
            "pca_behavior_k1",
            "pca_behavior_k2",
            "pca_behavior_k3",
            "pca_behavior_k4",
            "pca_behavior_k5",
        ],
        "selection": "all_dimensions_reported_no_test_chosen_winner_or_confirmatory_claim",
        "reference": "predeclared_generator_endpoint_not_redefined_by_representation",
    }
    frozen = output / "probe-protocol.json"
    if frozen.is_file() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("probe source/protocol changed; preserve evidence and choose another directory")
    frozen.write_text(json.dumps(protocol, indent=2) + "\n")
    cells = []
    all_rows = []
    for original in saved["cells"]:
        cell_id = original["cell_id"]
        path = output / f"{cell_id}.json"
        if path.is_file():
            cell = json.loads(path.read_text())
            if cell["protocol"] != content_fingerprint(protocol):
                raise ValueError("probe cache signature changed")
            cells.append(cell)
            all_rows.extend(cell["rows"])
            continue
        logger.info("Compression probe %s", cell_id)
        generator = dict(saved["protocol"]["config"]["mechanism"]["generator"])
        generator.update(scenario=original["scenario"], seed=original["seed"])
        trials = mechanism_control_trials(make_mechanism_control(MechanismControlConfig(**generator)))
        train = trials[original["training_trial_ids"]]
        evaluation = trials[original["evaluation_trial_ids"]]
        assert content_fingerprint(train) == original["training_fingerprint"]
        assert content_fingerprint(evaluation) == original["evaluation_fingerprint"]
        assert content_fingerprint(evaluation[:, 0]) == original["evaluation_initial_fingerprint"]
        parent_folder = base / "cells" / cell_id
        artifact_path = parent_folder / "endpoints.json"
        assert digest(artifact_path) == original["artifacts"]["endpoints.json"]
        original_endpoints = {
            name: endpoint_from_dict(value) for name, value in json.loads(artifact_path.read_text()).items()
        }
        full = original_endpoints["full_neural_behavior"]
        assert isinstance(full, FrozenEndpoint) and full.readout_weights is not None
        raw_weights = full.readout_weights @ full.encoder_weights
        raw_bias = float(full.readout_weights @ full.encoder_bias + full.readout_intercept)
        score = FrozenEndpoint(
            "behavior_score_1d",
            raw_weights[None, :],
            np.array([raw_bias]),
            "linear",
            np.ones(1),
            metadata={
                "construction": "exact_affine_factorization_of_previously_fitted_behavioral_decoder",
                "parent_fingerprint": full.to_dict()["fingerprint"],
                "original_training_ownership": full.metadata,
                "uses_evaluation_labels_for_fitting": False,
                "claim": "one_dimensional_binary_readout_not_autonomous_nonlinear_manifold",
            },
        )
        endpoints = {
            "behavior_reference": original_endpoints["behavior_reference"],
            "full_neural_behavior": full,
            "behavior_score_1d": score,
            "pca_behavior_k2": original_endpoints["pca_behavior"],
        }
        mechanism_config = saved["protocol"]["config"]["mechanism"]
        rollout_config = MechanismRolloutConfig(
            **{**mechanism_config["rollout"], "seed": original["rollout_seed"]}
        )
        geometry = []
        for k in (1, 2, 3, 4, 5):
            if k != 2:
                arms = fit_endpoint_arms(
                    train,
                    endpoint_index=generator["n_neurons"] - 1,
                    target_length=rollout_config.target_length,
                    n_components=k,
                    validation_fraction=mechanism_config["fit"]["validation_fraction"],
                    decoder_cs=tuple(saved["protocol"]["config"]["decoder_cs"]),
                    seed=saved["protocol"]["config"]["endpoint_seed"],
                )
                endpoints[f"pca_behavior_k{k}"] = arms["pca_behavior"]
            endpoint = endpoints[f"pca_behavior_k{k}"]
            assert isinstance(endpoint, FrozenEndpoint)
            components = np.asarray(endpoint.metadata["pca_components"])
            geometry.append(
                {
                    "dimension": k,
                    "variance_fraction": sum(endpoint.metadata["pca_explained_variance_ratio"]),
                    "behavior_axis_squared_projection": float(np.sum(components[:, -1] ** 2)),
                    "best_inner_validation_decoder_accuracy": max(
                        endpoint.metadata["decoder_validation_accuracy_by_c"]
                    ),
                }
            )
        (output / f"{cell_id}-endpoints.json").write_text(
            json.dumps({name: endpoint.to_dict() for name, endpoint in endpoints.items()}, indent=2) + "\n"
        )
        natural_disagreements = int(
            np.count_nonzero(full.source_states(trials) != score.source_states(trials))
        )
        oracle = original["methods"]["oracle"]["model"]
        truth = paired_endpoint_probabilities(
            AdditiveMechanism(**oracle),
            evaluation[:, 0],
            original["interventions"],
            endpoints=endpoints,
            config=rollout_config,
            origin_ids=np.array(original["evaluation_trial_ids"]),
        )
        results = {}
        rows = []
        total_event_disagreements = 0
        for method, baseline in original["methods"].items():
            run = (
                truth
                if method == "oracle"
                else paired_endpoint_probabilities(
                    AdditiveMechanism(**baseline["model"]),
                    evaluation[:, 0],
                    original["interventions"],
                    endpoints=endpoints,
                    config=rollout_config,
                    origin_ids=np.array(original["evaluation_trial_ids"]),
                )
            )
            differences = sum(
                int(
                    np.count_nonzero(
                        run["events"]["full_neural_behavior"][branch]
                        != run["events"]["behavior_score_1d"][branch]
                    )
                )
                for branch in run["events"]["full_neural_behavior"]
            )
            total_event_disagreements += differences
            metrics = semantics_errors(truth, run)
            with np.load(parent_folder / f"{method}-events.npz") as previous:
                for branch, events in run["events"]["full_neural_behavior"].items():
                    np.testing.assert_array_equal(events, previous[f"full_neural_behavior|{branch}"])
            archive = output / f"{cell_id}-{method}-events.npz"
            np.savez_compressed(
                archive,
                **{
                    f"{arm}|{branch}": value
                    for arm, branches in run["events"].items()
                    for branch, value in branches.items()
                },
            )
            results[method] = {
                "metrics": metrics,
                "score_event_disagreements": differences,
                "events_sha256": digest(archive),
            }
            for arm, values in metrics.items():
                rows.append(
                    {
                        "seed": original["seed"],
                        "scenario": original["scenario"],
                        "method": method,
                        "endpoint": arm,
                        "dimension": 1
                        if arm == "behavior_score_1d"
                        else (
                            6
                            if arm == "full_neural_behavior"
                            else int(arm[-1])
                            if arm.startswith("pca_behavior_k")
                            else 1
                        ),
                        "behavioral_mae": values["end_to_end_mae"],
                        "within_chart_mae": values["within_chart_mae"],
                        "semantic_mae": values["semantic_mae"],
                    }
                )
        cell = {
            "cell_id": cell_id,
            "protocol": content_fingerprint(protocol),
            "scenario": original["scenario"],
            "seed": original["seed"],
            "natural_score_label_disagreements": natural_disagreements,
            "rollout_score_event_disagreements": total_event_disagreements,
            "frozen_parent_events_replayed_exactly": True,
            "geometry": geometry,
            "results": results,
            "rows": rows,
        }
        path.write_text(json.dumps(_json_safe(cell), indent=2, allow_nan=False) + "\n")
        cells.append(cell)
        all_rows.extend(rows)
    with (output / "seed-metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "protocol": protocol,
        "execution_complete": len(cells) == len(saved["cells"]),
        "cells": len(cells),
        "methods_per_cell": len(saved["neural_dynamics"]),
        "endpoint_arms": len(protocol["arms"]),
        "independent_development_seeds": 5,
        "new_independent_confirmatory_seeds": 0,
        "new_native_or_mechanism_fits": 0,
        "natural_score_label_disagreements": sum(c["natural_score_label_disagreements"] for c in cells),
        "rollout_score_event_disagreements": sum(c["rollout_score_event_disagreements"] for c in cells),
        "rollout_config_template": saved["protocol"]["config"]["mechanism"]["rollout"],
        "effective_seed_by_cell": {cell["cell_id"]: cell["rollout_seed"] for cell in saved["cells"]},
        "PCA_dimension_fit_calls": 40,
        "score_fit_calls": 0,
    }
    (output / "probe-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Completed: %s", summary)


if __name__ == "__main__":
    main()
