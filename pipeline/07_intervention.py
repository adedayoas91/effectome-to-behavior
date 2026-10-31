"""Stage 7b: validate surrogate predictor-feature sensitivity for preliminary candidates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from effectome.attribution import qualify_candidate_drivers
from effectome.connectivity import ConnectivityConfig
from effectome.intervention import fit_linear_surrogate, run_predictor_feature_sensitivity
from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact
from effectome.workflows.feature_ownership import load_or_fit_owned_features

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    iv = cast(dict[str, Any], OmegaConf.to_container(cfg.intervention, resolve=True))

    strict_outer_ownership = bool(iv.get("strict_outer_ownership", True))
    if strict_outer_ownership:
        connectivity_config = cast(dict[str, Any], OmegaConf.to_container(cfg.connectivity, resolve=True))
        owned = load_or_fit_owned_features(
            art / "windows.pkl", art / "outer_owned_features.pkl",
            ConnectivityConfig(**connectivity_config), iv,
        )
        series, manifold, community = owned.connectivity, owned.manifold, owned.community
        fit_indices, evaluation_indices, fit_scope_id = (
            owned.fit_indices, owned.evaluation_indices, owned.fit_scope_id
        )
        attribution = None
    else:
        series = load_artifact(art / "connectivity.pkl")
        manifold = load_artifact(art / "manifold.pkl")
        attribution = load_artifact(art / "attribution.pkl")
        community = None
        fit_indices = None
        evaluation_indices = None
        fit_scope_id = None

    declared_behavior_keys = [str(key) for key in iv["behavior_keys"]]
    available_behavior_keys = []
    report = {}
    for key in declared_behavior_keys:
        if key not in series.behavior_per_window:
            logger.warning("behavior '%s' missing from windows; skipping", key)
            report[key] = {
                "status": "unavailable",
                "fail_safe_reason": "behavior_missing",
            }
            continue
        available_behavior_keys.append(key)
    multiplicity_family = tuple(
        f"{key}:feature-edit" for key in declared_behavior_keys
    )

    for bkey in available_behavior_keys:
        beh = series.behavior_per_window[bkey]
        endpoint = str(iv["endpoint"])
        values = np.asarray(beh)
        categorical_behavior = np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype,
            np.bool_,
        )
        if categorical_behavior and endpoint in {"behavior", "joint"}:
            logger.warning(
                "behavior '%s' is categorical; using manifold endpoint for feature sensitivity",
                bkey,
            )
            endpoint = "manifold"
        if strict_outer_ownership:
            assert community is not None
            candidate_result = qualify_candidate_drivers(
                series,
                community,
                manifold,
                beh,
                bkey,
                lag=int(iv["lag"]),
                n_folds=int(iv["n_folds"]),
                embargo=int(iv["embargo"]),
                seed=int(iv["seed"]),
                fit_indices=fit_indices,
                evaluation_indices=evaluation_indices,
                fit_scope_id=fit_scope_id,
                require_feature_ownership=True,
            )
        else:
            assert attribution is not None
            candidate_result = attribution[bkey]
        preliminary_scores = [
            score for score in candidate_result.scores if score.is_candidate
        ]
        top_nodes = [score.node for score in preliminary_scores[: int(iv["top_k"])]]
        hypothesis_id = f"{bkey}:feature-edit"
        surrogate = fit_linear_surrogate(
            series,
            manifold,
            beh,
            lag=int(iv["lag"]),
            ridge_alpha=float(iv["ridge_alpha"]),
            n_folds=int(iv["n_folds"]),
            embargo=int(iv["embargo"]),
            min_skill=float(iv["min_skill"]),
            endpoint=endpoint,
            fit_indices=fit_indices,
            evaluation_indices=evaluation_indices,
            candidate_nodes=top_nodes,
            min_absolute_skill=float(iv["min_absolute_skill"]),
            max_shift=float(iv["max_distribution_shift"]),
            require_feature_ownership=strict_outer_ownership,
            fit_scope_id=fit_scope_id,
            strict_outer_ownership=strict_outer_ownership,
            dose_scales=iv.get("dose_scales"),
            n_controls=int(iv["n_controls"]),
            multiplicity_family=multiplicity_family,
            hypothesis_id=hypothesis_id,
            control_alpha=float(iv["control_alpha"]),
            control_resolution_policy=str(iv["control_resolution_policy"]),
            increment_interval_level=float(iv["increment_interval_level"]),
            interval_seed=int(iv["seed"]),
        )
        perturb = run_predictor_feature_sensitivity(
            surrogate,
            series,
            manifold,
            beh,
            node_indices=top_nodes,
            scale=float(iv["scale"]),
            n_controls=int(iv["n_controls"]),
            endpoint=endpoint,
            seed=int(iv["seed"]),
            dose_scales=iv.get("dose_scales"),
            hypothesis_id=hypothesis_id,
        )
        report[bkey] = {
            "surrogate": surrogate,
            "feature_sensitivity": perturb,
            "preliminary_candidate_nodes": top_nodes,
            "candidate_screen": candidate_result,
            "fit_scope_id": fit_scope_id,
        }
    save_artifact(report, art / "intervention.pkl")
    logger.info("Stage 7b done: feature-sensitivity reports for %d behavior targets", len(report))


if __name__ == "__main__":
    main()
