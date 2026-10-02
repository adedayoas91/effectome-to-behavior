"""Stage 7b: validate surrogate predictor-feature sensitivity for preliminary candidates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from effectome.intervention import fit_linear_surrogate, run_predictor_feature_sensitivity
from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    iv_cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "conf" / "intervention" / "default.yaml")
    iv = cast(dict[str, Any], OmegaConf.to_container(iv_cfg, resolve=True))

    series = load_artifact(art / "connectivity.pkl")
    manifold = load_artifact(art / "manifold.pkl")
    attribution = load_artifact(art / "attribution.pkl")

    report = {}
    for bkey in iv["behavior_keys"]:
        if bkey not in series.behavior_per_window:
            logger.warning("behavior '%s' missing from windows; skipping", bkey)
            continue
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
        )
        preliminary_scores = [score for score in attribution[bkey].scores if score.is_candidate]
        top_nodes = [score.node for score in preliminary_scores[: int(iv["top_k"])]]
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
        )
        report[bkey] = {
            "surrogate": surrogate,
            "feature_sensitivity": perturb,
            "preliminary_candidate_nodes": top_nodes,
        }
    save_artifact(report, art / "intervention.pkl")
    logger.info("Stage 7b done: feature-sensitivity reports for %d behavior targets", len(report))


if __name__ == "__main__":
    main()
