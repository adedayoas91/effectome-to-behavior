"""Stage 7b: validate a surrogate and perturb preliminary predictive candidates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from effectome.intervention import fit_linear_surrogate, run_virtual_perturbation
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
        surrogate = fit_linear_surrogate(
            series,
            manifold,
            beh,
            lag=int(iv["lag"]),
            ridge_alpha=float(iv["ridge_alpha"]),
            n_folds=int(iv["n_folds"]),
            embargo=int(iv["embargo"]),
            min_skill=float(iv["min_skill"]),
        )
        preliminary_scores = [score for score in attribution[bkey].scores if score.is_candidate]
        top_nodes = [score.node for score in preliminary_scores[: int(iv["top_k"])]]
        perturb = run_virtual_perturbation(
            surrogate,
            series,
            manifold,
            beh,
            node_indices=top_nodes,
            scale=float(iv["scale"]),
            n_controls=int(iv["n_controls"]),
            endpoint=str(iv["endpoint"]),
            seed=int(iv["seed"]),
            dose_scales=iv.get("dose_scales"),
        )
        report[bkey] = {
            "surrogate": surrogate,
            "perturbation": perturb,
            "preliminary_candidate_nodes": top_nodes,
        }
    save_artifact(report, art / "intervention.pkl")
    logger.info("Stage 7b done: intervention reports for %d behavior targets", len(report))


if __name__ == "__main__":
    main()
