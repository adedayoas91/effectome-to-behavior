"""Stage 7b: validate a surrogate and perturb preliminary predictive candidates."""

from __future__ import annotations

import logging
from pathlib import Path

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
    iv = OmegaConf.to_container(iv_cfg, resolve=True)

    series = load_artifact(art / "connectivity.pkl")
    manifold = load_artifact(art / "manifold.pkl")
    attribution = load_artifact(art / "attribution.pkl")

    report = {}
    for bkey in iv["behavior_keys"]:
        beh = series.behavior_per_window[bkey]
        surrogate = fit_linear_surrogate(
            series,
            manifold,
            beh,
            lag=iv["lag"],
            ridge_alpha=iv["ridge_alpha"],
            n_folds=iv["n_folds"],
            embargo=iv["embargo"],
            min_skill=iv["min_skill"],
        )
        preliminary_scores = [score for score in attribution[bkey].scores if score.is_candidate]
        top_nodes = [score.node for score in preliminary_scores[: iv["top_k"]]]
        perturb = run_virtual_perturbation(
            surrogate,
            series,
            manifold,
            beh,
            node_indices=top_nodes,
            scale=iv["scale"],
            n_controls=iv["n_controls"],
            endpoint=iv["endpoint"],
            seed=iv["seed"],
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
