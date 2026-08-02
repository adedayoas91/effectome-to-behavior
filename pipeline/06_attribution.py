"""Stage 7a: signed node attribution and candidate-driver qualification."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from effectome.attribution import qualify_candidate_drivers
from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    at_cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "conf" / "attribution" / "default.yaml")
    at = OmegaConf.to_container(at_cfg, resolve=True)

    series = load_artifact(art / "connectivity.pkl")
    community = load_artifact(art / "community.pkl")
    manifold = load_artifact(art / "manifold.pkl")

    report = {}
    for bkey in at["behavior_keys"]:
        beh = series.behavior_per_window[bkey]
        report[bkey] = qualify_candidate_drivers(
            series,
            community,
            manifold,
            beh,
            bkey,
            lag=at["lag"],
            n_folds=at["n_folds"],
            embargo=at["embargo"],
            seed=at["seed"],
            matched_control_percentile=at["matched_control_percentile"],
        )
    save_artifact(report, art / "attribution.pkl")
    logger.info("Stage 7a done: attribution for %d behavior targets", len(report))


if __name__ == "__main__":
    main()
