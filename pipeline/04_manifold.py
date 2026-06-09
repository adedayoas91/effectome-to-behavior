"""Stage 5: learn a behavioral manifold from neural activity."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from effectome.manifold import ManifoldConfig, ManifoldFactory
from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact
from effectome.viz import plot_manifold

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    fig = Path(cfg.paths.figures)

    recording = load_artifact(art / "recording.pkl")
    man_cfg = ManifoldConfig(**OmegaConf.to_container(cfg.manifold, resolve=True))
    embedder = ManifoldFactory(man_cfg)
    embedding = embedder.embed(recording.traces.T, recording.behavior)

    save_artifact({"embedding": embedding, "method": man_cfg.name}, art / "manifold.pkl")
    color = recording.behavior.get(man_cfg.behavior_key, np.zeros(embedding.shape[0]))
    if embedding.shape[1] >= 2:
        plot_manifold(
            embedding, color, f"behavioral manifold ({man_cfg.name})",
            fig / f"manifold_{man_cfg.name}.png",
        )
    logger.info("Stage 5 done: %s embedding %s", man_cfg.name, embedding.shape)


if __name__ == "__main__":
    main()
