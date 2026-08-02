"""Stage 4: detect (evolving) neural communities from the dynamic effectome."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from effectome.community import CommunityConfig, CommunityFactory
from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)

    series = load_artifact(art / "connectivity.pkl")
    com_cfg_data = cast(dict[str, Any], OmegaConf.to_container(cfg.community, resolve=True))
    com_cfg = CommunityConfig(**com_cfg_data)
    detector = CommunityFactory(com_cfg)
    communities = detector.run(series)

    save_artifact(communities, art / "community.pkl")
    logger.info(
        "Stage 4 done: %s, communities/window min=%d max=%d",
        communities.method,
        int(communities.n_communities_per_window.min()),
        int(communities.n_communities_per_window.max()),
    )


if __name__ == "__main__":
    main()
