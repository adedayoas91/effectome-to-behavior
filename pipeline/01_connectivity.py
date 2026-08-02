"""Stage 2: infer a connectivity matrix per window (the dynamic effectome)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact
from effectome.viz import plot_connectivity_matrix

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    fig = Path(cfg.paths.figures)

    windows = load_artifact(art / "windows.pkl")
    connectivity_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.connectivity, resolve=True))
    conn_cfg = ConnectivityConfig(**connectivity_cfg)
    estimator = ConnectivityFactory(conn_cfg)
    series = estimator.run(windows)

    save_artifact(series, art / "connectivity.pkl")
    plot_connectivity_matrix(
        series.matrices.mean(axis=0),
        f"mean effectome ({series.method})",
        fig / f"mean_connectivity_{series.method}.png",
    )
    logger.info(
        "Stage 2 done: %d %s matrices (directed=%s signed=%s weighted=%s mode=%s)",
        series.n_windows,
        series.method,
        series.directed,
        series.signed,
        series.weighted,
        series.diagnostics.get("estimation_mode"),
    )


if __name__ == "__main__":
    main()
