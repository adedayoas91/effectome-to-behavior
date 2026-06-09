"""Stage 0-1: load -> preprocess -> window. Writes recording + windowed segments."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from effectome.data_module import (
    PreprocessConfig,
    WindowConfig,
    get_loader,
    make_windows,
    preprocess,
)
from effectome.utils import set_seed
from effectome.utils.io import save_artifact

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    recording = get_loader(data_cfg["name"])(data_cfg)

    pre = preprocess(recording, PreprocessConfig(**OmegaConf.to_container(cfg.preprocess, resolve=True)))
    win_cfg = WindowConfig(**OmegaConf.to_container(cfg.windowing, resolve=True))
    windows = make_windows(pre, win_cfg)

    save_artifact(pre, art / "recording.pkl")
    save_artifact(windows, art / "windows.pkl")
    logger.info("Stage 0-1 done: %d neurons, %d windows", pre.n_neurons, windows.n_windows)


if __name__ == "__main__":
    main()
