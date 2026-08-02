"""Stage 0-1: load -> preprocess -> window. Writes recording + windowed segments."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from effectome.data_module.loaders import get_loader
from effectome.data_module.preprocess import (
    PreprocessConfig,
)
from effectome.data_module.preprocess import (
    preprocess as preprocess_recording,
)
from effectome.data_module.windowing import WindowConfig, make_windows
from effectome.utils import set_seed
from effectome.utils.io import save_artifact

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)

    data_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.data, resolve=True))
    recording = get_loader(data_cfg["name"])(data_cfg)

    preprocess_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.preprocess, resolve=True))
    windowing_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.windowing, resolve=True))
    pre = preprocess_recording(recording, PreprocessConfig(**preprocess_cfg))
    win_cfg = WindowConfig(**windowing_cfg)
    windows = make_windows(pre, win_cfg)

    save_artifact(pre, art / "recording.pkl")
    save_artifact(windows, art / "windows.pkl")
    logger.info(
        "Stage 0-1 done: dataset=%s recording=%s neurons=%d windows=%d mode=%s",
        pre.identity.dataset_id,
        pre.identity.recording_id,
        pre.n_neurons,
        windows.n_windows,
        windows.metadata.get("window_mode", win_cfg.mode),
    )


if __name__ == "__main__":
    main()
