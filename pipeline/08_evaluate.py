"""Stage 8: run the split-owned probability-level M0/M1 evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact
from effectome.workflows.evaluation import EvaluationConfig, run_primary_evaluation


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    windows = load_artifact(art / "windows.pkl")
    eval_cfg_data = cast(dict[str, Any], OmegaConf.to_container(cfg.evaluation, resolve=True))
    result = run_primary_evaluation(windows, EvaluationConfig(**eval_cfg_data))
    save_artifact(result, art / "evaluation.pkl")
    print(
        "Primary evaluation: "
        f"{result.diagnostics['available_folds']} available folds, "
        f"{result.diagnostics['unavailable_folds']} unavailable folds"
    )
    for fold in result.manifest.outer_folds:
        print(
            f"  {fold.fold_id}: {fold.status} "
            f"train={len(fold.train_indices)} test={len(fold.test_indices)}"
            + (f" reason={fold.failure_reason}" if fold.failure_reason else "")
        )


if __name__ == "__main__":
    main()
