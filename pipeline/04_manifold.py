"""Stage 5: learn a behavioral manifold and align target-window latent codes to anchors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from effectome.data_module.schema import ArtifactProvenance
from effectome.manifold import ManifoldArtifact, ManifoldConfig, ManifoldFactory, TargetSlice
from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact
from effectome.viz import plot_manifold

logger = logging.getLogger(__name__)


def _target_slices(window_starts: np.ndarray, history_length: int, target_length: int) -> list[TargetSlice]:
    return [
        TargetSlice(int(start + history_length - target_length), int(start + history_length))
        for start in window_starts
    ]


def _target_slices_for_connectivity(
    connectivity,
    history_length: int,
    target_length: int,
) -> list[TargetSlice]:
    if connectivity.anchors:
        target_slices = [
            TargetSlice(int(anchor.target_start), int(anchor.target_stop)) for anchor in connectivity.anchors
        ]
        if any(ts.length != target_length for ts in target_slices):
            raise ValueError("connectivity anchors do not match manifold target_length")
        return target_slices
    return _target_slices(connectivity.window_starts, history_length, target_length)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    fig = Path(cfg.paths.figures)

    recording = load_artifact(art / "recording.pkl")
    man_cfg_data = cast(dict[str, Any], OmegaConf.to_container(cfg.manifold, resolve=True))
    man_cfg = ManifoldConfig(**man_cfg_data)
    embedder = ManifoldFactory(man_cfg)
    embedder.fit(recording.traces.T, recording.behavior)
    full_embedding = embedder.transform(recording.traces.T)

    connectivity = load_artifact(art / "connectivity.pkl")
    history_length = int(
        cfg.windowing.history_length
        if cfg.windowing.mode == "temporal" and cfg.windowing.history_length is not None
        else cfg.windowing.length
    )
    target_slices = _target_slices_for_connectivity(connectivity, history_length, man_cfg.target_length)
    window_embedding = embedder.transform_targets(recording.traces.T, target_slices, recording.behavior)

    model_path = embedder.save(art / "manifold_model.pkl")
    artifact = ManifoldArtifact(
        method=man_cfg.name,
        behavior_key=man_cfg.behavior_key,
        full_embedding=full_embedding,
        window_embedding=window_embedding,
        target_slices=target_slices,
        target_length=man_cfg.target_length,
        model_path=str(model_path),
        window_starts=np.asarray(connectivity.window_starts, dtype=np.int64),
        anchors=list(connectivity.anchors),
        provenance=ArtifactProvenance(
            identity=connectivity.provenance.identity,
            stage="manifold",
            source=connectivity.provenance.source,
            configuration_id=man_cfg.name,
            random_seed=man_cfg.seed,
            code_version=connectivity.provenance.code_version,
            fit_data_ids=(recording.identity.recording_id,),
            transform_data_ids=(recording.identity.recording_id,),
            axis_conventions={
                "full_embedding": "sample_or_target_window,latent_dimension",
                "window_embedding": "anchor,latent_dimension",
            },
            metadata={"upstream_stage": connectivity.provenance.stage},
        ),
        metadata={
            "n_timepoints": int(recording.n_timepoints),
            "window_count": int(len(target_slices)),
            "history_length": history_length,
            "anchors": list(connectivity.anchors),
            "target_alignment": "anchor_target_interval" if connectivity.anchors else "window_start_fallback",
            "full_embedding_indexing": ("target_window_end" if man_cfg.name == "bunddle" else "sample"),
            "full_embedding_sample_offset": man_cfg.target_length - 1 if man_cfg.name == "bunddle" else 0,
        },
    )
    save_artifact(artifact, art / "manifold.pkl")

    color = recording.behavior.get(man_cfg.behavior_key, np.zeros(recording.n_timepoints))
    if man_cfg.name == "bunddle":
        color = color[man_cfg.target_length - 1 :]
    if full_embedding.shape[1] >= 2:
        plot_manifold(
            full_embedding,
            color,
            f"behavioral manifold ({man_cfg.name})",
            fig / f"manifold_{man_cfg.name}.png",
        )
    logger.info(
        "Stage 5 done: %s full=%s aligned=%s target_length=%d",
        man_cfg.name,
        full_embedding.shape,
        window_embedding.shape,
        man_cfg.target_length,
    )


if __name__ == "__main__":
    main()
