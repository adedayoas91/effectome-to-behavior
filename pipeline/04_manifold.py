"""Stage 5: learn a behavioral manifold and align target-window latent codes to anchors."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from effectome.data_module.schema import ArtifactProvenance
from effectome.linking import DependencySupport, anchor_group_labels, purged_blocked_splits
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


def _cross_fitted_window_embedding(
    recording,
    connectivity,
    target_slices: list[TargetSlice],
    man_cfg: ManifoldConfig,
    linking_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Fit classical manifold transforms without exposing held-out raw dependencies."""
    if not connectivity.anchors:
        raise ValueError("cross-fitted manifold learning requires typed temporal anchors")
    if man_cfg.name != "classical" or man_cfg.method != "pca":
        raise ValueError(
            "prospective cross-fitting currently requires classical PCA because fold-specific "
            "UMAP/learned coordinates do not share an identified coordinate system; use those "
            "methods only as retrospective sensitivity analyses"
        )
    preprocess_dependency = recording.metadata.get("preprocess_dependency", {})
    if preprocess_dependency.get("kind") == "global":
        raise ValueError(
            "cross-fitting cannot repair recording-global preprocessing; rerun preprocessing "
            "without global detrending/z-scoring/deconvolution"
        )
    dependency_support = DependencySupport(
        lag_extension=int(linking_cfg.get("lag_extension", 0)),
        preprocessing_past=max(
            int(linking_cfg.get("preprocessing_past_support", 0)),
            int(preprocess_dependency.get("past_support", 0)),
        ),
        preprocessing_future=max(
            int(linking_cfg.get("preprocessing_future_support", 0)),
            int(preprocess_dependency.get("future_support", 0)),
        ),
    )
    groups = anchor_group_labels(
        connectivity.anchors,
        group_by=str(linking_cfg.get("group_by", "recording")),
    )
    splits = purged_blocked_splits(
        connectivity.n_windows,
        man_cfg.n_folds,
        embargo=int(linking_cfg.get("embargo", 0)),
        anchors=connectivity.anchors,
        groups=groups,
        dependency_support=dependency_support,
    )
    embeddings = np.full(
        (connectivity.n_windows, man_cfg.n_dims),
        np.nan,
        dtype=np.float32,
    )
    fold_ids = np.full(connectivity.n_windows, -1, dtype=int)
    fit_sample_counts: list[int] = []
    neural = np.asarray(recording.traces.T)
    for fold_id, (train, test) in enumerate(splits):
        fit_mask = np.zeros(recording.n_timepoints, dtype=bool)
        for idx in train:
            anchor = connectivity.anchors[int(idx)]
            start = max(0, anchor.context_start - dependency_support.total_past)
            stop = min(
                recording.n_timepoints,
                anchor.context_stop + dependency_support.preprocessing_future,
            )
            fit_mask[start:stop] = True
        fit_indices = np.flatnonzero(fit_mask)
        if fit_indices.size <= man_cfg.n_dims:
            raise ValueError("too few dependency-purged samples to fit the manifold fold")
        fold_behavior = {
            key: np.asarray(values)[fit_indices] for key, values in recording.behavior.items()
        }
        fold_embedder = ManifoldFactory(man_cfg)
        fold_embedder.fit(neural[fit_indices], fold_behavior)
        test_slices = [target_slices[int(idx)] for idx in test]
        embeddings[test] = fold_embedder.transform_targets(
            neural,
            test_slices,
            recording.behavior,
        )
        fold_ids[test] = fold_id
        fit_sample_counts.append(int(fit_indices.size))
    if np.any(fold_ids < 0) or not np.all(np.isfinite(embeddings)):
        raise RuntimeError("cross-fitted manifold did not produce every held-out anchor")
    return embeddings, fold_ids, fit_sample_counts


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    fig = Path(cfg.paths.figures)

    recording = load_artifact(art / "recording.pkl")
    man_cfg_data = cast(dict[str, Any], OmegaConf.to_container(cfg.manifold, resolve=True))
    man_cfg = ManifoldConfig(**man_cfg_data)
    connectivity = load_artifact(art / "connectivity.pkl")
    history_length = int(
        cfg.windowing.history_length
        if cfg.windowing.mode == "temporal" and cfg.windowing.history_length is not None
        else cfg.windowing.length
    )
    target_slices = _target_slices_for_connectivity(connectivity, history_length, man_cfg.target_length)

    # The full-data model is retained only for retrospective visualization and
    # serialization. Downstream predictive features use the fold-specific path.
    embedder = ManifoldFactory(man_cfg)
    embedder.fit(recording.traces.T, recording.behavior)
    full_embedding = embedder.transform(recording.traces.T)
    fold_ids = None
    fit_sample_counts: list[int] = []
    if man_cfg.cross_fit:
        linking_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.linking, resolve=True))
        window_embedding, fold_ids, fit_sample_counts = _cross_fitted_window_embedding(
            recording,
            connectivity,
            target_slices,
            man_cfg,
            linking_cfg,
        )
        window_embedding_mode = "cross_fitted"
    else:
        window_embedding = embedder.transform_targets(
            recording.traces.T, target_slices, recording.behavior
        )
        window_embedding_mode = "retrospective_full_fit"

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
            "window_embedding_mode": window_embedding_mode,
            "cross_fit_fold_ids": fold_ids,
            "cross_fit_fit_sample_counts": fit_sample_counts,
            "prospective_eligible": bool(man_cfg.cross_fit),
            "fold_coordinate_contract": (
                "ordered deterministic PCA coordinates; derivatives must break at fold boundaries"
                if man_cfg.cross_fit
                else "single full-fit coordinate system for retrospective use"
            ),
            "full_embedding_claim_boundary": "retrospective_visualization_only",
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
