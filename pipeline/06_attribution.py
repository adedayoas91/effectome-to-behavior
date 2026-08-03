"""Stage 7a: signed node attribution and candidate-driver qualification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from effectome.attribution import qualify_candidate_drivers
from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact

logger = logging.getLogger(__name__)


def _manifold_anchor_metadata(manifold):
    anchors = getattr(manifold, "anchors", None)
    if anchors:
        return anchors
    return manifold.metadata.get("anchors")


def _validate_manifold_handoff(series, manifold) -> None:
    if not series.anchors:
        return
    if len(manifold.target_slices) != len(series.anchors):
        raise ValueError("manifold target slices do not align 1:1 with connectivity anchors")
    expected_slices = [(int(anchor.target_start), int(anchor.target_stop)) for anchor in series.anchors]
    observed_slices = [(int(ts.start), int(ts.stop)) for ts in manifold.target_slices]
    if observed_slices != expected_slices:
        raise ValueError("manifold target slices are not aligned to connectivity target anchors")
    manifold_anchors = _manifold_anchor_metadata(manifold)
    if manifold_anchors is not None:
        if len(manifold_anchors) != len(series.anchors):
            raise ValueError("manifold anchor metadata does not align 1:1 with connectivity anchors")
        for series_anchor, manifold_anchor in zip(series.anchors, manifold_anchors, strict=True):
            if (
                series_anchor.recording_id != manifold_anchor.recording_id
                or series_anchor.target_start != manifold_anchor.target_start
                or series_anchor.target_stop != manifold_anchor.target_stop
            ):
                raise ValueError("manifold anchor metadata no longer matches connectivity anchors")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    at_cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "conf" / "attribution" / "default.yaml")
    at = cast(dict[str, Any], OmegaConf.to_container(at_cfg, resolve=True))

    series = load_artifact(art / "connectivity.pkl")
    community = load_artifact(art / "community.pkl")
    manifold = load_artifact(art / "manifold.pkl")
    _validate_manifold_handoff(series, manifold)

    report = {}
    for bkey in at["behavior_keys"]:
        if bkey not in series.behavior_per_window:
            logger.warning("behavior '%s' missing from windows; skipping", bkey)
            continue
        beh = series.behavior_per_window[bkey]
        report[bkey] = qualify_candidate_drivers(
            series,
            community,
            manifold,
            beh,
            bkey,
            lag=int(at["lag"]),
            n_folds=int(at["n_folds"]),
            embargo=int(at["embargo"]),
            seed=int(at["seed"]),
            matched_control_percentile=float(at["matched_control_percentile"]),
        )
    save_artifact(report, art / "attribution.pkl")
    logger.info("Stage 7a done: attribution for %d behavior targets", len(report))


if __name__ == "__main__":
    main()
