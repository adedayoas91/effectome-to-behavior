"""Stage 6: link effectome dynamics to behavior, manifold dynamics, and community reconfiguration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from effectome.linking import (
    anchor_group_labels,
    association_with_null,
    community_features,
    connectivity_features,
    decode_behavior,
    incremental_decode_behavior,
    lead_lag,
    manifold_speed,
    state_features,
    valid_positive_lag_origins,
)
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


def _split_inputs_for_linking(series, manifold, group_by: str):
    _validate_manifold_handoff(series, manifold)
    if not series.anchors:
        return None, None
    return series.anchors, anchor_group_labels(series.anchors, group_by=group_by)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    lk = cast(dict[str, Any], OmegaConf.to_container(cfg.linking, resolve=True))

    series = load_artifact(art / "connectivity.pkl")
    states = load_artifact(art / "graph_states.pkl")
    community = load_artifact(art / "community.pkl")
    manifold = load_artifact(art / "manifold.pkl")
    anchors, groups = _split_inputs_for_linking(series, manifold, str(lk.get("group_by", "recording")))

    conn_feat = connectivity_features(series)
    state_feat = state_features(states.labels, states.n_states)
    com_feat = community_features(community)
    manifold_dyn = manifold_speed(manifold.window_embedding, groups=groups).reshape(-1, 1)

    report: dict = {
        "decoding": [],
        "association": {},
        "lead_lag": {},
        "incremental": {},
        "positive_lag_incremental": {},
        "splitter": {
            "mode": "anchor_aware_grouped_purged" if anchors is not None else "purged_blocked",
            "group_by": str(lk.get("group_by", "recording")) if anchors is not None else None,
            "embargo": int(lk["embargo"]),
        },
    }
    for bkey in lk["behavior_keys"]:
        if bkey not in series.behavior_per_window:
            logger.warning("behavior '%s' missing from windows; skipping", bkey)
            continue
        beh = series.behavior_per_window[bkey]
        beh = beh.astype(np.int64) if np.allclose(beh, np.round(beh)) else beh.astype(float)

        feature_sets = [
            ("connectivity", conn_feat),
            ("state", state_feat),
            ("community", com_feat),
            ("manifold_dynamics", manifold_dyn),
        ]
        for name, feats in feature_sets:
            res = decode_behavior(
                feats,
                beh,
                name,
                bkey,
                n_folds=lk["n_folds"],
                seed=lk["seed"],
                embargo=lk["embargo"],
                anchors=anchors,
                groups=groups,
            )
            report["decoding"].append(res)

        assoc = association_with_null(
            states.labels,
            beh,
            n_null=lk["n_null"],
            seed=lk["seed"],
            n_bins=lk["n_bins"],
            null_kind=lk["null_kind"],
            block_length=lk["block_length"],
            groups=groups,
        )
        report["association"][bkey] = assoc

        ll = lead_lag(
            states.labels,
            beh,
            max_lag=lk["max_lag"],
            n_bins=lk["n_bins"],
            target_name=bkey,
            groups=groups,
        )
        report["lead_lag"][bkey] = ll

        baseline = np.concatenate([state_feat, manifold.window_embedding], axis=1)
        inc = incremental_decode_behavior(
            baseline,
            com_feat,
            beh,
            n_folds=lk["n_folds"],
            seed=lk["seed"],
            embargo=lk["embargo"],
            anchors=anchors,
            groups=groups,
        )
        report["incremental"][bkey] = inc

        positive_lag = int(lk.get("positive_lag", 1))
        community_mode = getattr(community, "mode", None)
        if community_mode != "prospective":
            logger.warning(
                "community mode is %r; positive-lag community prediction requires prospective mode",
                community_mode,
            )
            report["positive_lag_incremental"][bkey] = {
                "lag": positive_lag,
                "status": "invalid_future_aware_community_features",
                "community_mode": community_mode,
                "claim_boundary": "not_eligible_for_prospective_interpretation",
            }
            continue
        origins = valid_positive_lag_origins(
            series.n_windows,
            positive_lag,
            anchors=series.anchors if series.anchors else None,
        )
        if origins.size < 4:
            logger.warning(
                "behavior '%s' has too few within-segment samples at lag %d; skipping positive-lag model",
                bkey,
                positive_lag,
            )
            continue
        lag_anchors = [series.anchors[int(idx)] for idx in origins] if series.anchors else None
        lag_groups: list[object] | None = (
            anchor_group_labels(
                lag_anchors, group_by=str(lk.get("group_by", "recording"))
            ).tolist()
            if lag_anchors
            else None
        )
        future_behavior = beh[origins + positive_lag]
        current_behavior = beh[origins].astype(float).reshape(-1, 1)
        predictive_baseline = np.concatenate(
            [current_behavior, manifold.window_embedding[origins], conn_feat[origins]],
            axis=1,
        )
        prospective_community = com_feat[origins]
        positive_inc = incremental_decode_behavior(
            predictive_baseline,
            prospective_community,
            future_behavior,
            n_folds=lk["n_folds"],
            seed=lk["seed"],
            embargo=lk["embargo"],
            anchors=lag_anchors,
            groups=lag_groups,
        )
        report["positive_lag_incremental"][bkey] = {
            "lag": positive_lag,
            "result": positive_inc,
            "baseline": "current_behavior_plus_manifold_plus_raw_effectome",
            "community_mode": community_mode,
            "claim_boundary": "predictive_increment_only_not_causal_mediation",
        }

    save_artifact(report, art / "linking.pkl")
    _print_summary(report)


def _print_summary(report: dict) -> None:
    print("\n=== Effectome-to-Behavior: linking summary ===")
    print("\nDecoding (purged blocked CV):")
    for r in report["decoding"]:
        print(
            f"  {r.behavior_key:<12} from {r.feature_set:<18} "
            f"{r.mean_score:6.3f} ± {r.std_score:.3f}  [{r.task}; embargo={r.embargo}]"
        )
    print("\nState↔behavior association vs temporal null:")
    for bkey, a in report["association"].items():
        print(f"  {bkey:<12} MI={a.statistic:.4f}  z={a.z_score:5.2f}  p={a.p_value:.4f}  [{a.null_kind}]")
    print("\nLead-lag (positive lag = connectivity leads target):")
    for bkey, ll in report["lead_lag"].items():
        print(
            f"  {bkey:<12} best_lag={ll.best_lag:+d}  "
            f"xcorr={ll.best_value:.3f}  leads={ll.connectivity_leads}"
        )
    print("\nIncremental community gain over state+manifold baseline:")
    for bkey, inc in report["incremental"].items():
        print(
            f"  {bkey:<12} baseline={inc.baseline_score:.3f}  full={inc.full_score:.3f}  gain={inc.gain:.3f}"
        )
    print("\nPositive-lag community gain over current behavior+manifold+effectome baseline:")
    for bkey, payload in report["positive_lag_incremental"].items():
        if "result" not in payload:
            print(f"  {bkey:<12} skipped [{payload['status']}]")
            continue
        inc = payload["result"]
        print(
            f"  {bkey:<12} lag={payload['lag']}  baseline={inc.baseline_score:.3f}  "
            f"full={inc.full_score:.3f}  gain={inc.gain:.3f}"
        )
    print()


if __name__ == "__main__":
    main()
