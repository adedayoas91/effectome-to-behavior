"""Stage 6: link effectome dynamics to behavior, manifold dynamics, and community reconfiguration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from effectome.linking import (
    DependencySupport,
    activity_magnitude_features,
    anchor_continuity_labels,
    anchor_group_labels,
    association_with_null,
    combined_continuity_groups,
    community_features,
    community_reconfiguration_features,
    connectivity_features,
    decode_behavior,
    future_manifold_displacement,
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
    recording = load_artifact(art / "recording.pkl")
    anchors, groups = _split_inputs_for_linking(series, manifold, str(lk.get("group_by", "recording")))
    continuity_groups = (
        anchor_continuity_labels(
            series.anchors,
            group_by=str(lk.get("group_by", "recording")),
        )
        if series.anchors
        else None
    )
    preprocess_dependency = recording.metadata.get("preprocess_dependency", {})
    if (
        preprocess_dependency.get("kind") == "global"
        and not bool(lk.get("allow_global_preprocessing", False))
    ):
        raise ValueError(
            "recording-global preprocessing cannot support prospective linking; rerun with "
            "fold/window-fitted transforms or set linking.allow_global_preprocessing=true "
            "for a clearly labelled retrospective sensitivity analysis"
        )
    dependency_support = DependencySupport(
        lag_extension=int(lk.get("lag_extension", 0)),
        preprocessing_past=max(
            int(lk.get("preprocessing_past_support", 0)),
            int(preprocess_dependency.get("past_support", 0)),
        ),
        preprocessing_future=max(
            int(lk.get("preprocessing_future_support", 0)),
            int(preprocess_dependency.get("future_support", 0)),
        ),
    )

    conn_feat = connectivity_features(series)
    state_feat = state_features(states.labels, states.n_states)
    com_feat = community_features(community)
    reconfiguration_feat = community_reconfiguration_features(community)
    manifold_prospective = bool(manifold.metadata.get("prospective_eligible", False))
    fold_ids = manifold.metadata.get("cross_fit_fold_ids") if manifold_prospective else None
    manifold_groups = combined_continuity_groups(
        continuity_groups,
        None if fold_ids is None else np.asarray(fold_ids),
        series.n_windows,
    )
    manifold_dyn = manifold_speed(
        manifold.window_embedding,
        groups=manifold_groups,
    ).reshape(-1, 1)
    activity_feat = (
        activity_magnitude_features(recording.traces.T, list(series.anchors))
        if series.anchors
        else np.empty((series.n_windows, 0), dtype=float)
    )

    report: dict = {
        "decoding": [],
        "association": {},
        "lead_lag": {},
        "lead_lag_activity_adjusted": {},
        "incremental": {},
        "positive_lag_incremental": {},
        "effectome_to_future_manifold": {},
        "splitter": {
            "mode": "anchor_aware_grouped_purged" if anchors is not None else "purged_blocked",
            "group_by": str(lk.get("group_by", "recording")) if anchors is not None else None,
            "continuity_boundaries": "recording_hard_gap_and_sparse_bad_frame",
            "embargo": int(lk["embargo"]),
            "raw_dependency_support": {
                "lag_extension": dependency_support.lag_extension,
                "preprocessing_past": dependency_support.preprocessing_past,
                "preprocessing_future": dependency_support.preprocessing_future,
                "preprocess_kind": preprocess_dependency.get("kind", "undeclared"),
            },
        },
        "manifold_feature_mode": manifold.metadata.get(
            "window_embedding_mode", "undeclared_retrospective"
        ),
        "manifold_derivative_boundaries": (
            "recording_hard_gap_sparse_bad_frame_and_cross_fit_fold"
        ),
    }
    switching_count = np.rint(
        reconfiguration_feat[:, 0] * float(community.labels.shape[1])
    ).astype(int)
    report["manifold_alignment"] = {
        "state_to_manifold_speed_lead_lag": lead_lag(
            states.labels,
            manifold_dyn[:, 0],
            max_lag=lk["max_lag"],
            n_bins=lk["n_bins"],
            target_name="manifold_speed",
            groups=manifold_groups,
        ),
        "state_to_manifold_speed_association": association_with_null(
            states.labels,
            manifold_dyn[:, 0],
            n_null=lk["n_null"],
            seed=lk["seed"],
            n_bins=lk["n_bins"],
            null_kind=lk["null_kind"],
            block_length=lk["block_length"],
            groups=manifold_groups,
        ),
        "switching_count_to_manifold_speed_association": association_with_null(
            switching_count,
            manifold_dyn[:, 0],
            n_null=lk["n_null"],
            seed=lk["seed"] + 1,
            n_bins=lk["n_bins"],
            null_kind=lk["null_kind"],
            block_length=lk["block_length"],
            groups=manifold_groups,
        ),
        "mode": "prospective_cross_fitted" if manifold_prospective else "retrospective_descriptive",
        "claim_boundary": (
            "cross_fitted_current_state_alignment"
            if manifold_prospective
            else "full_fit_manifold_correspondence_only_not_prospective_or_causal"
        ),
    }
    positive_lag = int(lk.get("positive_lag", 1))
    if not manifold_prospective:
        report["effectome_to_future_manifold"] = {
            "lag": positive_lag,
            "status": "invalid_non_cross_fitted_manifold",
            "claim_boundary": "retrospective_manifold_not_eligible_for_prospective_prediction",
        }
    else:
        manifold_origins = valid_positive_lag_origins(
            series.n_windows,
            positive_lag,
            anchors=series.anchors if series.anchors else None,
        )
        if manifold_groups is not None and manifold_origins.size:
            group_arr = np.asarray(manifold_groups, dtype=object)
            within_chart = (
                group_arr[manifold_origins]
                == group_arr[manifold_origins + positive_lag]
            )
            manifold_origins = manifold_origins[within_chart]
        if manifold_origins.size < 4:
            report["effectome_to_future_manifold"] = {
                "lag": positive_lag,
                "status": "too_few_within_segment_within_chart_transitions",
            }
        else:
            _, future_manifold_distance = future_manifold_displacement(
                manifold.window_embedding,
                manifold_origins,
                lag=positive_lag,
                groups=manifold_groups,
            )
            manifold_lag_anchors = (
                [series.anchors[int(idx)] for idx in manifold_origins]
                if series.anchors
                else None
            )
            manifold_future_anchors = (
                [series.anchors[int(idx + positive_lag)] for idx in manifold_origins]
                if series.anchors
                else None
            )
            manifold_lag_groups = (
                np.asarray(manifold_groups, dtype=object)[manifold_origins].tolist()
                if manifold_groups is not None
                else None
            )
            raw_effectome_result = decode_behavior(
                conn_feat[manifold_origins],
                future_manifold_distance,
                "connectivity",
                "future_manifold_displacement",
                n_folds=lk["n_folds"],
                seed=lk["seed"],
                embargo=lk["embargo"],
                anchors=manifold_lag_anchors,
                groups=manifold_lag_groups,
                dependency_support=dependency_support,
                outcome_anchors=manifold_future_anchors,
            )
            manifold_report: dict[str, Any] = {
                "lag": positive_lag,
                "n_transitions": int(manifold_origins.size),
                "outcome": "euclidean_norm_of_z_future_minus_z_current",
                "raw_effectome_decode": raw_effectome_result,
                "coordinate_contract": (
                    "magnitude_is_rotation_reflection_invariant; transitions never cross folds, "
                    "recordings, or declared gaps"
                ),
                "claim_boundary": "prospective_prediction_not_interventional_causation",
            }
            if getattr(community, "mode", None) == "prospective":
                manifold_baseline = np.concatenate(
                    [
                        manifold_dyn[manifold_origins],
                        activity_feat[manifold_origins],
                        conn_feat[manifold_origins],
                    ],
                    axis=1,
                )
                manifold_report["community_reconfiguration_increment"] = (
                    incremental_decode_behavior(
                        manifold_baseline,
                        reconfiguration_feat[manifold_origins],
                        future_manifold_distance,
                        n_folds=lk["n_folds"],
                        seed=lk["seed"],
                        embargo=lk["embargo"],
                        anchors=manifold_lag_anchors,
                        groups=manifold_lag_groups,
                        dependency_support=dependency_support,
                        outcome_anchors=manifold_future_anchors,
                    )
                )
                manifold_report["community_feature_contract"] = (
                    "switching_fraction_plus_neuron_resolved_allegiance_changes"
                )
            else:
                manifold_report["community_reconfiguration_status"] = (
                    "invalid_future_aware_community_features"
                )
            report["effectome_to_future_manifold"] = manifold_report
    if manifold_prospective:
        report["lead_lag"]["manifold_speed"] = lead_lag(
            states.labels,
            manifold_dyn[:, 0],
            max_lag=lk["max_lag"],
            n_bins=lk["n_bins"],
            target_name="manifold_speed",
            groups=manifold_groups,
        )
        report["lead_lag_activity_adjusted"]["manifold_speed"] = lead_lag(
            states.labels,
            manifold_dyn[:, 0],
            max_lag=lk["max_lag"],
            n_bins=lk["n_bins"],
            target_name="manifold_speed",
            groups=manifold_groups,
            controls=activity_feat if activity_feat.shape[1] else None,
        )
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
        ]
        if manifold_prospective:
            feature_sets.append(("manifold_dynamics", manifold_dyn))
        if activity_feat.shape[1]:
            feature_sets.append(("activity_magnitude", activity_feat))
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
                dependency_support=dependency_support,
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
            groups=continuity_groups,
        )
        report["association"][bkey] = assoc

        ll = lead_lag(
            states.labels,
            beh,
            max_lag=lk["max_lag"],
            n_bins=lk["n_bins"],
            target_name=bkey,
            groups=continuity_groups,
        )
        report["lead_lag"][bkey] = ll
        report["lead_lag_activity_adjusted"][bkey] = lead_lag(
            states.labels,
            beh,
            max_lag=lk["max_lag"],
            n_bins=lk["n_bins"],
            target_name=bkey,
            groups=continuity_groups,
            controls=activity_feat if activity_feat.shape[1] else None,
        )

        baseline_parts = [state_feat, activity_feat]
        if manifold_prospective:
            baseline_parts.append(manifold_dyn)
        baseline = np.concatenate(baseline_parts, axis=1)
        inc = incremental_decode_behavior(
            baseline,
            com_feat,
            beh,
            n_folds=lk["n_folds"],
            seed=lk["seed"],
            embargo=lk["embargo"],
            anchors=anchors,
            groups=groups,
            dependency_support=dependency_support,
        )
        report["incremental"][bkey] = inc

        community_mode = getattr(community, "mode", None)
        if not manifold_prospective:
            report["positive_lag_incremental"][bkey] = {
                "lag": positive_lag,
                "status": "invalid_non_cross_fitted_manifold",
                "community_mode": community_mode,
                "claim_boundary": "not_eligible_for_prospective_interpretation",
            }
            continue
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
        future_anchors = (
            [series.anchors[int(idx + positive_lag)] for idx in origins]
            if series.anchors
            else None
        )
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
            [
                current_behavior,
                manifold_dyn[origins],
                activity_feat[origins],
                conn_feat[origins],
            ],
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
            dependency_support=dependency_support,
            outcome_anchors=future_anchors,
        )
        report["positive_lag_incremental"][bkey] = {
            "lag": positive_lag,
            "result": positive_inc,
            "baseline": "current_behavior_plus_manifold_speed_plus_activity_plus_raw_effectome",
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
