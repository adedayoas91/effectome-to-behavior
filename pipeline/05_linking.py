"""Stage 6: link effectome dynamics to behavior, manifold dynamics, and community reconfiguration."""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from effectome.linking import (
    association_with_null,
    community_features,
    connectivity_features,
    decode_behavior,
    incremental_decode_behavior,
    lead_lag,
    manifold_speed,
    state_features,
)
from effectome.utils import set_seed
from effectome.utils.io import load_artifact, save_artifact

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    lk = OmegaConf.to_container(cfg.linking, resolve=True)

    series = load_artifact(art / "connectivity.pkl")
    states = load_artifact(art / "graph_states.pkl")
    community = load_artifact(art / "community.pkl")
    manifold = load_artifact(art / "manifold.pkl")

    conn_feat = connectivity_features(series)
    state_feat = state_features(states.labels, states.n_states)
    com_feat = community_features(community)
    manifold_dyn = manifold_speed(manifold.window_embedding).reshape(-1, 1)

    report: dict = {"decoding": [], "association": {}, "lead_lag": {}, "incremental": {}}
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
        )
        report["association"][bkey] = assoc

        ll = lead_lag(states.labels, beh, max_lag=lk["max_lag"], n_bins=lk["n_bins"], target_name=bkey)
        report["lead_lag"][bkey] = ll

        baseline = np.concatenate([state_feat, manifold.window_embedding], axis=1)
        inc = incremental_decode_behavior(
            baseline,
            com_feat,
            beh,
            n_folds=lk["n_folds"],
            seed=lk["seed"],
            embargo=lk["embargo"],
        )
        report["incremental"][bkey] = inc

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
        print(
            f"  {bkey:<12} MI={a.statistic:.4f}  z={a.z_score:5.2f}  p={a.p_value:.4f}  [{a.null_kind}]"
        )
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
    print()


if __name__ == "__main__":
    main()
