"""Stage 6: link effectome dynamics to behavior (decoding, MI, lead-lag).

This is the headline analysis: does the dynamic effectome carry behavioral information, and do
connectivity-state transitions precede behavioral transitions? Results are saved and printed.
"""

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
    lead_lag,
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

    conn_feat = connectivity_features(series)
    state_feat = state_features(states.labels, states.n_states)
    com_feat = community_features(community)

    report: dict = {"decoding": [], "association": {}, "lead_lag": {}}
    for bkey in lk["behavior_keys"]:
        if bkey not in series.behavior_per_window:
            logger.warning("behavior '%s' missing from windows; skipping", bkey)
            continue
        beh = series.behavior_per_window[bkey]
        beh = beh.astype(np.int64) if np.allclose(beh, np.round(beh)) else beh

        feature_sets = [("connectivity", conn_feat), ("state", state_feat), ("community", com_feat)]
        for name, feats in feature_sets:
            res = decode_behavior(feats, beh, name, bkey, n_folds=lk["n_folds"], seed=lk["seed"])
            report["decoding"].append(res)
            logger.info(
                "decode %s from %s: %.3f ± %.3f (%s)", bkey, name, res.mean_score, res.std_score, res.task
            )

        assoc = association_with_null(
            states.labels, beh, n_null=lk["n_null"], seed=lk["seed"], n_bins=lk["n_bins"]
        )
        report["association"][bkey] = assoc
        logger.info("state<->%s MI=%.4f z=%.2f p=%.4f", bkey, assoc.statistic, assoc.z_score, assoc.p_value)

        ll = lead_lag(states.labels, beh, max_lag=lk["max_lag"], n_bins=lk["n_bins"])
        report["lead_lag"][bkey] = ll
        logger.info(
            "lead-lag %s: best_lag=%d value=%.3f connectivity_leads=%s",
            bkey, ll.best_lag, ll.best_value, ll.connectivity_leads,
        )

    save_artifact(report, art / "linking.pkl")
    _print_summary(report)


def _print_summary(report: dict) -> None:
    print("\n=== Effectome-to-Behavior: linking summary ===")
    print("\nDecoding (cross-validated):")
    for r in report["decoding"]:
        print(
            f"  {r.behavior_key:<12} from {r.feature_set:<12} "
            f"{r.mean_score:6.3f} ± {r.std_score:.3f}  [{r.task}]"
        )
    print("\nState↔behavior association vs time-shuffle null:")
    for bkey, a in report["association"].items():
        print(f"  {bkey:<12} MI={a.statistic:.4f}  z={a.z_score:5.2f}  p={a.p_value:.4f}")
    print("\nLead-lag (positive lag = connectivity leads behavior):")
    for bkey, ll in report["lead_lag"].items():
        print(
            f"  {bkey:<12} best_lag={ll.best_lag:+d}  xcorr={ll.best_value:.3f}  "
            f"leads={ll.connectivity_leads}"
        )
    print()


if __name__ == "__main__":
    main()
