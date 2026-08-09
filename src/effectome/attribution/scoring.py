"""Signed attribution and preliminary predictive-candidate screening for Stage 6/7."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from effectome.data_module.schema import CommunitySeries, ConnectivitySeries, TemporalAnchor
from effectome.linking import anchor_group_labels, incremental_decode_behavior
from effectome.manifold import ManifoldArtifact


def signed_node_roles(series: ConnectivitySeries) -> dict[str, np.ndarray]:
    """Signed outgoing/incoming summaries per node and window."""
    w = np.asarray(series.matrices, dtype=float)
    pos = np.maximum(w, 0.0)
    neg = np.maximum(-w, 0.0)
    return {
        "outgoing_pos": pos.sum(axis=2),
        "outgoing_neg": neg.sum(axis=2),
        "incoming_pos": pos.sum(axis=1),
        "incoming_neg": neg.sum(axis=1),
        "net_outgoing": w.sum(axis=2),
        "net_incoming": w.sum(axis=1),
    }


def community_switch_rates(series: CommunitySeries) -> np.ndarray:
    """Fraction of transitions where each neuron's community label changes."""
    labels = np.asarray(series.labels)
    if labels.shape[0] < 2:
        return np.zeros(labels.shape[1], dtype=float)
    switching = labels[1:] != labels[:-1]
    valid = np.ones(labels.shape[0] - 1, dtype=bool)
    if series.boundary_indices is not None:
        for boundary in np.asarray(series.boundary_indices, dtype=int):
            if 0 < boundary < labels.shape[0]:
                valid[boundary - 1] = False
    if not np.any(valid):
        return np.zeros(labels.shape[1], dtype=float)
    return switching[valid].mean(axis=0)


@dataclass
class DriverScore:
    node: int
    predictive_gain: float
    outgoing_sign: float
    switch_rate: float
    control_pvalue: float
    matched_control_threshold: float
    is_candidate: bool
    candidate_stage: str = "screened_out"
    is_validated_driver: bool = False
    provenance: dict[str, object] = field(default_factory=dict)


@dataclass
class CandidateDriverResult:
    scores: list[DriverScore]
    behavior_key: str
    target_name: str
    matched_control_percentile: float
    n_controls: int
    status: str = "preliminary_predictive_screen"
    provenance: dict[str, object] = field(default_factory=dict)


def _baseline_features(behavior: np.ndarray, manifold_window: np.ndarray) -> np.ndarray:
    b = np.asarray(behavior, dtype=float).reshape(-1, 1)
    return np.concatenate([b, manifold_window], axis=1)


def _anchor_key(anchor: TemporalAnchor) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        anchor.dataset_id,
        anchor.recording_id,
        anchor.animal_id,
        anchor.session_id,
        anchor.segment_id,
    )


def _valid_lag_origins(series: ConnectivitySeries, lag: int) -> np.ndarray:
    if not series.anchors:
        return np.arange(series.n_windows - lag, dtype=int)
    origins: list[int] = []
    for origin in range(series.n_windows - lag):
        path = series.anchors[origin : origin + lag + 1]
        if all(_anchor_key(anchor) == _anchor_key(path[0]) for anchor in path) and not (
            any(anchor.gap_after for anchor in path[:-1]) or any(anchor.gap_before for anchor in path[1:])
        ):
            origins.append(origin)
    return np.asarray(origins, dtype=int)


def qualify_candidate_drivers(
    series: ConnectivitySeries,
    community: CommunitySeries,
    manifold: ManifoldArtifact,
    behavior: np.ndarray,
    behavior_key: str,
    lag: int = 1,
    n_folds: int = 5,
    embargo: int = 0,
    seed: int = 42,
    matched_control_percentile: float = 95.0,
) -> CandidateDriverResult:
    """Screen neurons for preliminary predictive-candidate status."""
    if lag <= 0:
        raise ValueError("lag must be positive for candidate-driver qualification")
    roles = signed_node_roles(series)
    switch_rate = community_switch_rates(community)
    net_out = roles["net_outgoing"]

    origins = _valid_lag_origins(series, lag)
    if origins.size < 4:
        raise ValueError("not enough within-segment positive-lag samples for attribution")
    targets = origins + lag
    baseline = _baseline_features(behavior[origins], manifold.window_embedding[origins])
    target = np.asarray(behavior[targets])
    split_anchors = [series.anchors[int(idx)] for idx in origins] if series.anchors else None
    outcome_anchors = [series.anchors[int(idx)] for idx in targets] if series.anchors else None
    groups: list[object] | None = (
        anchor_group_labels(split_anchors, group_by="recording").tolist() if split_anchors else None
    )
    rng = np.random.default_rng(seed)

    scores: list[DriverScore] = []
    n_nodes = net_out.shape[1]
    for node in range(n_nodes):
        feature = net_out[origins, node : node + 1]
        inc = incremental_decode_behavior(
            baseline,
            feature,
            target,
            n_folds=n_folds,
            seed=seed,
            embargo=embargo,
            anchors=split_anchors,
            groups=groups,
            outcome_anchors=outcome_anchors,
        )
        control_gains: list[float] = []
        sampled_controls: list[int] = []
        others = np.array([i for i in range(n_nodes) if i != node], dtype=int)
        draw_count = min(max(8, n_folds), len(others)) if len(others) else 0
        if draw_count > 0:
            sampled = rng.choice(others, size=draw_count, replace=False)
            sampled_controls = sampled.tolist()
            for other in sampled:
                ctrl = incremental_decode_behavior(
                    baseline,
                    net_out[origins, other : other + 1],
                    target,
                    n_folds=n_folds,
                    seed=seed,
                    embargo=embargo,
                    anchors=split_anchors,
                    groups=groups,
                    outcome_anchors=outcome_anchors,
                )
                control_gains.append(ctrl.gain)
        percentile = np.percentile(control_gains, matched_control_percentile) if control_gains else 0.0
        if control_gains:
            exceedances = int(np.sum(np.asarray(control_gains) >= inc.gain))
            pvalue = float((exceedances + 1) / (len(control_gains) + 1))
        else:
            pvalue = 1.0
        is_preliminary = bool(inc.gain > percentile and inc.gain > 0)
        scores.append(
            DriverScore(
                node=node,
                predictive_gain=float(inc.gain),
                outgoing_sign=float(np.sign(net_out[:, node].mean())),
                switch_rate=float(switch_rate[node]),
                control_pvalue=pvalue,
                matched_control_threshold=float(percentile),
                is_candidate=is_preliminary,
                candidate_stage=("preliminary_predictive_candidate" if is_preliminary else "screened_out"),
                is_validated_driver=False,
                provenance={
                    "screen": "predictive_gain_vs_matched_controls",
                    "matched_control_nodes": sampled_controls,
                    "claim_boundary": "screening_only_not_a_validated_driver",
                },
            )
        )

    scores.sort(key=lambda s: s.predictive_gain, reverse=True)
    return CandidateDriverResult(
        scores=scores,
        behavior_key=behavior_key,
        target_name="future_behavior",
        matched_control_percentile=float(matched_control_percentile),
        n_controls=max(0, len(scores) - 1),
        status="preliminary_predictive_screen",
        provenance={
            "lag": int(lag),
            "n_folds": int(n_folds),
            "embargo": int(embargo),
            "seed": int(seed),
            "claim_boundary": "candidate labels here are preliminary predictive screens",
            "split_mode": "anchor_aware_grouped_purged" if split_anchors else "purged_blocked",
        },
    )
