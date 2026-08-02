"""Signed node/group attribution and candidate-driver qualification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from effectome.data_module.schema import CommunitySeries, ConnectivitySeries
from effectome.linking import incremental_decode_behavior
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
    return (labels[1:] != labels[:-1]).mean(axis=0)


@dataclass
class DriverScore:
    node: int
    predictive_gain: float
    outgoing_sign: float
    switch_rate: float
    control_pvalue: float
    is_candidate: bool


@dataclass
class CandidateDriverResult:
    scores: list[DriverScore]
    behavior_key: str
    target_name: str
    matched_control_percentile: float
    n_controls: int


def _baseline_features(behavior: np.ndarray, manifold_window: np.ndarray) -> np.ndarray:
    b = np.asarray(behavior, dtype=float).reshape(-1, 1)
    return np.concatenate([b, manifold_window], axis=1)


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
    """Score neurons by incremental predictive value of their signed outgoing influence."""
    if lag <= 0:
        raise ValueError("lag must be positive for candidate-driver qualification")
    roles = signed_node_roles(series)
    switch_rate = community_switch_rates(community)
    net_out = roles["net_outgoing"]

    baseline = _baseline_features(behavior[:-lag], manifold.window_embedding[:-lag])
    target = np.asarray(behavior[lag:])
    rng = np.random.default_rng(seed)

    scores: list[DriverScore] = []
    n_nodes = net_out.shape[1]
    for node in range(n_nodes):
        feature = net_out[:-lag, node : node + 1]
        inc = incremental_decode_behavior(
            baseline,
            feature,
            target,
            n_folds=n_folds,
            seed=seed,
            embargo=embargo,
        )
        control_gains = []
        others = np.array([i for i in range(n_nodes) if i != node], dtype=int)
        draw_count = min(max(8, n_folds), len(others)) if len(others) else 0
        if draw_count > 0:
            sampled = rng.choice(others, size=draw_count, replace=False)
            for other in sampled:
                ctrl = incremental_decode_behavior(
                    baseline,
                    net_out[:-lag, other : other + 1],
                    target,
                    n_folds=n_folds,
                    seed=seed,
                    embargo=embargo,
                )
                control_gains.append(ctrl.gain)
        percentile = np.percentile(control_gains, matched_control_percentile) if control_gains else 0.0
        pvalue = float((np.asarray(control_gains) >= inc.gain).mean()) if control_gains else 1.0
        scores.append(
            DriverScore(
                node=node,
                predictive_gain=float(inc.gain),
                outgoing_sign=float(np.sign(net_out[:, node].mean())),
                switch_rate=float(switch_rate[node]),
                control_pvalue=pvalue,
                is_candidate=bool(inc.gain > percentile and inc.gain > 0),
            )
        )

    scores.sort(key=lambda s: s.predictive_gain, reverse=True)
    return CandidateDriverResult(
        scores=scores,
        behavior_key=behavior_key,
        target_name="future_behavior",
        matched_control_percentile=float(matched_control_percentile),
        n_controls=max(0, len(scores) - 1),
    )
