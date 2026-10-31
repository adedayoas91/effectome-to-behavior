"""Signed attribution and preliminary predictive-candidate screening for Stage 6/7."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from effectome.data_module.schema import CommunitySeries, ConnectivitySeries, TemporalAnchor
from effectome.linking import (
    anchor_group_labels,
    incremental_decode_behavior,
    valid_positive_lag_origins,
)
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
    fit_scope_id: str | None = None
    candidate_nodes: tuple[int, ...] = ()
    category_vocabulary: tuple[object, ...] = ()
    provenance: dict[str, object] = field(default_factory=dict)


def _baseline_features(behavior: np.ndarray, manifold_window: np.ndarray) -> np.ndarray:
    values = np.asarray(behavior)
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        categories = np.unique(values)
        b = (values[:, None] == categories[None, :]).astype(float)
    else:
        b = np.asarray(values, dtype=float).reshape(-1, 1)
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
    return valid_positive_lag_origins(
        series.n_windows,
        lag,
        series.anchors if series.anchors else None,
    )


def _validated_ownership_indices(
    n_windows: int,
    fit_indices: np.ndarray | None,
    evaluation_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    fit = np.arange(n_windows, dtype=int) if fit_indices is None else np.asarray(fit_indices)
    evaluation = (
        np.array([], dtype=int)
        if evaluation_indices is None
        else np.asarray(evaluation_indices)
    )
    for name, indices in (("fit_indices", fit), ("evaluation_indices", evaluation)):
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= n_windows):
            raise ValueError(f"{name} are outside the available windows")
        if len(indices) != len(np.unique(indices)):
            raise ValueError(f"{name} must be unique")
    if np.intersect1d(fit, evaluation).size:
        raise ValueError("fit and evaluation ownership must be disjoint")
    return fit.astype(int, copy=False), evaluation.astype(int, copy=False)


def _artifact_fit_scope_id(artifact: object) -> str | None:
    provenance = getattr(artifact, "provenance", None)
    provenance_scope = getattr(provenance, "fit_scope_id", None)
    if provenance_scope:
        return str(provenance_scope)
    for attribute in ("diagnostics", "metadata"):
        values = getattr(artifact, attribute, {})
        if isinstance(values, dict) and values.get("fit_scope_id"):
            return str(values["fit_scope_id"])
    return None


def _training_switch_rates(
    community: CommunitySeries,
    fit_indices: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(community.labels)
    fit_set = set(int(index) for index in fit_indices)
    boundaries = (
        set()
        if community.boundary_indices is None
        else set(int(index) for index in np.asarray(community.boundary_indices, dtype=int))
    )
    valid_transitions = [
        index
        for index in range(1, labels.shape[0])
        if index in fit_set and index - 1 in fit_set and index not in boundaries
    ]
    if not valid_transitions:
        return np.zeros(labels.shape[1], dtype=float)
    current = labels[np.asarray(valid_transitions, dtype=int)]
    previous = labels[np.asarray(valid_transitions, dtype=int) - 1]
    return np.mean(current != previous, axis=0)


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
    fit_indices: np.ndarray | None = None,
    evaluation_indices: np.ndarray | None = None,
    fit_scope_id: str | None = None,
    require_feature_ownership: bool = False,
) -> CandidateDriverResult:
    """Screen neurons for preliminary status using only the owned training partition."""
    if lag <= 0:
        raise ValueError("lag must be positive for candidate-driver qualification")
    owned_fit, owned_evaluation = _validated_ownership_indices(
        series.n_windows,
        fit_indices,
        evaluation_indices,
    )
    series_scope = _artifact_fit_scope_id(series)
    community_scope = _artifact_fit_scope_id(community)
    manifold_scope = _artifact_fit_scope_id(manifold)
    if require_feature_ownership:
        if fit_scope_id is None:
            raise ValueError("feature ownership requires an explicit fit_scope_id")
        if (
            series_scope != fit_scope_id
            or community_scope != fit_scope_id
            or manifold_scope != fit_scope_id
        ):
            raise ValueError(
                "connectivity, community, and manifold fit_scope_id values must match "
                "the requested outer scope"
            )
    roles = signed_node_roles(series)
    switch_rate = _training_switch_rates(community, owned_fit)
    net_out = roles["net_outgoing"]

    all_origins = _valid_lag_origins(series, lag)
    fit_set = set(int(index) for index in owned_fit)
    origins = np.asarray(
        [
            origin
            for origin in all_origins
            if int(origin) in fit_set and int(origin + lag) in fit_set
        ],
        dtype=int,
    )
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
                outgoing_sign=float(np.sign(net_out[owned_fit, node].mean())),
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
    candidate_nodes = tuple(score.node for score in scores if score.is_candidate)
    behavior_values = np.asarray(behavior)[owned_fit]
    category_vocabulary: tuple[object, ...] = ()
    if np.issubdtype(behavior_values.dtype, np.integer) or np.issubdtype(
        behavior_values.dtype,
        np.bool_,
    ):
        category_vocabulary = tuple(np.unique(behavior_values).tolist())
    return CandidateDriverResult(
        scores=scores,
        behavior_key=behavior_key,
        target_name="future_behavior",
        matched_control_percentile=float(matched_control_percentile),
        n_controls=max(0, len(scores) - 1),
        status="preliminary_predictive_screen",
        fit_scope_id=fit_scope_id,
        candidate_nodes=candidate_nodes,
        category_vocabulary=category_vocabulary,
        provenance={
            "lag": int(lag),
            "n_folds": int(n_folds),
            "embargo": int(embargo),
            "seed": int(seed),
            "claim_boundary": "candidate labels here are preliminary predictive screens",
            "split_mode": "anchor_aware_grouped_purged" if split_anchors else "purged_blocked",
            "fit_scope_id": fit_scope_id,
            "fit_indices": owned_fit.tolist(),
            "evaluation_indices": owned_evaluation.tolist(),
            "candidate_nodes": list(candidate_nodes),
            "category_vocabulary": list(category_vocabulary),
            "selection_frozen_in_training": True,
            "feature_ownership_verified": bool(
                fit_scope_id is not None
                and series_scope == fit_scope_id
                and community_scope == fit_scope_id
                and manifold_scope == fit_scope_id
            ),
            "community_fit_scope_id": community_scope,
        },
    )
