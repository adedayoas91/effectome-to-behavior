"""Dependency-light surrogate validation and virtual perturbation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor
from effectome.linking.decoding import anchor_group_labels, purged_blocked_splits
from effectome.manifold import ManifoldArtifact


def _flatten_connectivity(series: ConnectivitySeries) -> np.ndarray:
    return series.matrices.reshape(series.n_windows, -1)


def _anchor_source_key(
    anchor: TemporalAnchor,
) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        anchor.dataset_id,
        anchor.recording_id,
        anchor.animal_id,
        anchor.session_id,
        anchor.segment_id,
    )


def _design_origins(series: ConnectivitySeries, lag: int) -> np.ndarray:
    """Return origin rows whose future target stays inside one continuous recording segment."""
    if not series.anchors:
        return np.arange(series.n_windows - lag, dtype=int)

    origins: list[int] = []
    for origin in range(series.n_windows - lag):
        path = series.anchors[origin : origin + lag + 1]
        same_source = all(_anchor_source_key(anchor) == _anchor_source_key(path[0]) for anchor in path)
        crosses_gap = any(anchor.gap_after for anchor in path[:-1]) or any(
            anchor.gap_before for anchor in path[1:]
        )
        if same_source and not crosses_gap:
            origins.append(origin)
    return np.asarray(origins, dtype=int)


def _build_design(
    series: ConnectivitySeries, manifold: ManifoldArtifact, behavior: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    if lag <= 0:
        raise ValueError("lag must be positive")
    origins = _design_origins(series, lag)
    if origins.size < 4:
        raise ValueError("not enough within-segment transitions to fit a surrogate")
    targets = origins + lag
    conn = _flatten_connectivity(series)
    cur_latent = np.asarray(manifold.window_embedding, dtype=float)
    cur_behavior = np.asarray(behavior, dtype=float).reshape(-1, 1)
    x = np.concatenate([conn[origins], cur_latent[origins], cur_behavior[origins]], axis=1)
    y = np.concatenate([cur_latent[targets], cur_behavior[targets]], axis=1)
    return x, y


@dataclass
class SurrogateValidation:
    status: str
    mean_skill: float
    fold_skills: list[float]
    min_skill: float
    fail_safe_reason: str | None = None


@dataclass
class LinearSurrogateModel:
    ridge_alpha: float
    lag: int
    validation: SurrogateValidation
    model: Ridge | None = None
    feature_shape: tuple[int, int] | None = None
    metadata: dict = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("surrogate model is not fitted")
        return np.asarray(self.model.predict(x), dtype=float)


@dataclass
class DoseResponsePoint:
    scale: float
    effect_size: float
    control_effects: list[float]
    control_pvalue: float
    sham_effect: float


@dataclass
class PerturbationResult:
    status: str
    node_indices: list[int]
    scale: float
    effect_size: float
    control_effects: list[float]
    control_pvalue: float
    endpoint: str
    dose_response: list[DoseResponsePoint] = field(default_factory=list)
    dose_response_monotonic: bool = False
    validation_status: str = "unvalidated_counterfactual"
    fail_safe_reason: str | None = None
    candidate_stage: str = "preliminary_predictive_candidate"
    control_strategy: str = "strength_matched_random_nodes_plus_sham"
    provenance: dict[str, object] = field(default_factory=dict)


def fit_linear_surrogate(
    series: ConnectivitySeries,
    manifold: ManifoldArtifact,
    behavior: np.ndarray,
    lag: int = 1,
    ridge_alpha: float = 1.0,
    n_folds: int = 5,
    embargo: int = 0,
    min_skill: float = 0.0,
) -> LinearSurrogateModel:
    """Fit and validate a ridge surrogate for next latent+behavior state."""
    x, y = _build_design(series, manifold, behavior, lag)
    origins = _design_origins(series, lag)
    split_anchors = [series.anchors[int(idx)] for idx in origins] if series.anchors else None
    groups: list[object] | None = (
        anchor_group_labels(split_anchors, group_by="recording").tolist() if split_anchors else None
    )
    fold_skills: list[float] = []
    connectivity_feature_count = series.n_neurons**2
    baseline_x = x[:, connectivity_feature_count:]
    for train, test in purged_blocked_splits(
        len(x),
        n_folds,
        embargo=embargo,
        anchors=split_anchors,
        groups=groups,
    ):
        model = Ridge(alpha=ridge_alpha)
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        baseline_model = Ridge(alpha=ridge_alpha).fit(baseline_x[train], y[train])
        baseline_pred = baseline_model.predict(baseline_x[test])
        skill = float(r2_score(y[test], pred) - r2_score(y[test], baseline_pred))
        fold_skills.append(skill)

    mean_skill = float(np.mean(fold_skills)) if fold_skills else float("-inf")
    validation = SurrogateValidation(
        status="valid" if mean_skill > min_skill else "invalid",
        mean_skill=mean_skill,
        fold_skills=fold_skills,
        min_skill=min_skill,
        fail_safe_reason=None if mean_skill > min_skill else "surrogate_validation_failed",
    )
    final = Ridge(alpha=ridge_alpha).fit(x, y)
    return LinearSurrogateModel(
        ridge_alpha=ridge_alpha,
        lag=lag,
        validation=validation,
        model=final,
        feature_shape=x.shape,
        metadata={
            "target_dim": int(y.shape[1]),
            "design_origins": origins,
            "split_mode": "anchor_aware_grouped_purged" if split_anchors else "purged_blocked",
            "validation_baseline": "autoregressive_latent_plus_behavior_without_connectivity",
        },
    )


def _perturb_series(series: ConnectivitySeries, node_indices: list[int], scale: float) -> ConnectivitySeries:
    mats = np.array(series.matrices, copy=True)
    for node in node_indices:
        mats[:, node, :] *= scale
    return ConnectivitySeries(
        matrices=mats,
        window_starts=np.array(series.window_starts, copy=True),
        method=series.method,
        directed=series.directed,
        behavior_per_window=series.behavior_per_window,
        anchors=list(series.anchors),
        signed=series.signed,
        weighted=series.weighted,
        storage=series.storage,
        weight_semantics=series.weight_semantics,
        diagnostics=dict(series.diagnostics),
        provenance=series.provenance,
    )


def _mean_endpoint_effect(
    model: LinearSurrogateModel,
    series: ConnectivitySeries,
    manifold: ManifoldArtifact,
    behavior: np.ndarray,
    node_indices: list[int],
    scale: float,
    endpoint: str,
    baseline_endpoint: np.ndarray,
) -> float:
    perturbed = _perturb_series(series, node_indices, scale)
    pert_x, _ = _build_design(perturbed, manifold, behavior, model.lag)
    pert_pred = model.predict(pert_x)
    pert_endpoint = pert_pred[:, -1] if endpoint == "behavior" else np.linalg.norm(pert_pred[:, :-1], axis=1)
    return float(np.mean(pert_endpoint - baseline_endpoint))


def _sample_matched_control_sets(
    series: ConnectivitySeries,
    node_indices: list[int],
    n_controls: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    if n_controls <= 0 or not node_indices:
        return []

    strengths = np.mean(np.abs(series.matrices), axis=(0, 2))
    all_nodes = np.arange(series.n_neurons)
    eligible = np.array([n for n in all_nodes if n not in node_indices], dtype=int)
    if len(eligible) < len(node_indices):
        return []

    target_strength = float(np.mean(strengths[node_indices]))
    distances = np.abs(strengths[eligible] - target_strength)
    ranked = eligible[np.argsort(distances)]
    pool_size = min(len(ranked), max(len(node_indices), len(node_indices) * 4))
    pool = ranked[:pool_size]

    if len(pool) == len(node_indices):
        return [pool.tolist() for _ in range(n_controls)]

    weights = 1.0 / (np.abs(strengths[pool] - target_strength) + 1e-6)
    weights = weights / weights.sum()
    control_sets: list[list[int]] = []
    for _ in range(n_controls):
        draw = rng.choice(pool, size=len(node_indices), replace=False, p=weights)
        control_sets.append(draw.tolist())
    return control_sets


def _dose_response_monotonic(points: list[DoseResponsePoint]) -> bool:
    if len(points) < 2:
        return False
    ordered = sorted(points, key=lambda point: point.scale, reverse=True)
    abs_effects = [abs(point.effect_size) for point in ordered]
    return all(curr <= nxt + 1e-9 for curr, nxt in zip(abs_effects, abs_effects[1:], strict=False))


def run_virtual_perturbation(
    model: LinearSurrogateModel,
    series: ConnectivitySeries,
    manifold: ManifoldArtifact,
    behavior: np.ndarray,
    node_indices: list[int],
    scale: float = 0.0,
    n_controls: int = 32,
    endpoint: str = "behavior",
    seed: int = 42,
    dose_scales: list[float] | None = None,
) -> PerturbationResult:
    """Run graded node perturbations against matched random and sham controls."""
    provenance = {
        "claim_boundary": "model_based_counterfactual_not_biological_causation",
        "endpoint": endpoint,
        "n_controls": int(n_controls),
        "seed": int(seed),
    }
    if model.validation.status != "valid" or model.model is None:
        return PerturbationResult(
            status="invalid",
            node_indices=node_indices,
            scale=scale,
            effect_size=0.0,
            control_effects=[],
            control_pvalue=1.0,
            endpoint=endpoint,
            validation_status="unvalidated_counterfactual",
            fail_safe_reason=model.validation.fail_safe_reason or "surrogate_validation_failed",
            candidate_stage="preliminary_predictive_candidate",
            provenance=provenance | {"surrogate_status": model.validation.status},
        )
    if not node_indices:
        return PerturbationResult(
            status="invalid",
            node_indices=node_indices,
            scale=scale,
            effect_size=0.0,
            control_effects=[],
            control_pvalue=1.0,
            endpoint=endpoint,
            validation_status="unvalidated_counterfactual",
            fail_safe_reason="no_preliminary_candidate_nodes",
            candidate_stage="screened_out",
            provenance=provenance | {"surrogate_status": model.validation.status},
        )

    lag = model.lag
    base_x, _ = _build_design(series, manifold, behavior, lag)
    base_pred = model.predict(base_x)
    baseline_endpoint = (
        base_pred[:, -1] if endpoint == "behavior" else np.linalg.norm(base_pred[:, :-1], axis=1)
    )

    doses = dose_scales or [scale]
    unique_doses = list(dict.fromkeys(float(dose) for dose in doses))
    rng = np.random.default_rng(seed)
    control_sets = _sample_matched_control_sets(series, node_indices, n_controls, rng)
    sham_effect = _mean_endpoint_effect(
        model,
        series,
        manifold,
        behavior,
        node_indices,
        1.0,
        endpoint,
        baseline_endpoint,
    )

    dose_response: list[DoseResponsePoint] = []
    for dose in unique_doses:
        effect = _mean_endpoint_effect(
            model,
            series,
            manifold,
            behavior,
            node_indices,
            dose,
            endpoint,
            baseline_endpoint,
        )
        control_effects: list[float] = []
        for control_nodes in control_sets:
            ctrl_effect = _mean_endpoint_effect(
                model,
                series,
                manifold,
                behavior,
                control_nodes,
                dose,
                endpoint,
                baseline_endpoint,
            )
            control_effects.append(ctrl_effect)
        if control_effects:
            exceedances = int(np.sum(np.abs(control_effects) >= abs(effect)))
            pval = float((exceedances + 1) / (len(control_effects) + 1))
        else:
            pval = 1.0
        dose_response.append(
            DoseResponsePoint(
                scale=float(dose),
                effect_size=effect,
                control_effects=control_effects,
                control_pvalue=pval,
                sham_effect=sham_effect,
            )
        )

    primary = min(dose_response, key=lambda point: point.scale)
    strongest_exceeds_controls = bool(primary.control_effects) and primary.control_pvalue < 0.05
    monotonic = _dose_response_monotonic(dose_response)
    counterfactually_supported = (
        monotonic and strongest_exceeds_controls and abs(primary.effect_size) > abs(sham_effect) + 1e-9
    )
    return PerturbationResult(
        status="valid",
        node_indices=node_indices,
        scale=primary.scale,
        effect_size=primary.effect_size,
        control_effects=primary.control_effects,
        control_pvalue=primary.control_pvalue,
        endpoint=endpoint,
        dose_response=dose_response,
        dose_response_monotonic=monotonic,
        validation_status=(
            "counterfactually_supported" if counterfactually_supported else "unvalidated_counterfactual"
        ),
        fail_safe_reason=(None if counterfactually_supported else "dose_response_or_control_gate_failed"),
        candidate_stage=(
            "counterfactually_supported_candidate"
            if counterfactually_supported
            else "preliminary_predictive_candidate"
        ),
        provenance=provenance
        | {
            "surrogate_status": model.validation.status,
            "matched_control_sets": control_sets,
            "dose_scales": unique_doses,
            "primary_scale": primary.scale,
            "claim_status": (
                "counterfactual_support_only_not_a_validated_biological_driver"
                if counterfactually_supported
                else "counterfactual_gate_not_met"
            ),
        },
    )
