"""Dependency-light surrogate validation and predictor-feature sensitivity utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor
from effectome.linking.decoding import (
    anchor_group_labels,
    purged_blocked_splits,
    valid_positive_lag_origins,
)
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
    return valid_positive_lag_origins(
        series.n_windows,
        lag,
        series.anchors if series.anchors else None,
    )


_SURROGATE_ENDPOINTS = {"joint", "manifold", "behavior"}
_PERTURBATION_ENDPOINTS = {"manifold", "behavior"}


def _validate_surrogate_endpoint(endpoint: str) -> None:
    if endpoint not in _SURROGATE_ENDPOINTS:
        raise ValueError(
            f"unknown surrogate endpoint {endpoint!r}; expected one of {sorted(_SURROGATE_ENDPOINTS)}"
        )


def _behavior_design(
    behavior: np.ndarray,
    n_windows: int,
    *,
    categories: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None]:
    """Return behavior values and stable current-state covariates.

    Integer-valued behavior is treated as categorical state rather than as an
    ordered scalar.  The returned categories can be reused when rebuilding the
    design matrix for a fitted model.
    """
    values = np.asarray(behavior)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("behavior must be a one-dimensional array")
    if values.shape[0] != n_windows:
        raise ValueError(f"behavior length {values.shape[0]} does not match connectivity windows {n_windows}")

    categorical = np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_)
    if not categorical:
        continuous = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(continuous)):
            raise ValueError("behavior contains non-finite values")
        return continuous, continuous.reshape(-1, 1), "continuous", None

    fitted_categories = np.unique(values) if categories is None else np.asarray(categories)
    if fitted_categories.ndim != 1 or fitted_categories.size == 0:
        raise ValueError("categorical behavior requires at least one fitted category")
    known = np.isin(values, fitted_categories)
    if not np.all(known):
        unknown = np.unique(values[~known]).tolist()
        raise ValueError(f"behavior contains categories not seen during surrogate fitting: {unknown}")
    one_hot = (values[:, None] == fitted_categories[None, :]).astype(float)
    return values, one_hot, "categorical_one_hot", fitted_categories


def _build_design(
    series: ConnectivitySeries,
    manifold: ManifoldArtifact,
    behavior: np.ndarray,
    lag: int,
    *,
    endpoint: str = "joint",
    behavior_categories: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_surrogate_endpoint(endpoint)
    if lag <= 0:
        raise ValueError("lag must be positive")
    origins = _design_origins(series, lag)
    if origins.size < 4:
        raise ValueError("not enough within-segment transitions to fit a surrogate")
    targets = origins + lag
    conn = _flatten_connectivity(series)
    cur_latent = np.asarray(manifold.window_embedding, dtype=float)
    if cur_latent.shape[0] != series.n_windows:
        raise ValueError(
            "manifold window count does not match connectivity windows: "
            f"{cur_latent.shape[0]} != {series.n_windows}"
        )
    behavior_values, behavior_covariates, encoding, _ = _behavior_design(
        behavior,
        series.n_windows,
        categories=behavior_categories,
    )
    if encoding == "categorical_one_hot" and endpoint in {"joint", "behavior"}:
        raise ValueError(
            "integer behavior labels are categorical and cannot be regressed as a behavior target; "
            "fit with endpoint='manifold' or provide a continuous behavior target"
        )

    x = np.concatenate([conn[origins], cur_latent[origins], behavior_covariates[origins]], axis=1)
    future_latent = cur_latent[targets]
    if endpoint == "manifold":
        y = future_latent
    else:
        future_behavior = np.asarray(behavior_values[targets], dtype=float).reshape(-1, 1)
        y = (
            future_behavior
            if endpoint == "behavior"
            else np.concatenate([future_latent, future_behavior], axis=1)
        )
    return x, y


@dataclass
class SurrogateValidation:
    status: str
    mean_skill: float
    fold_skills: list[float]
    min_skill: float
    absolute_skills: list[float] = field(default_factory=list)
    min_absolute_skill: float = float("-inf")
    distribution_shift: float | None = None
    fail_safe_reason: str | None = None


@dataclass
class LinearSurrogateModel:
    ridge_alpha: float
    lag: int
    validation: SurrogateValidation
    model: Ridge | None = None
    feature_shape: tuple[int, int] | None = None
    metadata: dict = field(default_factory=dict)
    endpoint: str = "joint"

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("surrogate model is not fitted")
        return np.asarray(self.model.predict(x), dtype=float)


@dataclass
class DoseResponsePoint:
    scale: float
    effect_size: float | None
    control_effects: list[float]
    control_pvalue: float | None
    sham_effect: float | None


@dataclass
class PerturbationResult:
    status: str
    node_indices: list[int]
    scale: float
    effect_size: float | None
    control_effects: list[float]
    control_pvalue: float | None
    endpoint: str
    dose_response: list[DoseResponsePoint] = field(default_factory=list)
    dose_response_monotonic: bool = False
    validation_status: str = "unvalidated_feature_sensitivity"
    fail_safe_reason: str | None = None
    candidate_stage: str = "preliminary_predictive_candidate"
    control_strategy: str = "strength_matched_random_nodes_plus_sham_sensitivity_check"
    analysis_kind: str = "predictor_feature_sensitivity"
    biological_intervention_identified: bool = False
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
    endpoint: str = "joint",
    fit_indices: np.ndarray | None = None,
    evaluation_indices: np.ndarray | None = None,
    candidate_nodes: list[int] | None = None,
    min_absolute_skill: float = float("-inf"),
    max_shift: float = float("inf"),
    require_feature_ownership: bool = False,
) -> LinearSurrogateModel:
    """Fit and validate a ridge surrogate for a future endpoint.

    ``joint`` preserves the historical latent-plus-behavior target.  Integer
    behavior codes are accepted as one-hot current-state covariates for a
    manifold endpoint, but are rejected when behavior itself is a regression
    target because numeric code distances have no categorical meaning.
    """
    _validate_surrogate_endpoint(endpoint)
    _, _, behavior_encoding, behavior_categories = _behavior_design(behavior, series.n_windows)
    x, y = _build_design(
        series,
        manifold,
        behavior,
        lag,
        endpoint=endpoint,
        behavior_categories=behavior_categories,
    )
    origins = _design_origins(series, lag)
    owned_fit = np.arange(series.n_windows, dtype=int) if fit_indices is None else np.asarray(fit_indices)
    owned_evaluation = (
        np.array([], dtype=int)
        if evaluation_indices is None
        else np.asarray(evaluation_indices)
    )
    for name, indices in (("fit_indices", owned_fit), ("evaluation_indices", owned_evaluation)):
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= series.n_windows):
            raise ValueError(f"{name} are outside the available windows")
        if len(indices) != len(np.unique(indices)):
            raise ValueError(f"{name} must be unique")
    if np.intersect1d(owned_fit, owned_evaluation).size:
        raise ValueError("fit and evaluation ownership must be disjoint")
    fit_set = set(int(index) for index in owned_fit)
    evaluation_set = set(int(index) for index in owned_evaluation)
    fit_rows = np.array(
        [
            row
            for row, origin in enumerate(origins)
            if int(origin) in fit_set and int(origin + lag) in fit_set
        ],
        dtype=int,
    )
    evaluation_rows = np.array(
        [
            row
            for row, origin in enumerate(origins)
            if int(origin) in evaluation_set and int(origin + lag) in evaluation_set
        ],
        dtype=int,
    )
    if fit_rows.size < 4:
        raise ValueError("fit ownership leaves too few complete lagged examples")
    fit_origins = origins[fit_rows]
    split_anchors = [series.anchors[int(idx)] for idx in fit_origins] if series.anchors else None
    outcome_anchors = (
        [series.anchors[int(idx + lag)] for idx in fit_origins] if series.anchors else None
    )
    groups: list[object] | None = (
        anchor_group_labels(split_anchors, group_by="recording").tolist() if split_anchors else None
    )
    fold_skills: list[float] = []
    connectivity_feature_count = series.n_neurons**2
    fit_x = x[fit_rows]
    fit_y = y[fit_rows]
    baseline_x = fit_x[:, connectivity_feature_count:]
    absolute_skills: list[float] = []
    for train, test in purged_blocked_splits(
        len(fit_x),
        n_folds,
        embargo=embargo,
        anchors=split_anchors,
        groups=groups,
        outcome_anchors=outcome_anchors,
    ):
        model = Ridge(alpha=ridge_alpha)
        model.fit(fit_x[train], fit_y[train])
        pred = model.predict(fit_x[test])
        baseline_model = Ridge(alpha=ridge_alpha).fit(baseline_x[train], fit_y[train])
        baseline_pred = baseline_model.predict(baseline_x[test])
        absolute = float(r2_score(fit_y[test], pred))
        skill = float(absolute - r2_score(fit_y[test], baseline_pred))
        absolute_skills.append(absolute)
        fold_skills.append(skill)

    mean_skill = float(np.mean(fold_skills)) if fold_skills else float("-inf")
    mean_absolute = float(np.mean(absolute_skills)) if absolute_skills else float("-inf")
    shift = None
    if evaluation_rows.size:
        center = fit_x.mean(axis=0)
        scale = np.where(fit_x.std(axis=0) > 1e-12, fit_x.std(axis=0), 1.0)
        shift = float(np.max(np.abs((x[evaluation_rows].mean(axis=0) - center) / scale)))
    ownership_verified = bool(
        series.diagnostics.get("fit_scope_id")
        and getattr(manifold, "metadata", {}).get("fit_scope_id")
    )
    reason = None
    if require_feature_ownership and not ownership_verified:
        reason = "learned_feature_ownership_unverified"
    elif mean_skill <= min_skill:
        reason = "surrogate_validation_failed"
    elif mean_absolute < min_absolute_skill:
        reason = "absolute_skill_gate_failed"
    elif shift is not None and shift > max_shift:
        reason = "distribution_shift_gate_failed"
    validation = SurrogateValidation(
        status="valid" if reason is None else "invalid",
        mean_skill=mean_skill,
        fold_skills=fold_skills,
        min_skill=min_skill,
        absolute_skills=absolute_skills,
        min_absolute_skill=min_absolute_skill,
        distribution_shift=shift,
        fail_safe_reason=reason,
    )
    final = Ridge(alpha=ridge_alpha).fit(fit_x, fit_y)
    candidate_nodes = [] if candidate_nodes is None else [int(node) for node in candidate_nodes]
    if any(node < 0 or node >= series.n_neurons for node in candidate_nodes):
        raise ValueError("candidate_nodes are outside the connectivity node axis")
    matching_strengths = np.mean(
        np.abs(series.matrices[owned_fit]), axis=(0, 2)
    ).tolist()
    return LinearSurrogateModel(
        ridge_alpha=ridge_alpha,
        lag=lag,
        validation=validation,
        endpoint=endpoint,
        model=final,
        feature_shape=(int(fit_x.shape[0]), int(fit_x.shape[1])),
        metadata={
            "target_dim": int(y.shape[1]),
            "manifold_target_dim": int(manifold.window_embedding.shape[1]),
            "endpoint": endpoint,
            "behavior_covariate_encoding": behavior_encoding,
            "behavior_categories": (
                behavior_categories.tolist() if behavior_categories is not None else None
            ),
            "design_origins": origins,
            "fit_origins": fit_origins.tolist(),
            "evaluation_origins": origins[evaluation_rows].tolist(),
            "fit_indices": owned_fit.tolist(),
            "evaluation_indices": owned_evaluation.tolist(),
            "candidate_nodes": candidate_nodes,
            "matching_strengths": matching_strengths,
            "feature_ownership_verified": ownership_verified,
            "split_mode": "anchor_aware_grouped_purged" if split_anchors else "purged_blocked",
            "validation_baseline": "autoregressive_latent_plus_behavior_without_connectivity",
            "analysis_kind": "predictor_feature_sensitivity",
            "biological_intervention_identified": False,
        },
    )


def _perturb_series(series: ConnectivitySeries, node_indices: list[int], scale: float) -> ConnectivitySeries:
    mats = np.array(series.matrices, copy=True)
    source_lagged = getattr(series, "lagged_matrices", None)
    lagged = np.array(source_lagged, copy=True) if source_lagged is not None else None
    for node in node_indices:
        mats[:, node, :] *= scale
        if lagged is not None:
            lagged[:, :, node, :] *= scale
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
        lagged_matrices=lagged,
    )


def _fitted_endpoint(model: LinearSurrogateModel) -> str:
    endpoint = str(getattr(model, "endpoint", model.metadata.get("endpoint", "joint")))
    _validate_surrogate_endpoint(endpoint)
    return endpoint


def _validate_perturbation_endpoint(model: LinearSurrogateModel, endpoint: str) -> str:
    if endpoint not in _PERTURBATION_ENDPOINTS:
        raise ValueError(
            f"unknown perturbation endpoint {endpoint!r}; expected one of {sorted(_PERTURBATION_ENDPOINTS)}"
        )
    fitted_endpoint = _fitted_endpoint(model)
    if fitted_endpoint != "joint" and endpoint != fitted_endpoint:
        raise ValueError(
            f"perturbation endpoint {endpoint!r} is incompatible with a surrogate fitted for "
            f"endpoint {fitted_endpoint!r}"
        )
    return fitted_endpoint


def _prediction_endpoint(model: LinearSurrogateModel, predictions: np.ndarray, endpoint: str) -> np.ndarray:
    fitted_endpoint = _validate_perturbation_endpoint(model, endpoint)
    pred = np.asarray(predictions, dtype=float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if pred.ndim != 2:
        raise ValueError(f"surrogate predictions must be 2D, got shape {pred.shape}")
    expected_dim = int(model.metadata.get("target_dim", pred.shape[1]))
    if pred.shape[1] != expected_dim:
        raise ValueError(
            f"surrogate prediction dimension {pred.shape[1]} does not match fitted target "
            f"dimension {expected_dim}"
        )

    if fitted_endpoint == "behavior":
        if pred.shape[1] != 1:
            raise ValueError("behavior surrogate must predict exactly one target column")
        return pred[:, 0]
    if fitted_endpoint == "manifold":
        return pred

    manifold_dim = int(model.metadata.get("manifold_target_dim", pred.shape[1] - 1))
    if manifold_dim <= 0 or manifold_dim >= pred.shape[1]:
        raise ValueError("joint surrogate target layout must contain manifold and behavior columns")
    if endpoint == "behavior":
        return pred[:, manifold_dim]
    return pred[:, :manifold_dim]


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
    fitted_endpoint = _fitted_endpoint(model)
    categories = model.metadata.get("behavior_categories")
    pert_x, _ = _build_design(
        perturbed,
        manifold,
        behavior,
        model.lag,
        endpoint=fitted_endpoint,
        behavior_categories=np.asarray(categories) if categories is not None else None,
    )
    evaluation_origins = model.metadata.get("evaluation_origins", [])
    if evaluation_origins:
        all_origins = _design_origins(perturbed, model.lag)
        selected = np.flatnonzero(np.isin(all_origins, np.asarray(evaluation_origins)))
        pert_x = pert_x[selected]
    pert_pred = model.predict(pert_x)
    pert_endpoint = _prediction_endpoint(model, pert_pred, endpoint)
    endpoint_delta = pert_endpoint - baseline_endpoint
    if endpoint == "manifold":
        return float(np.mean(np.linalg.norm(endpoint_delta, axis=1)))
    return float(np.mean(endpoint_delta))


def _sample_matched_control_sets(
    series: ConnectivitySeries,
    node_indices: list[int],
    n_controls: int,
    rng: np.random.Generator,
    *,
    fitted_strengths: np.ndarray | None = None,
) -> list[list[int]]:
    if n_controls <= 0 or not node_indices:
        return []

    del rng
    strengths = (
        np.mean(np.abs(series.matrices), axis=(0, 2))
        if fitted_strengths is None
        else np.asarray(fitted_strengths, dtype=float)
    )
    all_nodes = np.arange(series.n_neurons)
    eligible = np.array([n for n in all_nodes if n not in node_indices], dtype=int)
    if len(eligible) < len(node_indices):
        return []

    target_strength = float(np.mean(strengths[node_indices]))
    distances = np.abs(strengths[eligible] - target_strength)
    ranked = eligible[np.argsort(distances)]
    pool_size = min(len(ranked), max(len(node_indices), len(node_indices) * 4))
    pool = ranked[:pool_size]

    candidates = [list(items) for items in combinations(pool.tolist(), len(node_indices))]
    candidates.sort(
        key=lambda items: (
            abs(float(np.mean(strengths[items])) - target_strength),
            tuple(items),
        )
    )
    return candidates[:n_controls]


def _dose_response_monotonic(points: list[DoseResponsePoint]) -> bool:
    if len(points) < 2:
        return False
    ordered = sorted(points, key=lambda point: point.scale, reverse=True)
    if any(point.effect_size is None for point in ordered):
        return False
    abs_effects = [abs(point.effect_size) for point in ordered if point.effect_size is not None]
    return all(curr <= nxt + 1e-9 for curr, nxt in zip(abs_effects, abs_effects[1:], strict=False))


def run_predictor_feature_sensitivity(
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
    """Run graded predictor-feature edits against matched random and sham controls."""
    fitted_endpoint = _validate_perturbation_endpoint(model, endpoint)
    provenance = {
        "analysis_kind": "predictor_feature_sensitivity",
        "biological_intervention_identified": False,
        "claim_boundary": "predictor_feature_sensitivity_not_biological_intervention",
        "endpoint": endpoint,
        "n_controls": int(n_controls),
        "seed": int(seed),
        "surrogate_endpoint": fitted_endpoint,
        "effect_metric": (
            "mean_euclidean_latent_displacement" if endpoint == "manifold" else "mean_signed_behavior_change"
        ),
    }
    if model.validation.status != "valid" or model.model is None:
        return PerturbationResult(
            status="invalid",
            node_indices=node_indices,
            scale=scale,
            effect_size=None,
            control_effects=[],
            control_pvalue=None,
            endpoint=endpoint,
            validation_status="unavailable",
            fail_safe_reason=model.validation.fail_safe_reason or "surrogate_validation_failed",
            candidate_stage="preliminary_predictive_candidate",
            provenance=provenance | {"surrogate_status": model.validation.status},
        )
    if not node_indices:
        return PerturbationResult(
            status="invalid",
            node_indices=node_indices,
            scale=scale,
            effect_size=None,
            control_effects=[],
            control_pvalue=None,
            endpoint=endpoint,
            validation_status="unavailable",
            fail_safe_reason="no_preliminary_candidate_nodes",
            candidate_stage="screened_out",
            provenance=provenance | {"surrogate_status": model.validation.status},
        )

    lag = model.lag
    categories = model.metadata.get("behavior_categories")
    base_x, _ = _build_design(
        series,
        manifold,
        behavior,
        lag,
        endpoint=fitted_endpoint,
        behavior_categories=np.asarray(categories) if categories is not None else None,
    )
    evaluation_origins = model.metadata.get("evaluation_origins", [])
    if evaluation_origins:
        all_origins = _design_origins(series, lag)
        selected = np.flatnonzero(np.isin(all_origins, np.asarray(evaluation_origins)))
        base_x = base_x[selected]
    base_pred = model.predict(base_x)
    baseline_endpoint = _prediction_endpoint(model, base_pred, endpoint)

    doses = dose_scales or [scale]
    unique_doses = list(dict.fromkeys(float(dose) for dose in doses))
    rng = np.random.default_rng(seed)
    fitted_strengths = model.metadata.get("matching_strengths")
    control_sets = _sample_matched_control_sets(
        series,
        node_indices,
        n_controls,
        rng,
        fitted_strengths=(
            None if fitted_strengths is None else np.asarray(fitted_strengths, dtype=float)
        ),
    )
    minimum_pvalue = 1.0 / (len(control_sets) + 1)
    control_resolution_available = minimum_pvalue <= 0.05
    provenance.update(
        {
            "unique_controls": len(control_sets),
            "minimum_attainable_pvalue": minimum_pvalue,
            "control_status": (
                "available"
                if control_resolution_available
                else "insufficient_control_resolution"
            ),
        }
    )
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
        if control_effects and control_resolution_available:
            exceedances = int(np.sum(np.abs(control_effects) >= abs(effect)))
            pval = float((exceedances + 1) / (len(control_effects) + 1))
        else:
            pval = None
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
    strongest_exceeds_controls = (
        bool(primary.control_effects) and primary.control_pvalue is not None and primary.control_pvalue < 0.05
    )
    monotonic = _dose_response_monotonic(dose_response)
    sensitivity_supported = (
        primary.effect_size is not None
        and monotonic
        and strongest_exceeds_controls
        and abs(primary.effect_size) > abs(sham_effect) + 1e-9
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
            "model_sensitivity_supported" if sensitivity_supported else "model_sensitivity_unconfirmed"
        ),
        fail_safe_reason=(None if sensitivity_supported else "dose_response_or_control_gate_failed"),
        candidate_stage="preliminary_predictive_candidate",
        provenance=provenance
        | {
            "surrogate_status": model.validation.status,
            "matched_control_sets": control_sets,
            "dose_scales": unique_doses,
            "primary_scale": primary.scale,
            "claim_status": (
                "model_feature_sensitivity_supported_not_biological_intervention"
                if sensitivity_supported
                else "feature_sensitivity_gate_not_met"
            ),
        },
    )


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
    """Backward-compatible name for predictor-feature sensitivity analysis."""
    return run_predictor_feature_sensitivity(
        model,
        series,
        manifold,
        behavior,
        node_indices=node_indices,
        scale=scale,
        n_controls=n_controls,
        endpoint=endpoint,
        seed=seed,
        dose_scales=dose_scales,
    )
