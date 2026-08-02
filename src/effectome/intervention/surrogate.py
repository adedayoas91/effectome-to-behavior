"""Dependency-light generative surrogate and virtual perturbation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from effectome.data_module.schema import ConnectivitySeries
from effectome.linking.decoding import purged_blocked_splits
from effectome.manifold import ManifoldArtifact


def _flatten_connectivity(series: ConnectivitySeries) -> np.ndarray:
    return series.matrices.reshape(series.n_windows, -1)


def _build_design(
    series: ConnectivitySeries, manifold: ManifoldArtifact, behavior: np.ndarray, lag: int
) -> tuple[np.ndarray, np.ndarray]:
    if lag <= 0:
        raise ValueError("lag must be positive")
    conn = _flatten_connectivity(series)
    cur_latent = np.asarray(manifold.window_embedding, dtype=float)
    cur_behavior = np.asarray(behavior, dtype=float).reshape(-1, 1)
    x = np.concatenate([conn[:-lag], cur_latent[:-lag], cur_behavior[:-lag]], axis=1)
    y = np.concatenate([cur_latent[lag:], cur_behavior[lag:]], axis=1)
    return x, y


@dataclass
class SurrogateValidation:
    status: str
    mean_skill: float
    fold_skills: list[float]
    min_skill: float


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
class PerturbationResult:
    status: str
    node_indices: list[int]
    scale: float
    effect_size: float
    control_effects: list[float]
    control_pvalue: float
    endpoint: str


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
    fold_skills: list[float] = []
    for train, test in purged_blocked_splits(len(x), n_folds, embargo=embargo):
        model = Ridge(alpha=ridge_alpha)
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        baseline = np.repeat(y[train].mean(axis=0, keepdims=True), len(test), axis=0)
        skill = float(r2_score(y[test], pred) - r2_score(y[test], baseline))
        fold_skills.append(skill)

    mean_skill = float(np.mean(fold_skills)) if fold_skills else float("-inf")
    validation = SurrogateValidation(
        status="valid" if mean_skill > min_skill else "invalid",
        mean_skill=mean_skill,
        fold_skills=fold_skills,
        min_skill=min_skill,
    )
    final = Ridge(alpha=ridge_alpha).fit(x, y)
    return LinearSurrogateModel(
        ridge_alpha=ridge_alpha,
        lag=lag,
        validation=validation,
        model=final,
        feature_shape=x.shape,
        metadata={"target_dim": int(y.shape[1])},
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
    )


def run_virtual_perturbation(
    model: LinearSurrogateModel,
    series: ConnectivitySeries,
    manifold: ManifoldArtifact,
    behavior: np.ndarray,
    node_indices: list[int],
    scale: float = 0.0,
    n_controls: int = 16,
    endpoint: str = "behavior",
    seed: int = 42,
) -> PerturbationResult:
    """Run graded node perturbations against matched random controls."""
    if model.validation.status != "valid" or model.model is None:
        return PerturbationResult(
            status="invalid",
            node_indices=node_indices,
            scale=scale,
            effect_size=0.0,
            control_effects=[],
            control_pvalue=1.0,
            endpoint=endpoint,
        )

    lag = model.lag
    base_x, _ = _build_design(series, manifold, behavior, lag)
    base_pred = model.predict(base_x)
    baseline_endpoint = (
        base_pred[:, -1] if endpoint == "behavior" else np.linalg.norm(base_pred[:, :-1], axis=1)
    )

    perturbed = _perturb_series(series, node_indices, scale)
    pert_x, _ = _build_design(perturbed, manifold, behavior, lag)
    pert_pred = model.predict(pert_x)
    pert_endpoint = pert_pred[:, -1] if endpoint == "behavior" else np.linalg.norm(pert_pred[:, :-1], axis=1)
    effect = float(np.mean(pert_endpoint - baseline_endpoint))

    rng = np.random.default_rng(seed)
    all_nodes = np.arange(series.n_neurons)
    eligible = np.array([n for n in all_nodes if n not in node_indices], dtype=int)
    control_effects: list[float] = []
    if len(eligible) >= len(node_indices) and len(node_indices) > 0:
        for _ in range(n_controls):
            control_nodes = rng.choice(eligible, size=len(node_indices), replace=False).tolist()
            ctrl_series = _perturb_series(series, control_nodes, scale)
            ctrl_x, _ = _build_design(ctrl_series, manifold, behavior, lag)
            ctrl_pred = model.predict(ctrl_x)
            ctrl_endpoint = (
                ctrl_pred[:, -1] if endpoint == "behavior" else np.linalg.norm(ctrl_pred[:, :-1], axis=1)
            )
            control_effects.append(float(np.mean(ctrl_endpoint - baseline_endpoint)))

    pval = float((np.abs(control_effects) >= abs(effect)).mean()) if control_effects else 1.0
    return PerturbationResult(
        status="valid",
        node_indices=node_indices,
        scale=scale,
        effect_size=effect,
        control_effects=control_effects,
        control_pvalue=pval,
        endpoint=endpoint,
    )
