"""Nested chronological evaluation of optional graph summaries.

Targets are graph histories wholly after the predictor history. This evaluates
predictive compression of estimated graphs, not identification of neural regimes.
Upstream graph/preprocessing provenance remains a separate acceptance requirement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from effectome.data_module.schema import ConnectivitySeries

from .graph_states import GraphStateConfig, GraphStateModel, boundary_indices_for_series, fit_graph_states


@dataclass(frozen=True)
class StateEvaluationConfig:
    graph: GraphStateConfig = field(
        default_factory=lambda: GraphStateConfig(
            n_states=1,
            metric="frobenius",
            feature_representation="signed_lag_offdiag",
            assignment="kmeans",
            pca_components=8,
        )
    )
    candidate_counts: tuple[int, ...] = (1, 2, 3, 4, 6)
    assignments: tuple[str, ...] = ("kmeans", "gaussian_mixture")
    outer_test_fraction: float = 0.25
    inner_validation_fraction: float = 0.25
    min_train_pairs: int = 4
    ridge_penalties: tuple[float, ...] = (0.1, 1.0, 10.0)
    # Proposed predictive-compression margin; not a biological effect threshold.
    minimum_relative_gain: float = 0.01

    def __post_init__(self) -> None:
        if not 0 < self.outer_test_fraction < 1 or not 0 < self.inner_validation_fraction < 1:
            raise ValueError("state evaluation fractions must lie in (0, 1)")
        if self.min_train_pairs < 2 or not self.candidate_counts or 1 not in self.candidate_counts:
            raise ValueError("state selection needs min_train_pairs>=2 and a one-state candidate")
        if min(self.candidate_counts) < 1 or not self.assignments:
            raise ValueError("invalid state selection candidates")
        if not set(self.assignments) <= {"kmeans", "gaussian_mixture"}:
            raise ValueError("state selection supports kmeans and gaussian_mixture")
        if not self.ridge_penalties or any(not np.isfinite(x) or x <= 0 for x in self.ridge_penalties):
            raise ValueError("ridge penalties must be finite and positive")
        if not np.isfinite(self.minimum_relative_gain) or not 0 <= self.minimum_relative_gain < 1:
            raise ValueError("minimum_relative_gain must lie in [0, 1)")
        if self.graph.feature_representation != "signed_lag_offdiag":
            raise ValueError("prospective state evaluation requires signed_lag_offdiag features")


@dataclass
class StateEvaluationResult:
    status: str
    selected_model: GraphStateModel | None = None
    selected_algorithm: str = "unavailable"
    selected_n_states: int = 1
    state_selected: bool = False
    fit_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    evaluation_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected_algorithm": self.selected_algorithm,
            "selected_n_states": self.selected_n_states,
            "state_selected": self.state_selected,
            "fit_indices": self.fit_indices.tolist(),
            "evaluation_indices": self.evaluation_indices.tolist(),
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
        }


def _group(anchor: Any) -> tuple[Any, ...]:
    return anchor.dataset_id, anchor.recording_id, anchor.animal_id, anchor.session_id, anchor.segment_id


def future_graph_pairs(series: ConnectivitySeries) -> np.ndarray:
    """Pair histories within a segment without using an observed part of the target."""
    if len(series.anchors) != series.n_windows:
        raise ValueError("prospective state evaluation requires typed raw-history anchors")
    groups: dict[tuple[Any, ...], list[int]] = {}
    boundaries = set(boundary_indices_for_series(series).tolist())
    segment = 0
    previous_invalid = False
    for index, anchor in enumerate(series.anchors):
        if index in boundaries or previous_invalid or not anchor.valid:
            segment += 1
        previous_invalid = not anchor.valid
        if anchor.valid:
            groups.setdefault((*_group(anchor), segment), []).append(index)
    pairs = []
    for indices in groups.values():
        starts = np.array([series.anchors[i].context_start for i in indices])
        if np.any(np.diff(starts) <= 0):
            raise ValueError("graph histories must be strictly chronological within each segment")
        for index in indices:
            first = int(np.searchsorted(starts, series.anchors[index].context_stop, side="left"))
            if first < len(indices):
                pairs.append((index, indices[first]))
    return np.asarray(sorted(pairs), dtype=int).reshape(-1, 2)


def _split_pairs(
    series: ConnectivitySeries,
    pairs: np.ndarray,
    fraction: float,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray]:
    cut = int(np.floor(len(pairs) * (1 - fraction)))
    if cut < 1 or cut >= len(pairs):
        raise ValueError("too few future graph pairs for a chronological split")
    test = pairs[cut:]
    protected: dict[tuple[Any, ...], int] = {}
    for index in test.ravel():
        anchor = series.anchors[int(index)]
        key = _group(anchor)
        protected[key] = min(protected.get(key, anchor.context_start), anchor.context_start)
    train = []
    for pair in pairs[:cut]:
        if all(
            _group(series.anchors[int(i)]) not in protected
            or series.anchors[int(i)].context_stop <= protected[_group(series.anchors[int(i)])]
            for i in pair
        ):
            train.append(pair)
    training = np.asarray(train, dtype=int).reshape(-1, 2)
    if len(training) < minimum or not len(test):
        raise ValueError("raw-history purging leaves too few state-evaluation pairs")
    return training, test


def _fit_state(
    series: ConnectivitySeries,
    cfg: StateEvaluationConfig,
    indices: np.ndarray,
    assignment: str,
    count: int,
) -> GraphStateModel:
    graph = replace(
        cfg.graph, n_states=count, select_k=False, assignment=assignment, analysis_scope="prospective"
    )
    return fit_graph_states(series, graph, fit_indices=indices)


def _state_inputs(model: GraphStateModel, series: ConnectivitySeries, pairs: np.ndarray) -> np.ndarray:
    return model.transform_proba(series)[pairs[:, 0]]


def _loss(prediction: np.ndarray, target: np.ndarray) -> float:
    loss = float(np.mean((prediction - target) ** 2))
    if not np.isfinite(loss):
        raise ValueError("nonfinite state prediction loss")
    return loss


def evaluate_state_models(
    series: ConnectivitySeries,
    cfg: StateEvaluationConfig | None = None,
) -> StateEvaluationResult:
    """Select summaries using inner data, then evaluate the frozen choice once.

    Candidate state codes predict the same future feature vectors as constant and
    continuous ridge baselines. No outer-test risk selects an algorithm or state
    count. A selected multistate summary is not a validated latent-regime claim.
    """
    cfg = cfg or StateEvaluationConfig()
    diagnostics: dict[str, Any] = {
        "analysis_scope": "prospective_predictive_compression",
        "claim_eligibility": "development_only",
        "regime_claim_eligible": False,
        "response": "signed_lag_graph_features_in_training_fitted_coordinates",
        "target_support": "entire_target_history_begins_after_predictor_history_ends",
        "upstream_provenance": "graph_and_preprocessing_ownership_must_be_verified_separately",
        "configuration": asdict(cfg),
    }
    try:
        pairs = future_graph_pairs(series)
        outer_train, outer_test = _split_pairs(series, pairs, cfg.outer_test_fraction, cfg.min_train_pairs)
        inner_train, inner_test = _split_pairs(
            series,
            outer_train,
            cfg.inner_validation_fraction,
            cfg.min_train_pairs,
        )
    except ValueError as exc:
        return StateEvaluationResult(status="unavailable", diagnostics={**diagnostics, "reason": str(exc)})

    inner_indices = np.unique(inner_train)
    reference = _fit_state(series, cfg, inner_indices, "kmeans", 1)
    features = reference.transform_features(series)
    x, y = features[inner_train[:, 0]], features[inner_train[:, 1]]
    vx, vy = features[inner_test[:, 0]], features[inner_test[:, 1]]
    constant_loss = _loss(np.broadcast_to(y.mean(axis=0), vy.shape), vy)
    trials: list[dict[str, Any]] = [{"algorithm": "constant", "count": 1, "loss": constant_loss}]
    continuous = []
    for penalty in cfg.ridge_penalties:
        predictor = Ridge(alpha=penalty).fit(x, y)
        continuous.append((_loss(predictor.predict(vx), vy), penalty))
    continuous_loss, continuous_penalty = min(continuous)
    trials.append(
        {"algorithm": "continuous", "count": 1, "loss": continuous_loss, "ridge_penalty": continuous_penalty}
    )
    baseline_algorithm = "constant" if constant_loss <= continuous_loss else "continuous"
    baseline_loss = min(constant_loss, continuous_loss)
    selected_algorithm, selected_count, selected_penalty = baseline_algorithm, 1, continuous_penalty
    selected_loss = baseline_loss
    for assignment in cfg.assignments:
        for count in sorted(set(cfg.candidate_counts)):
            if count == 1 or count > len(inner_indices):
                continue
            try:
                model = _fit_state(series, cfg, inner_indices, assignment, count)
                train_inputs = _state_inputs(model, series, inner_train)
                validation_inputs = _state_inputs(model, series, inner_test)
                choices = []
                for penalty in cfg.ridge_penalties:
                    predictor = Ridge(alpha=penalty).fit(train_inputs, y)
                    choices.append((_loss(predictor.predict(validation_inputs), vy), penalty))
                loss, penalty = min(choices)
                trials.append(
                    {
                        "algorithm": assignment,
                        "count": count,
                        "loss": loss,
                        "ridge_penalty": penalty,
                        "status": "available",
                    }
                )
                if loss < selected_loss and loss < baseline_loss * (1 - cfg.minimum_relative_gain):
                    selected_algorithm, selected_count, selected_penalty = assignment, count, penalty
                    selected_loss = loss
            except (ValueError, np.linalg.LinAlgError) as exc:
                trials.append(
                    {"algorithm": assignment, "count": count, "status": "unavailable", "reason": str(exc)}
                )

    fit_indices = np.unique(outer_train)
    clustered = selected_algorithm in {"kmeans", "gaussian_mixture"}
    model = _fit_state(
        series, cfg, fit_indices, selected_algorithm if clustered else "kmeans", selected_count
    )
    features = model.transform_features(series)
    x, y = features[outer_train[:, 0]], features[outer_train[:, 1]]
    tx, ty = features[outer_test[:, 0]], features[outer_test[:, 1]]
    constant_mse = _loss(np.broadcast_to(y.mean(axis=0), ty.shape), ty)
    continuous_predictor = Ridge(alpha=continuous_penalty).fit(x, y)
    continuous_mse = _loss(continuous_predictor.predict(tx), ty)
    if clustered:
        predictor = Ridge(alpha=selected_penalty).fit(_state_inputs(model, series, outer_train), y)
        heldout_mse = _loss(predictor.predict(_state_inputs(model, series, outer_test)), ty)
    else:
        heldout_mse = constant_mse if selected_algorithm == "constant" else continuous_mse
    probabilities = model.transform_proba(series)[outer_test[:, 0]]
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
    baseline_outer_mse = constant_mse if baseline_algorithm == "constant" else continuous_mse
    relative_gain = (baseline_outer_mse - heldout_mse) / max(baseline_outer_mse, 1e-12)
    diagnostics.update(
        {
            "inner_trials": trials,
            "inner_selected_loss": selected_loss,
            "inner_baseline_algorithm": baseline_algorithm,
            "inner_baseline_loss": baseline_loss,
            "outer_train_pairs": outer_train.tolist(),
            "outer_test_pairs": outer_test.tolist(),
            "inner_train_pairs": inner_train.tolist(),
            "inner_validation_pairs": inner_test.tolist(),
            "selection_uses_outer_test": False,
            "predictive_validation_passed": bool(clustered and relative_gain > cfg.minimum_relative_gain),
            "interpretation": (
                "state code retained only for predictive compression; "
                "raw-null calibration still required"
            ),
        }
    )
    return StateEvaluationResult(
        status="available",
        selected_model=model,
        selected_algorithm=selected_algorithm,
        selected_n_states=selected_count,
        state_selected=clustered and selected_count > 1,
        fit_indices=fit_indices,
        evaluation_indices=outer_test[:, 0],
        metrics={
            "heldout_mse": heldout_mse,
            "constant_mse": constant_mse,
            "continuous_mse": continuous_mse,
            "relative_gain": relative_gain,
            "mean_membership_entropy": float(entropy.mean()),
            "forecast_pair_count": float(len(outer_test)),
        },
        diagnostics=diagnostics,
    )


def fit_frozen_state_transitions(
    series: ConnectivitySeries,
    result: StateEvaluationResult,
    transition_cfg: Any,
) -> Any:
    """Fit transitions only on selected training codes and score untouched codes."""
    from .transitions import fit_transition_model, score_transition_model

    if result.status != "available" or result.selected_model is None:
        raise ValueError("frozen transitions require an available state evaluation")
    labels = result.selected_model.labels
    original_boundaries = set(boundary_indices_for_series(series).tolist())

    def boundaries_for(indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            [0]
            + [
                j
                for j in range(1, len(indices))
                if indices[j] != indices[j - 1] + 1 or int(indices[j]) in original_boundaries
            ],
            dtype=int,
        )

    train = result.fit_indices
    test = result.evaluation_indices
    model = fit_transition_model(
        labels[train],
        result.selected_n_states,
        boundaries_for(train),
        transition_cfg,
    )
    score = score_transition_model(model, labels[test], boundaries_for(test))
    occupancy = np.bincount(labels[train], minlength=result.selected_n_states).astype(float)
    occupancy += transition_cfg.laplace
    occupancy /= occupancy.sum()
    baseline = replace(model, transition_matrix=np.tile(occupancy, (result.selected_n_states, 1)))
    baseline_score = score_transition_model(baseline, labels[test], boundaries_for(test))
    model.heldout_transition_log_likelihood = score.total_log_score
    model.heldout_baseline_log_likelihood = baseline_score.total_log_score
    model.heldout_delta_log_likelihood = score.total_log_score - baseline_score.total_log_score
    model.heldout_n_transitions = score.n_scored
    model.per_transition_log_scores = score.log_scores
    model.null_scope = "not_run_frozen_training_transition_prediction"
    return model
