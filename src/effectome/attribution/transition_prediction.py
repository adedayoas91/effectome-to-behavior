"""Paired prediction and feature-omission evidence for shared manifold transitions.

The caller owns raw-time purging and the common event table. All fitted moments,
decoder parameters, and omission refits here use ``train_rows`` only. Importance
measures a graph feature block's predictive contribution, not neuronal necessity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from effectome.linking.stats import brier_score, classification_log_loss

_PROBABILITY_FLOOR = 1e-12


def _indices(values: np.ndarray, name: str, n_events: int) -> np.ndarray:
    indices = np.asarray(values)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or indices.size == 0
        or np.any(indices < 0)
        or np.any(indices >= n_events)
        or np.any(np.diff(indices) <= 0)
    ):
        raise ValueError(f"{name} must be nonempty, strictly increasing integer event indices")
    return indices.astype(int)


def _states(values: np.ndarray, name: str, n_events: int, n_states: int) -> np.ndarray:
    states = np.asarray(values)
    if (
        states.shape != (n_events,)
        or not np.issubdtype(states.dtype, np.integer)
        or np.any(states < 0)
        or np.any(states >= n_states)
    ):
        raise ValueError(f"{name} must contain integer states in [0, n_states)")
    return states.astype(int)


def _fit_probabilities(
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    *,
    n_states: int,
    c: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(features[train])
    classes = np.unique(targets[train])
    metadata: dict[str, Any] = {
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "observed_training_classes": classes.tolist(),
        "n_training_events": int(train.size),
    }
    probabilities = np.full((evaluation.size, n_states), _PROBABILITY_FLOOR)
    if classes.size == 1:
        probabilities[:, int(classes[0])] = 1.0
        metadata.update({"kind": "training_occupancy", "coefficients": [], "intercept": []})
    else:
        model = LogisticRegression(C=c, max_iter=10_000, random_state=seed).fit(
            scaler.transform(features[train]), targets[train]
        )
        if np.any(np.asarray(model.n_iter_) >= model.max_iter):
            raise RuntimeError("transition logistic predictor did not converge within its iteration limit")
        probabilities[:, model.classes_.astype(int)] = model.predict_proba(
            scaler.transform(features[evaluation])
        )
        metadata.update(
            kind="regularized_logistic",
            coefficients=model.coef_.tolist(),
            intercept=model.intercept_.tolist(),
            iterations=model.n_iter_.tolist(),
        )
    probabilities = np.maximum(probabilities, _PROBABILITY_FLOOR)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, metadata


def _binary_log_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    probabilities = np.clip(probabilities, _PROBABILITY_FLOOR, 1 - _PROBABILITY_FLOOR)
    return float(-np.mean(np.where(targets, np.log(probabilities), np.log1p(-probabilities))))


def _score(probabilities: np.ndarray, sources: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    n_states = probabilities.shape[1]
    exits = targets != sources
    exit_probabilities = 1.0 - probabilities[np.arange(sources.size), sources]
    conditional_destination_loss = None
    if np.any(exits):
        destination = probabilities[exits].copy()
        destination[np.arange(int(exits.sum())), sources[exits]] = 0.0
        destination /= destination.sum(axis=1, keepdims=True)
        conditional_destination_loss = classification_log_loss(
            destination, targets[exits], class_order=list(range(n_states))
        )
    return {
        "n_origins": int(sources.size),
        "n_exits": int(exits.sum()),
        "log_loss": classification_log_loss(probabilities, targets, class_order=list(range(n_states))),
        "brier": brier_score(probabilities, targets, class_order=list(range(n_states))),
        "exit_log_loss": _binary_log_loss(exit_probabilities, exits),
        "exit_brier": float(np.mean((exit_probabilities - exits) ** 2)),
        "conditional_destination_log_loss": conditional_destination_loss,
    }


def _transition_matrices(
    sources: np.ndarray, targets: np.ndarray, baseline: np.ndarray, augmented: np.ndarray, n_states: int
) -> dict[str, Any]:
    counts = np.zeros((n_states, n_states), dtype=int)
    np.add.at(counts, (sources, targets), 1)
    source_counts = counts.sum(axis=1)
    empirical: list[list[float] | None] = []
    baseline_rows: list[list[float] | None] = []
    augmented_rows: list[list[float] | None] = []
    for state in range(n_states):
        selected = sources == state
        if source_counts[state] == 0:
            empirical.append(None)
            baseline_rows.append(None)
            augmented_rows.append(None)
        else:
            empirical.append((counts[state] / source_counts[state]).tolist())
            baseline_rows.append(baseline[selected].mean(axis=0).tolist())
            augmented_rows.append(augmented[selected].mean(axis=0).tolist())
    return {
        "counts": counts.tolist(),
        "source_counts": source_counts.tolist(),
        "empirical": empirical,
        "baseline_predicted": baseline_rows,
        "augmented_predicted": augmented_rows,
        "unsupported_rows": np.flatnonzero(source_counts == 0).tolist(),
        "scope": "same_heldout_origins_fixed_horizon",
    }


def _displacement_scores(
    baseline: np.ndarray,
    augmented: np.ndarray,
    displacement: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    c: float,
) -> dict[str, Any]:
    predictions = []
    metadata = []
    for features in (baseline, augmented):
        scaler = StandardScaler().fit(features[train])
        model = Ridge(alpha=1.0 / c).fit(scaler.transform(features[train]), displacement[train])
        predictions.append(np.asarray(model.predict(scaler.transform(features[evaluation]))))
        metadata.append({
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficients": model.coef_.tolist(),
            "intercept": np.asarray(model.intercept_).tolist(),
        })
    baseline_mse, augmented_mse = (
        float(np.mean((prediction - displacement[evaluation]) ** 2)) for prediction in predictions
    )
    return {
        "baseline_mse": baseline_mse,
        "augmented_mse": augmented_mse,
        "mse_gain": baseline_mse - augmented_mse,
        "baseline_predictions": predictions[0].tolist(),
        "augmented_predictions": predictions[1].tolist(),
        "training_parameters": {"baseline": metadata[0], "augmented": metadata[1]},
        "ridge_alpha": 1.0 / c,
    }


def evaluate_transition_predictors(
    baseline: np.ndarray,
    graph_features: np.ndarray,
    source_states: np.ndarray,
    target_states: np.ndarray,
    *,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    feature_blocks: Mapping[str, Sequence[int]],
    n_states: int,
    seed: int = 42,
    c: float = 1.0,
    future_displacement: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare common-event transition predictors and refitted graph-block omissions.

    The state vocabulary is declared before evaluation. Source-state one-hot
    indicators are appended automatically to both models. Every A-to-B omission
    score uses all evaluation origins in A, including stays and other destinations.
    The fixed ``c`` is never selected from held-out outcomes. Continuous predictions
    use a Ridge penalty of ``1 / c``. The caller must supply causally available
    features and enforce raw-history/outcome purging before invoking this function.
    """
    if isinstance(n_states, bool) or not isinstance(n_states, (int, np.integer)) or n_states < 1:
        raise ValueError("n_states must be a positive integer")
    if not np.isfinite(c) or c <= 0:
        raise ValueError("c must be finite and positive")
    base = np.asarray(baseline, dtype=float)
    graph = np.asarray(graph_features, dtype=float)
    if (
        base.ndim != 2 or graph.ndim != 2 or base.shape[0] != graph.shape[0]
        or not np.all(np.isfinite(base)) or not np.all(np.isfinite(graph))
    ):
        raise ValueError("baseline and graph_features must be finite aligned event-by-feature matrices")
    n_events = base.shape[0]
    train = _indices(train_rows, "train_rows", n_events)
    evaluation = _indices(evaluation_rows, "evaluation_rows", n_events)
    if np.intersect1d(train, evaluation).size:
        raise ValueError("train_rows and evaluation_rows must be disjoint")
    sources = _states(source_states, "source_states", n_events, n_states)
    targets = _states(target_states, "target_states", n_events, n_states)
    blocks: dict[str, np.ndarray] = {}
    for name, columns in feature_blocks.items():
        values = np.asarray(columns)
        if (
            not isinstance(name, str) or not name or values.ndim != 1 or values.size == 0
            or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values >= graph.shape[1])
            or np.unique(values).size != values.size
        ):
            raise ValueError("feature_blocks need nonempty names and unique integer graph-column indices")
        blocks[name] = values.astype(int)
    displacement = None
    if future_displacement is not None:
        displacement = np.asarray(future_displacement, dtype=float)
        if displacement.ndim != 2 or displacement.shape[0] != n_events or not np.all(
            np.isfinite(displacement)
        ) or displacement.shape[1] == 0:
            raise ValueError("future_displacement must be a finite event-by-coordinate matrix")
    conditioned_base = np.column_stack((base, np.eye(n_states)[sources]))
    augmented = np.column_stack((conditioned_base, graph))
    baseline_probabilities, baseline_fit = _fit_probabilities(
        conditioned_base, targets, train, evaluation, n_states=n_states, c=c, seed=seed
    )
    augmented_probabilities, augmented_fit = _fit_probabilities(
        augmented, targets, train, evaluation, n_states=n_states, c=c, seed=seed
    )
    eval_sources, eval_targets = sources[evaluation], targets[evaluation]
    baseline_score = _score(baseline_probabilities, eval_sources, eval_targets)
    augmented_score = _score(augmented_probabilities, eval_sources, eval_targets)
    omissions: list[dict[str, Any]] = []
    for name, block_indices in blocks.items():
        retained = np.ones(graph.shape[1], dtype=bool)
        retained[block_indices] = False
        omitted_features = np.column_stack((conditioned_base, graph[:, retained]))
        probabilities, _ = _fit_probabilities(
            omitted_features, targets, train, evaluation, n_states=n_states, c=c, seed=seed
        )
        for source in range(n_states):
            selected = eval_sources == source
            if not np.any(selected):
                continue
            for destination in range(n_states):
                if source == destination:
                    continue
                observed = eval_targets[selected] == destination
                full_loss = _binary_log_loss(augmented_probabilities[selected, destination], observed)
                omitted_loss = _binary_log_loss(probabilities[selected, destination], observed)
                omissions.append({
                    "unit": name,
                    "source_state": source,
                    "destination_state": destination,
                    "n_origins": int(selected.sum()),
                    "n_events": int(observed.sum()),
                    "full_log_loss": full_loss,
                    "omitted_log_loss": omitted_loss,
                    "omission_delta_log_loss": omitted_loss - full_loss,
                    "n_omitted_features": int(block_indices.size),
                    "is_validated_driver": False,
                })
    return {
        "status": "development_predictive_comparison",
        "class_order": list(range(n_states)),
        "train_rows": train.tolist(),
        "evaluation_rows": evaluation.tolist(),
        "baseline_probabilities": baseline_probabilities.tolist(),
        "augmented_probabilities": augmented_probabilities.tolist(),
        "baseline_scores": baseline_score,
        "augmented_scores": augmented_score,
        "log_loss_gain": baseline_score["log_loss"] - augmented_score["log_loss"],
        "transition_matrices": _transition_matrices(
            eval_sources, eval_targets, baseline_probabilities, augmented_probabilities, n_states
        ),
        "omission_scores": omissions,
        "training_parameters": {"baseline": baseline_fit, "augmented": augmented_fit},
        "training_source_counts": np.bincount(sources[train], minlength=n_states).tolist(),
        "displacement": None if displacement is None else _displacement_scores(
            conditioned_base, augmented, displacement, train, evaluation, c
        ),
        "claim_eligibility": "development_only_not_biological_necessity",
        "raw_dependency_ownership": "caller_enforced",
    }
