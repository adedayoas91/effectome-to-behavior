"""Grouped prospective forecasts from activity and origin-owned graph features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

_RECENT_LAGS = (0, 1, 2, 4, 8, 16)


def transition_features(
    neural: np.ndarray,
    behavior: np.ndarray,
    codes: np.ndarray,
    origins: np.ndarray,
    *,
    history_length: int,
    horizon: int,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return past features, origin states, and future changes in trial-major order.

    Origin indices identify the last complete sample. Histories end at that
    sample and use the same support as the graph. The behavioral target is the
    terminal future-window mean minus the observed behavior at the origin.
    """
    for name, value in (
        ("history_length", history_length),
        ("horizon", horizon),
        ("target_length", target_length),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if history_length < 2 or target_length > horizon:
        raise ValueError("lag moments require two history samples and the target must be strictly future")
    x, b, z = (np.asarray(values, dtype=float) for values in (neural, behavior, codes))
    if any(
        values.ndim != 3
        or not values.size
        or values.shape[:2] != x.shape[:2]
        or not np.isfinite(values).all()
        for values in (x, b, z)
    ):
        raise ValueError("neural, behavior, and codes must be finite aligned trial-by-time arrays")
    t = np.asarray(origins)
    if (
        t.ndim != 2
        or t.shape[0] != x.shape[0]
        or not t.size
        or not np.issubdtype(t.dtype, np.integer)
        or np.any(t < history_length - 1)
        or np.any(t > x.shape[1] - 1 - horizon)
        or np.any(t[:, 1:] <= t[:, :-1])
    ):
        raise ValueError("origins must be ordered integers with in-trial history and future support")
    t = t.astype(np.intp, copy=False)
    lags = np.asarray([lag for lag in _RECENT_LAGS if lag < history_length], dtype=int)
    expected_history = np.concatenate((x, b, z), axis=-1)
    features = []
    for trial, row in enumerate(t):
        for origin in row:
            recent = expected_history[trial, origin - lags]
            history = expected_history[trial, origin + 1 - history_length : origin + 1]
            # Retain marginal summaries and unselected lagged dependence. This
            # gives the graph-free model access to the same longer raw history.
            previous = np.tanh(history[:-1, : x.shape[-1]])
            following = history[1:, : x.shape[-1]]
            lag_covariance = (
                (previous - previous.mean(axis=0)).T
                @ (following - following.mean(axis=0))
                / (history_length - 1)
            )
            features.append(
                np.concatenate(
                    (recent.ravel(), history.mean(axis=0), history.std(axis=0), lag_covariance.ravel())
                )
            )
    trials = np.repeat(np.arange(len(x)), t.shape[1])
    flat_origins = t.ravel()
    terminal_offsets = np.arange(horizon - target_length + 1, horizon + 1)
    target = b[trials[:, None], flat_origins[:, None] + terminal_offsets].mean(axis=1)
    return np.stack(features), x[trials, flat_origins], target - b[trials, flat_origins]


def _design(baseline: np.ndarray, states: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    history, initial = (np.asarray(values, dtype=float) for values in (baseline, states))
    if (
        history.ndim != 2
        or initial.ndim != 2
        or not history.size
        or not initial.size
        or len(history) != len(initial)
        or not np.isfinite(history).all()
        or not np.isfinite(initial).all()
    ):
        raise ValueError("baseline and states must be finite event-aligned matrices")
    if weights is None:
        return history
    interaction = np.asarray(weights, dtype=float)
    if (
        interaction.shape != (len(initial), initial.shape[1], initial.shape[1])
        or not np.isfinite(interaction).all()
    ):
        raise ValueError("weights must be finite event-by-source-by-target matrices")
    action = np.einsum("es,est->et", np.tanh(initial), interaction)
    return np.column_stack((history, action))


@dataclass(frozen=True)
class TransitionForecast:
    """A fitted forecast with its graph input contract and grouped tuning record."""

    model: Pipeline
    baseline_width: int
    state_width: int
    uses_weights: bool
    tuning: dict[str, Any]

    def predict(
        self, baseline: np.ndarray, states: np.ndarray, weights: np.ndarray | None = None
    ) -> np.ndarray:
        if self.uses_weights != (weights is not None):
            raise ValueError("weights must be supplied exactly for graph-augmented forecasts")
        design = _design(baseline, states, weights)
        if np.shape(baseline)[1] != self.baseline_width or np.shape(states)[1] != self.state_width:
            raise ValueError("prediction baseline and state widths must match fitting")
        predicted = np.asarray(self.model.predict(design), dtype=float)
        if not np.isfinite(predicted).all():
            raise ValueError("forecasts must be finite")
        return predicted


def _candidate_model(family: str, alpha: float, gamma_scale: float, width: int) -> Pipeline:
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    # Centering the outputs gives the kernel an intercept. Scale the kernel
    # width by dimension so the same gamma grid applies to every feature arm.
    kernel = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma_scale / width)
    return make_pipeline(
        StandardScaler(),
        TransformedTargetRegressor(regressor=kernel, transformer=StandardScaler(with_std=False)),
    )


def fit_transition_forecast(
    baseline: np.ndarray,
    states: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> TransitionForecast:
    """Select from a common fixed grid by out-of-fold MAE across whole training trials."""
    design = _design(baseline, states, weights)
    target = np.asarray(y, dtype=float)
    trials = np.asarray(groups)
    if target.ndim != 2 or not target.size or len(target) != len(design) or not np.isfinite(target).all():
        raise ValueError("y must be a finite event-by-behavior matrix")
    if trials.shape != (len(design),) or trials.dtype.kind not in "iuU":
        raise ValueError("groups must identify each event's training trial by integer or string")
    unique, trial_index, counts = np.unique(trials, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        raise ValueError("forecast tuning requires at least two whole training trials")
    folds = list(GroupKFold(n_splits=min(5, len(unique))).split(design, groups=trials))
    candidates = [
        *(("ridge", alpha, 0.0) for alpha in (0.1, 1.0, 10.0, 100.0)),
        *(("rbf", alpha, scale) for alpha in (0.1, 1.0, 10.0) for scale in (0.25, 1.0)),
    ]
    scores = []
    for family, alpha, scale in candidates:
        error = np.empty(len(design))
        for train, validation in folds:
            model = _candidate_model(family, alpha, scale, design.shape[1])
            model.fit(design[train], target[train])
            error[validation] = np.abs(model.predict(design[validation]) - target[validation]).mean(axis=1)
        scores.append(float(np.mean(np.bincount(trial_index, weights=error) / counts)))
    selected = int(np.argmin(scores))
    model = _candidate_model(*candidates[selected], design.shape[1])
    model.fit(design, target)
    return TransitionForecast(
        model,
        np.shape(baseline)[1],
        np.shape(states)[1],
        weights is not None,
        {
            "fit_scope": "supplied_training_trials_only",
            "selection_metric": "mean_over_trials_of_event_behavior_mae",
            "selected_candidate": selected,
            "candidates": [
                {"family": family, "alpha": alpha, "gamma_scale": scale, "validation_mae": scores[index]}
                for index, (family, alpha, scale) in enumerate(candidates)
            ],
            "folds": [
                {
                    "train_trials": np.unique(trials[train]).tolist(),
                    "validation_trials": np.unique(trials[validation]).tolist(),
                }
                for train, validation in folds
            ],
        },
    )
