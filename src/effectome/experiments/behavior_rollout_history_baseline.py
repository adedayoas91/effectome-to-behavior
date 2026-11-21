"""Fit a direct behavior forecast from graph-free neural-history features."""

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from .behavior_readout import FrozenBehaviorMap
from .behavior_transition_forecast import fit_transition_forecast, transition_features


def fit_history_baseline(
    training: np.ndarray,
    training_groups: np.ndarray,
    neural: np.ndarray,
    points: np.ndarray,
    readout: FrozenBehaviorMap,
    *,
    history: int,
    horizon: int,
    target_length: int,
    stride: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit absolute decoded future behavior and apply it to held-out histories.

    Training windows from the same original trial share a validation group.
    The frozen encoder and decoder provide only samplewise transformations
    of neural activity. Forecasts use the same terminal interval and
    trailing history as the rollout arms.
    """
    for name, value in (
        ("history", history),
        ("horizon", horizon),
        ("target_length", target_length),
        ("stride", stride),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    train = np.asarray(training)
    test = np.asarray(neural)
    groups = np.asarray(training_groups)
    test_points = np.asarray(points)
    if any(values.ndim != 3 or not values.size for values in (train, test)):
        raise ValueError("training and test observations need nonempty trial, time, and neural axes")
    if groups.shape != (len(train),):
        raise ValueError("training groups must identify the original trial of every training window")
    if (
        test_points.ndim != 1
        or not test_points.size
        or test_points.dtype.kind not in "iu"
        or np.any(test_points < history - 1)
        or np.any(test_points >= test.shape[1])
        or np.any(test_points[1:] <= test_points[:-1])
    ):
        raise ValueError("test forecast points must be ordered integers with an observed trailing history")

    train_points = np.arange(history - 1, train.shape[1] - horizon, stride)
    settings = {"history_length": history, "horizon": horizon, "target_length": target_length}
    train_readout = readout.predict(train)
    train_features, train_states, train_changes = transition_features(
        train,
        train_readout,
        readout.encode(train),
        np.broadcast_to(train_points, (len(train), len(train_points))),
        **settings,
    )
    train_target = train_changes + train_readout[:, train_points].reshape(train_changes.shape)
    model = fit_transition_forecast(
        train_features, train_states, train_target, np.repeat(groups, len(train_points))
    )

    # The shared helper also computes targets. Pad beyond the last observed
    # origin so its unused test targets require no additional observations.
    observed = test[:, : int(test_points[-1]) + 1]
    padded = np.pad(observed, ((0, 0), (0, horizon), (0, 0)), mode="edge")
    test_features, test_states, _ = transition_features(
        padded,
        readout.predict(padded),
        readout.encode(padded),
        np.broadcast_to(test_points, (len(test), len(test_points))),
        **settings,
    )
    predictions = model.predict(test_features, test_states).reshape(len(test), len(test_points), 2)
    return predictions, {
        **model.tuning,
        "target": "absolute_terminal_window_mean_of_frozen_behavior_readout",
        "training_windows": len(train),
        "training_events": len(train_features),
    }
