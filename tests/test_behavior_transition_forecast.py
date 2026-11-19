import json

import numpy as np
import pytest

from effectome.experiments.behavior_transition_forecast import (
    fit_transition_forecast,
    transition_features,
)


def test_forecast_features_and_targets_use_declared_time_support() -> None:
    rng = np.random.default_rng(32)
    neural = rng.normal(size=(2, 20, 3))
    behavior = rng.normal(size=(2, 20, 2))
    codes = rng.normal(size=(2, 20, 2))
    origins = np.array([[7, 15], [8, 16]], dtype=np.uint64)
    settings = {"history_length": 4, "horizon": 3, "target_length": 2}
    baseline, states, y = transition_features(neural, behavior, codes, origins, **settings)

    flat = [(trial, int(t)) for trial, times in enumerate(origins) for t in times]
    np.testing.assert_array_equal(states, np.stack([neural[trial, t] for trial, t in flat]))
    np.testing.assert_array_equal(
        y, np.stack([behavior[trial, t + 2 : t + 4].mean(axis=0) - behavior[trial, t] for trial, t in flat])
    )

    alternate_neural = neural.copy()
    alternate_behavior = behavior.copy()
    alternate_codes = codes.copy()
    for trial, t in enumerate(origins[:, -1].astype(int)):
        alternate_neural[trial, t + 1 :] += 50
        alternate_behavior[trial, t + 1 :] += 30
        alternate_codes[trial, t + 1 :] += 20
    altered, altered_states, altered_y = transition_features(
        alternate_neural, alternate_behavior, alternate_codes, origins, **settings
    )
    np.testing.assert_array_equal(altered, baseline)
    np.testing.assert_array_equal(altered_states, states)
    np.testing.assert_allclose(altered_y[1::2] - y[1::2], 30)
    np.testing.assert_array_equal(altered_y[::2], y[::2])


@pytest.mark.parametrize("origin", [-1, 2, 8, 4.5])
def test_forecast_rejects_origins_without_integer_in_trial_support(origin: float) -> None:
    values = np.zeros((2, 10, 2))
    with pytest.raises(ValueError, match="origins"):
        transition_features(
            values,
            values,
            values,
            np.array([[origin], [origin]]),
            history_length=4,
            horizon=2,
            target_length=2,
        )


def test_grouped_tuning_uses_whole_trials_and_same_grid_for_graph_arms() -> None:
    rng = np.random.default_rng(45)
    trials = np.repeat(np.arange(14), 2)
    states = np.repeat(rng.normal(size=(14, 2)), 2, axis=0)
    weights = np.zeros((len(states), 2, 2))
    weights[:, 0, 1] = np.tile([1.0, -1.0], 14)
    action = weights[:, 0, 1] * np.tanh(states[:, 0])
    y = np.column_stack((action, -action / 2))
    training = trials < 10
    baseline = fit_transition_forecast(states[training], states[training], y[training], trials[training])
    augmented = fit_transition_forecast(
        states[training], states[training], y[training], trials[training], weights=weights[training]
    )
    assert baseline.tuning["folds"] == augmented.tuning["folds"]
    validation_groups = []
    for fold in augmented.tuning["folds"]:
        assert set(fold["train_trials"]).isdisjoint(fold["validation_trials"])
        validation_groups.extend(fold["validation_trials"])
    assert sorted(validation_groups) == list(range(10))
    for standard, graph in zip(baseline.tuning["candidates"], augmented.tuning["candidates"], strict=True):
        assert [standard[key] for key in ("family", "alpha", "gamma_scale")] == [
            graph[key] for key in ("family", "alpha", "gamma_scale")
        ]
    json.dumps(augmented.tuning, allow_nan=False)

    plain = baseline.predict(states[~training], states[~training])
    with_graph = augmented.predict(states[~training], states[~training], weights[~training])
    assert np.abs(y[~training] - with_graph).mean() < 0.1 * np.abs(y[~training] - plain).mean()
    with pytest.raises(ValueError, match="weights"):
        augmented.predict(states[~training], states[~training])
