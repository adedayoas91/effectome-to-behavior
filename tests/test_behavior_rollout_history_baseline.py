import json

import numpy as np

from effectome.experiments.behavior_readout import FrozenBehaviorMap, FrozenDenseLayer
from effectome.experiments.behavior_rollout_history_baseline import fit_history_baseline


def _readout() -> FrozenBehaviorMap:
    return FrozenBehaviorMap(
        "two_behavior_channels",
        np.zeros(2),
        np.ones(2),
        (FrozenDenseLayer(np.eye(2), np.array([4.0, -2.0])),),
    )


def test_history_baseline_forecasts_absolute_behavior_with_whole_trial_folds() -> None:
    training = np.broadcast_to([1.0, 0.0], (6, 24, 2))
    groups = np.repeat([10, 11, 12], 2)
    test = np.broadcast_to([50.0, -50.0], (2, 24, 2))
    points = np.array([8, 16])
    predicted, tuning = fit_history_baseline(
        training, groups, test, points, _readout(), history=6, horizon=4, target_length=2, stride=4
    )

    assert predicted.shape == (2, 2, 2)
    np.testing.assert_allclose(predicted, np.broadcast_to([5.0, -2.0], predicted.shape))
    assert tuning["training_windows"] == 6
    assert tuning["training_events"] == 24
    assert tuning["target"] == "absolute_terminal_window_mean_of_frozen_behavior_readout"
    validation = []
    for fold in tuning["folds"]:
        assert set(fold["train_trials"]).isdisjoint(fold["validation_trials"])
        validation.extend(fold["validation_trials"])
    assert sorted(validation) == [10, 11, 12]
    json.dumps(tuning, allow_nan=False)


def test_history_baseline_uses_only_neural_samples_through_each_origin() -> None:
    rng = np.random.default_rng(114)
    training = rng.normal(size=(6, 24, 2))
    groups = np.repeat([10, 11, 12], 2)
    test = rng.normal(size=(2, 24, 2))
    settings = {"history": 6, "horizon": 4, "target_length": 2, "stride": 4}
    points = np.array([8, 16])
    readout = _readout()
    original, selected = fit_history_baseline(training, groups, test, points, readout, **settings)

    mutated = test.copy()
    mutated[:, 9:17] += 50
    mutated[:, 17:] = np.nan
    changed, repeat = fit_history_baseline(training, groups, mutated, points, readout, **settings)
    truncated, _ = fit_history_baseline(
        training, groups, test[:, :17], points, readout, **settings
    )

    np.testing.assert_allclose(changed[:, 0], original[:, 0], atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(truncated, original)
    assert not np.allclose(changed[:, 1], original[:, 1])
    assert repeat == selected
