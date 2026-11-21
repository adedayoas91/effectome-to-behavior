"""Verify that local-refit choices depend on available forecasts only."""

import numpy as np
import pytest

from effectome.experiments.behavior_rollout_online_policy import refit_decisions


def test_irregular_origins_use_exact_maturity_full_lookback_and_trial_boundaries():
    origins = np.array([[0, 2, 5, 8, 9, 15], [40, 41, 43, 44, 45, 46]])
    targets = np.zeros((*origins.shape, 2))
    pooled = np.full_like(targets, 0.5)
    local = np.repeat(
        np.array([[0.25, 0.25, 0.5, 0.25, 0, 0], [1, 1, 1, 1, 1, 1]])[:, :, None],
        repeats=2,
        axis=2,
    )
    selected = refit_decisions(origins, pooled, local, targets, horizon=3, lookback=2, margin=0.125)
    np.testing.assert_array_equal(
        selected, [[False, False, True, False, False, True], [False, False, False, False, False, False]]
    )
    assert selected.dtype == np.dtype(bool)


def test_changing_unavailable_outcomes_and_forecasts_preserves_earlier_decisions():
    origins = np.arange(12)[None, :]
    targets = np.zeros((*origins.shape, 2))
    pooled = np.full_like(targets, 0.25)
    local = np.zeros_like(targets)
    kwargs = {"horizon": 3, "lookback": 2, "margin": 0.1}
    before = refit_decisions(origins, pooled, local, targets, **kwargs)

    # The changed outcomes belong to origins whose first maturity is sample 7.
    targets[:, 4:] = 1
    pooled[:, 4:] = 0.75
    after = refit_decisions(origins, pooled, local, targets, **kwargs)
    np.testing.assert_array_equal(after[:, :7], before[:, :7])
    assert before[0, 7]
    assert not after[0, 7]


def test_matured_forecast_losses_average_both_behavior_coordinates():
    origins = np.array([[0, 1, 2]])
    targets = np.zeros((*origins.shape, 2))
    pooled = np.broadcast_to([1.0, 0.0], targets.shape)
    local = np.broadcast_to([0.5, 0.75], targets.shape)
    selected = refit_decisions(origins, pooled, local, targets, horizon=1, lookback=1, margin=0)
    assert not selected.any()


def test_forecast_count_shorter_than_lookback_keeps_pooling():
    origins = np.array([[3, 6]])
    values = np.zeros((*origins.shape, 2))
    selected = refit_decisions(origins, values + 1, values, values, horizon=2, lookback=3, margin=0)
    np.testing.assert_array_equal(selected, [[False, False]])


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"origins": np.array([[2, 2]])}, "increase strictly"),
        ({"origins": np.array([[0.0, 1.0]])}, "integer sample indices"),
        ({"origins": np.array([[-1, 0]])}, "nonnegative integer"),
        ({"pooled": np.zeros((1, 2, 1))}, "pooled must be finite real"),
        ({"local": np.zeros((1, 2, 2), dtype=complex)}, "local must be finite real"),
        ({"observed_targets": np.full((1, 2, 2), np.nan)}, "observed_targets must be finite real"),
        ({"horizon": 0}, "horizon"),
        ({"lookback": True}, "lookback"),
        ({"margin": float("inf")}, "margin"),
    ],
)
def test_invalid_inputs_fail_clearly(changed, message):
    values = np.zeros((1, 2, 2))
    args = {
        "origins": np.array([[0, 1]]),
        "pooled": values,
        "local": values,
        "observed_targets": values,
        "horizon": 1,
        "lookback": 1,
        "margin": 0.0,
    }
    with pytest.raises(ValueError, match=message):
        refit_decisions(**(args | changed))
