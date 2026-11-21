"""Regression tests for rapid fallback using completed forecasts."""

import numpy as np

from effectome.experiments.behavior_rollout_online_policy import (
    fast_refit_decisions,
    fast_refit_loss_decisions,
    refit_decisions,
)


def _forecasts(gains: list[int]) -> tuple[np.ndarray, ...]:
    values = np.asarray(gains)[None, :, None]
    pooled = np.repeat(np.maximum(values, 0), 2, axis=-1)
    local = np.repeat(np.maximum(-values, 0), 2, axis=-1)
    return np.arange(len(gains))[None, :], pooled, local, np.zeros_like(pooled)


def test_fast_fallback_waits_for_new_forecasts_before_resuming():
    traces = _forecasts([1] * 7 + [-1, -1] + [1] * 6)
    slow = refit_decisions(*traces, horizon=1, lookback=6, margin=0)
    fast = fast_refit_decisions(*traces, horizon=1, lookback=6)
    assert slow[0, 9:13].all()
    assert fast[0, 8]
    assert not fast[0, 9:13].any()
    assert fast[0, 13:].all()


def test_an_unmatured_target_cannot_trigger_fast_fallback():
    origins, pooled, local, observed = _forecasts([1] * 14)
    baseline = fast_refit_decisions(origins, pooled, local, observed, horizon=3, lookback=4)
    observed[:, 8:] = 3
    changed = fast_refit_decisions(origins, pooled, local, observed, horizon=3, lookback=4)
    np.testing.assert_array_equal(changed[:, :12], baseline[:, :12])
    assert baseline[0, 12] and not changed[0, 12]


def test_maturing_multiple_forecasts_between_origins_does_not_skip_losses():
    _, pooled, local, observed = _forecasts([1, 1, 1, 1, -1, -1, 1, 1])
    origins = np.array([[0, 1, 2, 3, 10, 11, 12, 20]])
    slow = refit_decisions(origins, pooled, local, observed, horizon=6, lookback=6, margin=0)
    fast = fast_refit_decisions(origins, pooled, local, observed, horizon=6, lookback=6)
    assert slow[0, -1] and not fast[0, -1]


def test_custom_loss_has_the_same_maturity_and_fallback_rule():
    origins, pooled, local, target = _forecasts([1] * 6 + [-1] * 3 + [1] * 5)
    settings = dict(horizon=2, lookback=3, margin=0.0, consecutive_losses=3, recovery_forecasts=1)
    pool_loss = np.abs(pooled - target).mean(axis=-1)
    local_loss = np.abs(local - target).mean(axis=-1)
    reference = fast_refit_decisions(origins, pooled, local, target, **settings)
    replay = fast_refit_loss_decisions(origins, pool_loss, local_loss, **settings)
    np.testing.assert_array_equal(replay, reference)
    changed = local_loss.copy()
    changed[:, 6:] += 5
    future = fast_refit_loss_decisions(origins, pool_loss, changed, **settings)
    np.testing.assert_array_equal(future[:, :8], replay[:, :8])
