"""Verify that tuning evaluates the same policy used for held-out forecasts."""

from itertools import product

import numpy as np

from synthetic.behavior_rollout_tuning import Policy, _grid_masks, _rank, _replay


def test_grid_matches_production_rule_with_delayed_and_batched_targets():
    rng = np.random.default_rng(580)
    shape = (5, 28, 2)
    case = {
        "origin_samples": rng.integers(1, 15, size=shape[1]).cumsum(),
        "pooled": rng.integers(-2, 3, size=shape) / 2,
        "local": rng.integers(-2, 3, size=shape) / 2,
        "decoded_observations": rng.integers(-2, 3, size=shape) / 2,
    }
    policies = [
        Policy(window, margin, losses, recovery, 32)
        for window, margin, losses, recovery in product((1, 4, 12, 29), (0.0, 0.25), (1, 2, 29), (1, 4, 29))
    ]
    for policy, mask in zip(policies, _grid_masks(case, policies), strict=True):
        np.testing.assert_array_equal(mask, _replay(case, policy))


def test_ranking_prioritizes_both_forecast_panels_and_protects_effect_accuracy():
    baseline = {
        "both_panels_pass": False,
        "worst_seed_count": 13,
        "driver_mae": 0.21,
        "early_gain": 0.01,
        "active_effect_mae": 0.075,
        "effect_eligible": True,
    }
    eligible = {**baseline, "both_panels_pass": True, "worst_seed_count": 15, "driver_mae": 0.202}
    poor_effect = {**eligible, "driver_mae": 0.18, "effect_eligible": False}
    poor_early = {**baseline, "driver_mae": 0.19}
    assert _rank([baseline, eligible, poor_effect, poor_early]) == eligible
    assert _rank([baseline, poor_effect, {**poor_early, "worst_seed_count": 12}]) == baseline
