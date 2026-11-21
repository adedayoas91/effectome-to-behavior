"""Check the matched support controls and their within-seed aggregation."""

import numpy as np
import pytest

from synthetic.behavior_rollout_support import _replicate_mean, _support_masks


def test_shuffles_preserve_directed_parent_counts_and_are_reproducible():
    learned = np.zeros((6, 6), dtype=bool)
    learned[0, 1:] = True
    learned[[2, 3], 1] = True
    learned[2, 0] = True
    first = _support_masks(learned, 91003, 3)
    second = _support_masks(learned, 91003, 3)

    np.testing.assert_array_equal(first["learned"], learned)
    assert not np.diag(first["dense"]).any()
    assert first["dense"].sum() == 30
    for name in ("shuffle_1", "shuffle_2", "shuffle_3"):
        np.testing.assert_array_equal(first[name], second[name])
        np.testing.assert_array_equal(first[name].sum(axis=0), learned.sum(axis=0))
        assert not np.diag(first[name]).any()
    assert any(not np.array_equal(first[name], learned) for name in ("shuffle_1", "shuffle_2", "shuffle_3"))


def test_support_rejects_self_edges():
    with pytest.raises(ValueError, match="without self edges"):
        _support_masks(np.eye(3, dtype=bool), 91004, 2)


def test_shuffles_are_one_seed_row_and_preserve_unavailable_effects():
    common = {"scenario": "constant", "seed": 91000, "origins": 12, "effect_origins": 4}
    rows = [
        {"arm": "shuffle_1", **common, "forecast_mae": 0.25, "active_effect_mae": None},
        {"arm": "shuffle_2", **common, "forecast_mae": 0.5, "active_effect_mae": None},
    ]
    assert _replicate_mean(rows, "shuffled") == {
        "arm": "shuffled", **common, "forecast_mae": 0.375, "active_effect_mae": None,
    }
