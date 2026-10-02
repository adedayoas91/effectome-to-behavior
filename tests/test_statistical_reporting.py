"""Statistical reporting primitive tests for Appendix-C WP9."""

from __future__ import annotations

import numpy as np

from effectome.linking import (
    benjamini_yekutieli,
    brier_score,
    classification_log_loss,
    plus_one_monte_carlo_pvalue,
    require_monte_carlo_resolution,
)


def test_benjamini_yekutieli_known_values_and_empty_family():
    p = np.array([0.01, 0.04, 0.03])
    q = benjamini_yekutieli(p)
    harmonic = 1.0 + 1.0 / 2.0 + 1.0 / 3.0
    ranked = np.array([0.01, 0.03, 0.04])
    expected_ranked = np.minimum.accumulate(
        (ranked * 3 * harmonic / np.array([1, 2, 3]))[::-1]
    )[::-1]
    expected = np.empty(3)
    expected[np.argsort(p, kind="mergesort")] = np.clip(expected_ranked, 0.0, 1.0)

    assert np.allclose(q, expected)
    assert benjamini_yekutieli(np.array([])).size == 0


def test_classification_log_loss_respects_fixed_class_order():
    probs = np.array([[0.2, 0.8], [0.75, 0.25]])
    observed = np.array(["b", "a"])

    loss = classification_log_loss(probs, observed, class_order=["a", "b"])

    assert np.isclose(loss, np.mean([-np.log(0.8), -np.log(0.75)]))


def test_brier_score_multiclass_example():
    probs = np.array([[0.7, 0.2, 0.1]])
    observed = np.array(["x"])

    score = brier_score(probs, observed, class_order=["x", "y", "z"])

    assert np.isclose(score, (0.3**2 + 0.2**2 + 0.1**2))


def test_plus_one_monte_carlo_and_resolution_check():
    assert plus_one_monte_carlo_pvalue(0, 999) == 0.001
    require_monte_carlo_resolution(999, 0.001)
    try:
        require_monte_carlo_resolution(999, 0.0005)
    except ValueError as exc:
        assert "too small" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected insufficient resolution to fail")
