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
from effectome.linking.stats import (
    binomial_proportion_interval,
    finite_control_accounting,
    raw_block_sensitivity_refits,
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


def test_finite_control_accounting_distinguishes_unique_pool_from_draws():
    accounting = finite_control_accounting(
        ["a", "a", "b", "c"], requested_draws=999, alpha_min=0.01
    )
    assert accounting.unique_controls == 3
    assert accounting.sampling_method == "enumeration"
    assert accounting.effective_draws == 3
    assert accounting.resolution_sufficient is False


def test_binomial_calibration_interval_contains_rate():
    interval = binomial_proportion_interval(5, 100, confidence=0.95)
    assert interval.low < 0.05 < interval.high
    assert interval.n_success == 5
    assert interval.n_trials == 100


def test_raw_block_sensitivity_refits_every_resample_deterministically():
    calls: list[tuple[int, int]] = []

    def refit(sample, block_length, seed):
        calls.append((block_length, seed))
        return {"first_source": int(sample.source_indices[0])}

    first = raw_block_sensitivity_refits(
        {"signal": np.arange(40)}, block_length=8, seed=5, refit=refit
    )
    second = raw_block_sensitivity_refits(
        {"signal": np.arange(40)}, block_length=8, seed=5, refit=lambda s, b, r: {
            "first_source": int(s.source_indices[0])
        }
    )
    assert [row.block_length for row in first] == [4, 8, 16]
    assert [row.result for row in first] == [row.result for row in second]
    assert len(calls) == 3
