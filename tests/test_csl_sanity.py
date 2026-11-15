"""Sanity metrics must distinguish successful recovery from empty consensus."""

import numpy as np
import pytest

from effectome.experiments.csl_sanity import (
    CSLSanityConfig,
    graph_recovery_metrics,
    make_csl_sanity_data,
    support_agreement,
)


def test_planted_chain_has_two_signed_links_and_null_has_none():
    config = CSLSanityConfig(n_samples=1024)
    chain, truth = make_csl_sanity_data(config, "chain", 4)
    null, null_truth = make_csl_sanity_data(config, "null", 4)
    assert set(zip(*np.where(truth), strict=True)) == {(0, 1), (1, 2)}
    assert not null_truth.any()
    np.testing.assert_array_equal(chain[:, [0, 3]], null[:, [0, 3]])
    assert np.corrcoef(chain[:-1, 0], chain[1:, 1])[0, 1] > .5
    assert np.corrcoef(chain[:-1, 1], chain[1:, 2])[0, 1] < -.5


def test_empty_agreement_is_not_positive_recovery():
    empty = np.zeros((4, 4), bool)
    truth = empty.copy()
    truth[0, 1] = truth[1, 2] = True
    agreement = support_agreement(empty, empty)
    assert agreement["jaccard"] is None
    assert agreement["both_empty"]
    metrics = graph_recovery_metrics(empty, truth)
    assert metrics["recall"] == metrics["f1"] == 0
    assert metrics["precision"] is None
    assert not metrics["exact_support"]


def test_off_diagonal_metric_counts_and_orientation():
    truth = np.eye(3, dtype=bool)
    truth[0, 1] = True
    inferred = truth.T
    result = graph_recovery_metrics(inferred, truth)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (0, 1, 1, 4)


@pytest.mark.parametrize("changes", [{"seeds": ()}, {"seeds": (1, 1)}, {"n_samples": 64}])
def test_bad_sanity_design_is_rejected(changes):
    with pytest.raises(ValueError):
        CSLSanityConfig(**changes)
