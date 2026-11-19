"""Check raw tail dependencies and prospective split isolation."""

import numpy as np

from synthetic.preflight_tail_transition import TailOrigin, _chronological_split, _raw_tail_intervals


def test_tail_target_excludes_the_origin_sample() -> None:
    times = np.arange(20) / 4

    history, target = _raw_tail_intervals(times, 2.0, 1.0, 1.0) or (None, None)

    assert history == (4, 9)
    assert target == (9, 13)
    assert _raw_tail_intervals(times, 4.0, 1.0, 1.0) is None


def test_chronological_purge_detects_overlap_only_in_raw_tail() -> None:
    safe = TailOrigin(3, 1.0, (0, 4), (0, 3), (3, 6), 0)
    overlaps_tail_target = TailOrigin(4, 2.0, (1, 5), (1, 5), (5, 7), 0)
    test = TailOrigin(8, 3.0, (5, 9), (6, 8), (8, 10), 0)

    train, evaluation, train_candidates = _chronological_split([safe, overlaps_tail_target, test], 3.0)

    assert train_candidates == 2
    assert train == [safe]
    assert evaluation == [test]
