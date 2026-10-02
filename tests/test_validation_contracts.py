"""Small deterministic acceptance fixtures; never execute the publication grid."""
import numpy as np
import pytest

from effectome.experiments.validation import (
    ValidationBudget,
    benchmark_cells,
    run_validation_cells,
)
from effectome.linking.stats import RegisteredFamily, raw_block_resample
from effectome.utils.synthetic import SyntheticConfig, certify_common_metric, make_synthetic_recording


def test_switching_unstable_pair_rejected():
    pair = np.array([[[0.5, 2.0], [0.0, 0.5]], [[0.5, 0.0], [2.0, 0.5]]])
    with pytest.raises(ValueError, match="certificate"):
        certify_common_metric(pair, rho=0.95)


def test_multilag_effective_certificate_and_clock():
    clock = np.cumsum(np.linspace(0.08, 0.12, 40))
    rec = make_synthetic_recording(SyntheticConfig(
        n_neurons=4, n_timepoints=40, max_lag=2, instantaneous_density=0.8,
        instantaneous_strength=0.7, native_time=tuple(clock)))
    cert = rec.metadata["stability_certificate"]
    assert cert["worst_factor"] <= cert["rho"] ** 2 + 1e-10
    assert rec.metadata["burn_in"] >= 1000
    np.testing.assert_array_equal(rec.time, clock)
    for companion in rec.metadata["effective_companions"]:
        np.testing.assert_array_equal(companion[4:, :4], np.eye(4))
    assert rec.metadata["truth_contract"]["observed_projection"] == "unavailable"


def test_finite_grid_counts_and_unique_attempts():
    cells = benchmark_cells(include_grouping=True)
    assert len(cells) == 1520
    assert len({c.cell_id for c in cells}) == 1520
    assert sum(c.suite == "core" for c in cells) == 500
    assert sum(c.suite == "independent_calibration" for c in cells) == 600


def test_registered_failures_stay_in_family():
    family = RegisteredFamily(("a", "b", "c"))
    rows = family.adjust({"a": (0.01, "ok"), "b": (None, "failed"), "c": (0.03, "ok")})
    assert rows["a"]["q_value"] == pytest.approx(0.055)
    assert rows["b"]["p_value"] is None
    assert rows["b"]["status"] == "failed"
    with pytest.raises(ValueError):
        family.adjust({"a": (0.01, "ok")})


def test_raw_blocks_preserve_labels_masks_and_reset_joins():
    values = np.arange(20)
    out = raw_block_resample({"signal": values[:, None], "labels": values.astype(str),
                              "mask": values % 2 == 0}, block_length=4, seed=1)
    np.testing.assert_array_equal(out.arrays["labels"], out.source_indices.astype(str))
    np.testing.assert_array_equal(out.arrays["mask"], out.source_indices % 2 == 0)
    eligible = out.valid_origins(history=2, horizon=1)
    assert not np.any(eligible[out.join_before])
    for i in np.flatnonzero(eligible):
        assert len(set(out.segment_ids[i - 2:i + 2])) == 1


def test_validation_runner_persists_failures_and_resumes(tmp_path):
    cells = benchmark_cells()[:2]
    observed: list[str] = []

    def evaluate(cell, config):
        observed.append(cell.cell_id)
        if cell == cells[1]:
            raise RuntimeError("registered failure")
        return {"seed": config["seed"], "mask": np.array([True, False])}

    attempts = run_validation_cells(
        cells,
        output_dir=tmp_path,
        master_seed=9,
        config_factory=lambda _cell, seed: {"seed": seed},
        evaluator=evaluate,
        budget=ValidationBudget(timeout_seconds=5, memory_limit_mb=512),
    )
    assert [attempt.status for attempt in attempts] == ["completed", "failed"]
    assert len(list(tmp_path.rglob("*.json"))) == 2
    resumed = run_validation_cells(
        cells,
        output_dir=tmp_path,
        master_seed=9,
        config_factory=lambda _cell, seed: {"seed": seed},
        evaluator=lambda _cell, _config: (_ for _ in ()).throw(AssertionError("reran")),
    )
    assert all(attempt.resumed for attempt in resumed)
    assert len(observed) == 2
