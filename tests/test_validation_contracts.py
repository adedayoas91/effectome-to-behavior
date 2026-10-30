"""Small deterministic acceptance fixtures; never execute the publication grid."""
import numpy as np
import pytest

from effectome.experiments.validation import (
    DevelopmentSmokeConfig,
    ValidationBudget,
    benchmark_cells,
    edge_recovery_metrics,
    run_development_smoke,
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
    budget = ValidationBudget(timeout_seconds=5, memory_limit_mb=512)

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
        budget=budget,
    )
    assert [attempt.status for attempt in attempts] == ["completed", "failed"]
    assert len(list(tmp_path.rglob("*.json"))) == 2
    resumed = run_validation_cells(
        cells,
        output_dir=tmp_path,
        master_seed=9,
        config_factory=lambda _cell, seed: {"seed": seed},
        evaluator=evaluate,
        budget=budget,
    )
    assert all(attempt.resumed for attempt in resumed)
    assert len(observed) == 2
    assert attempts[0].actual_config == {"seed": attempts[0].seed}
    assert attempts[0].graph_id
    assert attempts[0].trajectory_id
    assert attempts[0].input_hash
    assert attempts[0].implementation_hash
    assert attempts[0].evaluator_hash
    assert attempts[0].runtime_method == "time.perf_counter"
    assert attempts[0].memory_method == "tracemalloc_python_peak"

    stale = run_validation_cells(
        cells,
        output_dir=tmp_path,
        master_seed=9,
        config_factory=lambda _cell, seed: {"seed": seed, "revision": 2},
        evaluator=evaluate,
        budget=budget,
    )
    assert not any(attempt.resumed for attempt in stale)
    assert len(observed) == 4


def test_edge_recovery_reports_lag_resolved_prevalence_and_signed_fdp():
    truth = np.zeros((2, 3, 3))
    estimate = np.zeros_like(truth)
    truth[0, 0, 1] = 0.8
    truth[1, 2, 1] = -0.5
    estimate[0, 0, 1] = -0.9  # selected true edge with wrong sign
    estimate[0, 1, 2] = 0.4  # selected null edge
    metrics = edge_recovery_metrics(truth, estimate, selection_threshold=0.1)
    assert len(metrics["by_lag"]) == 2
    assert metrics["by_lag"][0]["prevalence_baseline"] == pytest.approx(1 / 6)
    assert metrics["overall"]["signed_fdp"] == pytest.approx(1.0)


def test_notebook_friendly_development_smoke_runs_without_callbacks(tmp_path):
    result = run_development_smoke(
        DevelopmentSmokeConfig(
            output_dir=tmp_path,
            n_neurons=4,
            n_timepoints=160,
            history_length=32,
            target_length=8,
            stride=8,
            max_cells=1,
        )
    )
    assert result.label == "smoke"
    assert len(result.attempts) == 1
    assert result.attempts[0].status == "completed"
    assert result.summary["completed"] == 1
    assert result.attempts[0].result["pipeline"] == "preprocess_window_primary_evaluation"


def test_development_smoke_checkpoint_tracks_window_geometry(tmp_path):
    common = dict(
        output_dir=tmp_path,
        n_neurons=4,
        n_timepoints=160,
        target_length=8,
        stride=8,
        max_cells=1,
    )
    first = run_development_smoke(
        DevelopmentSmokeConfig(history_length=32, **common)
    )
    changed = run_development_smoke(
        DevelopmentSmokeConfig(history_length=40, **common)
    )

    assert first.attempts[0].resumed is False
    assert changed.attempts[0].resumed is False
    assert first.attempts[0].config_hash != changed.attempts[0].config_hash
