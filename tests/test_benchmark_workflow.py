"""Executable benchmark contracts on bounded raw processes, never publication jobs."""

import json
from dataclasses import replace

import numpy as np
import pytest

from effectome.experiments.benchmark import (
    BenchmarkConfig,
    _benchmark_report,
    _calibrate_cell,
    _null_statistic,
    _score_recording,
    benchmark_preflight,
    run_benchmark,
)
from effectome.experiments.validation import (
    ValidationAttempt,
    ValidationBudget,
    ValidationCell,
    _atomic_attempt,
    synthetic_config_for_cell,
)
from effectome.utils.synthetic import SyntheticConfig, make_synthetic_recording


def test_registered_dimensions_native_clock_and_physical_dwell():
    baseline = ValidationCell("stress", "recurrent_geometric", 0, 0, "dwell_short")
    config = synthetic_config_for_cell(baseline, master_seed=260926)
    assert config.n_neurons == 109
    assert config.n_timepoints == 3137
    assert config.fps == 2.9046296296296297
    assert config.regime_dwell == round(15 * config.fps)
    assert config.measurement_snr == 3
    assert config.self_lag_coefficients == (0.5, 0.1)
    for suite in ("nonlinear", "independent_calibration"):
        resolved = synthetic_config_for_cell(ValidationCell(suite, "variance", 0, 0), master_seed=260926)
        assert resolved.n_neurons == 109
        assert resolved.fps == config.fps
    small = synthetic_config_for_cell(ValidationCell("core", "constant", 0, 0, "n20"), master_seed=260926)
    assert small.n_neurons == 20


def test_graph_stream_shared_across_observation_replicates_but_not_trajectories():
    first = synthetic_config_for_cell(
        ValidationCell("core", "constant", 3, 0, "n20"), master_seed=260925, profile="smoke"
    )
    second = synthetic_config_for_cell(
        ValidationCell("core", "constant", 3, 1, "n20"), master_seed=260925, profile="smoke"
    )
    assert first.graph_seed == second.graph_seed
    assert first.seed != second.seed
    records = [
        make_synthetic_recording(replace(item, n_neurons=4, behavior_drivers=2, n_timepoints=80))
        for item in (first, second)
    ]
    np.testing.assert_array_equal(
        records[0].metadata["true_lag_graphs"], records[1].metadata["true_lag_graphs"]
    )
    assert not np.array_equal(records[0].traces, records[1].traces)


def test_fixed_interaction_variance_null_changes_noise_not_coefficients():
    common = dict(
        n_neurons=4,
        n_timepoints=1200,
        n_states=1,
        scenario="constant",
        coupling=0.01,
        behavior_drivers=2,
        seed=8,
    )
    baseline = make_synthetic_recording(SyntheticConfig(**common))
    shifted = make_synthetic_recording(SyntheticConfig(variance_ratio=4, **common))
    np.testing.assert_array_equal(baseline.metadata["true_lag_graphs"], shifted.metadata["true_lag_graphs"])
    assert np.var(shifted.traces[:, 700:]) > 2 * np.var(baseline.traces[:, 700:])
    np.testing.assert_array_equal(shifted.traces[:, :590], baseline.traces[:, :590])


def test_preflight_publication_is_complete_and_does_not_fit(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not generate raw data")

    monkeypatch.setattr("effectome.experiments.benchmark.make_synthetic_recording", forbidden)
    config = BenchmarkConfig(output_dir=tmp_path, profile="publication", master_seed=260926, max_cells=None)
    result = benchmark_preflight(config)
    assert result["selected_cells"] == 1400
    assert result["suite_counts"]["independent_calibration"] == 600
    assert result["node_counts"] == [20, 109]
    assert result["native_fps"] == [2.9046296296296297]
    assert len(result["cells"]) == 1400
    assert list(tmp_path.iterdir()) == []


def test_raw_benchmark_all_predictive_adapters_share_origins_and_persist_scores(tmp_path):
    config = BenchmarkConfig(output_dir=tmp_path, max_origins=2, solver_max_iterations=1000)
    report = run_benchmark(config)
    assert report["counts"]["completed"] == 1
    assert report["publication_grid_executed"] is False
    result = report["attempts"][0]["result"]
    assert result["predictions"]
    assert result["prediction_scores"]
    comparison = result["neural_comparison"]
    assert set(comparison["methods"]) == set(config.methods)
    for name, method in comparison["methods"].items():
        assert method["issued"] == 2, (name, method)
        assert method["failed"] == 0
        for origin in method["origins"]:
            assert origin["standardized_mse"] >= 0
            assert origin["target_seconds"] > origin["issue_seconds"]
            assert origin["sample_convention"] == "issue_sample_is_exclusive_observation_stop"
            assert origin["recovery"]["observed_projection"]["status"] == "unavailable"
    assert all(pair["origin_ids"] == [1, 2] for pair in comparison["paired"].values())
    assert (tmp_path / "report.json").is_file()
    resumed = run_benchmark(config)
    assert resumed["attempts"][0]["resumed"] is True
    assert resumed["attempts"][0]["result"]["input_hash"] == result["input_hash"]


def test_recurrent_three_regime_endpoint_is_binary_and_raw_adapter_issues(tmp_path):
    synthetic = SyntheticConfig(
        n_neurons=4, n_timepoints=200, max_lag=2, n_states=3, regime_dwell=10, behavior_drivers=2, seed=17
    )
    recording = make_synthetic_recording(synthetic)
    assert set(np.unique(recording.behavior["motif"])) == {0, 1}
    result = _score_recording(
        ValidationCell("core", "recurrent_geometric", 0, 0),
        synthetic,
        BenchmarkConfig(output_dir=tmp_path, max_origins=1),
    )
    assert result["n_predictions"] > 0
    assert all(record["class_order"] == (0, 1) for record in result["predictions"])


def test_null_family_failures_cannot_disappear_from_maximum_or_prediction_test():
    result = {
        "neural_comparison": {
            "methods": {
                "constant_sparse_var": {"status": "available", "failed": 0, "origins": [{"origin": 1}]},
                "windowed_sparse_var": {"status": "unavailable", "failed": 1, "origins": []},
            },
            "paired": {"windowed_sparse_var": {"status": "unavailable", "origin_ids": []}},
        }
    }
    with pytest.raises(ValueError, match="every registered"):
        _null_statistic(result, "variance")
    prediction = {
        "n_matched_predictions": 3,
        "n_baseline_only": 0,
        "n_candidate_only": 0,
        "n_failed_baseline": 0,
        "n_failed_candidate": 1,
        "equal_animal_delta_log_loss": 0.5,
    }
    with pytest.raises(ValueError, match="complete registered"):
        _null_statistic({"paired_prediction": prediction}, "endpoint")


def test_full_pipeline_null_draws_are_fresh_seeded_and_fail_closed(monkeypatch, tmp_path):
    seeds = []

    def statistic(cell, synthetic, config):
        seeds.append((synthetic.graph_seed, synthetic.seed))
        if len(seeds) == 3:
            raise ValueError("solver failed")
        return {
            "paired_prediction": {
                "n_matched_predictions": 3,
                "n_baseline_only": 0,
                "n_candidate_only": 0,
                "n_failed_baseline": 0,
                "n_failed_candidate": 0,
                "equal_animal_delta_log_loss": 0.5,
            },
            "evaluation_diagnostics": {
                "unavailable_folds": 0,
                "available_folds": 1,
                "folds": {"0": {"test_examples": 3}},
            },
        }

    monkeypatch.setattr("effectome.experiments.benchmark._score_recording", statistic)
    cell = ValidationCell("independent_calibration", "endpoint", 0, 0)
    synthetic = synthetic_config_for_cell(cell, master_seed=260925, profile="smoke")
    config = BenchmarkConfig(output_dir=tmp_path, methods=("constant_sparse_var", "windowed_sparse_var"))
    observed = statistic(cell, synthetic, config)
    result = _calibrate_cell(cell, synthetic, config, observed)
    assert result["status"] == "unavailable"
    assert result["p_value"] is None
    assert result["reject"] is False
    assert len({pair[0] for pair in seeds}) == 1
    assert len({pair[1] for pair in seeds}) == len(seeds)


def test_resolution_and_publication_calibration_freeze_rejected_before_fitting(tmp_path):
    with pytest.raises(ValueError, match="resolution"):
        BenchmarkConfig(output_dir=tmp_path, calibration_draws=9)
    config = BenchmarkConfig(
        output_dir=tmp_path, profile="publication", master_seed=260926, suites=("independent_calibration",)
    )
    with pytest.raises(ValueError, match="freeze calibration"):
        run_benchmark(config)


def test_failed_attempt_never_marks_full_publication_grid_complete(tmp_path):
    config = BenchmarkConfig(output_dir=tmp_path, profile="publication", master_seed=260926)
    cell = ValidationCell("core", "constant", 0, 0)
    attempt = ValidationAttempt(
        cell.cell_id,
        "failed",
        1,
        "cfg",
        1,
        1,
        ValidationBudget(),
        {},
        {},
        "g",
        "t",
        "i",
        "impl",
        "evaluator",
        failure_reason="bad_solver",
    )
    report = _benchmark_report(config, {"full_registered_grid": True}, [cell], [attempt])
    assert report["full_registered_grid_attempted"] is True
    assert report["full_registered_grid_executed"] is False
    assert report["publication_grid_executed"] is False


def test_unavailable_scores_serialize_as_null_in_strict_json(tmp_path):
    path = tmp_path / "unavailable.json"
    _atomic_attempt(path, {"status": "unavailable", "score": float("nan"), "other": np.float64(float("inf"))})
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    result = json.loads(text)
    assert result == {"status": "unavailable", "score": None, "other": None}


def test_complete_raw_null_family_uses_deterministic_plus_one_tail(monkeypatch, tmp_path):
    def statistic(cell, synthetic, config):
        return {
            "paired_prediction": {
                "n_matched_predictions": 4,
                "n_baseline_only": 0,
                "n_candidate_only": 0,
                "n_failed_baseline": 0,
                "n_failed_candidate": 0,
                "equal_animal_delta_log_loss": np.random.default_rng(synthetic.seed).normal(),
            },
            "evaluation_diagnostics": {
                "unavailable_folds": 0,
                "available_folds": 1,
                "folds": {"0": {"test_examples": 4}},
            },
        }

    monkeypatch.setattr("effectome.experiments.benchmark._score_recording", statistic)
    cell = ValidationCell("independent_calibration", "endpoint", 1, 0)
    synthetic = synthetic_config_for_cell(cell, master_seed=260925, profile="smoke")
    config = BenchmarkConfig(output_dir=tmp_path)
    observed = statistic(cell, synthetic, config)
    first = _calibrate_cell(cell, synthetic, config, observed)
    second = _calibrate_cell(cell, synthetic, config, observed)
    assert first == second
    assert first["status"] == "available"
    count = sum(value >= first["statistic"] for value in first["null_statistics"])
    assert first["p_value"] == (1 + count) / 20
    assert len(first["seed_mapping"]) == 19


def test_empty_selection_and_nonfinite_null_statistic_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="no cells"):
        benchmark_preflight(BenchmarkConfig(output_dir=tmp_path, cell_offset=9999))
    with pytest.raises(ValueError, match="finite"):
        _null_statistic(
            {
                "paired_prediction": {
                    "n_matched_predictions": 1,
                    "n_baseline_only": 0,
                    "n_candidate_only": 0,
                    "n_failed_baseline": 0,
                    "n_failed_candidate": 0,
                    "equal_animal_delta_log_loss": float("nan"),
                }
            },
            "endpoint",
        )


def test_unavailable_folds_and_unissued_expected_pairs_fail_complete_pipeline(tmp_path):
    paired = {
        "n_matched_predictions": 2,
        "n_baseline_only": 0,
        "n_candidate_only": 0,
        "n_failed_baseline": 0,
        "n_failed_candidate": 0,
        "equal_animal_delta_log_loss": 0.1,
    }
    result = {
        "paired_prediction": paired,
        "evaluation_diagnostics": {
            "unavailable_folds": 1,
            "available_folds": 1,
            "folds": {"0": {"test_examples": 2}},
        },
    }
    with pytest.raises(ValueError, match="evaluator folds"):
        _null_statistic(result, "endpoint")
    result["evaluation_diagnostics"]["unavailable_folds"] = 0
    result["evaluation_diagnostics"]["folds"]["0"]["test_examples"] = 3
    with pytest.raises(ValueError, match="expected registered"):
        _null_statistic(result, "endpoint")
    cell = ValidationCell("core", "constant", 0, 0)
    attempt = ValidationAttempt(
        cell.cell_id,
        "completed",
        1,
        "c",
        1,
        1,
        ValidationBudget(),
        result,
        {},
        "g",
        "t",
        "i",
        "impl",
        "evaluator",
    )
    report = _benchmark_report(
        BenchmarkConfig(output_dir=tmp_path), {"full_registered_grid": True}, [cell], [attempt]
    )
    assert report["full_registered_grid_executed"] is False


def test_origin_bounded_publication_manifest_cannot_be_a_full_grid(tmp_path):
    config = BenchmarkConfig(
        output_dir=tmp_path, profile="publication", master_seed=260926, max_cells=None, max_origins=1
    )
    report = benchmark_preflight(config)
    assert report["full_registered_manifest"] is True
    assert report["full_registered_grid"] is False
