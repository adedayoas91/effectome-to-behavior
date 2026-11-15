"""Validation manifests preserve independent units, failures and artifact lineage."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import effectome.experiments.framework_validation as validation
from effectome.connectivity import ConnectivityConfig
from effectome.experiments.framework_validation import (
    FrameworkValidationConfig,
    framework_validation_cells,
    load_framework_validation_profile,
    run_framework_validation,
)
from effectome.experiments.framework_validation_analysis import analyze_framework_validation


def test_profiles_declare_independent_seeds_complete_matrix_and_stationary_control():
    for profile in ("smoke", "development", "manuscript"):
        config = load_framework_validation_profile(profile)
        cells = framework_validation_cells(config)
        assert len({cell["cell_id"] for cell in cells}) == len(cells)
        assert "stationary" in config.scenarios
        assert config.seeds == tuple(range(len(config.seeds)))
        assert config.estimator_profile == "csl_sanity"
        assert min(config.history_lengths) >= 256
    assert len(load_framework_validation_profile("manuscript").seeds) == 20
    with pytest.raises(ValueError, match="unique"):
        FrameworkValidationConfig(seeds=(0, 0))
    with pytest.raises(ValueError, match="integers"):
        FrameworkValidationConfig(seeds=(0.5,))
    with pytest.raises(ValueError, match="unique"):
        FrameworkValidationConfig(scenarios=("stationary", "stationary"))


def test_real_validation_retains_failed_method_and_resumes_only_identical_inputs(tmp_path, monkeypatch):
    config = FrameworkValidationConfig(
        seeds=(0,), scenarios=("stationary",), n_samples=512,
        history_lengths=(64,), dwell_lengths=(128,), estimator_profile="smoke",
    )
    methods = {
        "partial": ConnectivityConfig(name="correlation", partial=True),
        "unavailable": ConnectivityConfig(name="not_a_method"),
    }
    first = run_framework_validation(config, tmp_path, methods=methods)
    assert first["attempted_methods"] == 2
    assert first["available_methods"] == 1
    assert not first["publication_suite_complete"]
    assert {row["status"] for row in first["rows"]} == {"available", "unavailable"}
    second = run_framework_validation(config, tmp_path, methods=methods)
    assert second["cells"][0]["reused"]
    changed = run_framework_validation(replace(config, n_samples=640), tmp_path, methods=methods)
    assert not changed["cells"][0]["reused"]
    assert changed["manifest_fingerprint"] != first["manifest_fingerprint"]
    assert first["resolved_protocol"]["method_configs"]["partial"]["partial"]
    assert first["resolved_protocol"]["generator_defaults"]["fps"] == 10.0
    original_loader = validation.load_cross_method_profile

    def changed_reference(profile):
        base, defaults = original_loader(profile)
        return replace(base, n_manifold_states=2), defaults

    monkeypatch.setattr(validation, "load_cross_method_profile", changed_reference)
    changed = run_framework_validation(config, tmp_path, methods=methods)
    assert not changed["cells"][0]["reused"]
    assert changed["manifest_fingerprint"] != first["manifest_fingerprint"]


def _report(gains=(0.1, 0.2, 0.3)):
    seeds = tuple(range(len(gains)))
    planned = [{"spec_id": "spec", "seed": seed, "scenario": "stationary"} for seed in seeds]
    rows = [
        {**cell, "method": "method", "status": "available", "log_loss_gain": gain}
        for cell, gain in zip(planned, gains, strict=True)
    ]
    return {
        "profile": "smoke",
        "config": {"seeds": seeds},
        "planned_cells": planned,
        "planned_methods": ["method", "failed"],
        "rows": rows,
        "manifest_fingerprint": "test",
        "publication_suite_complete": False,
        "attempted_methods": len(rows),
        "available_methods": len(rows),
    }


def test_seed_analysis_exports_real_figures_and_uses_full_planned_family(tmp_path):
    report = _report()
    result = analyze_framework_validation(report, tmp_path, bootstrap_replicates=200)
    assert result["planned_contrasts"] == 2
    success = next(row for row in result["summaries"] if row["method"] == "method")
    assert success["n_available_seeds"] == 3
    assert success["mean_gain_nats"] == pytest.approx(0.2)
    assert success["holm_p"] == min(1, success["raw_sign_test_p"] * 2)
    assert success["mean_gain_95ci"] is not None
    assert (tmp_path / "figures/figure-01-predictive-gain.pdf").stat().st_size > 100
    assert (tmp_path / "figures/figure-02-availability.pdf").stat().st_size > 100
    for name in (
        "analysis-report.md",
        "stats-appendix.md",
        "figure-catalog.md",
        "summary.tex",
        "numeric-summary.md",
        "manuscript_results.tex",
    ):
        assert (tmp_path / name).is_file()
    assert not json.loads((tmp_path / "statistics.json").read_text())["scientific_validation_passed"]


def test_small_or_incomplete_seed_sets_do_not_fabricate_uncertainty_or_inference(tmp_path):
    report = _report((0.1, 0.2))
    result = analyze_framework_validation(report, tmp_path, bootstrap_replicates=100)
    success = next(row for row in result["summaries"] if row["method"] == "method")
    assert success["mean_gain_95ci"] is None
    assert success["raw_sign_test_p"] is None
    report = _report()
    report["rows"][0]["status"] = "unavailable"
    result = analyze_framework_validation(report, tmp_path, bootstrap_replicates=100)
    success = next(row for row in result["summaries"] if row["method"] == "method")
    assert not success["inferentially_complete"]
    assert success["holm_p"] is None
    report["rows"].append(report["rows"][-1])
    with pytest.raises(ValueError, match="duplicate"):
        analyze_framework_validation(report, tmp_path, bootstrap_replicates=100)

    report = _report((0.1, 0.2, 0.3, 0.4))
    report["rows"][0]["status"] = "unavailable"
    result = analyze_framework_validation(report, tmp_path, bootstrap_replicates=100)
    success = next(row for row in result["summaries"] if row["method"] == "method")
    assert success["n_available_seeds"] == 3
    assert success["mean_gain_95ci"] is None


def test_custom_matrix_cannot_claim_declared_manuscript_suite_complete(tmp_path, monkeypatch):
    config = FrameworkValidationConfig(
        profile="manuscript",
        seeds=tuple(range(20)),
        scenarios=("stationary",),
        n_samples=256,
        history_lengths=(64,),
    )

    def fake_comparison(windows, comparison_config, *, method_configs, **kwargs):
        return {
            "methods": {method: {"status": "unavailable", "reason": "test skip"} for method in method_configs}
        }

    monkeypatch.setattr(validation, "run_cross_method_comparison", fake_comparison)
    report = run_framework_validation(config, tmp_path)
    assert report["execution_complete"]
    assert len(report["planned_methods"]) == 4
    assert not report["publication_suite_complete"]


def test_empty_execution_retains_all_planned_contrasts(tmp_path):
    report = _report()
    report["rows"] = []
    report["available_methods"] = report["attempted_methods"] = 0
    result = analyze_framework_validation(report, tmp_path, bootstrap_replicates=100)
    assert all(row["mean_gain_nats"] is None for row in result["summaries"])
    assert result["planned_contrasts"] == 2
    assert not result["publication_suite_complete"]
