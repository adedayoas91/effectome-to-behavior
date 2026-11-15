"""Study collection preserves missing methods and verifies shared targets."""

from dataclasses import replace

from effectome.experiments.driver_study import collect_driver_study
from effectome.experiments.driver_validation import load_driver_validation_profile
from effectome.workflows.cross_method import load_cross_method_profile


def test_empty_collection_is_incomplete_with_all_four_methods_and_full_denominators(tmp_path):
    config = replace(load_driver_validation_profile("smoke"), positive_seeds=(0,), null_seeds=(0,))
    _, methods = load_cross_method_profile("csl_sanity")
    report = collect_driver_study(config, tmp_path, methods)
    assert not report["execution_complete"]
    assert not report["synthetic_driver_recovery_passed"]
    assert report["attempted_method_runs"] == 0
    assert len(report["planned_methods"]) == 4
    assert all(cell["completed_methods"] == 0 for cell in report["cells"])
    assert report["acceptance"]["overall_status"] == "inconclusive"
