"""Planted controls check known mathematics, not discovery from noisy finite samples."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from effectome.experiments.transition_examples import (
    TRANSITION_EXAMPLE_SCENARIOS,
    TransitionExampleConfig,
    make_transition_example,
    make_transition_example_windows,
)


@pytest.mark.parametrize("scenario", TRANSITION_EXAMPLE_SCENARIOS)
def test_examples_have_finite_reproducible_activity_and_no_truth_behavior(scenario):
    config = TransitionExampleConfig(scenario=scenario, n_samples=384, trial_length=128, seed=7)
    recording = make_transition_example(config)
    recording.validate()
    repeated = make_transition_example(config)
    np.testing.assert_array_equal(recording.traces, repeated.traces)
    assert recording.traces.shape == (4, 384)
    assert np.all(np.isfinite(recording.traces))
    np.testing.assert_allclose(np.diff(recording.time), 1 / config.fps)
    assert recording.behavior == {}
    assert recording.metadata["truth_scope"] == "metadata_only_not_predictors_or_behavior"
    for matrix in recording.metadata["true_coefficients"]:
        assert np.max(np.abs(np.linalg.eigvals(matrix))) < 1
    other = make_transition_example(TransitionExampleConfig(
        scenario=scenario, n_samples=384, trial_length=128, seed=8
    ))
    assert not np.array_equal(recording.traces, other.traces)


def test_distinct_rotating_circuits_have_the_same_exact_stationary_gaussian_marginal():
    recording = make_transition_example(TransitionExampleConfig(n_samples=300, regime_dwell=64))
    coefficients = recording.metadata["true_coefficients"]
    innovations = recording.metadata["true_innovation_covariances"]
    covariance = recording.metadata["true_stationary_covariances"]
    assert not np.allclose(coefficients[0], coefficients[1])
    assert np.sign(coefficients[0, 0, 1]) != np.sign(coefficients[1, 0, 1])
    for matrix, noise, stationary in zip(coefficients, innovations, covariance, strict=True):
        np.testing.assert_allclose(stationary, np.eye(recording.n_neurons), atol=1e-12)
        np.testing.assert_allclose(matrix.T @ stationary @ matrix + noise, stationary, atol=1e-12)
    np.testing.assert_array_equal(recording.metadata["true_stationary_means"], np.zeros((2, 4)))
    assert recording.metadata["identical_stationary_marginals"]
    # Do not assert equality of two random empirical means/covariances.


def test_one_to_one_control_has_distinct_stationary_activity_centers():
    recording = make_transition_example(TransitionExampleConfig(scenario="one_to_one", n_samples=300))
    means = recording.metadata["true_stationary_means"]
    assert means[0, 0] < 0 < means[1, 0]
    for mean, matrix, intercept in zip(
        means, recording.metadata["true_coefficients"], recording.metadata["true_intercepts"], strict=True
    ):
        np.testing.assert_allclose(mean @ matrix + intercept, mean, atol=1e-12)
    assert not recording.metadata["identical_stationary_marginals"]


def test_fixed_circuit_initial_conditions_are_declared_independent_trial_boundaries():
    recording = make_transition_example(TransitionExampleConfig(
        scenario="fixed_circuit", n_samples=384, trial_length=128
    ))
    assert recording.metadata["true_coefficients"].shape == (1, 4, 4)
    np.testing.assert_array_equal(recording.metadata["true_regime"], np.zeros(384, dtype=int))
    np.testing.assert_array_equal(recording.traces[0, [0, 128, 256]], [4.0, -4.0, 4.0])
    windows = make_transition_example_windows(recording, history_length=32, target_length=8, stride=16)
    for anchor in windows.anchors:
        assert any(
            start <= anchor.context_start < anchor.context_stop <= stop
            for start, stop in recording.metadata["valid_ranges"]
        )
    assert any(anchor.gap_before for anchor in windows.anchors)
    assert not windows.metadata["neural_standardization"]["enabled"]
    assert windows.metadata["preprocess_dependency"]["future_support"] == 0
    assert windows.behavior_per_window == {}


@pytest.mark.parametrize("kwargs", [
    {"scenario": "unregistered"}, {"n_neurons": 1}, {"n_samples": 2},
    {"contraction": 1}, {"fixed_contraction": float("nan")},
    {"rotation_angle": 0}, {"noise_std": -1}, {"fps": 0}, {"seed": -1},
])
def test_invalid_example_design_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TransitionExampleConfig(**kwargs)


def test_synthetic_notebook_code_parses_and_run_defaults_remain_enabled():
    root = Path(__file__).resolve().parents[1] / "synthetic"
    paths = sorted(root.glob("*.ipynb"))
    assert len(paths) == 9
    assert (root / "06_mechanism_driver_validation.ipynb") in paths
    enabled = {}
    for path in paths:
        notebook = json.loads(path.read_text())
        assert notebook["metadata"]["language_info"]["version"].startswith("3.14")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                # Working notebooks may contain the user's completed runs.
                # Output-free publication is checked against fresh builder output below.
                tree = ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id.startswith("RUN"):
                                if isinstance(node.value.value, bool):
                                    enabled[target.id] = node.value.value
    assert enabled == {
        "RUN_SANITY": True, "RUN_COMPARISON": True, "RUN_VALIDATION": True,
        "RUN_DRIVER_VALIDATION": True, "RUN": True, "RUN_HISTORY_COMPARISON": True,
    }


def test_synthetic_builders_create_output_free_notebooks_without_clearing_saved_runs(tmp_path, monkeypatch):
    from notebooks._build_notebooks import write_notebook
    from synthetic import (
        _build_csl_sanity_notebook,
        _build_driver_notebook,
        _build_manifold_notebook,
        _build_mechanism_notebook,
        _build_replication_notebook,
        _build_validation_notebook,
    )

    def redirected(path, cells):
        write_notebook(tmp_path / path.name, cells)

    for module in (_build_csl_sanity_notebook, _build_driver_notebook, _build_mechanism_notebook,
                   _build_replication_notebook, _build_validation_notebook, _build_manifold_notebook):
        monkeypatch.setattr(module, "write_notebook", redirected)
        module.main()
    generated = list(tmp_path.glob("*.ipynb"))
    assert len(generated) == 6
    for path in generated:
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                ast.parse("".join(cell["source"]))


def _notebook_switches(source: str, switches: dict[str, bool]) -> ast.Module:
    """Test explicit inspection mode without launching the user's fitting workload."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in switches:
                node.value = ast.copy_location(ast.Constant(switches[target.id]), node.value)
    return ast.fix_missing_locations(tree)


def test_synthetic_notebooks_inspect_protocols_when_fitting_is_explicitly_disabled(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    comparison_inventory = None
    for path in sorted((root / "synthetic").glob("*.ipynb")):
        namespace = {"__name__": "__main__"}
        notebook = json.loads(path.read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            tree = _notebook_switches("".join(cell["source"]), {
                "RUN_SANITY": False, "RUN_COMPARISON": False, "RUN_VALIDATION": False,
                "RUN_DRIVER_VALIDATION": False, "RUN": False, "RUN_HISTORY_COMPARISON": False,
            })
            exec(compile(tree, f"{path}:cell-{index}", "exec"), namespace)
            if cell.get("id") == "project_imports":
                # Root discovery/imports remain real; reports go to the test's
                # directory so the three notebooks can share their saved run.
                namespace["ROOT"] = tmp_path
            if path.name == "04_framework_validation.ipynb" and "OUTPUT_DIRECTORY" in namespace:
                namespace["OUTPUT_DIRECTORY"] = tmp_path / "framework-validation"
            if path.name == "05_driver_recovery.ipynb" and "OUTPUT_DIRECTORY" in namespace:
                namespace["OUTPUT_DIRECTORY"] = tmp_path / "driver-validation"
                namespace["ANALYSIS_DIRECTORY"] = tmp_path / "driver-validation" / "analysis"
        if path.name == "02_shared_transition_comparison.ipynb":
            comparison_inventory = namespace
        plt.close("all")
    assert comparison_inventory is not None
    assert not comparison_inventory["RUN_COMPARISON"]
    assert comparison_inventory["report"] is None
    assert set(comparison_inventory["methods"]) == {
        "conventional_cgc", "conventional_cgc_star", "fast_cgc", "fast_cgc_star",
    }
    assert comparison_inventory["windows"].segments.shape[1] >= 256
    assert not comparison_inventory["report_path"].is_file()


def test_synthetic_notebook_prefixes_follow_documented_execution_order():
    root = Path(__file__).resolve().parents[1] / "synthetic"
    assert [path.name for path in sorted(root.glob("*.ipynb"))] == [
        "00_ground_truth.ipynb", "01_four_csl_sanity.ipynb",
        "02_shared_transition_comparison.ipynb", "03_correspondence_and_controls.ipynb",
        "04_framework_validation.ipynb", "05_driver_recovery.ipynb",
        "06_mechanism_driver_validation.ipynb",
        "07_mechanism_replication.ipynb",
        "08_learned_manifold_validation.ipynb",
    ]


def test_mechanism_notebook_runs_oracle_panel_from_nested_working_directory(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root / "synthetic" / "results")
    notebook = json.loads((root / "synthetic/06_mechanism_driver_validation.ipynb").read_text())
    namespace = {"__name__": "__main__"}
    output = tmp_path / "oracle-only"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = _notebook_switches(source, {
            "RUN": True, "ORACLE_ONLY": True, "RUN_HISTORY_COMPARISON": False,
        })
        exec(compile(tree, f"mechanism-notebook:cell-{index}", "exec"), namespace)
        if 'PROFILE = "smoke"' in source:
            namespace["OUTPUT"] = output
            namespace["HISTORY_OUTPUT"] = tmp_path / "history"
    report = json.loads((output / "mechanism-validation.json").read_text())
    assert report["execution_complete"]
    assert report["available_method_runs"] == report["attempted_method_runs"] == 4
    assert report["planned_methods"] == ["known_support", "empty_support"]
    assert (output / "run.log").is_file()
    assert not report["scientific_validation_passed"]


def test_native_notebook_fitting_stops_when_conventional_runtime_is_missing(tmp_path):
    from effectome.workflows.cross_method import load_cross_method_profile
    from notebooks._shared import require_conventional_runtime

    _, methods = load_cross_method_profile("csl_sanity")
    with pytest.raises(RuntimeError, match="uv sync --project tools/conventional-cgc"):
        require_conventional_runtime(methods, project_root=tmp_path)
    require_conventional_runtime({"fast_cgc": methods["fast_cgc"]}, project_root=tmp_path)
