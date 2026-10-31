"""Notebook entry points use the reviewed optional-state and role contracts."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from notebooks._build_notebooks import (
    notebook_community,
    notebook_probabilistic,
    notebook_state_calibration,
    notebook_states,
)

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def _keyword_calls(payload, function):
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == function
                ):
                    yield {keyword.arg: keyword.value for keyword in node.keywords}


def test_state_notebooks_use_nested_prospective_selection_instead_of_forced_states():
    paths = sorted(NOTEBOOKS.glob("**/clustering/*/02_cluster_markov.ipynb"))
    assert len(paths) == 24
    for path in paths:
        payload = json.loads(path.read_text())
        graph, = _keyword_calls(payload, "GraphStateConfig")
        assert ast.literal_eval(graph["n_states"]) == 1, path
        assert ast.literal_eval(graph["metric"]) == "frobenius", path
        assert ast.literal_eval(graph["analysis_scope"]) == "prospective", path
        assert ast.literal_eval(graph["feature_representation"]) == "signed_lag_offdiag", path
        evaluation, = _keyword_calls(payload, "run_graph_states_and_transitions")
        assert "selection_cfg" in evaluation, path
        assert ast.literal_eval(evaluation["analysis_scope"]) == "prospective", path
        assert ast.literal_eval(evaluation["on_unavailable"]) == "return", path
        assert "if states is None:" in _source(payload), path
        source = _source(payload)
        assert "continuous baselines" in source and "Raw-history purging" in source, path
        assert "Label-permutation tests do not establish neural regimes" in source, path


def test_role_and_hmm_notebooks_keep_their_distinct_optional_scope():
    for path in NOTEBOOKS.glob("**/communities/*/04_temporal_community.ipynb"):
        payload = json.loads(path.read_text())
        configuration, = _keyword_calls(payload, "CommunityConfig")
        extras = ast.literal_eval(configuration["extra"])
        assert extras["n_communities"] == 1, path
        assert extras["feature_representation"] == "signed_in_out", path
        source = _source(payload)
        assert "Temporal Neuron Roles" in source and "Leiden" in source, path
    for path in NOTEBOOKS.glob("**/probabilistic_states/*/03_hmm.ipynb"):
        payload = json.loads(path.read_text())
        configuration, = _keyword_calls(payload, "ProbabilisticStateConfig")
        assert ast.literal_eval(configuration["n_states"]) == 1, path
        assert "Optional descriptive HMM" in _source(payload), path


def test_generator_and_new_raw_calibration_notebook_use_the_same_contract():
    for cells in (
        notebook_states("cgc", "fastC-GC"), notebook_community("cgc", "fastC-GC"),
        notebook_probabilistic("cgc", "fastC-GC"), notebook_state_calibration(),
    ):
        for cell in cells:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
    graph, = _keyword_calls({"cells": notebook_states("cgc", "fastC-GC")}, "GraphStateConfig")
    assert ast.literal_eval(graph["n_states"]) == 1
    path = NOTEBOOKS / "validation" / "10_state_calibration.ipynb"
    payload = json.loads(path.read_text())
    source = _source(payload)
    assert 'state_calibration_profile("smoke")' in source
    assert "run_state_calibration(CALIBRATION_CFG)" in source
    assert "write_state_calibration_report" in source
    assert "not a validated neural-regime claim" in source


def test_manifold_defaults_are_supported_and_linking_handles_unavailable_states():
    for path in NOTEBOOKS.glob("**/manifolds/*/05_manifold.ipynb"):
        source = _source(json.loads(path.read_text()))
        assert 'MANIFOLD_CONFIG_NAME = "classical"' in source, path
        assert '"bundle_net_reference_recording" / "artifact.pkl"' not in source, path
    for path in NOTEBOOKS.glob("**/linking/*/06_linking.ipynb"):
        source = _source(json.loads(path.read_text()))
        assert 'if report.get("status") == "unavailable":' in source, path
