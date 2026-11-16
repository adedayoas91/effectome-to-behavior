"""Held-out structural diagnostics preserve labels, density controls and failures."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from effectome.experiments.graph_similarity_validation import (
    _fit_panel,
    graph_similarity_inventory,
    graph_similarity_probes,
    load_graph_similarity_profile,
    make_planted_graph_panel,
    render_graph_similarity_report,
    run_graph_similarity_validation,
)


def test_planted_panels_hold_density_and_sampling_fixed_across_rewiring():
    config = load_graph_similarity_profile("smoke")
    clean = make_planted_graph_panel(config, seed=2, rewiring_probability=0)
    corrupted = make_planted_graph_panel(config, seed=2, rewiring_probability=1)
    np.testing.assert_array_equal(clean.identity_labels, corrupted.identity_labels)
    np.testing.assert_array_equal(clean.topology_labels, corrupted.topology_labels)
    for panel in (clean, corrupted):
        graphs = panel.series.lagged_matrices
        assert graphs is not None
        np.testing.assert_array_equal(graphs.sum(axis=(1, 2, 3)), np.full(len(graphs), 2))
        assert not np.diagonal(graphs, axis1=2, axis2=3).any()
        assert not graphs[:, 1].any()
        assert not np.intersect1d(panel.fit_indices, panel.evaluation_indices).size
        assert panel.series.behavior_per_window == {}
        assert "identity_labels" not in panel.series.diagnostics
    assert np.all(np.any(clean.series.lagged_matrices != corrupted.series.lagged_matrices, axis=(1, 2, 3)))


def test_probes_distinguish_identity_direction_lag_and_known_WL_collision():
    rows = {row["probe"]: row for row in graph_similarity_probes(load_graph_similarity_profile("smoke"))}
    relabelled = rows["same_topology_different_neurons"]
    assert relabelled["exact_lag_preserving_isomorphic"]
    assert relabelled["edge_jaccard_distance"] > 0
    assert relabelled["directed_wl_distance"] == 0
    for key in ("outward_versus_inward_star", "same_edges_different_physical_lag"):
        assert not rows[key]["exact_lag_preserving_isomorphic"]
        assert rows[key]["directed_wl_distance"] > 0
        assert rows[key]["edge_jaccard_distance"] > 0
    collision = rows["six_cycle_versus_two_triangles_WL_collision"]
    assert not collision["exact_lag_preserving_isomorphic"]
    assert collision["directed_wl_distance"] == 0
    assert rows["both_empty_no_positive_recovery_evidence"]["both_empty"]


def test_archetype_labels_cannot_change_fitted_graph_codebook():
    config = replace(load_graph_similarity_profile("smoke"), metrics=("edge_jaccard",),
                     candidate_counts=(4,), train_repeats=2, evaluation_repeats=2)
    panel = make_planted_graph_panel(config, seed=1, rewiring_probability=0)
    altered = replace(panel, identity_labels=np.zeros_like(panel.identity_labels),
                      topology_labels=np.zeros_like(panel.topology_labels))
    before = _fit_panel(panel, config, seed=1, rewiring_probability=0)[0]
    after = _fit_panel(altered, config, seed=1, rewiring_probability=0)[0]
    assert before["status"] == after["status"] == "available"
    assert before["medoid_fingerprint"] == after["medoid_fingerprint"]
    np.testing.assert_array_equal(before["heldout_assignments"], after["heldout_assignments"])
    assert before["heldout_identity_ari"] == pytest.approx(1)
    assert after["heldout_identity_ari"] == 0


def test_validation_exports_unavailable_partitions_and_rejects_protocol_mixing(tmp_path):
    config = replace(load_graph_similarity_profile("smoke"), seeds=(0,),
                     rewiring_probabilities=(0,), candidate_counts=(3, 4),
                     train_repeats=2, evaluation_repeats=2)
    report = run_graph_similarity_validation(config, tmp_path)
    assert report["execution_complete"]
    assert report["attempted_fits"] == 4
    by_method = {(row["metric"], row["n_states"]): row for row in report["rows"]}
    assert by_method["directed_wl", 3]["status"] == "available"
    assert by_method["directed_wl", 4]["status"] == "unavailable"
    assert by_method["directed_wl", 4]["reason"]
    assert not report["scientific_validation_passed"]
    assert not report["native_CSL_refitted"]
    path = render_graph_similarity_report(report, tmp_path / "analysis")
    assert path.is_file()
    for name in ("seed-grouping.csv", "distance-probes.csv", "graph-characteristics.csv",
                 "figure-01-heldout-grouping.pdf", "figure-01-heldout-grouping.png",
                 "figure-02-distance-probes.pdf", "figure-02-distance-probes.png"):
        assert (path.parent / name).stat().st_size > 0
    stored = json.loads((tmp_path / "graph-similarity.json").read_text())
    assert stored["attempted_fits"] == report["attempted_fits"]
    with pytest.raises(ValueError, match="another graph similarity protocol"):
        run_graph_similarity_validation(replace(config, seeds=(1,)), tmp_path)


def test_inventory_cli_does_not_fit_or_write_results(tmp_path):
    config = load_graph_similarity_profile("smoke")
    assert graph_similarity_inventory(config)["planned_quantizer_fits"] == 36
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "inventory-only"
    process = subprocess.run(
        [sys.executable, "pipeline/16_graph_similarity.py", "--output-dir", str(output)],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert "Graph similarity inventory" in process.stderr
    assert not output.exists()


@pytest.mark.parametrize("changes", [
    {"seeds": (1, 1)}, {"seeds": (True,)}, {"candidate_counts": (0,)},
    {"rewiring_probabilities": (float("nan"),)}, {"metrics": ("unknown",)},
    {"physical_lags": (0.2, 0.1)}, {"train_repeats": 1},
])
def test_invalid_graph_diagnostic_protocol_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(load_graph_similarity_profile("smoke"), **changes)



def _inspect_graph_notebook(profile: str, root: Path, monkeypatch) -> dict:
    from notebooks import _shared

    repository = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repository)
    monkeypatch.setattr(_shared, "PROJECT_ROOT", root)
    notebook = json.loads((repository / "synthetic/07_graph_similarity.ipynb").read_text())
    namespace = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"]).replace('PROFILE = "smoke"', f'PROFILE = "{profile}"')
        exec(compile(source, "graph-notebook-inspection", "exec"), namespace)
    return namespace


def test_graph_notebook_profile_path_cannot_load_another_profiles_smoke_result(tmp_path, monkeypatch):
    smoke = tmp_path / "outputs/graph-similarity/smoke/smoke-01/graph-similarity.json"
    smoke.parent.mkdir(parents=True)
    smoke.write_text(json.dumps({"protocol": {"config": asdict(load_graph_similarity_profile("smoke"))}}))
    namespace = _inspect_graph_notebook("development", tmp_path, monkeypatch)
    assert namespace["OUTPUT_DIRECTORY"] == tmp_path / "outputs/graph-similarity/development/smoke-01"
    assert namespace["report"] is None
    assert not namespace["RUN"]


def test_graph_notebook_refuses_mismatched_saved_config_in_requested_path(tmp_path, monkeypatch, capsys):
    copied = tmp_path / "outputs/graph-similarity/development/smoke-01/graph-similarity.json"
    copied.parent.mkdir(parents=True)
    copied.write_text(json.dumps({"protocol": {"config": asdict(load_graph_similarity_profile("smoke"))}}))
    with pytest.raises(ValueError, match="Saved diagnostic configuration does not match"):
        _inspect_graph_notebook("development", tmp_path, monkeypatch)
    output = capsys.readouterr().out
    assert "Saved configuration:" in output
    assert "saved local run (not a new run)" in output
