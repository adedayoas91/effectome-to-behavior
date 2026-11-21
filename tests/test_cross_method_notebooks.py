"""Canonical recording notebooks share one raw-owned reference and preserve old results."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from effectome.connectivity import ConnectivityConfig
from effectome.data_module import WindowConfig
from effectome.experiments.transition_examples import TransitionExampleConfig, make_transition_example
from effectome.utils.io import save_artifact
from effectome.workflows.cross_method import CrossMethodConfig
from notebooks._build_notebooks import SHARED_TRANSITION_NOTICE_MARKER, migrate_shared_transition_notices
from notebooks._shared import make_run, run_shared_transition_comparison


def test_v2a_recordings_have_canonical_entrypoint_and_each_stage_points_to_it():
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for recording in (
        root / "v2a_rsns/220119_F2_run11",
        root / "v2a_rsns/220127_F4_run2",
    ):
        canonical = recording / "09_cross_method/shared_transitions.ipynb"
        assert canonical.is_file()
        code = "\n".join("".join(cell["source"]) for cell in json.loads(canonical.read_text())["cells"])
        assert 'stage_artifact_path(shared_run, "raw_recording")' in code
        assert "standardize_per_window=False" in code
        assert "run_shared_transition_comparison(" in code
        for path in recording.rglob("*.ipynb"):
            if path != canonical:
                assert SHARED_TRANSITION_NOTICE_MARKER in path.read_text(), path


def test_scope_migration_preserves_code_metadata_and_stored_error_outputs(tmp_path):
    path = tmp_path / "05_manifolds/cgc.ipynb"
    path.parent.mkdir()
    payload = {
        "metadata": {"language_info": {"version": "3.14.7"}},
        "cells": [
            {"cell_type": "markdown", "source": ["# Historical analysis\n"], "metadata": {}},
            {
                "cell_type": "code",
                "source": ["x=1\n"],
                "execution_count": 1,
                "metadata": {"user": True},
                "outputs": [
                    {"output_type": "error", "ename": "Example", "evalue": "stored", "traceback": []}
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload))
    assert migrate_shared_transition_notices(tmp_path) == [path]
    updated = json.loads(path.read_text())
    assert updated["metadata"] == payload["metadata"]
    assert updated["cells"][1] == payload["cells"][1]
    assert migrate_shared_transition_notices(tmp_path) == []


def test_canonical_helper_rebuilds_raw_windows_and_tracks_cache_settings(tmp_path):
    raw = make_transition_example(TransitionExampleConfig(n_samples=640))
    path = save_artifact(raw, tmp_path / "raw.pkl")
    run = make_run(tmp_path / "output", "shared", "cross-method")
    config = CrossMethodConfig(n_dims=2)
    window = WindowConfig(
        mode="temporal", history_length=48, target_length=8, stride=16, standardize_per_window=True
    )
    methods = {"partial": ConnectivityConfig(name="correlation", partial=True)}
    first = run_shared_transition_comparison(
        run, raw_recording_path=path, window_cfg=window, comparison_cfg=config, method_configs=methods
    )
    second = run_shared_transition_comparison(
        run, raw_recording_path=path, window_cfg=window, comparison_cfg=config, method_configs=methods
    )
    assert run.last_record.reused
    np.testing.assert_array_equal(first["reference"]["state_centers"], second["reference"]["state_centers"])
    third = run_shared_transition_comparison(
        run,
        raw_recording_path=path,
        window_cfg=window,
        comparison_cfg=replace(config, n_manifold_states=2),
        method_configs=methods,
    )
    assert not run.last_record.reused
    assert len(third["reference"]["state_centers"]) == 2
