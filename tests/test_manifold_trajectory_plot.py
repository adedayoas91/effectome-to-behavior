"""Tests for the interactive 3D manifold trajectory helper."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from effectome.utils.io import save_artifact
from scripts.plot_manifold_trajectory import (
    TrajectoryData,
    TrajectoryExplorer,
    load_trajectory_data,
    plot_trajectory,
    save_rotation_gif,
    smooth_trajectory_segments,
    valid_step_mask,
)


def _anchor(**overrides):
    values = {
        "dataset_id": "dataset",
        "recording_id": "recording",
        "animal_id": "animal",
        "session_id": None,
        "segment_id": None,
        "valid": True,
        "gap_before": False,
        "gap_after": False,
        "bad_frame_before": False,
        "bad_frame_after": False,
        "anchor_time_seconds": 60.0,
        "context_start": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _write_status(stage_dir, connectivity_digest="same"):
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "signature_payload": {
                    "dependencies": {"connectivity": {"sha256": connectivity_digest}}
                },
            }
        )
    )


def test_valid_step_mask_breaks_at_fold_gap_and_nonfinite_point():
    embedding = np.arange(18, dtype=float).reshape(6, 3)
    embedding[4, 0] = np.nan
    anchors = [_anchor() for _ in range(6)]
    anchors[2].gap_after = True
    folds = np.array([0, 0, 1, 1, 1, 1])

    mask = valid_step_mask(embedding, anchors=anchors, fold_ids=folds)

    assert mask.tolist() == [True, False, False, False, False]


def test_plot_trajectory_exports_rotating_gif(tmp_path):
    embedding = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.4],
            [1.4, 1.0, 0.8],
            [0.5, 1.5, 1.2],
        ]
    )
    data = TrajectoryData(
        embedding=embedding,
        labels=np.array([0, 0, 1, 1]),
        label_names={0: "state 0", 1: "state 1"},
        label_colors={0: "#0072B2", 1: "#E69F00"},
        times_minutes=np.arange(4, dtype=float),
        valid_steps=np.array([True, True, True]),
        title="Test trajectory",
    )

    fig, ax = plot_trajectory(data)
    destination = save_rotation_gif(fig, ax, tmp_path / "trajectory.gif", frames=3, fps=2, dpi=40)

    assert destination.is_file()
    assert destination.stat().st_size > 0
    plt.close(fig)


def test_smoothed_segments_preserve_observations_and_respect_breaks():
    embedding = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.2],
            [1.5, 1.0, 0.8],
            [2.0, 1.0, 1.0],
        ]
    )

    curves = smooth_trajectory_segments(
        embedding,
        np.array([True, True, False]),
        segment_count=3,
        samples_per_segment=10,
    )

    assert [curve.step for curve in curves] == [0, 1]
    for curve in curves:
        assert np.allclose(curve.points[0], embedding[curve.step])
        assert np.allclose(curve.points[-1], embedding[curve.step + 1])
    linear_first = np.linspace(embedding[0], embedding[1], 10)
    assert np.max(np.linalg.norm(curves[0].points - linear_first, axis=1)) > 1e-3


def test_explorer_controls_consecutive_window_and_effectome_panels():
    n_steps = 6
    data = TrajectoryData(
        embedding=np.column_stack(
            (
                np.linspace(0.0, 2.0, n_steps),
                np.sin(np.linspace(0.0, np.pi, n_steps)),
                np.cos(np.linspace(0.0, np.pi, n_steps)),
            )
        ),
        labels=np.array([0, 0, 1, 1, 0, 0]),
        label_names={0: "state 0", 1: "state 1"},
        label_colors={0: "#0072B2", 1: "#E69F00"},
        times_minutes=np.arange(n_steps, dtype=float),
        valid_steps=np.ones(n_steps - 1, dtype=bool),
        title="Test trajectory",
        effectomes=np.arange(n_steps * 9, dtype=float).reshape(n_steps, 3, 3),
    )

    explorer = TrajectoryExplorer(data, start_step=0, segment_count=2, active_step=1)
    assert explorer.selection == (0, 2, 1)
    explorer._next(None)
    assert explorer.selection == (2, 2, 2)
    assert len(explorer.active_effectome_ax.images) == 1
    assert len(explorer.mean_effectome_ax.images) == 1
    assert explorer.active_effectome_ax.images[0].get_clim() == (
        explorer.mean_effectome_ax.images[0].get_clim()
    )
    plt.close(explorer.figure)


def test_explorer_tail_navigation_preserves_page_size_and_round_trips():
    n_steps = 12
    data = TrajectoryData(
        embedding=np.column_stack(
            (
                np.linspace(0.0, 2.0, n_steps),
                np.sin(np.linspace(0.0, np.pi, n_steps)),
                np.cos(np.linspace(0.0, np.pi, n_steps)),
            )
        ),
        labels=np.zeros(n_steps, dtype=int),
        label_names={0: "state 0"},
        label_colors={0: "#0072B2"},
        times_minutes=np.arange(n_steps, dtype=float),
        valid_steps=np.ones(n_steps - 1, dtype=bool),
        title="Test trajectory",
        effectomes=np.zeros((n_steps, 2, 2)),
    )
    explorer = TrajectoryExplorer(data, start_step=0, segment_count=5)

    explorer._next(None)
    assert explorer.selection[:2] == (5, 5)
    explorer._next(None)
    assert explorer.selection[:2] == (6, 5)
    explorer._previous(None)
    assert explorer.selection[:2] == (5, 5)
    plt.close(explorer.figure)


def test_load_trajectory_rejects_stale_graph_window_alignment(tmp_path):
    run_root = tmp_path / "run"
    anchors = [_anchor(context_start=index, anchor_time_seconds=float(index)) for index in range(3)]
    manifold_dir = run_root / "cgc" / "stages" / "manifold"
    graph_dir = run_root / "cgc" / "stages" / "graph_states"
    save_artifact(
        SimpleNamespace(
            window_embedding=np.arange(9, dtype=float).reshape(3, 3),
            window_starts=np.array([0, 1, 2]),
            anchors=anchors,
            metadata={"cross_fit_fold_ids": None},
        ),
        manifold_dir / "artifact.pkl",
    )
    save_artifact(
        SimpleNamespace(labels=np.array([0, 1, 1]), window_starts=np.array([0, 1, 3])),
        graph_dir / "artifact.pkl",
    )
    connectivity_path = run_root / "cgc" / "stages" / "connectivity" / "artifact.pkl"
    save_artifact(
        SimpleNamespace(
            matrices=np.zeros((3, 2, 2)),
            window_starts=np.array([0, 1, 2]),
            behavior_per_window={"motif": np.array([0, 1, 1])},
        ),
        connectivity_path,
    )
    connectivity_digest = sha256(connectivity_path.read_bytes()).hexdigest()
    _write_status(manifold_dir, connectivity_digest)
    _write_status(graph_dir, connectivity_digest)

    with np.testing.assert_raises_regex(ValueError, "not aligned"):
        load_trajectory_data(run_root, method="cgc", color_by="graph")


def test_load_trajectory_verifies_and_returns_aligned_effectomes(tmp_path):
    run_root = tmp_path / "run"
    anchors = [_anchor(context_start=index, anchor_time_seconds=float(index)) for index in range(3)]
    manifold_dir = run_root / "cgc" / "stages" / "manifold"
    connectivity_path = run_root / "cgc" / "stages" / "connectivity" / "artifact.pkl"
    save_artifact(
        SimpleNamespace(
            window_embedding=np.arange(9, dtype=float).reshape(3, 3),
            window_starts=np.array([0, 1, 2]),
            anchors=anchors,
            metadata={"cross_fit_fold_ids": None},
        ),
        manifold_dir / "artifact.pkl",
    )
    matrices = np.arange(12, dtype=float).reshape(3, 2, 2)
    save_artifact(
        SimpleNamespace(
            matrices=matrices,
            window_starts=np.array([0, 1, 2]),
            behavior_per_window={"motif": np.array([0, 1, 1])},
        ),
        connectivity_path,
    )
    _write_status(manifold_dir, sha256(connectivity_path.read_bytes()).hexdigest())

    loaded = load_trajectory_data(run_root, method="cgc", color_by="motif")

    assert np.array_equal(loaded.effectomes, matrices)
    assert loaded.labels.tolist() == [0, 1, 1]


def test_visualization_notebook_exposes_segment_controls():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "visualization" / (
        "11_manifold_trajectory_3d.ipynb"
    )
    payload = json.loads(path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

    assert "TrajectoryExplorer" in source
    assert "SEGMENTS_PER_VIEW" in source
    assert "START_STEP" in source
    assert "ACTIVE_EFFECTOME_STEP" in source
    assert "centripetal Catmull--Rom" in source
