"""Plot a checkpointed manifold as an interactive 3D temporal trajectory.

The native Matplotlib window supports click-and-drag rotation. The same figure
can be exported as a rotating GIF without adding a plotting dependency.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from effectome.utils.io import load_artifact

ColorBy = Literal["motif", "graph", "hmm"]

MOTIF_NAMES = ("dt", "fwd", "nostate", "rev1", "rev2", "revsus", "slow", "vt")
MOTIF_COLORS = (
    "#999999",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
    "#E69F00",
    "#009E73",
    "#56B4E9",
)
STATE_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00")


@dataclass(frozen=True)
class TrajectoryData:
    """Aligned values needed to draw one manifold trajectory."""

    embedding: np.ndarray
    labels: np.ndarray
    label_names: dict[int, str]
    label_colors: dict[int, str]
    times_minutes: np.ndarray
    valid_steps: np.ndarray
    title: str
    effectomes: np.ndarray | None = None


@dataclass(frozen=True)
class SmoothSegment:
    """One interpolated transition that preserves its observed endpoints."""

    step: int
    points: np.ndarray


def _anchor_group(anchor: Any) -> tuple[Any, ...]:
    return tuple(
        getattr(anchor, field, None)
        for field in ("dataset_id", "recording_id", "animal_id", "session_id", "segment_id")
    )


def valid_step_mask(
    embedding: np.ndarray,
    *,
    anchors: Sequence[Any] | None = None,
    fold_ids: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Return which consecutive points may be joined without crossing a discontinuity."""
    values = np.asarray(embedding, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"embedding must have shape (steps, >=3), got {values.shape}")
    if values.shape[0] < 2:
        raise ValueError("embedding must contain at least two time steps")

    valid = np.isfinite(values[:-1, :3]).all(axis=1) & np.isfinite(values[1:, :3]).all(axis=1)

    if fold_ids is not None:
        folds = np.asarray(fold_ids)
        if folds.shape != (values.shape[0],):
            raise ValueError(f"fold_ids must have shape ({values.shape[0]},), got {folds.shape}")
        valid &= folds[:-1] == folds[1:]

    if anchors is not None:
        if len(anchors) != values.shape[0]:
            raise ValueError(f"anchors must contain {values.shape[0]} entries, got {len(anchors)}")
        for index, (current, following) in enumerate(zip(anchors[:-1], anchors[1:], strict=True)):
            crosses_boundary = (
                _anchor_group(current) != _anchor_group(following)
                or not bool(getattr(current, "valid", True))
                or not bool(getattr(following, "valid", True))
                or bool(getattr(current, "gap_after", False))
                or bool(getattr(following, "gap_before", False))
                or bool(getattr(current, "bad_frame_after", False))
                or bool(getattr(following, "bad_frame_before", False))
            )
            if crosses_boundary:
                valid[index] = False
    return valid


def _centripetal_catmull_rom(
    control_points: np.ndarray,
    *,
    samples: int,
) -> np.ndarray:
    """Interpolate the middle control-point pair with a centripetal Catmull--Rom curve."""
    if control_points.shape != (4, 3):
        raise ValueError(f"control_points must have shape (4, 3), got {control_points.shape}")
    if samples < 4:
        raise ValueError("curve samples must be at least 4")

    p0, p1, p2, p3 = control_points
    if np.linalg.norm(p2 - p1) <= np.finfo(float).eps:
        return np.repeat(p1[None, :], samples, axis=0)

    def next_knot(previous: float, start: np.ndarray, stop: np.ndarray) -> float:
        chord = max(float(np.linalg.norm(stop - start)), 1e-12)
        return previous + chord**0.5

    t0 = 0.0
    t1 = next_knot(t0, p0, p1)
    t2 = next_knot(t1, p1, p2)
    t3 = next_knot(t2, p2, p3)
    values = np.linspace(t1, t2, samples)
    column = values[:, None]

    a1 = (t1 - column) / (t1 - t0) * p0 + (column - t0) / (t1 - t0) * p1
    a2 = (t2 - column) / (t2 - t1) * p1 + (column - t1) / (t2 - t1) * p2
    a3 = (t3 - column) / (t3 - t2) * p2 + (column - t2) / (t3 - t2) * p3
    b1 = (t2 - column) / (t2 - t0) * a1 + (column - t0) / (t2 - t0) * a2
    b2 = (t3 - column) / (t3 - t1) * a2 + (column - t1) / (t3 - t1) * a3
    curve = (t2 - column) / (t2 - t1) * b1 + (column - t1) / (t2 - t1) * b2
    curve[0] = p1
    curve[-1] = p2
    return curve


def smooth_trajectory_segments(
    embedding: np.ndarray,
    valid_steps: np.ndarray,
    *,
    start_step: int = 0,
    segment_count: int | None = None,
    samples_per_segment: int = 16,
) -> list[SmoothSegment]:
    """Return rounded, endpoint-preserving curves for a consecutive step window."""
    values = np.asarray(embedding, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("embedding must be a finite array with shape (steps, 3)")
    steps = np.asarray(valid_steps, dtype=bool)
    if steps.shape != (values.shape[0] - 1,):
        raise ValueError(f"valid_steps must have shape ({values.shape[0] - 1},), got {steps.shape}")
    start, count = selection_bounds(values.shape[0], start_step, segment_count)
    stop = start + count

    curves: list[SmoothSegment] = []
    for step in range(start, stop):
        if not steps[step]:
            continue
        p1 = values[step]
        p2 = values[step + 1]
        p0 = values[step - 1] if step > 0 and steps[step - 1] else 2.0 * p1 - p2
        p3 = (
            values[step + 2]
            if step + 1 < steps.size and steps[step + 1]
            else 2.0 * p2 - p1
        )
        curves.append(
            SmoothSegment(
                step=step,
                points=_centripetal_catmull_rom(
                    np.stack((p0, p1, p2, p3)),
                    samples=samples_per_segment,
                ),
            )
        )
    return curves


def selection_bounds(
    n_steps: int,
    start_step: int,
    segment_count: int | None,
) -> tuple[int, int]:
    """Clamp a requested consecutive transition window to the available trajectory."""
    if n_steps < 2:
        raise ValueError("a trajectory needs at least two steps")
    start = min(max(int(start_step), 0), n_steps - 2)
    requested = n_steps - 1 - start if segment_count is None else int(segment_count)
    if requested < 1:
        raise ValueError("segment_count must be at least 1")
    return start, min(requested, n_steps - 1 - start)


def _category_style(labels: np.ndarray, color_by: ColorBy) -> tuple[dict[int, str], dict[int, str]]:
    categories = sorted(int(value) for value in np.unique(labels))
    if color_by == "motif":
        names = {
            value: MOTIF_NAMES[value] if 0 <= value < len(MOTIF_NAMES) else f"motif {value}"
            for value in categories
        }
        colors = {value: MOTIF_COLORS[value % len(MOTIF_COLORS)] for value in categories}
        return names, colors

    prefix = "graph state" if color_by == "graph" else "HMM state"
    names = {value: f"{prefix} {value}" for value in categories}
    colors = {value: STATE_COLORS[index % len(STATE_COLORS)] for index, value in enumerate(categories)}
    return names, colors


def _dependency_digest(stage_dir: Path, dependency: str) -> str:
    status_path = stage_dir / "status.json"
    try:
        status = json.loads(status_path.read_text())
        digest = status["signature_payload"]["dependencies"][dependency]["sha256"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot verify {stage_dir.name!r} alignment: {status_path} lacks a {dependency!r} digest"
        ) from error
    if status.get("status") != "completed":
        raise ValueError(f"stage {stage_dir} is not completed")
    return str(digest)


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_same_starts(reference: np.ndarray, candidate: np.ndarray, source: Path) -> None:
    if reference.shape != candidate.shape or not np.array_equal(reference, candidate):
        raise ValueError(f"{source} is not aligned with the manifold window starts")


def load_trajectory_data(
    run_root: Path,
    *,
    method: str,
    color_by: ColorBy,
    behavior_key: str = "motif",
) -> TrajectoryData:
    """Load trusted local checkpoints and align labels with the manifold windows."""
    manifold_path = run_root / method / "stages" / "manifold" / "artifact.pkl"
    manifold = load_artifact(manifold_path)
    embedding = np.asarray(manifold.window_embedding, dtype=float)
    if embedding.ndim != 2 or embedding.shape[1] < 3:
        raise ValueError(
            f"{manifold_path} has {embedding.shape[1] if embedding.ndim == 2 else 0} dimensions; "
            "a 3D trajectory requires at least three"
        )
    if not np.isfinite(embedding[:, :3]).all():
        raise ValueError(f"{manifold_path} contains non-finite values in its first three dimensions")

    metadata = dict(getattr(manifold, "metadata", {}))
    anchors = list(getattr(manifold, "anchors", [])) or metadata.get("anchors")
    if not anchors:
        raise ValueError(f"{manifold_path} has no temporal anchors; recording time cannot be verified")
    fold_ids = metadata.get("cross_fit_fold_ids")
    manifold_starts = np.asarray(
        getattr(manifold, "window_starts", None)
        if getattr(manifold, "window_starts", None) is not None
        else [anchor.context_start for anchor in anchors],
        dtype=int,
    )

    connectivity_path = run_root / method / "stages" / "connectivity" / "artifact.pkl"
    expected_connectivity_digest = _dependency_digest(manifold_path.parent, "connectivity")
    actual_connectivity_digest = _file_digest(connectivity_path)
    if actual_connectivity_digest != expected_connectivity_digest:
        raise ValueError(
            f"{connectivity_path} does not match the connectivity dependency recorded by {manifold_path}"
        )
    connectivity = load_artifact(connectivity_path)
    connectivity_starts = np.asarray(connectivity.window_starts, dtype=int)
    _require_same_starts(manifold_starts, connectivity_starts, connectivity_path)
    effectomes = np.asarray(connectivity.matrices, dtype=float)
    if effectomes.ndim != 3 or effectomes.shape[0] != embedding.shape[0]:
        raise ValueError(
            f"{connectivity_path} matrices have shape {effectomes.shape}; expected "
            f"({embedding.shape[0]}, nodes, nodes)"
        )

    if color_by == "motif":
        if behavior_key not in connectivity.behavior_per_window:
            available = ", ".join(sorted(connectivity.behavior_per_window))
            raise KeyError(f"behavior key {behavior_key!r} is unavailable; choose one of: {available}")
        labels = np.asarray(connectivity.behavior_per_window[behavior_key], dtype=int)
    else:
        stage = "graph_states" if color_by == "graph" else "probabilistic_states"
        state_dir = run_root / method / "stages" / stage
        state_path = state_dir / "artifact.pkl"
        state_artifact = load_artifact(state_path)
        labels = np.asarray(state_artifact.labels, dtype=int)
        state_digest = _dependency_digest(state_dir, "connectivity")
        if expected_connectivity_digest != state_digest:
            raise ValueError(f"{state_path} and {manifold_path} depend on different connectivity artifacts")
        state_starts = getattr(state_artifact, "window_starts", None)
        if state_starts is not None:
            _require_same_starts(manifold_starts, np.asarray(state_starts, dtype=int), state_path)

    if labels.shape != (embedding.shape[0],):
        raise ValueError(
            f"{color_by} labels have shape {labels.shape}, but the manifold has {embedding.shape[0]} windows"
        )

    times_minutes = np.asarray([float(anchor.anchor_time_seconds) for anchor in anchors]) / 60.0

    label_names, label_colors = _category_style(labels, color_by)
    method_label = method.replace("_", " ")
    return TrajectoryData(
        embedding=embedding[:, :3],
        labels=labels,
        label_names=label_names,
        label_colors=label_colors,
        times_minutes=times_minutes,
        valid_steps=valid_step_mask(embedding, anchors=anchors, fold_ids=fold_ids),
        title=f"{method_label}: 3D manifold trajectory colored by {color_by}",
        effectomes=effectomes,
    )


def _draw_trajectory_axis(
    ax: Axes3D,
    data: TrajectoryData,
    *,
    start_step: int,
    segment_count: int | None,
    curve_samples: int,
    arrow_stride: int,
    active_step: int | None = None,
    legend_outside: bool = True,
) -> tuple[Any, np.ndarray]:
    """Draw one selected trajectory window on an existing 3D axis."""
    if arrow_stride < 1:
        raise ValueError("arrow_stride must be at least 1")
    embedding = data.embedding
    if embedding.ndim != 2 or embedding.shape[1] != 3 or not np.isfinite(embedding).all():
        raise ValueError("trajectory embedding must be a finite array with shape (steps, 3)")

    start, count = selection_bounds(embedding.shape[0], start_step, segment_count)
    stop = start + count
    point_indices = np.arange(start, stop + 1, dtype=int)
    point_colors = np.asarray([data.label_colors[int(label)] for label in data.labels])
    curves = smooth_trajectory_segments(
        embedding,
        data.valid_steps,
        start_step=start,
        segment_count=count,
        samples_per_segment=curve_samples,
    )

    line_segments: list[np.ndarray] = []
    line_colors: list[str] = []
    for curve in curves:
        line_segments.extend(np.stack((curve.points[:-1], curve.points[1:]), axis=1))
        line_colors.extend([point_colors[curve.step + 1]] * (curve.points.shape[0] - 1))
    if line_segments:
        ax.add_collection3d(
            Line3DCollection(line_segments, colors=line_colors, linewidths=1.35, alpha=0.82)
        )

    arrow_starts: list[np.ndarray] = []
    arrow_vectors: list[np.ndarray] = []
    arrow_colors: list[str] = []
    for curve in curves[::arrow_stride]:
        arrow_index = max(1, min(curve.points.shape[0] - 2, int(curve.points.shape[0] * 0.72)))
        tangent = curve.points[arrow_index + 1] - curve.points[arrow_index - 1]
        tangent_norm = float(np.linalg.norm(tangent))
        chord = float(np.linalg.norm(embedding[curve.step + 1] - embedding[curve.step]))
        if tangent_norm <= np.finfo(float).eps or chord <= np.finfo(float).eps:
            continue
        arrow_starts.append(curve.points[arrow_index])
        arrow_vectors.append(tangent / tangent_norm * chord * 0.16)
        arrow_colors.append(point_colors[curve.step + 1])
    if arrow_starts:
        starts = np.asarray(arrow_starts)
        vectors = np.asarray(arrow_vectors)
        ax.quiver(
            starts[:, 0],
            starts[:, 1],
            starts[:, 2],
            vectors[:, 0],
            vectors[:, 1],
            vectors[:, 2],
            colors=arrow_colors,
            arrow_length_ratio=0.5,
            linewidths=0.9,
            alpha=0.9,
            normalize=False,
        )

    selected = embedding[point_indices]
    points = ax.scatter(
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        c=point_colors[point_indices],
        s=28,
        edgecolors="none",
        alpha=0.9,
        depthshade=True,
        picker=True,
    )
    transition_mask = data.valid_steps & (data.labels[:-1] != data.labels[1:])
    transition_indices = np.flatnonzero(transition_mask) + 1
    transition_indices = transition_indices[(transition_indices > start) & (transition_indices <= stop)]
    if transition_indices.size:
        transition_points = embedding[transition_indices]
        ax.scatter(
            transition_points[:, 0],
            transition_points[:, 1],
            transition_points[:, 2],
            s=72,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.0,
            depthshade=False,
        )

    ax.scatter(*embedding[start], marker="^", s=90, c="#111111", depthshade=False)
    ax.scatter(*embedding[stop], marker="*", s=135, c="#111111", depthshade=False)
    if active_step is not None:
        active = min(max(int(active_step), start), stop)
        ax.scatter(
            *embedding[active],
            marker="D",
            s=115,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.6,
            depthshade=False,
        )

    ax.set_xlabel("Manifold dimension 1")
    ax.set_ylabel("Manifold dimension 2")
    ax.set_zlabel("Manifold dimension 3")
    ax.set_title(
        f"{data.title}\nsteps {start}–{stop}; "
        f"{data.times_minutes[start]:.2f}–{data.times_minutes[stop]:.2f} min",
        pad=18,
    )
    ranges = np.ptp(selected, axis=0)
    ax.set_box_aspect(np.where(ranges > 0, ranges, 1.0), zoom=0.86)
    ax.view_init(elev=24, azim=-58)

    selected_labels = data.labels[point_indices]
    categories = sorted(int(value) for value in np.unique(selected_labels))
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=data.label_colors[value],
            markeredgecolor="none",
            label=f"{data.label_names[value]} (n={np.count_nonzero(selected_labels == value)})",
        )
        for value in categories
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="#111111",
                label="label transition",
            ),
            Line2D([0], [0], marker="^", linestyle="", color="#111111", label="view start"),
            Line2D([0], [0], marker="*", linestyle="", color="#111111", label="view end"),
        ]
    )
    if active_step is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor="#111111",
                label="active effectome",
            )
        )
    if legend_outside:
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 0.96), frameon=False)
    else:
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)
    return points, point_indices


def plot_trajectory(
    data: TrajectoryData,
    *,
    arrow_stride: int = 1,
    start_step: int = 0,
    segment_count: int | None = None,
    curve_samples: int = 16,
) -> tuple[Figure, Axes3D]:
    """Draw a rounded trajectory window with temporal arrows and transition markers."""
    fig = plt.figure(figsize=(10.5, 8.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    points, point_indices = _draw_trajectory_axis(
        ax,
        data,
        start_step=start_step,
        segment_count=segment_count,
        curve_samples=curve_samples,
        arrow_stride=arrow_stride,
    )
    instruction = ax.text2D(
        0.01,
        0.98,
        "Drag to rotate; scroll to zoom; click a point for its step, time, and label",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#444444",
    )
    selection = ax.text2D(0.01, 0.02, "", transform=ax.transAxes, va="bottom", fontsize=9)

    def show_selected_point(event: Any) -> None:
        if event.artist is not points or not len(event.ind):
            return
        index = int(point_indices[int(event.ind[0])])
        selection.set_text(
            f"step {index}; t={data.times_minutes[index]:.2f} min; "
            f"{data.label_names[int(data.labels[index])]}"
        )
        instruction.set_color("#777777")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", show_selected_point)
    fig.text(
        0.01,
        0.005,
        "Rounded curves interpolate observed points; arrows follow curve tangents. "
        "This is descriptive, not evidence of dynamical causation.",
        fontsize=8,
        color="#555555",
    )
    return fig, ax


def _effectome_limit(*matrices: np.ndarray) -> float:
    values = np.concatenate([np.asarray(matrix, dtype=float).ravel() for matrix in matrices])
    finite = np.abs(values[np.isfinite(values)])
    limit = float(np.quantile(finite, 0.995)) if finite.size else 1.0
    if limit <= np.finfo(float).eps:
        limit = 1.0
    return limit


def _draw_effectome(ax: Any, matrix: np.ndarray, title: str, *, limit: float) -> None:
    values = np.asarray(matrix, dtype=float)
    ax.clear()
    ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_title(f"{title}\nshared color range ±{limit:.3g}", fontsize=9)
    ax.set_xlabel("Target row")
    ax.set_ylabel("Source row")
    ax.spines[:].set_visible(False)


class TrajectoryExplorer:
    """Matplotlib controls for consecutive trajectory windows and synchronized effectomes."""

    def __init__(
        self,
        data: TrajectoryData,
        *,
        start_step: int = 0,
        segment_count: int = 25,
        active_step: int | None = None,
        curve_samples: int = 16,
        arrow_stride: int = 1,
    ) -> None:
        if data.effectomes is None:
            raise ValueError("TrajectoryExplorer requires aligned effectome matrices")
        if curve_samples < 4:
            raise ValueError("curve_samples must be at least 4")
        if arrow_stride < 1:
            raise ValueError("arrow_stride must be at least 1")
        self.data = data
        self.curve_samples = curve_samples
        self.arrow_stride = arrow_stride
        self._updating = False

        start, count = selection_bounds(data.embedding.shape[0], start_step, segment_count)
        active = start if active_step is None else min(max(int(active_step), start), start + count)
        self.figure = plt.figure(figsize=(14.5, 8.8))
        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=(1.75, 1.0),
            left=0.04,
            right=0.97,
            bottom=0.23,
            top=0.95,
            wspace=0.18,
            hspace=0.28,
        )
        self.trajectory_ax = self.figure.add_subplot(grid[:, 0], projection="3d")
        self.active_effectome_ax = self.figure.add_subplot(grid[0, 1])
        self.mean_effectome_ax = self.figure.add_subplot(grid[1, 1])

        start_ax = self.figure.add_axes((0.13, 0.145, 0.67, 0.025))
        count_ax = self.figure.add_axes((0.13, 0.10, 0.67, 0.025))
        active_ax = self.figure.add_axes((0.13, 0.055, 0.67, 0.025))
        previous_ax = self.figure.add_axes((0.83, 0.115, 0.07, 0.04))
        next_ax = self.figure.add_axes((0.91, 0.115, 0.06, 0.04))
        self.start_slider = Slider(
            start_ax,
            "Start step",
            0,
            data.embedding.shape[0] - 2,
            valinit=start,
            valstep=1,
        )
        self.count_slider = Slider(
            count_ax,
            "Segments",
            1,
            data.embedding.shape[0] - 1,
            valinit=count,
            valstep=1,
        )
        self.active_slider = Slider(
            active_ax,
            "Effectome step",
            0,
            data.embedding.shape[0] - 1,
            valinit=active,
            valstep=1,
        )
        self.previous_button = Button(previous_ax, "Previous")
        self.next_button = Button(next_ax, "Next")
        self.start_slider.on_changed(self._selection_changed)
        self.count_slider.on_changed(self._selection_changed)
        self.active_slider.on_changed(self._selection_changed)
        self.previous_button.on_clicked(self._previous)
        self.next_button.on_clicked(self._next)
        self._redraw()

    @staticmethod
    def _set_slider_value(slider: Slider, value: int) -> None:
        event_state = slider.eventson
        slider.eventson = False
        slider.set_val(value)
        slider.eventson = event_state

    @property
    def selection(self) -> tuple[int, int, int]:
        """Return the clamped start step, segment count, and active effectome step."""
        start, count = selection_bounds(
            self.data.embedding.shape[0],
            int(round(self.start_slider.val)),
            int(round(self.count_slider.val)),
        )
        active = min(max(int(round(self.active_slider.val)), start), start + count)
        return start, count, active

    def _selection_changed(self, _value: float) -> None:
        if self._updating:
            return
        self._updating = True
        start, count, active = self.selection
        self._set_slider_value(self.start_slider, start)
        self._set_slider_value(self.count_slider, count)
        self._set_slider_value(self.active_slider, active)
        self._updating = False
        self._redraw()

    def _previous(self, _event: Any) -> None:
        start, count, _ = self.selection
        max_start = self.data.embedding.shape[0] - 1 - count
        if start == max_start and start % count:
            previous = start // count * count
        else:
            previous = max(0, start - count)
        self.start_slider.set_val(previous)

    def _next(self, _event: Any) -> None:
        start, count, _ = self.selection
        max_start = self.data.embedding.shape[0] - 1 - count
        self.start_slider.set_val(min(max_start, start + count))

    def _redraw(self) -> None:
        start, count, active = self.selection
        stop = start + count
        self.trajectory_ax.clear()
        _draw_trajectory_axis(
            self.trajectory_ax,
            self.data,
            start_step=start,
            segment_count=count,
            curve_samples=self.curve_samples,
            arrow_stride=self.arrow_stride,
            active_step=active,
            legend_outside=False,
        )
        effectomes = np.asarray(self.data.effectomes)
        active_effectome = effectomes[active]
        mean_effectome = np.mean(effectomes[start : stop + 1], axis=0)
        shared_limit = _effectome_limit(active_effectome, mean_effectome)
        _draw_effectome(
            self.active_effectome_ax,
            active_effectome,
            f"Effectome step {active} ({self.data.times_minutes[active]:.2f} min)",
            limit=shared_limit,
        )
        _draw_effectome(
            self.mean_effectome_ax,
            mean_effectome,
            f"Mean effectome, steps {start}–{stop}",
            limit=shared_limit,
        )
        self.figure.canvas.draw_idle()

    def save_png(self, path: str | Path, *, dpi: int = 150) -> Path:
        """Save the currently selected view."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        return destination

    def save_gif(
        self,
        path: str | Path,
        *,
        frames: int = 72,
        fps: int = 12,
        dpi: int = 110,
    ) -> Path:
        """Save a rotating GIF of the current trajectory and effectome panels."""
        return save_rotation_gif(
            self.figure,
            self.trajectory_ax,
            Path(path),
            frames=frames,
            fps=fps,
            dpi=dpi,
        )


def save_rotation_gif(
    fig: Figure,
    ax: Axes3D,
    path: Path,
    *,
    frames: int = 72,
    fps: int = 12,
    dpi: int = 110,
) -> Path:
    """Save a full azimuthal rotation while preserving the current elevation."""
    if frames < 2:
        raise ValueError("frames must be at least 2")
    if fps < 1:
        raise ValueError("fps must be at least 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    initial_elevation = float(ax.elev)
    initial_azimuth = float(ax.azim)

    def rotate(frame: int) -> tuple[()]:
        ax.view_init(elev=initial_elevation, azim=initial_azimuth + 360.0 * frame / frames)
        return ()

    animation = FuncAnimation(fig, rotate, frames=frames, interval=1000 / fps, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    ax.view_init(elev=initial_elevation, azim=initial_azimuth)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Run directory containing shared/ and method/stages/, ending at the run ID.",
    )
    parser.add_argument("--method", default="cgc", help="Method directory, for example cgc or cgc_star.")
    parser.add_argument(
        "--color-by",
        nargs="+",
        choices=("motif", "graph", "hmm"),
        default=("motif",),
        help="Create one view for each requested label source.",
    )
    parser.add_argument("--behavior-key", default="motif")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis-output/manifold-3d"))
    parser.add_argument(
        "--arrow-stride",
        type=int,
        default=1,
        help="Draw every Nth valid arrow (default: every step).",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument(
        "--segment-count",
        type=int,
        default=None,
        help="Number of consecutive transitions to display (default: all remaining).",
    )
    parser.add_argument("--curve-samples", type=int, default=16)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF generation.")
    parser.add_argument("--show", action="store_true", help="Open draggable native Matplotlib windows.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[Figure] = []
    for color_by in args.color_by:
        data = load_trajectory_data(
            args.run_root,
            method=args.method,
            color_by=color_by,
            behavior_key=args.behavior_key,
        )
        fig, ax = plot_trajectory(
            data,
            arrow_stride=args.arrow_stride,
            start_step=args.start_step,
            segment_count=args.segment_count,
            curve_samples=args.curve_samples,
        )
        stem = f"manifold-3d-{args.method}-{color_by}"
        png_path = args.output_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {png_path}")
        if not args.no_gif:
            gif_path = save_rotation_gif(
                fig,
                ax,
                args.output_dir / f"{stem}.gif",
                frames=args.frames,
                fps=args.fps,
                dpi=args.dpi,
            )
            print(f"Saved {gif_path}")
        figures.append(fig)

    if args.show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)


if __name__ == "__main__":
    main()
