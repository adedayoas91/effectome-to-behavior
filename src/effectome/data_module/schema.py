"""Typed data contracts shared across pipeline stages.

These dataclasses are the stable interfaces between stages. Keep them backward-compatible:
downstream modules (connectivity, dynamics, community, manifold, linking) depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RecordingIdentity:
    """Stable dataset/recording identifiers carried through all artifacts."""

    dataset: str = "unknown"
    dataset_id: str = "unknown"
    recording_id: str = "recording-0"
    animal_id: str | None = None
    session_id: str | None = None
    segment_id: str | None = None


@dataclass
class NeuralRecording:
    """A whole-brain calcium-imaging recording with aligned behavior.

    Attributes:
        traces: Neural activity, shape (N_neurons, T_timepoints), float32 (e.g. dF/F).
        time: Timestamps in seconds, shape (T,).
        coords: Optional anatomical coordinates, shape (N, 3) or None.
        neuron_ids: Integer neuron identifiers, shape (N,).
        behavior: Mapping from behavior-variable name to a length-T array.
        fps: Sampling rate in Hz.
        metadata: Free-form provenance (source, ground-truth graphs for synthetic data, etc.).
        identity: Stable dataset/recording identifiers used to prevent accidental pooling.
    """

    traces: np.ndarray
    time: np.ndarray
    coords: np.ndarray | None
    neuron_ids: np.ndarray
    behavior: dict[str, np.ndarray]
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)
    identity: RecordingIdentity = field(default_factory=RecordingIdentity)

    @property
    def n_neurons(self) -> int:
        return int(self.traces.shape[0])

    @property
    def n_timepoints(self) -> int:
        return int(self.traces.shape[1])

    def validate(self) -> None:
        """Check internal shape consistency; raise ValueError on mismatch."""
        if self.traces.ndim != 2:
            raise ValueError(f"traces must be 2D (N, T), got shape {self.traces.shape}")
        n, t = self.traces.shape
        if self.time.ndim != 1:
            raise ValueError(f"time must be 1D, got shape {self.time.shape}")
        if self.time.shape[0] != t:
            raise ValueError(f"time length {self.time.shape[0]} != T {t}")
        if not np.all(np.isfinite(self.time)):
            raise ValueError("time contains non-finite values")
        if t > 1 and not np.all(np.diff(self.time) > 0):
            raise ValueError("time must be strictly increasing")
        if self.neuron_ids.shape[0] != n:
            raise ValueError(f"neuron_ids length {self.neuron_ids.shape[0]} != N {n}")
        if self.coords is not None and self.coords.shape[0] != n:
            raise ValueError(f"coords rows {self.coords.shape[0]} != N {n}")
        for name, arr in self.behavior.items():
            if arr.shape[0] != t:
                raise ValueError(f"behavior['{name}'] length {arr.shape[0]} != T {t}")


@dataclass(frozen=True)
class TemporalAnchor:
    """Typed zero-based temporal contract linking context/history to a target interval."""

    dataset_id: str
    recording_id: str
    animal_id: str | None
    session_id: str | None
    segment_id: str | None
    context_start: int
    context_stop: int
    target_start: int
    target_stop: int
    anchor_sample: int
    anchor_time_seconds: float
    sampling_rate_hz: float
    valid: bool = True
    invalid_reason: str | None = None
    gap_before: bool = False
    gap_after: bool = False
    original_source_indices: tuple[int, int] | None = None

    @property
    def history_length(self) -> int:
        return self.context_stop - self.context_start

    @property
    def target_length(self) -> int:
        return self.target_stop - self.target_start


@dataclass(frozen=True)
class Window:
    """A single time window (segment) of the recording."""

    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class WindowedSegments:
    """Sliding/behavior-aligned segments produced by the windowing stage.

    Attributes:
        segments: Array of shape (K_windows, L_length, N_neurons) — the per-window
            multivariate time series passed to connectivity estimators.
        windows: The K `Window` objects (start/stop indices into the recording).
        behavior_per_window: Mapping behavior name -> array (K,) summarizing each window.
        n_neurons: Number of neurons (N).
        fps: Sampling rate (Hz).
    metadata: Optional segmentation metadata. Overlapping calcium windows use
            this for core windows, center samples, sample weights, and overlap.
        anchors: Optional typed time contracts aligned 1:1 with `windows`.
    """

    segments: np.ndarray
    windows: list[Window]
    behavior_per_window: dict[str, np.ndarray]
    n_neurons: int
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)
    anchors: list[TemporalAnchor] = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return int(self.segments.shape[0])


@dataclass
class ConnectivitySeries:
    """A temporal sequence of inferred connectivity matrices (the dynamic effectome).

    Attributes:
        matrices: Stack of shape (K_windows, N, N). Entry [k, i, j] is the inferred
            influence of neuron i on neuron j in window k (convention: source -> target).
        window_starts: Start index of each window, shape (K,).
        method: Name of the connectivity estimator that produced these.
        directed: Whether matrices are directed (asymmetric) or symmetric.
        behavior_per_window: Behavior summary per window, carried through for linking.
        anchors: Temporal anchors aligned 1:1 with matrices when available.
        diagnostics: Estimator metadata, semantics, and run-time summary.
    """

    matrices: np.ndarray
    window_starts: np.ndarray
    method: str
    directed: bool
    behavior_per_window: dict[str, np.ndarray] = field(default_factory=dict)
    anchors: list[TemporalAnchor] = field(default_factory=list)
    signed: bool = True
    weighted: bool = True
    storage: str = "dense"
    weight_semantics: str = "effective_influence"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def n_windows(self) -> int:
        return int(self.matrices.shape[0])

    @property
    def n_neurons(self) -> int:
        return int(self.matrices.shape[1])


@dataclass
class CommunitySeries:
    """Per-window community partitions of the dynamic effectome.

    Attributes:
        labels: Array (K_windows, N) of integer community labels per neuron per window.
        method: Community-detection method name.
        n_communities_per_window: Array (K,) count of communities found in each window.
    """

    labels: np.ndarray
    method: str
    n_communities_per_window: np.ndarray
