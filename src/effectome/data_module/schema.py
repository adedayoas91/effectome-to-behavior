"""Typed data contracts shared across pipeline stages.

These dataclasses are the stable interfaces between stages. Keep them backward-compatible:
downstream modules (connectivity, dynamics, community, manifold, linking) depend on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _validate_anchor_alignment(
    anchors: list[TemporalAnchor], expected: int, *, window_starts: np.ndarray | None = None
) -> None:
    if anchors and len(anchors) != expected:
        raise ValueError(f"anchors length {len(anchors)} != expected length {expected}")
    if anchors and window_starts is not None and len(window_starts) == expected:
        for idx, anchor in enumerate(anchors):
            if int(window_starts[idx]) != anchor.context_start:
                raise ValueError(
                    "window_starts must align with anchor.context_start for every window"
                )


def _validate_behavior_alignment(behavior_per_window: dict[str, np.ndarray], expected: int) -> None:
    for name, arr in behavior_per_window.items():
        if arr.shape[0] != expected:
            raise ValueError(f"behavior_per_window['{name}'] length {arr.shape[0]} != {expected}")


@dataclass(frozen=True)
class RecordingIdentity:
    """Stable dataset/recording identifiers carried through all artifacts."""

    dataset: str = "unknown"
    dataset_id: str = "unknown"
    recording_id: str = "recording-0"
    animal_id: str | None = None
    session_id: str | None = None
    segment_id: str | None = None


@dataclass(frozen=True)
class ArtifactProvenance:
    """Typed provenance carried by stage artifacts without replacing free-form metadata."""

    identity: RecordingIdentity = field(default_factory=RecordingIdentity)
    stage: str = "unknown"
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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

    def __post_init__(self) -> None:
        if self.context_start < 0 or self.target_start < 0:
            raise ValueError("temporal anchor indices must be non-negative")
        if self.context_start >= self.context_stop:
            raise ValueError("context_start must be < context_stop")
        if self.target_start >= self.target_stop:
            raise ValueError("target_start must be < target_stop")
        if self.target_start < self.context_start or self.target_stop > self.context_stop:
            raise ValueError("target interval must lie inside the context interval")
        if self.anchor_sample != self.target_stop - 1:
            raise ValueError("anchor_sample must equal target_stop - 1")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.original_source_indices is not None:
            src_start, src_stop = self.original_source_indices
            if src_start < 0 or src_start >= src_stop:
                raise ValueError("original_source_indices must be an ordered half-open interval")

    @property
    def history_length(self) -> int:
        return self.context_stop - self.context_start

    @property
    def target_length(self) -> int:
        return self.target_stop - self.target_start

    @property
    def history_seconds(self) -> float:
        return self.history_length / self.sampling_rate_hz

    @property
    def target_seconds(self) -> float:
        return self.target_length / self.sampling_rate_hz


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
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)

    def __post_init__(self) -> None:
        if self.segments.ndim != 3:
            raise ValueError(f"segments must be 3D (K, L, N), got shape {self.segments.shape}")
        k, _, n = self.segments.shape
        if len(self.windows) != k:
            raise ValueError(f"windows length {len(self.windows)} != K {k}")
        if self.n_neurons != n:
            raise ValueError(f"n_neurons {self.n_neurons} != segment width {n}")
        _validate_behavior_alignment(self.behavior_per_window, k)
        _validate_anchor_alignment(self.anchors, k)

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
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)

    def __post_init__(self) -> None:
        if self.matrices.ndim != 3:
            raise ValueError(f"matrices must be 3D (K, N, N), got shape {self.matrices.shape}")
        k, n_in, n_out = self.matrices.shape
        if n_in != n_out:
            raise ValueError("connectivity matrices must be square")
        if self.window_starts.shape[0] != k:
            raise ValueError(f"window_starts length {self.window_starts.shape[0]} != K {k}")
        _validate_behavior_alignment(self.behavior_per_window, k)
        _validate_anchor_alignment(self.anchors, k, window_starts=self.window_starts)

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
    window_starts: np.ndarray | None = None
    anchors: list[TemporalAnchor] = field(default_factory=list)
    consensus_labels: np.ndarray | None = None
    signed: bool | None = None
    directed: bool | None = None
    resolution: float | None = None
    interlayer_coupling: float | None = None
    boundary_indices: np.ndarray | None = None
    mode: str | None = None
    flexibility: np.ndarray | None = None
    switching: np.ndarray | None = None
    switching_rate: float | None = None
    coassignment: np.ndarray | None = None
    stability: float | None = None
    objective_scores: np.ndarray | None = None
    n_runs: int | None = None
    provenance: ArtifactProvenance = field(default_factory=ArtifactProvenance)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.labels.ndim != 2:
            raise ValueError(f"labels must be 2D (K, N), got shape {self.labels.shape}")
        k, n = self.labels.shape
        if self.n_communities_per_window.shape[0] != k:
            raise ValueError(
                f"n_communities_per_window length {self.n_communities_per_window.shape[0]} != K {k}"
            )
        if self.window_starts is not None and self.window_starts.shape[0] != k:
            raise ValueError(f"window_starts length {self.window_starts.shape[0]} != K {k}")
        _validate_anchor_alignment(self.anchors, k, window_starts=self.window_starts)
        if self.consensus_labels is None:
            self.consensus_labels = np.array(self.labels, copy=True)
        elif self.consensus_labels.shape != self.labels.shape:
            raise ValueError(
                f"consensus_labels shape {self.consensus_labels.shape} != labels shape {self.labels.shape}"
            )
        if self.boundary_indices is not None:
            if self.boundary_indices.ndim != 1:
                raise ValueError("boundary_indices must be 1D when provided")
            if np.any(self.boundary_indices < 0) or np.any(self.boundary_indices > k):
                raise ValueError("boundary_indices must lie in [0, K]")
        if self.flexibility is not None and self.flexibility.shape != (n,):
            raise ValueError(f"flexibility shape {self.flexibility.shape} != ({n},)")
        if self.switching is not None and self.switching.shape != (max(k - 1, 0), n):
            raise ValueError(f"switching shape {self.switching.shape} is not ((K-1), N)")
        if self.coassignment is not None and self.coassignment.shape != (k, n, n):
            raise ValueError(f"coassignment shape {self.coassignment.shape} != ({k}, {n}, {n})")
        if self.switching_rate is not None and not 0.0 <= self.switching_rate <= 1.0:
            raise ValueError("switching_rate must lie in [0, 1]")
        if self.objective_scores is not None and self.objective_scores.ndim != 1:
            raise ValueError("objective_scores must be 1D")
        if self.n_runs is not None and self.n_runs <= 0:
            raise ValueError("n_runs must be positive when provided")
