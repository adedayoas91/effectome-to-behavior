"""Verified input and chronological timing for the BunDLe-Net worm pilot."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from effectome.data_module import EndpointMetadata, NeuralRecording, get_loader
from effectome.data_module.windowing import _round_samples_half_up
from effectome.workflows.resume import _file_digest


def load_worm(repo: Path, worm_index: int, *, data_root: Path | None = None) -> NeuralRecording:
    """Read one worm from the pinned export, preserving its source ROI IDs."""
    if isinstance(worm_index, bool) or not isinstance(worm_index, Integral) or worm_index < 0:
        raise ValueError("worm_index must be a nonnegative integer")
    config = OmegaConf.to_container(OmegaConf.load(repo / "conf/data/c_elegans.yaml"), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("C. elegans data configuration must be a mapping")
    base = (repo / str(config["path"]) if data_root is None else data_root.expanduser()).resolve()
    files, hashes = config["files"], config["downloaded_sha256"]
    if set(files.values()) != set(hashes):
        raise ValueError("every C. elegans source file must have a pinned checksum")
    for name, expected in hashes.items():
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(f"C. elegans source file missing: {path}")
        if _file_digest(path) != expected:
            raise ValueError(f"C. elegans source file SHA-256 mismatch: {path}")

    config.update(
        path=str(base),
        worm_index=int(worm_index),
        recording_id=f"bundle-net-worm-{worm_index}",
        animal_id=f"worm-{worm_index}",
        trace_field="deltaFOverF",
        signal_type="dff",
    )
    recording = get_loader(str(config["name"]))(config)
    recording.metadata["verified_source_sha256"] = dict(hashes)
    motifs_file = files["recordings"]
    recording.endpoint_metadata["motif"] = EndpointMetadata(
        endpoint_kind="fictive",
        label_vocabulary=tuple(recording.metadata["state_names"]),
        source_files=(str(base / motifs_file),),
        source_hashes=(hashes[motifs_file],),
        annotation_method="supplied_NoStim_Data_States",
        availability_rule="offline_annotation_for_fit_and_evaluation",
        verification_status="source_file_checksum_verified",
    )
    return recording


@dataclass(frozen=True)
class WormPlan:
    """Frame counts and eligible origins for one native worm recording."""

    fps: float
    frames: int
    readout_stop: int
    training_stop: int
    history: int
    discovery_history: int
    discovery_stride: int
    horizon: int
    target_length: int
    stride: int
    discovery_starts: np.ndarray
    origins: np.ndarray
    motif_counts: dict[str, dict[str, int]]

    def summary(self) -> dict[str, Any]:
        timing = {
            "history": self.history,
            "discovery_history": self.discovery_history,
            "discovery_stride": self.discovery_stride,
            "horizon": self.horizon,
            "target_length": self.target_length,
            "stride": self.stride,
        }
        return {
            "fps": self.fps,
            "frames": self.frames,
            "readout_stop": self.readout_stop,
            "training_stop": self.training_stop,
            **timing,
            "timing_seconds": {key: value / self.fps for key, value in timing.items()},
            "discovery_windows": len(self.discovery_starts),
            "forecast_origins": len(self.origins),
            "first_origin": int(self.origins[0]),
            "last_origin": int(self.origins[-1]),
            "motif_counts": self.motif_counts,
            "motifs_missing_from_readout": [
                name for name, count in self.motif_counts["readout"].items() if not count
            ],
        }


def _positive_seconds_in_frames(value: float, fps: float, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} duration must be positive and finite")
    frames = float(value) * fps
    if value <= 0 or not np.isfinite(frames):
        raise ValueError(f"{name} duration must be positive and finite")
    return max(1, _round_samples_half_up(frames))


def plan_worm(
    recording: NeuralRecording,
    *,
    readout_fraction: float,
    training_end_fraction: float,
    history_seconds: float,
    discovery_seconds: float,
    discovery_stride_seconds: float,
    horizon_seconds: float,
    target_seconds: float,
    stride_seconds: float,
) -> WormPlan:
    """Allocate disjoint readout, mechanism training and forecast blocks."""
    recording.validate()
    frames = recording.n_timepoints
    fps = float(recording.fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("worm sampling rate must be positive and finite")
    if recording.n_neurons < 2 or frames < 2 or not np.isfinite(recording.traces).all():
        raise ValueError("worm rollout requires finite neural observations")
    if (
        recording.neural_validity is not None and not np.all(recording.neural_validity)
        or "motif" in recording.outcome_validity and not np.all(recording.outcome_validity["motif"])
    ):
        raise ValueError("worm rollout requires valid neural and motif samples throughout")
    valid_ranges = recording.metadata.get("valid_ranges")
    if (
        recording.metadata.get("gap_intervals")
        or valid_ranges and [tuple(interval) for interval in valid_ranges] != [(0, frames)]
        or recording.source_indices is not None and np.any(np.diff(recording.source_indices) != 1)
        or not np.allclose(np.diff(recording.time), 1 / fps, rtol=0.05, atol=0)
    ):
        raise ValueError("worm rollout requires a contiguous clock matching the declared sampling rate")
    if (
        any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in (readout_fraction, training_end_fraction)
        )
        or not 0 < readout_fraction < training_end_fraction < 1
    ):
        raise ValueError("split fractions must satisfy 0 < readout < training end < 1")

    readout_stop = int(readout_fraction * frames)
    training_stop = int(training_end_fraction * frames)
    timings = {
        name: _positive_seconds_in_frames(seconds, fps, name)
        for name, seconds in (
            ("history", history_seconds),
            ("discovery_history", discovery_seconds),
            ("discovery_stride", discovery_stride_seconds),
            ("horizon", horizon_seconds),
            ("target_length", target_seconds),
            ("stride", stride_seconds),
        )
    }
    history = timings["history"]
    discovery_history = timings["discovery_history"]
    horizon = timings["horizon"]
    if history < 3 or discovery_history < 3 or timings["target_length"] > horizon:
        raise ValueError("worm rollout needs at least three history frames and a target within the horizon")
    discovery_starts = np.arange(
        readout_stop, training_stop - discovery_history + 1, timings["discovery_stride"], dtype=np.int64
    )
    origins = np.arange(training_stop + history - 1, frames - horizon, timings["stride"], dtype=np.int64)
    if readout_stop < 2 or not len(discovery_starts) or not len(origins):
        raise ValueError("worm recording is too short for the requested split and timing")

    labels = np.asarray(recording.behavior.get("motif"))
    names = tuple(recording.metadata.get("state_names", ()))
    if (
        labels.shape != (frames,)
        or labels.dtype.kind not in "iu"
        or not names
        or len(set(names)) != len(names)
        or np.any(labels < 0)
        or np.any(labels >= len(names))
    ):
        raise ValueError("worm motif values must match the declared vocabulary at every frame")
    motif_counts = {}
    for block, indices in (
        ("readout", slice(0, readout_stop)),
        ("mechanism_training", slice(readout_stop, training_stop)),
        ("forecast", slice(training_stop, frames)),
    ):
        counts = np.bincount(labels[indices].astype(np.intp), minlength=len(names))
        motif_counts[block] = dict(zip(names, map(int, counts), strict=True))
    if np.count_nonzero(list(motif_counts["readout"].values())) < 2:
        raise ValueError("worm readout block must contain at least two motifs")

    discovery_starts.setflags(write=False)
    origins.setflags(write=False)
    return WormPlan(
        fps=fps,
        frames=frames,
        readout_stop=readout_stop,
        training_stop=training_stop,
        **timings,
        discovery_starts=discovery_starts,
        origins=origins,
        motif_counts=motif_counts,
    )
