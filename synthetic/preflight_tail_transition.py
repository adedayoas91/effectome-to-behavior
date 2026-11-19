"""Inventory prospective V2a tail-motion forecast rows without fitting any model."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from effectome.data_module.loaders import load_v2a_rsns
from effectome.data_module.schema import NeuralRecording
from effectome.data_module.windowing import _crosses_interval, _round_samples_half_up, _valid_ranges
from effectome.workflows.cross_method import load_cross_method_profile

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_HISTORIES = (30.0, 60.0, 120.0, 180.0)


@dataclass(frozen=True)
class TailPreflightConfig:
    histories_seconds: tuple[float, ...] = _DEFAULT_HISTORIES
    behavior_history_seconds: float = 2.0
    future_seconds: float = 2.0
    stride_seconds: float = 2.0
    test_fraction: float = 0.25
    validation_fraction: float = 0.25
    nominal_tail_offset_seconds: float = 0.0

    def __post_init__(self) -> None:
        durations = (
            *self.histories_seconds,
            self.behavior_history_seconds,
            self.future_seconds,
            self.stride_seconds,
        )
        if not self.histories_seconds or len(set(self.histories_seconds)) != len(self.histories_seconds):
            raise ValueError("history durations must be nonempty and unique")
        if any(not np.isfinite(x) or x <= 0 for x in durations):
            raise ValueError("durations must be finite and positive")
        if any(x < self.behavior_history_seconds for x in self.histories_seconds):
            raise ValueError("neural histories must cover the declared behavior lookback")
        if any(not 0 < x < 1 for x in (self.test_fraction, self.validation_fraction)):
            raise ValueError("split fractions must lie strictly between zero and one")
        if not np.isfinite(self.nominal_tail_offset_seconds):
            raise ValueError("the nominal tail offset must be finite")


@dataclass(frozen=True)
class TailOrigin:
    origin: int
    time_seconds: float
    calcium_history: tuple[int, int]
    tail_history: tuple[int, int]
    tail_target: tuple[int, int]
    invalid_calcium_samples: int


def _raw_tail_intervals(
    times: np.ndarray, origin: float, lookback: float, horizon: float
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Select complete raw support on both sides with no interpolated origin sample."""
    left, right = origin - lookback, origin + horizon
    if left < times[0] or right > times[-1]:
        return None
    start = int(np.searchsorted(times, left, side="left"))
    split = int(np.searchsorted(times, origin, side="right"))
    stop = int(np.searchsorted(times, right, side="right"))
    if min(split - start, stop - split) < 2:
        return None
    return (start, split), (split, stop)


def _invalid_count(cumulative: np.ndarray, bounds: tuple[int, int]) -> int:
    return int(cumulative[bounds[1]] - cumulative[bounds[0]])


def _eligible_origins(
    recording: NeuralRecording,
    tail: np.ndarray,
    tail_time: np.ndarray,
    config: TailPreflightConfig,
    history_samples: int,
) -> tuple[list[TailOrigin], dict[str, int]]:
    stride = max(1, _round_samples_half_up(config.stride_seconds * recording.fps))
    invalid_neural = np.r_[0, np.cumsum(~np.isfinite(recording.traces).all(axis=0))]
    invalid_tail = np.r_[0, np.cumsum(~np.isfinite(tail))]
    ranges = _valid_ranges(recording)
    gaps = recording.metadata.get("gap_intervals", [])
    events = []
    counts = {
        "candidate_origins": 0,
        "missing_or_nonfinite_tail": 0,
        "outside_recording_or_crosses_hard_gap": 0,
        "invalid_origin_calcium": 0,
        "finite_only_rejects": 0,
    }
    # Keep the same stride phase at all history lengths.
    for index in range(0, recording.n_timepoints, stride):
        if index + 1 < history_samples:
            continue
        counts["candidate_origins"] += 1
        t0 = float(recording.time[index])
        start = index + 1 - history_samples
        horizon = t0 + config.future_seconds
        raw = _raw_tail_intervals(tail_time, t0, config.behavior_history_seconds, config.future_seconds)
        if raw is None or any(_invalid_count(invalid_tail, span) for span in raw):
            counts["missing_or_nonfinite_tail"] += 1
            continue
        prior, outcome = raw
        # The two modalities share the nominal clock, but have distinct raw indices.
        # The raw behavior lookback must also sit in the current valid segment.
        first = min(float(recording.time[start]), float(tail_time[prior[0]]))
        first_sample = int(np.searchsorted(recording.time, first, side="right")) - 1
        future_stop = int(np.searchsorted(recording.time, horizon, side="right"))
        if not any(
            recording.time[a] <= first and horizon <= recording.time[b - 1] for a, b in ranges
        ) or _crosses_interval(gaps, first_sample, future_stop):
            counts["outside_recording_or_crosses_hard_gap"] += 1
            continue
        if _invalid_count(invalid_neural, (index, index + 1)):
            counts["invalid_origin_calcium"] += 1
            continue
        bad = _invalid_count(invalid_neural, (start, index + 1))
        counts["finite_only_rejects"] += bool(bad)
        events.append(TailOrigin(index, t0, (start, index + 1), prior, outcome, bad))
    return events, counts


def _ends_before(earlier: TailOrigin, later: TailOrigin) -> bool:
    return (
        earlier.calcium_history[1] <= later.calcium_history[0]
        and earlier.tail_target[1] <= later.tail_history[0]
    )


def _chronological_split(
    events: list[TailOrigin], boundary: float
) -> tuple[list[TailOrigin], list[TailOrigin], int]:
    earlier = [event for event in events if event.time_seconds < boundary]
    later = [event for event in events if event.time_seconds >= boundary]
    # Fixed duration and ordered time make the first test origin's start the
    # earliest raw dependency in each modality.
    kept = [event for event in earlier if later and _ends_before(event, later[0])]
    return kept, later, len(earlier)


def _disjoint_count(events: list[TailOrigin]) -> int:
    selected: list[TailOrigin] = []
    for event in events:
        if not selected or _ends_before(selected[-1], event):
            selected.append(event)
    return len(selected)


def _split_stats(events: list[TailOrigin], end_time: float, config: TailPreflightConfig) -> dict:
    boundary = end_time * (1 - config.test_fraction)
    train, test, prior = _chronological_split(events, boundary)
    inner_boundary = train[-1].time_seconds * (1 - config.validation_fraction) if train else None
    inner_train, validation, inner_prior = _chronological_split(
        train, inner_boundary if inner_boundary is not None else 0.0
    )
    return {
        "events": len(events),
        "outer_boundary_seconds": boundary,
        "train_candidates_before_purge": prior,
        "train_origins": [event.origin for event in train],
        "test_origins": [event.origin for event in test],
        "train_dropped_for_test_overlap": prior - len(train) if test else None,
        "disjoint_test_origin_count": _disjoint_count(test),
        "inner_boundary_seconds": inner_boundary,
        "inner_train_origins": [event.origin for event in inner_train],
        "inner_validation_origins": [event.origin for event in validation],
        "inner_train_dropped_for_validation_overlap": inner_prior - len(inner_train) if validation else None,
        "disjoint_inner_validation_origin_count": _disjoint_count(validation),
        "outer_nonempty": bool(train and test),
        "outer_and_inner_nonempty": bool(train and test and inner_train and validation),
    }


def _source_files(folder: Path) -> dict[str, str]:
    names = {
        "analysis_info": "*_analysis_info.json",
        "traces": "*_cells_fluorescence_signals.npy",
        "coords": "*_registered_cells_pos.npy",
        "tail_angle": "*_tail_angle.npy",
        "emitter_cells": "*_emitter_cells.npy",
        "receiver_cells": "*_receiver_cells.npy",
    }
    result = {}
    for key, pattern in names.items():
        matches = sorted(folder.glob(pattern))
        if len(matches) != 1:
            raise ValueError(f"{folder}: expected one {pattern}, found {len(matches)}")
        result[key] = matches[0].name
    return result


def _inspect_recording(folder: Path, config: TailPreflightConfig, fast_threshold: float) -> dict:
    files = _source_files(folder)
    metadata = json.loads((folder / files["analysis_info"]).read_text())
    recording = load_v2a_rsns(
        {
            "path": str(folder),
            "files": files,
            "dataset_id": "v2a-rsns",
            "recording_id": folder.name,
            "animal_id": metadata["fishID"],
            "signal_type": "fluorescence",
            "cell_subset": "all",
            "bad_frame_indexing": "zero_based",
            "bad_frame_policy": "mask",
            "max_masked_gap_seconds": 1.0,
        }
    )
    tail = np.load(folder / files["tail_angle"], allow_pickle=False).reshape(-1)
    tail_fps = float(metadata["frameRateBeh"])
    if tail.size < 2 or not np.isfinite(tail_fps) or tail_fps <= 0:
        raise ValueError(f"{folder}: tail sample count and frame rate must be usable")
    tail_time = np.arange(tail.size) / tail_fps + config.nominal_tail_offset_seconds
    neural_end = float(recording.time[-1])
    evaluations = []
    for seconds in config.histories_seconds:
        samples = max(1, _round_samples_half_up(seconds * recording.fps))
        events, counts = _eligible_origins(recording, tail, tail_time, config, samples)
        finite = [event for event in events if event.invalid_calcium_samples == 0] if samples >= 3 else []
        floor = 1.0 / (samples - 1) if samples >= 3 else None
        evaluations.append(
            {
                "requested_history_seconds": seconds,
                "history_samples": samples,
                "elapsed_history_seconds": (samples - 1) / recording.fps,
                "fast_native_pvalue_floor": floor,
                "fast_native_threshold_attainable": floor is not None and floor <= fast_threshold,
                **counts,
                "finite_native_input": _split_stats(finite, neural_end, config),
                "geometry_only_if_short_masks_supported": _split_stats(events, neural_end, config),
                "origin_dependencies": [asdict(event) for event in events],
            }
        )
    return {
        "animal_id": metadata["fishID"],
        "recording_id": folder.name,
        "export_run_id": metadata["runID"],
        "source_files": files,
        "neural_shape": list(recording.traces.shape),
        "calcium_samples_per_second": recording.fps,
        "stride_calcium_samples": max(1, _round_samples_half_up(config.stride_seconds * recording.fps)),
        "tail_samples": len(tail),
        "tail_samples_per_second": tail_fps,
        "calcium_last_sample_seconds": neural_end,
        "tail_last_sample_nominal_seconds": float(tail_time[-1]),
        "tail_minus_calcium_last_sample_seconds": float(tail_time[-1] - neural_end),
        "finite_tail_samples": int(np.isfinite(tail).sum()),
        "bad_frame_indexing_assumption": "zero_based",
        "bad_frames": recording.metadata["bad_frames"],
        "hard_gap_intervals": recording.metadata["gap_intervals"],
        "histories": evaluations,
    }


def preflight_tail_transitions(data_root: Path, config: TailPreflightConfig) -> dict:
    folders = sorted(path.parent for path in data_root.glob("*/*_analysis_info.json"))
    if not folders or len(set(folders)) != len(folders):
        raise ValueError(f"{data_root}: expected export subdirectories with unique analysis metadata")
    _, methods = load_cross_method_profile("csl_sanity")
    fast = methods["fast_cgc"]
    fast_threshold = min(fast.alpha, float(fast.extra.get("beta", 0.005)))
    recordings = [_inspect_recording(folder, config, fast_threshold) for folder in folders]
    return {
        "scope": "read_only_nominal_clock_geometry_no_model_fit",
        "config": asdict(config),
        "dataset": str(data_root),
        "fish_count": len({entry["animal_id"] for entry in recordings}),
        "recording_count": len(recordings),
        "experimental_unit": "fish",
        "neuron_panel": "all observed fluorescence traces; no behavior-selected panel",
        "nominal_time_alignment": {
            "verified": False,
            "construction": "sample indices divided by reported frame rates with a nominal tail offset",
            "needs_confirmation": "shared start trigger, relative offset/drift, and frame indexing",
            "resampling": "none; raw tail-clock intervals with no boundary interpolation or extrapolation",
        },
        "target": {
            "interval": "(origin, origin + future_seconds]",
            "definition": "RMS angular speed from consecutive raw tail-angle samples inside the interval",
            "unit": "exported angle units per second",
            "uses_pre_origin_angle_sample": False,
            "scored_in_this_preflight": False,
        },
        "baseline_tail_history": "[origin - behavior_history_seconds, origin]",
        "raw_dependency_index_convention": "zero-based half-open intervals per modality",
        "fast_native_test_threshold": fast_threshold,
        "feasibility_limits": [
            "Finite native input counts enforce complete unmasked calcium history for upstream GC.",
            "The short-mask geometry is an upper bound and requires a mask-aware graph estimator.",
            "An attainable shift p-value and nonempty split do not establish adequate information or power.",
            "Overlapping origins are dependent, and four fish establish pilot coverage only.",
        ],
        "recordings": recordings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_ROOT / "data/v2a-RSNs")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "outputs/behavior-transition/empirical-preflight/tail-transition-preflight.json",
    )
    parser.add_argument("--histories", nargs="+", type=float, default=_DEFAULT_HISTORIES, metavar="SECONDS")
    parser.add_argument("--lookback", type=float, default=2.0, metavar="SECONDS")
    parser.add_argument("--future", type=float, default=2.0, metavar="SECONDS")
    parser.add_argument("--stride", type=float, default=2.0, metavar="SECONDS")
    parser.add_argument("--nominal-tail-offset", type=float, default=0.0, metavar="SECONDS")
    args = parser.parse_args()
    config = TailPreflightConfig(
        histories_seconds=tuple(args.histories),
        behavior_history_seconds=args.lookback,
        future_seconds=args.future,
        stride_seconds=args.stride,
        nominal_tail_offset_seconds=args.nominal_tail_offset,
    )
    report = preflight_tail_transitions(args.data_root, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
