"""Dataset loaders behind a registry.

Add a loader by decorating it with `@register_loader("name")`. The pipeline selects one via
`cfg.data.name`. The registry includes synthetic and generic loaders plus dataset-specific
adapters for the V2a-RSN zebrafish recordings and BunDLe-Net C. elegans recordings.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .schema import NeuralRecording, RecordingIdentity

logger = logging.getLogger(__name__)

LoaderFn = Callable[[dict], NeuralRecording]
LOADER_REGISTRY: dict[str, LoaderFn] = {}


def register_loader(name: str) -> Callable[[LoaderFn], LoaderFn]:
    """Register a dataset loader under `name`."""

    def deco(fn: LoaderFn) -> LoaderFn:
        if name in LOADER_REGISTRY:
            raise ValueError(f"loader '{name}' already registered")
        LOADER_REGISTRY[name] = fn
        return fn

    return deco


def get_loader(name: str) -> LoaderFn:
    """Resolve a loader by name; raise KeyError listing options if unknown."""
    if name not in LOADER_REGISTRY:
        raise KeyError(f"unknown loader '{name}'; available: {sorted(LOADER_REGISTRY)}")
    return LOADER_REGISTRY[name]


def _normalize_ranges(ranges: list[list[int]] | list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    if not ranges:
        return []
    out: list[tuple[int, int]] = []
    for start, stop in ranges:
        out.append((int(start), int(stop)))
    return out


def _identity_from_cfg(cfg: dict, *, dataset: str, path: Path | None = None) -> RecordingIdentity:
    default_recording_id = path.stem if path is not None else f"{dataset}-recording-0"
    recording_id = str(cfg.get("recording_id") or default_recording_id)
    return RecordingIdentity(
        dataset=dataset,
        dataset_id=str(cfg.get("dataset_id", dataset)),
        recording_id=recording_id,
        animal_id=cfg.get("animal_id"),
        session_id=cfg.get("session_id"),
        segment_id=cfg.get("segment_id"),
    )


def _metadata_from_cfg(cfg: dict, *, source: str, path: Path | None = None) -> dict:
    metadata = {
        "source": source,
        "signal_type": cfg.get("signal_type", "dff"),
        "valid_ranges": _normalize_ranges(cfg.get("valid_ranges")),
        "gap_intervals": _normalize_ranges(cfg.get("gap_intervals")),
    }
    if path is not None:
        metadata["path"] = str(path)
    return metadata


def _build_recording(
    *,
    traces: np.ndarray,
    time: np.ndarray,
    coords: np.ndarray | None,
    behavior: dict[str, np.ndarray],
    fps: float,
    metadata: dict,
    identity: RecordingIdentity,
    neuron_ids: np.ndarray | None = None,
) -> NeuralRecording:
    if neuron_ids is None:
        neuron_ids = np.arange(traces.shape[0])
    return NeuralRecording(
        traces=np.asarray(traces, dtype=np.float32),
        time=np.asarray(time, dtype=np.float64),
        coords=None if coords is None else np.asarray(coords),
        neuron_ids=np.asarray(neuron_ids),
        behavior={name: np.asarray(values) for name, values in behavior.items()},
        fps=float(fps),
        metadata=metadata,
        identity=identity,
    )


def _resolve_file(base: Path, files: dict, key: str, *, fallback: str | None = None) -> Path:
    relative = files.get(key)
    if relative is None and fallback is not None:
        relative = fallback
    if relative is None:
        raise KeyError(f"dataset config is missing files.{key}")
    path = (base / str(relative)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    return path


def _merge_index_ranges(indices: list[int] | np.ndarray) -> list[tuple[int, int]]:
    if len(indices) == 0:
        return []
    ordered = sorted({int(idx) for idx in indices})
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev + 1))
        start = prev = idx
    ranges.append((start, prev + 1))
    return ranges


def _merge_gap_intervals(*interval_groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points = sorted(
        (int(start), int(stop))
        for group in interval_groups
        for start, stop in group
        if int(stop) > int(start)
    )
    if not points:
        return []
    merged: list[list[int]] = [[points[0][0], points[0][1]]]
    for start, stop in points[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], stop)
        else:
            merged.append([start, stop])
    return [(start, stop) for start, stop in merged]


def _resolve_bad_frame_policy(
    cfg: dict,
    *,
    bad_frame_intervals: list[tuple[int, int]],
    fps: float,
) -> tuple[str, float | None, list[tuple[int, int]]]:
    """Resolve bad frames into hard boundaries without compressing recording time.

    ``mask`` preserves short invalid runs on the native clock so lagged estimators can
    discard only affected design rows. Runs longer than ``max_masked_gap_seconds``
    remain hard boundaries. ``boundaries`` retains the stricter legacy sensitivity in
    which every invalid run splits the recording.
    """

    policy = str(cfg.get("bad_frame_policy", "mask")).lower()
    if policy == "boundaries":
        return policy, None, list(bad_frame_intervals)
    if policy != "mask":
        raise ValueError("bad_frame_policy must be 'mask' or 'boundaries'")
    max_masked_gap_seconds = float(cfg.get("max_masked_gap_seconds", 1.0))
    if not np.isfinite(max_masked_gap_seconds) or max_masked_gap_seconds < 0:
        raise ValueError("max_masked_gap_seconds must be finite and non-negative")
    hard = [
        (start, stop)
        for start, stop in bad_frame_intervals
        if (stop - start) / float(fps) > max_masked_gap_seconds
    ]
    return policy, max_masked_gap_seconds, hard


def _interp_to_target_rate(
    values: np.ndarray,
    *,
    source_fps: float,
    target_length: int,
    target_fps: float,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    if source.size == target_length:
        return source.astype(np.float32, copy=False)
    source_time = np.arange(source.size, dtype=np.float64) / float(source_fps)
    target_time = np.arange(target_length, dtype=np.float64) / float(target_fps)
    return np.interp(target_time, source_time, source).astype(np.float32, copy=False)


def _select_v2a_cells(
    cfg: dict,
    *,
    n_cells: int,
    emitter_cells: np.ndarray,
    receiver_cells: np.ndarray,
) -> np.ndarray:
    subset = cfg.get("cell_subset", "emitter_receiver")
    if isinstance(subset, (list, tuple, np.ndarray)):
        selected = np.asarray(subset, dtype=np.int64)
    else:
        key = str(subset).lower()
        if key in {"all", "full"}:
            selected = np.arange(n_cells, dtype=np.int64)
        elif key in {"emitter", "emitters"}:
            selected = np.asarray(emitter_cells, dtype=np.int64)
        elif key in {"receiver", "receivers"}:
            selected = np.asarray(receiver_cells, dtype=np.int64)
        elif key in {"emitter_receiver", "emitters_receivers", "union", "combined"}:
            selected = np.unique(
                np.concatenate(
                    [
                        np.asarray(emitter_cells, dtype=np.int64),
                        np.asarray(receiver_cells, dtype=np.int64),
                    ]
                )
            )
        else:
            raise ValueError(f"unsupported v2a cell_subset {subset!r}")
    if selected.ndim != 1:
        raise ValueError("selected cell indices must be a 1D sequence")
    if selected.size == 0:
        raise ValueError("selected cell subset is empty")
    if selected.min() < 0 or selected.max() >= n_cells:
        raise ValueError(f"selected cell subset must lie within [0, {n_cells})")
    return np.unique(selected.astype(np.int64, copy=False))


def _matlab_reference_target(handle, node):
    import h5py

    target = node
    while isinstance(target, h5py.Reference):
        target = handle[target]
    return target


def _decode_matlab_string(handle, node) -> str:
    import h5py

    target = _matlab_reference_target(handle, node)
    arr = np.asarray(target).squeeze()
    if isinstance(arr, h5py.Reference):
        return _decode_matlab_string(handle, arr)
    if getattr(arr, "dtype", None) is object:
        flat = np.asarray(arr).reshape(-1)
        if flat.size == 1:
            return _decode_matlab_string(handle, flat[0])
        return "".join(_decode_matlab_string(handle, ref) for ref in flat)
    if np.asarray(arr).ndim == 0:
        scalar = np.asarray(arr).item()
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8")
        if isinstance(scalar, str):
            return scalar
        if isinstance(scalar, (int, np.integer)):
            return chr(int(scalar))
        if isinstance(scalar, (float, np.floating)) and float(scalar).is_integer():
            return str(int(scalar))
        return str(scalar)
    array = np.asarray(arr)
    if array.dtype.kind in {"u", "i"}:
        return "".join(chr(int(value)) for value in array.flat if int(value) != 0)
    if array.dtype.kind == "S":
        return b"".join(array.flat).decode("utf-8")
    return "".join(str(value) for value in array.flat)


def _load_matlab_string_vector(handle, node) -> list[str]:
    target = _matlab_reference_target(handle, node)
    refs = np.asarray(target)
    if refs.dtype != object:
        return [_decode_matlab_string(handle, target)]
    if refs.ndim == 1:
        return [_decode_matlab_string(handle, ref) for ref in refs]
    return [_decode_matlab_string(handle, refs[idx, 0]) for idx in range(refs.shape[0])]


def _load_order279(path: Path) -> list[str]:
    import h5py

    with h5py.File(path, "r") as handle:
        return _load_matlab_string_vector(handle, handle["Order279"])


def _load_class_ids(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle["ClassIDs_279"]).reshape(-1).astype(np.int64)


@register_loader("synthetic")
def load_synthetic(cfg: dict) -> NeuralRecording:
    """Build a synthetic recording with a known causal graph (for tests/quickstart)."""
    from effectome.utils.synthetic import SyntheticConfig, make_synthetic_recording

    syn_cfg = SyntheticConfig(**{k: v for k, v in cfg.items() if k in SyntheticConfig.__annotations__})
    rec = make_synthetic_recording(syn_cfg)
    rec.identity = _identity_from_cfg(cfg, dataset="synthetic")
    rec.metadata.update(_metadata_from_cfg(cfg, source="synthetic"))
    rec.validate()
    logger.info("Loaded synthetic recording: %d neurons x %d timepoints", rec.n_neurons, rec.n_timepoints)
    return rec


@register_loader("zebrafish")
def load_zebrafish(cfg: dict) -> NeuralRecording:
    """Load the larval-zebrafish FLFM dataset (PMC10545542).

    Expects an HDF5/NPZ export under `cfg['path']`. The exact key names are confirmed in
    Stage 0; the mapping below is the *contract* we normalize the raw export into. Update the
    key lookups once the on-disk schema is verified.
    """
    path = Path(cfg["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"zebrafish data not found at {path}. Place the downloaded export there or set "
            f"data.path. Until confirmed, use data=synthetic."
        )

    keys = cfg.get("keys", {})
    if path.suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as f:
            traces = np.asarray(f[keys.get("traces", "traces")], dtype=np.float32)
            fps = float(cfg.get("fps", 10.0))
            time = np.asarray(f[keys["time"]]) if "time" in keys else np.arange(traces.shape[1]) / fps
            coords = np.asarray(f[keys["coords"]]) if "coords" in keys else None
            behavior = {b: np.asarray(f[k]) for b, k in cfg.get("behavior_keys", {}).items()}
    else:
        npz = np.load(path, allow_pickle=True)
        traces = np.asarray(npz[keys.get("traces", "traces")], dtype=np.float32)
        fps = float(cfg.get("fps", 10.0))
        time = npz[keys["time"]] if "time" in keys else np.arange(traces.shape[1]) / fps
        coords = npz[keys["coords"]] if "coords" in keys else None
        behavior = {b: np.asarray(npz[k]) for b, k in cfg.get("behavior_keys", {}).items()}

    rec = _build_recording(
        traces=traces,
        time=np.asarray(time),
        coords=coords,
        behavior=behavior,
        fps=fps,
        metadata=_metadata_from_cfg(cfg, source="zebrafish", path=path),
        identity=_identity_from_cfg(cfg, dataset="zebrafish", path=path),
    )
    rec.validate()
    logger.info("Loaded zebrafish recording: %d neurons x %d timepoints", rec.n_neurons, rec.n_timepoints)
    return rec


@register_loader("v2a_rsns")
def load_v2a_rsns(cfg: dict) -> NeuralRecording:
    """Load a V2a-RSN zebrafish recording exported as directory-local NPY/JSON files."""
    base = Path(cfg["path"])
    if not base.is_dir():
        raise FileNotFoundError(f"v2a_rsns data directory not found at {base}")

    files = cfg.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("v2a_rsns loader requires a non-empty files mapping")

    info_path = _resolve_file(base, files, "analysis_info")
    traces_path = _resolve_file(base, files, "traces")
    tail_path = _resolve_file(base, files, "tail_angle")
    emitter_path = _resolve_file(base, files, "emitter_cells")
    receiver_path = _resolve_file(base, files, "receiver_cells")

    coords_source = str(cfg.get("coords_source", "registered")).lower()
    if coords_source == "registered":
        coords_path = _resolve_file(base, files, "coords")
        raw_fallback = files.get("coords", "").replace("registered_cells_pos", "cells_positions")
        raw_coords_path = (
            _resolve_file(base, files, "raw_coords", fallback=raw_fallback)
            if "raw_coords" in files or "registered_cells_pos" in str(files.get("coords", ""))
            else None
        )
    elif coords_source == "raw":
        raw_fallback = files.get("coords", "").replace("registered_cells_pos", "cells_positions")
        coords_path = _resolve_file(base, files, "raw_coords", fallback=raw_fallback)
        raw_coords_path = coords_path
    else:
        raise ValueError(f"unsupported coords_source {coords_source!r}")

    with info_path.open(encoding="utf-8") as handle:
        analysis_info = json.load(handle)

    traces = np.asarray(np.load(traces_path, allow_pickle=True), dtype=np.float32)
    coords = np.asarray(np.load(coords_path, allow_pickle=True), dtype=np.float32)
    tail_angle = np.asarray(np.load(tail_path, allow_pickle=True), dtype=np.float32).reshape(-1)
    emitter_cells = np.asarray(np.load(emitter_path, allow_pickle=True), dtype=np.int64).reshape(-1)
    receiver_cells = np.asarray(np.load(receiver_path, allow_pickle=True), dtype=np.int64).reshape(-1)

    n_cells, n_frames = traces.shape
    if coords.shape[0] != n_cells:
        raise ValueError(f"coords rows {coords.shape[0]} != traces neurons {n_cells}")
    if int(analysis_info.get("nCells", n_cells)) != n_cells:
        raise ValueError(f"analysis_info nCells {analysis_info.get('nCells')} != traces neurons {n_cells}")
    if int(analysis_info.get("nFramesSCAPE", n_frames)) != n_frames:
        raise ValueError(
            f"analysis_info nFramesSCAPE {analysis_info.get('nFramesSCAPE')} != traces frames {n_frames}"
        )

    fps = float(analysis_info["frameRateSCAPE"])
    behavior_fps = float(analysis_info["frameRateBeh"])
    tail_interp = _interp_to_target_rate(
        tail_angle,
        source_fps=behavior_fps,
        target_length=n_frames,
        target_fps=fps,
    )
    time = np.arange(n_frames, dtype=np.float64) / fps

    selected = _select_v2a_cells(
        cfg,
        n_cells=n_cells,
        emitter_cells=emitter_cells,
        receiver_cells=receiver_cells,
    )
    traces = traces[selected]
    coords = coords[selected]

    bad_frame_indexing = str(cfg.get("bad_frame_indexing", "zero_based")).lower()
    raw_bad_frames = [int(idx) for idx in analysis_info.get("bad_frames", [])]
    if bad_frame_indexing == "one_based":
        raw_bad_frames = [idx - 1 for idx in raw_bad_frames]
    elif bad_frame_indexing != "zero_based":
        raise ValueError("bad_frame_indexing must be 'zero_based' or 'one_based'")
    bad_frames = [idx for idx in raw_bad_frames if 0 <= idx < n_frames]
    bad_frame_intervals = _merge_index_ranges(bad_frames)
    bad_frame_policy, max_masked_gap_seconds, hard_bad_frame_intervals = (
        _resolve_bad_frame_policy(
            cfg,
            bad_frame_intervals=bad_frame_intervals,
            fps=fps,
        )
    )
    gap_intervals = _merge_gap_intervals(
        _normalize_ranges(cfg.get("gap_intervals")),
        hard_bad_frame_intervals,
    )
    if bad_frame_policy == "mask" and bad_frames:
        traces = traces.copy()
        traces[:, bad_frames] = np.nan

    behavior_sources = {"tail_angle": tail_interp}
    behavior = {
        str(name): np.asarray(behavior_sources[str(source)])
        for name, source in (cfg.get("behavior_keys") or {"continuous": "tail_angle"}).items()
    }

    metadata = _metadata_from_cfg(cfg, source="v2a_rsns", path=base)
    metadata.update(
        {
            "analysis_info": analysis_info,
            "behavior_fps": behavior_fps,
            "coords_source": coords_source,
            "selected_cell_subset": cfg.get("cell_subset", "emitter_receiver"),
            "selected_cell_indices": selected.tolist(),
            "emitter_cell_indices": emitter_cells.tolist(),
            "receiver_cell_indices": receiver_cells.tolist(),
            "bad_frames": bad_frames,
            "bad_frame_indexing": bad_frame_indexing,
            "bad_frame_policy": bad_frame_policy,
            "max_masked_gap_seconds": max_masked_gap_seconds,
            "bad_frame_intervals": bad_frame_intervals,
            "hard_bad_frame_intervals": hard_bad_frame_intervals,
            "gap_intervals": gap_intervals,
            "source_files": {name: str((base / str(relative)).resolve()) for name, relative in files.items()},
        }
    )
    if raw_coords_path is not None:
        metadata["raw_coords_path"] = str(raw_coords_path)

    rec = _build_recording(
        traces=traces,
        time=time,
        coords=coords,
        behavior=behavior,
        fps=fps,
        metadata=metadata,
        identity=_identity_from_cfg(cfg, dataset="v2a_rsns", path=base),
        neuron_ids=selected,
    )
    rec.validate()
    logger.info("Loaded V2a-RSN recording: %d neurons x %d timepoints", rec.n_neurons, rec.n_timepoints)
    return rec


@register_loader("bundle_net_c_elegans")
def load_bundle_net_c_elegans(cfg: dict) -> NeuralRecording:
    """Load one BunDLe-Net C. elegans worm from the MATLAB v7.3 HDF5 export."""
    import h5py

    base = Path(cfg["path"])
    if not base.is_dir():
        raise FileNotFoundError(f"bundle_net_c_elegans data directory not found at {base}")

    files = cfg.get("files", {})
    if not isinstance(files, dict) or not files:
        raise ValueError("bundle_net_c_elegans loader requires a non-empty files mapping")

    recordings_path = _resolve_file(base, files, "recordings")
    class_ids_path = _resolve_file(base, files, "class_ids")
    order_path = _resolve_file(base, files, "neuron_order")
    order279 = _load_order279(order_path)
    class_ids = _load_class_ids(class_ids_path)
    if class_ids.shape[0] != len(order279):
        raise ValueError("ClassIDs_279 and Order279 must have matching lengths")

    state_names = [str(name) for name in cfg.get("state_names", [])]

    with h5py.File(recordings_path, "r") as handle:
        root = handle["NoStim_Data"]
        worm_index = int(cfg.get("worm_index", 0))
        n_worms = int(root["deltaFOverF_bc"].shape[0])
        if worm_index < 0 or worm_index >= n_worms:
            raise IndexError(f"worm_index {worm_index} outside available range [0, {n_worms})")

        traces = np.asarray(handle[root["deltaFOverF_bc"][worm_index, 0]], dtype=np.float32)
        time = np.asarray(handle[root["tv"][worm_index, 0]], dtype=np.float64).reshape(-1)
        fps = float(np.asarray(handle[root["fps"][worm_index, 0]]).squeeze())
        raw_neuron_names = _load_matlab_string_vector(handle, root["NeuronNames"][worm_index, 0])
        dataset_name = _decode_matlab_string(handle, root["dataset"][worm_index, 0])

        states_group = _matlab_reference_target(handle, root["States"][worm_index, 0])
        available_state_names = sorted(states_group.keys())
        if not state_names:
            state_names = available_state_names
        missing_states = sorted(set(state_names) - set(available_state_names))
        if missing_states:
            raise KeyError(f"bundle_net_c_elegans missing state channels: {missing_states}")
        state_indicators = {
            name: np.asarray(states_group[name]).reshape(-1).astype(np.float32)
            for name in state_names
        }

    if time.shape[0] != traces.shape[1]:
        raise ValueError(f"time length {time.shape[0]} != traces frames {traces.shape[1]}")
    if any(values.shape[0] != traces.shape[1] for values in state_indicators.values()):
        raise ValueError("every BunDLe-Net state indicator must align with the trace length")

    state_matrix = np.vstack([state_indicators[name] for name in state_names])
    motif = np.argmax(state_matrix, axis=0).astype(np.int64)
    state_sums = state_matrix.sum(axis=0)
    if not np.allclose(state_sums, 1.0):
        raise ValueError("BunDLe-Net states are expected to be one-hot per timepoint")

    canonical_lookup = {name: idx for idx, name in enumerate(order279)}
    canonical_indices: list[int] = []
    canonical_names: list[str] = []
    canonical_class_ids: list[int] = []
    unresolved_names: list[str] = []
    for raw_name in raw_neuron_names:
        if raw_name.isdigit():
            idx = int(raw_name) - 1
            if 0 <= idx < len(order279):
                canonical_indices.append(idx)
                canonical_names.append(order279[idx])
                canonical_class_ids.append(int(class_ids[idx]))
                continue
        canonical_idx = canonical_lookup.get(raw_name)
        if canonical_idx is not None:
            canonical_indices.append(canonical_idx)
            canonical_names.append(order279[canonical_idx])
            canonical_class_ids.append(int(class_ids[canonical_idx]))
            continue
        unresolved_names.append(raw_name)
        canonical_indices.append(-1)
        canonical_names.append(raw_name)
        canonical_class_ids.append(-1)

    if unresolved_names:
        neuron_ids = np.arange(traces.shape[0], dtype=np.int64)
    else:
        neuron_ids = np.asarray(canonical_indices, dtype=np.int64)

    source_raw_neuron_names = list(raw_neuron_names)
    source_canonical_neuron_names = list(canonical_names)
    excluded_names = [str(name) for name in cfg.get("exclude_neuron_names", [])]
    exclude_name_source = str(cfg.get("exclude_neuron_name_source", "raw")).lower()
    if exclude_name_source == "raw":
        exclusion_names = raw_neuron_names
    elif exclude_name_source == "canonical":
        exclusion_names = canonical_names
    else:
        raise ValueError("exclude_neuron_name_source must be 'raw' or 'canonical'")
    excluded_name_set = set(excluded_names)
    keep = np.asarray([name not in excluded_name_set for name in exclusion_names], dtype=bool)
    excluded_present = [name for name in excluded_names if name in exclusion_names]
    excluded_missing = [name for name in excluded_names if name not in exclusion_names]
    traces = traces[keep]
    neuron_ids = neuron_ids[keep]
    raw_neuron_names = [name for name, retain in zip(raw_neuron_names, keep, strict=True) if retain]
    canonical_names = [name for name, retain in zip(canonical_names, keep, strict=True) if retain]
    canonical_indices = [idx for idx, retain in zip(canonical_indices, keep, strict=True) if retain]
    canonical_class_ids = [
        class_id for class_id, retain in zip(canonical_class_ids, keep, strict=True) if retain
    ]
    unresolved_names = [name for name in unresolved_names if name in raw_neuron_names]

    behavior_sources = {"motif": motif, **state_indicators}
    behavior = {
        str(name): np.asarray(behavior_sources[str(source)])
        for name, source in (cfg.get("behavior_keys") or {"motif": "motif"}).items()
    }

    metadata = _metadata_from_cfg(cfg, source="bundle_net_c_elegans", path=base)
    metadata.update(
        {
            "bundle_dataset_name": dataset_name,
            "worm_index": worm_index,
            "state_names": state_names,
            "state_name_to_id": {name: idx for idx, name in enumerate(state_names)},
            "state_indicators": state_indicators,
            "raw_neuron_names": raw_neuron_names,
            "canonical_neuron_names": canonical_names,
            "canonical_neuron_indices": canonical_indices,
            "canonical_class_ids": canonical_class_ids,
            "unresolved_neuron_names": unresolved_names,
            "source_neuron_count": len(source_raw_neuron_names),
            "source_raw_neuron_names": source_raw_neuron_names,
            "source_canonical_neuron_names": source_canonical_neuron_names,
            "exclude_neuron_name_source": exclude_name_source,
            "requested_excluded_neuron_names": excluded_names,
            "excluded_neuron_names": excluded_present,
            "requested_but_absent_excluded_neuron_names": excluded_missing,
            "source_url": cfg.get("source_url"),
            "source_commit": cfg.get("source_commit"),
            "source_files": {name: str((base / str(relative)).resolve()) for name, relative in files.items()},
        }
    )

    rec = _build_recording(
        traces=traces,
        time=time,
        coords=None,
        behavior=behavior,
        fps=fps,
        metadata=metadata,
        identity=_identity_from_cfg(cfg, dataset="bundle_net_c_elegans", path=base),
        neuron_ids=neuron_ids,
    )
    rec.validate()
    logger.info(
        "Loaded BunDLe-Net C. elegans recording: worm=%d, %d neurons x %d timepoints",
        worm_index,
        rec.n_neurons,
        rec.n_timepoints,
    )
    return rec


@register_loader("c_elegans")
def load_c_elegans(cfg: dict) -> NeuralRecording:
    """Load a C. elegans whole-brain calcium recording under the common contract."""
    path = Path(cfg["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"c_elegans data not found at {path}. Place the exported recording there or set data.path."
        )

    keys = cfg.get("keys", {})
    if path.suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as f:
            traces = np.asarray(f[keys.get("traces", "traces")], dtype=np.float32)
            fps = float(cfg.get("fps", 2.0))
            time = np.asarray(f[keys["time"]]) if "time" in keys else np.arange(traces.shape[1]) / fps
            coords = np.asarray(f[keys["coords"]]) if "coords" in keys else None
            neuron_ids = (
                np.asarray(f[keys["neuron_ids"]])
                if "neuron_ids" in keys
                else np.arange(traces.shape[0])
            )
            behavior = {b: np.asarray(f[k]) for b, k in cfg.get("behavior_keys", {}).items()}
    else:
        npz = np.load(path, allow_pickle=True)
        traces = np.asarray(npz[keys.get("traces", "traces")], dtype=np.float32)
        fps = float(cfg.get("fps", 2.0))
        time = np.asarray(npz[keys["time"]]) if "time" in keys else np.arange(traces.shape[1]) / fps
        coords = np.asarray(npz[keys["coords"]]) if "coords" in keys else None
        neuron_ids = (
            np.asarray(npz[keys["neuron_ids"]])
            if "neuron_ids" in keys
            else np.arange(traces.shape[0])
        )
        behavior = {b: np.asarray(npz[k]) for b, k in cfg.get("behavior_keys", {}).items()}

    rec = _build_recording(
        traces=traces,
        time=time,
        coords=coords,
        behavior=behavior,
        fps=fps,
        metadata=_metadata_from_cfg(cfg, source="c_elegans", path=path),
        identity=_identity_from_cfg(cfg, dataset="c_elegans", path=path),
        neuron_ids=neuron_ids,
    )
    rec.validate()
    logger.info("Loaded C. elegans recording: %d neurons x %d timepoints", rec.n_neurons, rec.n_timepoints)
    return rec
