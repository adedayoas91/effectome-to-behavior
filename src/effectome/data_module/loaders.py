"""Dataset loaders behind a registry.

Add a loader by decorating it with `@register_loader("name")`. The pipeline selects one via
`cfg.data.name`. Two loaders ship: a synthetic loader (ground-truth, no files needed) and a
zebrafish loader for the PMC10545542 FLFM dataset (schema confirmed during Stage 0).
"""

from __future__ import annotations

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
