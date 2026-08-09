"""Focused tests for dataset-specific loaders."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from effectome.data_module import get_loader


def _write_char_dataset(handle: h5py.File, path: str, text: str) -> h5py.Dataset:
    data = np.array([[ord(char)] for char in text], dtype=np.uint16)
    return handle.create_dataset(path, data=data)


def _write_ref_vector(
    handle: h5py.File,
    path: str,
    values: list[str],
    *,
    prefix: str,
) -> h5py.Dataset:
    refs = np.empty((len(values), 1), dtype=h5py.ref_dtype)
    for idx, value in enumerate(values):
        node = _write_char_dataset(handle, f"{prefix}_{idx}", value)
        refs[idx, 0] = node.ref
    return handle.create_dataset(path, data=refs)


def _write_bundle_fixture(base: Path) -> dict:
    recordings_path = base / "NoStim_Data.mat"
    order_path = base / "Order279.mat"
    class_ids_path = base / "ClassIDs_279.mat"

    with h5py.File(order_path, "w") as handle:
        _write_ref_vector(handle, "Order279", ["AVA", "AVB"], prefix="order_name")

    with h5py.File(class_ids_path, "w") as handle:
        handle.create_dataset("ClassIDs_279", data=np.array([[1.0], [3.0]], dtype=np.float64))

    with h5py.File(recordings_path, "w") as handle:
        root = handle.create_group("NoStim_Data")
        traces = handle.create_dataset(
            "worm0_deltaFOverF_bc",
            data=np.array([[0.1, 0.2, 0.4, 0.8], [0.3, 0.6, 0.9, 1.2]], dtype=np.float64),
        )
        fps = handle.create_dataset("worm0_fps", data=np.array([[2.0]], dtype=np.float64))
        tv = handle.create_dataset("worm0_tv", data=np.array([[0.0, 0.5, 1.0, 1.5]], dtype=np.float64))
        dataset_name = _write_char_dataset(handle, "worm0_dataset_name", "bundle-test")
        neuron_names = _write_ref_vector(handle, "worm0_neuron_names", ["1", "2"], prefix="neuron_name")

        states = handle.create_group("worm0_states")
        states.create_dataset("fwd", data=np.array([[1.0], [0.0], [0.0], [1.0]], dtype=np.float64))
        states.create_dataset("rev", data=np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64))

        ref_shape = (1, 1)
        root.create_dataset("deltaFOverF_bc", data=np.array([[traces.ref]], dtype=h5py.ref_dtype))
        root.create_dataset("fps", data=np.array([[fps.ref]], dtype=h5py.ref_dtype))
        root.create_dataset("tv", data=np.array([[tv.ref]], dtype=h5py.ref_dtype))
        root.create_dataset("NeuronNames", data=np.array([[neuron_names.ref]], dtype=h5py.ref_dtype))
        root.create_dataset("States", data=np.array([[states.ref]], dtype=h5py.ref_dtype))
        root.create_dataset("dataset", data=np.array([[dataset_name.ref]], dtype=h5py.ref_dtype))
        root.create_dataset(
            "deltaFOverF",
            shape=ref_shape,
            dtype=h5py.ref_dtype,
            data=np.array([[traces.ref]], dtype=h5py.ref_dtype),
        )
        root.create_dataset(
            "derivs",
            shape=ref_shape,
            dtype=h5py.ref_dtype,
            data=np.array([[traces.ref]], dtype=h5py.ref_dtype),
        )
        root.create_dataset(
            "Opts",
            shape=ref_shape,
            dtype=h5py.ref_dtype,
            data=np.array([[states.ref]], dtype=h5py.ref_dtype),
        )
        root.create_dataset(
            "stateParams",
            shape=ref_shape,
            dtype=h5py.ref_dtype,
            data=np.array([[states.ref]], dtype=h5py.ref_dtype),
        )

    return {
        "recordings": recordings_path.name,
        "class_ids": class_ids_path.name,
        "neuron_order": order_path.name,
    }


def test_v2a_rsns_loader_interpolates_behavior_and_tracks_boundaries(tmp_path):
    base = tmp_path / "v2a"
    base.mkdir()

    info = {
        "frameRateBeh": 8.0,
        "frameRateSCAPE": 2.0,
        "nCells": 5,
        "nFramesSCAPE": 4,
        "bad_frames": [1, 2],
        "fishID": "fish-1",
        "runID": "run-1",
    }
    (base / "analysis_info.json").write_text(json.dumps(info), encoding="utf-8")
    np.save(base / "traces.npy", np.arange(20, dtype=np.float32).reshape(5, 4))
    np.save(base / "registered_coords.npy", np.arange(15, dtype=np.float32).reshape(5, 3))
    np.save(base / "raw_coords.npy", (100 + np.arange(15, dtype=np.float32)).reshape(5, 3))
    np.save(base / "tail.npy", np.arange(16, dtype=np.float32))
    np.save(base / "emitters.npy", np.array([0, 2], dtype=np.int64))
    np.save(base / "receivers.npy", np.array([3, 4], dtype=np.int64))

    rec = get_loader("v2a_rsns")(
        {
            "name": "v2a_rsns",
            "path": str(base),
            "dataset_id": "v2a-rsns",
            "recording_id": "fish-1-run-1",
            "animal_id": "fish-1",
            "cell_subset": "emitter_receiver",
            "coords_source": "raw",
            "bad_frame_policy": "boundaries",
            "bad_frame_indexing": "zero_based",
            "behavior_keys": {"continuous": "tail_angle"},
            "files": {
                "analysis_info": "analysis_info.json",
                "traces": "traces.npy",
                "coords": "registered_coords.npy",
                "raw_coords": "raw_coords.npy",
                "tail_angle": "tail.npy",
                "emitter_cells": "emitters.npy",
                "receiver_cells": "receivers.npy",
            },
        }
    )

    assert rec.traces.shape == (4, 4)
    assert rec.coords is not None
    assert rec.coords.tolist() == (100 + np.arange(15, dtype=np.float32).reshape(5, 3))[[0, 2, 3, 4]].tolist()
    assert rec.neuron_ids.tolist() == [0, 2, 3, 4]
    assert np.allclose(rec.behavior["continuous"], [0.0, 4.0, 8.0, 12.0])
    assert rec.metadata["bad_frame_intervals"] == [(1, 3)]
    assert rec.metadata["gap_intervals"] == [(1, 3)]
    assert rec.metadata["bad_frame_indexing"] == "zero_based"
    assert rec.metadata["coords_source"] == "raw"


def test_v2a_rsns_loader_masks_sparse_bad_frames_without_compressing_time(tmp_path):
    base = tmp_path / "v2a"
    base.mkdir()
    info = {
        "frameRateBeh": 10.0,
        "frameRateSCAPE": 10.0,
        "nCells": 2,
        "nFramesSCAPE": 20,
        "bad_frames": [2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    }
    (base / "analysis_info.json").write_text(json.dumps(info), encoding="utf-8")
    np.save(base / "traces.npy", np.arange(40, dtype=np.float32).reshape(2, 20))
    np.save(base / "coords.npy", np.arange(6, dtype=np.float32).reshape(2, 3))
    np.save(base / "tail.npy", np.arange(20, dtype=np.float32))
    np.save(base / "emitters.npy", np.array([0], dtype=np.int64))
    np.save(base / "receivers.npy", np.array([1], dtype=np.int64))

    rec = get_loader("v2a_rsns")(
        {
            "name": "v2a_rsns",
            "path": str(base),
            "bad_frame_policy": "mask",
            "max_masked_gap_seconds": 0.5,
            "files": {
                "analysis_info": "analysis_info.json",
                "traces": "traces.npy",
                "coords": "coords.npy",
                "tail_angle": "tail.npy",
                "emitter_cells": "emitters.npy",
                "receiver_cells": "receivers.npy",
            },
        }
    )

    assert rec.n_timepoints == 20
    assert np.array_equal(rec.time, np.arange(20, dtype=float) / 10.0)
    assert np.all(np.isnan(rec.traces[:, 2]))
    assert np.all(np.isnan(rec.traces[:, 10:20]))
    assert rec.metadata["bad_frame_intervals"] == [(2, 3), (10, 20)]
    assert rec.metadata["hard_bad_frame_intervals"] == [(10, 20)]
    assert rec.metadata["gap_intervals"] == [(10, 20)]
    assert rec.metadata["bad_frame_policy"] == "mask"


def test_bundle_net_c_elegans_loader_builds_motif_and_canonical_names(tmp_path):
    base = tmp_path / "bundle"
    base.mkdir()
    files = _write_bundle_fixture(base)

    rec = get_loader("bundle_net_c_elegans")(
        {
            "name": "bundle_net_c_elegans",
            "path": str(base),
            "dataset_id": "bundle-net-c-elegans",
            "recording_id": "worm-0",
            "animal_id": "worm-0",
            "worm_index": 0,
            "state_names": ["fwd", "rev"],
            "behavior_keys": {"motif": "motif"},
            "source_url": "https://example.test/bundle",
            "source_commit": "deadbeef",
            "files": files,
        }
    )

    assert rec.traces.shape == (2, 4)
    assert np.allclose(rec.time, [0.0, 0.5, 1.0, 1.5])
    assert rec.fps == 2.0
    assert rec.behavior["motif"].tolist() == [0, 1, 1, 0]
    assert rec.neuron_ids.tolist() == [0, 1]
    assert rec.metadata["bundle_dataset_name"] == "bundle-test"
    assert rec.metadata["state_names"] == ["fwd", "rev"]
    assert rec.metadata["canonical_neuron_names"] == ["AVA", "AVB"]
    assert rec.metadata["canonical_class_ids"] == [1, 3]
    assert rec.metadata["source_url"] == "https://example.test/bundle"
    assert rec.metadata["source_commit"] == "deadbeef"


def test_bundle_net_loader_reproduces_raw_name_exclusion(tmp_path):
    base = tmp_path / "bundle"
    base.mkdir()
    files = _write_bundle_fixture(base)

    rec = get_loader("bundle_net_c_elegans")(
        {
            "name": "bundle_net_c_elegans",
            "path": str(base),
            "worm_index": 0,
            "state_names": ["fwd", "rev"],
            "exclude_neuron_name_source": "raw",
            "exclude_neuron_names": ["1", "not-present"],
            "files": files,
        }
    )

    assert rec.traces.shape == (1, 4)
    assert rec.metadata["raw_neuron_names"] == ["2"]
    assert rec.metadata["excluded_neuron_names"] == ["1"]
    assert rec.metadata["requested_but_absent_excluded_neuron_names"] == ["not-present"]
