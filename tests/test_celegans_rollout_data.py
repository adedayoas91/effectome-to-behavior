"""Verify the worm input lineage and the raw-time split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf
from test_dataset_loaders import _write_bundle_fixture

from effectome.data_module import NeuralRecording, get_loader
from effectome.experiments.celegans_rollout_data import load_worm, plan_worm

REQUEST = {
    "readout_fraction": 0.35,
    "training_end_fraction": 0.65,
    "history_seconds": 2.0,
    "discovery_seconds": 3.0,
    "discovery_stride_seconds": 1.5,
    "horizon_seconds": 1.0,
    "target_seconds": 0.5,
    "stride_seconds": 1.5,
}


def _recording(frames: int = 40) -> NeuralRecording:
    motif = np.resize(np.array([0, 1]), frames)
    motif[-1] = 2
    return NeuralRecording(
        traces=np.ones((2, frames)),
        time=np.arange(frames) / 2,
        coords=None,
        neuron_ids=np.array([3, 8]),
        behavior={"motif": motif},
        fps=2.0,
        metadata={"state_names": ["fwd", "rev", "other"]},
    )


def test_worm_plan_uses_disjoint_blocks_and_actual_terminal_frame() -> None:
    plan = plan_worm(_recording(), **REQUEST)

    assert (plan.readout_stop, plan.training_stop) == (14, 26)
    assert (plan.history, plan.discovery_history, plan.discovery_stride) == (4, 6, 3)
    assert (plan.horizon, plan.target_length, plan.stride) == (2, 1, 3)
    np.testing.assert_array_equal(plan.discovery_starts, [14, 17, 20])
    np.testing.assert_array_equal(plan.origins, [29, 32, 35])
    assert plan.discovery_starts[-1] + plan.discovery_history == plan.training_stop
    assert plan.origins[0] - plan.history + 1 == plan.training_stop
    assert plan.origins[-1] + plan.horizon < plan.frames
    summary = json.loads(json.dumps(plan.summary(), allow_nan=False))
    assert summary["timing_seconds"]["horizon"] == 1.0
    assert summary["motifs_missing_from_readout"] == ["other"]
    assert summary["motif_counts"]["forecast"]["other"] == 1


@pytest.mark.parametrize("declared", [False, True])
def test_worm_plan_rejects_clock_gaps(declared: bool) -> None:
    recording = _recording()
    if declared:
        recording.metadata["gap_intervals"] = [(22, 23)]
    else:
        recording.time[22:] += 0.5
    with pytest.raises(ValueError, match="contiguous clock"):
        plan_worm(recording, **REQUEST)


def test_worm_plan_rejects_nonfinite_data_and_short_block() -> None:
    with pytest.raises(ValueError, match="too short"):
        plan_worm(_recording(12), **REQUEST)
    recording = _recording()
    recording.traces[0, -1] = np.nan
    with pytest.raises(ValueError, match="finite neural observations"):
        plan_worm(recording, **REQUEST)


def test_worm_loader_verifies_files_and_uses_unfiltered_source(tmp_path: Path) -> None:
    data = tmp_path / "external_data"
    data.mkdir()
    files = _write_bundle_fixture(data, raw_neuron_names=["AVAR", "1", "2"])
    with h5py.File(data / files["recordings"], "r+") as handle:
        raw = handle.create_dataset("distinct_raw_signal", data=np.full((3, 4), 7.0))
        handle["NoStim_Data/deltaFOverF"][0, 0] = raw.ref
    hashes = {name: hashlib.sha256((data / name).read_bytes()).hexdigest() for name in files.values()}
    config = {
        "name": "bundle_net_c_elegans",
        "path": "missing-default-path",
        "files": files,
        "downloaded_sha256": hashes,
        "state_names": ["fwd", "rev"],
        "exclude_neuron_names": ["AVAR"],
    }
    (tmp_path / "conf/data").mkdir(parents=True)
    OmegaConf.save(config, tmp_path / "conf/data/c_elegans.yaml")
    worm = load_worm(tmp_path, 0, data_root=data)

    assert worm.identity.recording_id == "bundle-net-worm-0"
    assert worm.identity.animal_id == "worm-0"
    assert worm.metadata["trace_field"] == "deltaFOverF"
    assert worm.neuron_ids.tolist() == [1, 2]
    np.testing.assert_array_equal(worm.traces, np.full((2, 4), 7.0))
    provenance = worm.endpoint_metadata["motif"]
    assert provenance.endpoint_kind == "fictive"
    assert not provenance.prospective_input_verified
    assert provenance.label_vocabulary == ("fwd", "rev")
    assert provenance.source_hashes == (hashes[files["recordings"]],)

    default = get_loader("bundle_net_c_elegans")({**config, "path": str(data)})
    assert default.metadata["trace_field"] == "deltaFOverF_bc"
    assert not np.array_equal(default.traces, worm.traces)
    with (data / files["class_ids"]).open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="SHA-256 mismatch.*ClassIDs_279"):
        load_worm(tmp_path, 0, data_root=data)
