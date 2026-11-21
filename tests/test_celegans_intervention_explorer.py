"""Regression tests for paired interventions on a saved worm forecast."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

from effectome.data_module.schema import NeuralRecording, RecordingIdentity
from effectome.experiments.behavior_readout import FrozenDenseLayer
from effectome.experiments.behavior_transition_graphs import fit_transition_mechanism
from effectome.experiments.celegans_motif_readout import MotifReadout
from effectome.experiments.celegans_rollout_data import plan_worm
from effectome.experiments.mechanism_validation import _write_json
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig
from effectome.workflows.resume import content_fingerprint
from scripts import celegans_intervention_explorer as explorer_module

_WORM = 4
_ROIS = (2, 19, 41)


@dataclass(frozen=True)
class ForecastFixture:
    explorer: explorer_module.Explorer
    case: Path
    origin: np.ndarray
    neural: np.ndarray
    models: tuple[AdditiveMechanism, ...]
    readout: MotifReadout
    config: MechanismRolloutConfig
    factual: np.ndarray


def _paired_reference(
    initial: np.ndarray,
    model: AdditiveMechanism,
    readout: MotifReadout,
    config: MechanismRolloutConfig,
    origin: int,
    removed: tuple[int, ...] = (),
) -> np.ndarray:
    """Run the single-mechanism step reference using the saved per-origin seed."""
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, origin]))
    simulated = np.repeat(initial[None], config.n_rollouts, axis=0)
    target = []
    for step in range(config.horizon):
        simulated = model.step(simulated, rng.normal(size=simulated.shape), removed)
        if step >= config.horizon - config.target_length:
            target.append(readout.predict(simulated))
    return np.mean(target, axis=(0, 1))


@pytest.fixture
def saved_forecast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ForecastFixture:
    frames = 40
    motifs = ["forward", "pause", "reverse"]
    neural = np.random.default_rng(7).normal(size=(frames, len(_ROIS)))
    mean, scale = np.array([0.8, -0.1, 0.5]), np.array([2.0, 0.7, 0.4])
    recording = NeuralRecording(
        traces=(neural * scale + mean).T,
        time=np.arange(frames) / 2,
        coords=None,
        neuron_ids=np.array(_ROIS),
        behavior={"motif": np.arange(frames) % len(motifs)},
        fps=2.0,
        metadata={
            "state_names": motifs,
            "canonical_neuron_names": ["A", "B", "C"],
            "raw_neuron_names": ["ROI A", "ROI B", "ROI C"],
        },
        identity=RecordingIdentity(recording_id="worm-4-fixture"),
    )
    plan = plan_worm(
        recording,
        readout_fraction=0.35,
        training_end_fraction=0.65,
        history_seconds=2.0,
        discovery_seconds=3.0,
        discovery_stride_seconds=1.5,
        horizon_seconds=1.5,
        target_seconds=1.0,
        stride_seconds=1.5,
    )
    pooled_cross = np.array([[0.0, 0.75, 0.0], [0.0, 0.0, 0.22], [0.2, -0.7, 0.0]])
    pooled = AdditiveMechanism(
        np.array([0.2, -0.1, 0.3]),
        np.array([0.25, 0.3, 0.45]),
        pooled_cross,
        np.array([0.2, 0.4, 0.3]),
        pooled_cross != 0,
    )
    readout = MotifReadout(
        input_mean=np.zeros(len(_ROIS)),
        input_scale=np.ones(len(_ROIS)),
        layers=(
            FrozenDenseLayer(np.eye(len(_ROIS)), np.zeros(len(_ROIS)), "tanh"),
            FrozenDenseLayer(np.array([[0.6, 0.3, -0.8], [0.4, -0.7, 0.4]]), np.zeros(2), "tanh"),
            FrozenDenseLayer(np.array([[1.4, -0.2], [-0.5, 0.8], [0.3, -1.0]]), np.zeros(len(motifs))),
        ),
        n_classes=len(motifs),
        classes=(0, 1, 2),
        metadata={},
    )
    seed, draws, penalty = 100, 15, 0.8
    config = MechanismRolloutConfig(plan.horizon, plan.target_length, draws, seed + _WORM)
    selected = np.arange(len(plan.origins)) % 2 == 1
    models = tuple(
        fit_transition_mechanism(
            neural[origin - plan.history + 1 : origin + 1][None], pooled.support, penalty
        )
        if use_local
        else pooled
        for origin, use_local in zip(plan.origins, selected, strict=True)
    )
    factual = np.stack(
        [
            _paired_reference(neural[origin], model, readout, config, int(origin))
            for origin, model in zip(plan.origins, models, strict=True)
        ]
    )

    root = tmp_path / "fixture-run"
    case_path = root / f"worm-{_WORM}" / "forecasts.json"
    inventory = {"worm_index": _WORM, "recording_id": recording.identity.recording_id}
    preflight = {
        "signature": "fixture-signature",
        "protocol": {"settings": {"rollout": {"seed": seed, "draws": draws, "ridge_alpha": penalty}}},
        "worms": [inventory, {"worm_index": 5, "recording_id": "unfinished"}],
    }
    case = {
        "signature": content_fingerprint((preflight["signature"], inventory)),
        "artifacts": {},
        "origins": plan.origins,
        "motif_names": motifs,
        "selected_local": selected,
        "readout": readout.to_dict(),
        "pooled_fit": asdict(pooled),
        "neural_scale": scale,
        "neural_mean": mean,
        "forecasts": {"selected": factual},
    }
    _write_json(case_path, case)
    monkeypatch.setattr(explorer_module, "_load_preflight", lambda *_: preflight)
    monkeypatch.setattr(explorer_module, "_collect_worm", lambda *_: (recording, plan))
    explorer = explorer_module.Explorer(tmp_path, root)
    return ForecastFixture(explorer, case_path, plan.origins, neural, models, readout, config, factual)


def test_explorer_inventory_exposes_only_finished_worm_and_recording_local_ids(saved_forecast):
    explorer = saved_forecast.explorer
    assert explorer.describe() == {
        "run_id": "fixture-run",
        "worms": [{"id": _WORM, "label": "worm-4-fixture"}],
    }
    data = explorer.worm(_WORM)
    assert data["motifs"] == ["forward", "pause", "reverse"]
    assert [neuron["id"] for neuron in data["neurons"]] == list(_ROIS)
    assert [neuron["outgoing_edges"] for neuron in data["neurons"]] == [1, 1, 2]
    assert [point["sample"] for point in data["origins"]] == saved_forecast.origin.tolist()
    assert [point["model"] for point in data["origins"]] == ["pooled", "local", "pooled"]


@pytest.mark.parametrize(("position", "model_name"), [(0, "pooled"), (1, "local")])
def test_explorer_group_cut_replays_one_joint_forecast(saved_forecast, position, model_name):
    trial = saved_forecast
    origin = int(trial.origin[position])
    baseline = trial.explorer.intervene(_WORM, origin, [])
    assert baseline["model"] == model_name
    assert baseline["removed_edges"] == 0
    np.testing.assert_allclose(baseline["factual"], trial.factual[position], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(baseline["disconnected"], baseline["factual"])
    np.testing.assert_array_equal(baseline["delta_pp"], np.zeros(3))

    group = trial.explorer.intervene(_WORM, origin, [41, 2])
    expected = _paired_reference(
        trial.neural[origin], trial.models[position], trial.readout, trial.config, origin, (0, 2)
    )
    assert group["off"] == [2, 41]
    assert group["removed_edges"] == 3
    np.testing.assert_array_equal(group["factual"], baseline["factual"])
    np.testing.assert_allclose(group["disconnected"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(group["delta_pp"], 100 * (expected - trial.factual[position]), atol=1e-10)
    np.testing.assert_allclose(sum(group["delta_pp"]), 0, rtol=0, atol=1e-10)
    assert np.max(np.abs(group["delta_pp"])) > 0.01
    assert trial.explorer.intervene(_WORM, origin, [2, 41]) == group
    assert trial.explorer.intervene(_WORM, origin, []) == baseline


def test_explorer_rejects_unknown_and_ineligible_inputs(saved_forecast):
    explorer = saved_forecast.explorer
    origin = int(saved_forecast.origin[0])
    with pytest.raises(ValueError, match="completed forecasts"):
        explorer.worm(5)
    with pytest.raises(ValueError, match="held-out grid"):
        explorer.intervene(_WORM, origin + 1, [2])
    with pytest.raises(ValueError, match="not found"):
        explorer.intervene(_WORM, origin, [0])
    with pytest.raises(ValueError, match="at most once"):
        explorer.intervene(_WORM, origin, [2, 2])


def test_explorer_requires_parity_with_saved_factual_probabilities(saved_forecast):
    case_path = saved_forecast.case
    case = json.loads(case_path.read_text())
    case["forecasts"]["selected"][0] = [1.0, 0.0, 0.0]
    case_path.write_text(json.dumps(case))
    with pytest.raises(RuntimeError, match="saved factual forecast"):
        saved_forecast.explorer.intervene(_WORM, int(saved_forecast.origin[0]), [2])


def test_explorer_query_parsing():
    assert explorer_module._integer_query({"worm": ["4"]}, "worm") == 4
    assert explorer_module._off_query({}) == ()
    assert explorer_module._off_query({"off": [""]}) == ()
    assert explorer_module._off_query({"off": ["41,2"]}) == (2, 41)
    for field in ([""], ["-1"], ["٣"], ["1", "2"], ["3.0"]):
        with pytest.raises(ValueError, match="valid worm integer"):
            explorer_module._integer_query({"worm": field}, "worm")
    for field in (["1,2,"], ["1,-2"], ["1,2", "3"]):
        with pytest.raises(ValueError, match="ROI IDs"):
            explorer_module._off_query({"off": field})
