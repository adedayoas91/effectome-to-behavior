"""Verify input ownership, paired innovations, and native rollout evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig
from effectome.data_module.schema import Window
from effectome.experiments import behavior_transition_graphs
from effectome.experiments.behavior_readout import FrozenBehaviorMap, FrozenDenseLayer
from effectome.experiments.behavior_transition_controls import TRANSITION_SCENARIOS
from effectome.experiments.behavior_transition_graphs import trailing_histories
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig
from effectome.workflows.effectome_rollout import forecast_mechanisms, run_effectome_rollout
from synthetic.effectome_rollout import GAIN_MARGIN, SEEDS, decision


def _readout():
    return FrozenBehaviorMap(
        "independent_learned_behavior",
        np.zeros(2),
        np.ones(2),
        (FrozenDenseLayer(np.array([[0.3, -0.7], [0.0, 1.0]]), np.zeros(2), "tanh"),),
    )


def _mechanism(coupling):
    cross = np.array([[0.0, coupling], [0.0, 0.0]])
    return AdditiveMechanism(np.zeros(2), np.array([0.75, 0.4]), cross, np.array([0.05, 0.12]), cross != 0)


def _options(tmp_path):
    return {
        "method": ConnectivityConfig(name="fast_cgc"),
        "fps": 10.0,
        "evidence": tmp_path / "native.json",
        "signature": "test-protocol",
        "readout": _readout(),
        "interventions": {"source": (0,)},
        "config": MechanismRolloutConfig(horizon=3, target_length=2, n_rollouts=13, seed=71),
    }


def test_core_fit_uses_trailing_histories_and_rejects_stale_native_evidence(tmp_path, monkeypatch):
    captured = []

    def factory(_config):
        def run(segments):
            captured.append(segments)
            support = np.array([[False, True], [False, False]])
            return SimpleNamespace(matrices=np.broadcast_to(support, (2, 2, 2)), diagnostics={})

        return SimpleNamespace(run=run)

    monkeypatch.setattr(behavior_transition_graphs, "ConnectivityFactory", factory)

    data = np.random.default_rng(12).normal(size=(2, 10, 2))
    origins = np.array([[6], [5]])
    origin_ids = np.array([19, 2])
    histories, windows = trailing_histories(data, origins, 5)
    options = _options(tmp_path)
    result, models = run_effectome_rollout(histories, windows, origin_ids, **options)

    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0].segments, np.stack((data[0, 2:7], data[1, 1:6])))
    assert captured[0].windows == [Window(2, 7), Window(11, 16)]
    assert all(model.metadata["fit_pairs"] == 4 for model in models)

    data[0, 7:] += 100
    data[1, 6:] -= 100
    unchanged, windows = trailing_histories(data, origins, 5)
    replay, _ = run_effectome_rollout(unchanged, windows, origin_ids, **options)
    assert len(captured) == 1
    np.testing.assert_array_equal(result.factual, replay.factual)
    np.testing.assert_array_equal(result.effects["source"], replay.effects["source"])

    changed = unchanged.copy()
    changed[0, 0, 0] += 1
    with pytest.raises(ValueError, match="native support evidence belongs to another study"):
        run_effectome_rollout(changed, windows, origin_ids, **options)
    assert len(captured) == 1


def test_forecasts_preserve_order_and_share_innovations_by_origin():
    positive, negative = _mechanism(0.8), _mechanism(-0.3)
    mechanisms = (positive, negative, positive)
    states = np.array([[1.0, 0.0], [2.0, 0.7], [-1.0, 0.0]])
    origin_ids = np.array([45, 6, 12])
    readout = _readout()
    config = MechanismRolloutConfig(horizon=3, target_length=2, n_rollouts=13, seed=71)
    result = forecast_mechanisms(mechanisms, states, origin_ids, readout, {"source": (0,)}, config)

    for index, (mechanism, state, origin) in enumerate(zip(mechanisms, states, origin_ids, strict=True)):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, int(origin)]))
        draws = rng.normal(size=(config.horizon, config.n_rollouts, len(state)))
        means = []
        for interrupted in ((), (0,)):
            neural = np.repeat(state[None], config.n_rollouts, axis=0)
            decoded = []
            for draw in draws:
                neural = mechanism.step(neural, draw, disconnected=interrupted)
                decoded.append(readout.predict(neural))
            means.append(np.mean(decoded[-config.target_length :], axis=(0, 1)))
        np.testing.assert_allclose(result.factual[index], means[0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(result.effects["source"][index], means[0] - means[1], rtol=0, atol=1e-15)


def test_rollout_rejects_wrong_history_shapes_and_duplicate_origin_ids(tmp_path):
    options = _options(tmp_path)
    with pytest.raises(ValueError, match="origin-by-time-by-neuron"):
        run_effectome_rollout(np.zeros((3, 2)), [Window(0, 3)], np.array([8]), **options)
    with pytest.raises(ValueError, match="raw dependencies"):
        run_effectome_rollout(np.zeros((1, 4, 2)), [Window(0, 3)], np.array([8]), **options)
    with pytest.raises(ValueError, match="distinct nonnegative integer identifier"):
        forecast_mechanisms(
            (_mechanism(0.8), _mechanism(-0.3)),
            np.ones((2, 2)),
            np.array([8, 8]),
            _readout(),
            {"source": (0,)},
            MechanismRolloutConfig(),
        )


def test_confirmation_checks_whole_seed_pool_and_all_control_gates():
    seeds = SEEDS["confirmation"]
    rows = [
        {
            "seed": seed,
            "scenario": scenario,
            "dynamic_vs_pooled_gain": 2 * GAIN_MARGIN if scenario == "driver_switch" else 0,
        }
        for scenario in TRANSITION_SCENARIOS
        for seed in seeds
    ]
    assert decision(rows, seeds)["forecast_confirmation_passed"]
    assert not decision(rows[:-1], seeds)["forecast_confirmation_passed"]
    nuisance = [row for row in rows if row["scenario"] == "nuisance_switch"]
    for row in nuisance[:6]:
        row["dynamic_vs_pooled_gain"] = GAIN_MARGIN
    evidence = decision(rows, seeds)
    assert evidence["tests"]["nuisance_switch"]["gains_on_expected_side"] == 14
    assert not evidence["forecast_confirmation_passed"]
