"""Acceptance checks for frozen readouts, full seed pools and calibration ordering."""

from dataclasses import replace

import numpy as np
import pytest

from effectome.experiments.behavior_readout import FrozenBehaviorMap, fit_linear_behavior_readouts
from effectome.experiments.behavior_supervised import (
    ARMS,
    behavior_supervised_inventory,
    run_behavior_supervised,
)
from effectome.experiments.behavior_supervised_config import (
    BehaviorDecisionConfig,
    load_behavior_supervised_profile,
)
from effectome.experiments.behavior_supervised_metrics import assess_confirmation, calibrate_behavior_scores
from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.distributed_behavior_controls import (
    DistributedBehaviorConfig,
    make_distributed_behavior_control,
)
from effectome.manifold import ManifoldConfig, ManifoldFactory
from effectome.manifold.behavior_dynamics import BehaviorDynamicsManifold, build_behavior_pair_indices


def test_training_pairs_use_whole_trials_and_preserve_pair_directions():
    lengths = (6, 9, 11)
    pairs = build_behavior_pair_indices(lengths, 8, 2)
    current, following, trial = (pairs[key] for key in ("current_indices", "next_indices", "trial_ids"))
    borders = np.cumsum(lengths)
    np.testing.assert_array_equal(following, current + 1)
    np.testing.assert_array_equal(np.searchsorted(borders, current, side="right"), trial)
    np.testing.assert_array_equal(np.searchsorted(borders, following, side="right"), trial)
    assert np.bincount(trial).tolist() == [8, 8, 8]


def test_linear_readout_replay_fingerprint_and_training_ownership():
    data = make_distributed_behavior_control(DistributedBehaviorConfig(n_trials=8, trial_length=32))
    original = fit_linear_behavior_readouts(
        data.neural_trials[:4], data.behavior_trials[:4], max_pairs_per_trial=8, early_samples=4
    )
    assert set(original) == {"pca_2d", "pca_4d", "pls_2d", "full_linear"}
    for readout in original.values():
        artifact = readout.to_dict()
        restored = FrozenBehaviorMap.from_dict(artifact)
        np.testing.assert_array_equal(
            readout.predict(data.neural_trials[4:]), restored.predict(data.neural_trials[4:])
        )
        assert restored.metadata["inner_training_trial_ids"] == [0, 1, 2]
        assert restored.metadata["inner_validation_trial_ids"] == [3]
        assert restored.metadata["refit_trial_ids"] == [0, 1, 2, 3]
        assert not readout.layers[0].weights.flags.writeable
        artifact["name"] = "modified"
        with pytest.raises(ValueError, match="fingerprint"):
            FrozenBehaviorMap.from_dict(artifact)


def test_nonlinear_registered_pointwise_backend_rejects_unavailable_runtime(tmp_path):
    shape = (4, 10, 10)
    neural = np.arange(np.prod(shape), dtype=float).reshape(-1, 10)
    with pytest.raises(ValueError, match="pointwise"):
        ManifoldFactory(ManifoldConfig(name="bunddle_behavior", target_length=4))
    estimator = ManifoldFactory(
        ManifoldConfig(
            name="bunddle_behavior",
            target_length=1,
            n_dims=2,
            extra={"trial_lengths": [10] * 4, "runtime_python": str(tmp_path / "missing")},
        )
    )
    assert isinstance(estimator, BehaviorDynamicsManifold)
    with pytest.raises(FileNotFoundError, match="behavior-encoder"):
        estimator.fit(neural, {"continuous": neural[:, :2]})


def _cell(phase, seed, scenario, score, *, present=True):
    return {
        "phase": phase,
        "seed": seed,
        "scenario": scenario,
        "methods": {
            "known": {
                "status": "available" if present else "unavailable",
                "metrics": {
                    "bunddle_behavior_2d": {
                        "candidate_scores": {"neuron:0": score, "group:pair": score},
                        "eligible_units": ["neuron:0", "group:pair"] if scenario == "driver" else [],
                        "end_to_end_mae": 0.01,
                        "semantic_mae": 0.01,
                        "positive_unit_mae": 0.02,
                        "rollout_collapsed": False,
                    },
                },
            }
        },
        "natural_evaluation": {"bunddle_behavior_2d": {"origin_mae": 0.02, "collapsed": False}},
    }


def test_calibration_rejects_wrong_pools_incomplete_evidence_and_duplicate_seeds():
    config = BehaviorDecisionConfig()
    seeds = tuple(range(50000, 50019))
    cells = [_cell("calibration", seed, "null", 0.16 if seed == seeds[-1] else 0.03) for seed in seeds]
    arms = [config.primary_readout]
    cutoffs = calibrate_behavior_scores(cells, ["known"], arms, seeds, config)
    assert cutoffs["rank"] == 19
    assert cutoffs["thresholds"]["known"][arms[0]]["cutoff"] == 0.16
    with pytest.raises(ValueError, match="declared separate null"):
        calibrate_behavior_scores(cells[:-1], ["known"], arms, seeds, config)
    with pytest.raises(ValueError, match="declared separate null"):
        calibrate_behavior_scores(cells + cells[-1:], ["known"], arms, seeds, config)
    cells[-1]["methods"]["known"]["status"] = "unavailable"
    with pytest.raises(ValueError, match="unavailable"):
        calibrate_behavior_scores(cells, ["known"], arms, seeds, config)


def test_confirmation_keeps_null_and_group_gates_separate():
    cfg = BehaviorDecisionConfig()
    cals = tuple(range(50000, 50019))
    calibration = calibrate_behavior_scores(
        [_cell("calibration", seed, "null", 0.05) for seed in cals],
        ["known"],
        [cfg.primary_readout],
        cals,
        cfg,
    )
    positive, null = tuple(range(60000, 60020)), tuple(range(70000, 70060))
    cells = [_cell("positive", seed, "driver", 0.2) for seed in positive] + [
        _cell("null-test", seed, "null", 0.1) for seed in null
    ]
    decision = assess_confirmation(cells, calibration, cfg, ["known"], positive, null)["dynamics"]["known"]
    assert decision["passed"]
    assert decision["null_false_seeds"] == 0
    assert decision["null_exact_upper"] == pytest.approx(0.04870291331)
    assert decision["recovery"]["group"]["eligible_seeds"] == 20
    assert decision["recovery"]["neuron"]["eligible_seeds"] == 20
    cells[0]["methods"]["known"]["metrics"][cfg.primary_readout]["eligible_units"] = ["group:pair"]
    cells[-1]["methods"]["known"]["status"] = "unavailable"
    decision = assess_confirmation(cells, calibration, cfg, ["known"], positive, null)["dynamics"]["known"]
    assert not decision["passed"]
    assert decision["gates"]["group_power"] and not decision["gates"]["neuron_power"]
    assert decision["null_exact_upper"] is None


def test_profiles_and_confirmation_require_same_procedure(tmp_path, monkeypatch):
    dev = load_behavior_supervised_profile("development")
    conf = load_behavior_supervised_profile("confirmation")
    assert dev.planned_cells == 10 and conf.planned_cells == 99
    inventory = behavior_supervised_inventory(dev, dict.fromkeys(CSL_METHODS))
    assert inventory["planned_native_window_fits"] == 360
    assert inventory["readout_arms"] == list(ARMS)
    assert replace(conf, profile="development") == dev
    with pytest.raises(ValueError, match="reserved"):
        replace(dev, development_seeds=(40004,))
    monkeypatch.setattr(
        "effectome.experiments.behavior_supervised._study_identity",
        lambda config, methods: {"source": "current"},
    )
    path = tmp_path / "development.json"
    path.write_text('{"profile":"development","execution_complete":true,"procedure_fingerprint":"other"}')
    with pytest.raises(ValueError, match="identical code"):
        run_behavior_supervised(
            conf, tmp_path / "confirm", methods=dict.fromkeys(CSL_METHODS), development_report=path
        )
    assert not (tmp_path / "confirm").exists()


def test_encoder_worker_trains_and_replays_frozen_weights():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    runtime = root / "tools/behavior-encoder/.venv/bin/python"
    if not runtime.is_file():
        pytest.skip("isolated behavior encoder runtime is not installed")
    data = make_distributed_behavior_control(DistributedBehaviorConfig(n_trials=8, trial_length=32))
    training = data.neural_trials[:4].reshape(-1, 10)
    behavior = data.behavior_trials[:4].reshape(-1, 2)
    model = ManifoldFactory(
        ManifoldConfig(
            name="bunddle_behavior",
            target_length=1,
            n_dims=2,
            extra={
                "trial_lengths": [32] * 4,
                "runtime_python": str(runtime),
                "dynamics_weights": (0.05, 0.2),
                "hidden_widths": (12, 8),
                "max_epochs": 4,
                "min_epochs": 2,
                "patience": 2,
                "max_pairs_per_trial": 8,
                "early_samples": 4,
            },
        )
    )
    assert isinstance(model, BehaviorDynamicsManifold)
    model.fit(training, {"continuous": behavior})
    artifact = model.export_artifact()
    training_metadata = artifact["metadata"]["training"]
    assert training_metadata["inner_training_trial_ids"] == [0, 1, 2]
    assert training_metadata["inner_validation_trial_ids"] == [3]
    assert training_metadata["final_training_trial_ids"] == [0, 1, 2, 3]
    assert artifact["metadata"]["numpy_prediction_parity_max_abs_error"] < 1e-4
    holdout = data.neural_trials[4:].reshape(-1, 10)
    frozen = FrozenBehaviorMap.from_export(artifact)
    np.testing.assert_array_equal(frozen.predict(holdout), model.predict_behavior(holdout))
    np.testing.assert_array_equal(frozen.encode(holdout), model.transform(holdout))


def test_confirmation_freezes_calibration_before_test_jobs(tmp_path, monkeypatch):
    import json

    from effectome.experiments.behavior_supervised import _run_study

    base = load_behavior_supervised_profile("confirmation")
    config = replace(
        base,
        calibration_seeds=(50000, 50001),
        positive_seeds=(60000, 60001),
        null_seeds=(70000, 70001),
        decisions=replace(base.decisions, calibration_alpha=0.34, min_positive_seeds=2, min_null_seeds=2),
    )
    methods = dict.fromkeys(CSL_METHODS)
    lanes = ("oracle", "known_support", "empty_support", *methods)
    seen = []

    def fake_cell(job):
        _, phase, scenario, seed, *_ = job
        seen.append(phase)
        if phase != "calibration":
            cutoff = json.loads((tmp_path / "calibration.json").read_text())
            assert cutoff["calibration_seeds"] == [50000, 50001]
            assert cutoff["thresholds"]["oracle"][config.decisions.primary_readout]["cutoff"] == 0.15
        cell = _cell(phase, seed, scenario, 0.2 if phase == "positive" else 0.15)
        row = cell["methods"]["known"]["metrics"][config.decisions.primary_readout]
        cell["methods"] = {
            method: {"status": "available", "metrics": dict.fromkeys(ARMS, row)} for method in lanes
        }
        cell["cell_id"] = f"{phase}-{seed}"
        return cell

    monkeypatch.setattr("effectome.experiments.behavior_supervised._run_cell", fake_cell)
    monkeypatch.setattr(
        "effectome.experiments.behavior_supervised.render_behavior_supervised_results", lambda *args: None
    )
    result = _run_study(config, tmp_path, 1, methods, {"source": "test"})
    assert seen == ["calibration", "calibration", "positive", "positive", "null-test", "null-test"]
    assert result["execution_complete"] and result["confirmatory_executed"]
    assert not result["scientific_validation_passed"]  # Two null seeds cannot resolve the 5% bound.
