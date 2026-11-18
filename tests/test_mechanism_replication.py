"""Protocol ownership, blocked inference and untouched calibration/test separation."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.csl_sanity import CSL_METHODS
from effectome.experiments.mechanism_metrics import mechanism_recovery_metrics
from effectome.experiments.mechanism_replication_analysis import summarize_replication
from effectome.experiments.mechanism_replication_config import (
    MechanismReplicationConfig,
    load_mechanism_replication_profile,
)


def _config():
    return replace(MechanismReplicationConfig(), positive_seeds=(20000, 20001),
                   null_seeds=(30000, 30001), min_positive_seeds=2, min_null_seeds=2,
                   max_null_probability=.95, bootstrap_replicates=20)


def _cell(config, scenario, seed, methods, root, signature):
    positive = scenario == "driver"
    truth_values = {"neuron:0": .5 if positive else 0., "neuron:1": 0.,
                    "group:driver_pair": .5 if positive else 0.}
    truth = {"source_states": [0, 1], "factual_transition_probability": [.5, .5],
             "probability_reductions": {k: [v, v] for k, v in truth_values.items()}}
    results = {}
    for name in ("known_support", "empty_support", *methods):
        values = truth_values.copy()
        if not positive and root.name == "calibration" and name in methods:
            values["neuron:0"] = .15
        if name == "empty_support" or config.discovery_history_length == 256:
            values = dict.fromkeys(values, 0.)
        estimated = {"factual_transition_probability": [.5, .5],
                     "probability_reductions": {k: [v, v] for k, v in values.items()}}
        results[name] = {
            "status": "available", "support_recovery": {"recall": 0. if name == "empty_support" else 1.},
            "metrics": mechanism_recovery_metrics(truth, estimated, np.asarray([False, True]),
                                                   effect_threshold=.1, nomination_threshold=.1),
            "rollout": estimated,
        }
    return {"scenario": scenario, "seed": seed, "methods": results, "signature": signature,
            "training_input_fingerprint": str(seed), "evaluation_initial_state_fingerprint": str(seed),
            "oracle": truth}


@pytest.fixture
def fake_runner(monkeypatch):
    import effectome.experiments.mechanism_replication as module

    monkeypatch.setattr(module, "implementation_fingerprint", lambda: "fixed-test-source")
    calls = []

    def execute(config, scenario, seed, methods, root, signature):
        if root.name != "calibration":
            assert (root.parent.parent / "calibration.json").is_file()
        calls.append((root.name, scenario, seed))
        return _cell(config, scenario, seed, methods, root, signature)

    monkeypatch.setattr(module, "_run_cell", execute)
    return module, calls


def test_freeze_calibration_before_tests_and_keep_paired_inputs(fake_runner, tmp_path):
    module, calls = fake_runner
    methods = {name: ConnectivityConfig(name=name) for name in CSL_METHODS}
    report = module.run_mechanism_replication(_config(), tmp_path, methods=methods, workers=1)
    assert calls[:19] == [("calibration", "null", seed) for seed in _config().calibration_seeds]
    assert all(phase != "calibration" for phase, _, _ in calls[19:])
    assert report["stage"] == "complete" and report["execution_complete"]
    assert report["bounded_control_replication_passed"]
    assert not report["scientific_validation_passed"]
    for name in methods:
        assert report["calibration"][name]["operational_threshold"] == .15
        assert report["summary"][name]["positive"]["nomination_precision"]["mean"] == 1
        assert report["paired_history"][name]["paired_inputs_verified"]
        assert report["paired_history"][name]["delta_effect_mae"]["mean"] < 0
    assert (tmp_path / "analysis/analysis-report.md").is_file()


def test_unavailable_calibration_never_submits_evaluation(fake_runner, monkeypatch, tmp_path):
    module, _ = fake_runner
    calls = []

    def missing(config, scenario, seed, methods, root, signature):
        calls.append(root.name)
        cell = _cell(config, scenario, seed, methods, root, signature)
        cell["methods"]["fast_cgc"] = {"status": "unavailable", "reason": "runtime missing"}
        return cell

    monkeypatch.setattr(module, "_run_cell", missing)
    report = module.run_mechanism_replication(_config(), tmp_path, workers=1,
                                             methods={"fast_cgc": ConnectivityConfig(name="fast_cgc")})
    assert set(calls) == {"calibration"}
    assert report["stage"] == "blocked_calibration" and not report["execution_complete"]
    assert report["calibration"]["fast_cgc"]["operational_threshold"] is None
    assert not report["supported_native_methods"]


def test_changed_protocol_cannot_reuse_inspected_test_seeds(fake_runner, tmp_path):
    module, _ = fake_runner
    module.run_mechanism_replication(_config(), tmp_path, workers=1, methods={})
    with pytest.raises(ValueError, match="protocol/source changed"):
        module.run_mechanism_replication(replace(_config(), nomination_floor=.2), tmp_path,
                                         workers=1, methods={})


def test_paired_input_change_is_rejected(fake_runner, monkeypatch, tmp_path):
    module, _ = fake_runner

    def changed(config, scenario, seed, methods, root, signature):
        cell = _cell(config, scenario, seed, methods, root, signature)
        if root.name == "comparison":
            cell["training_input_fingerprint"] = "changed-inputs"
        return cell

    monkeypatch.setattr(module, "_run_cell", changed)
    with pytest.raises(ValueError, match="paired history comparison"):
        module.run_mechanism_replication(_config(), tmp_path, workers=1, methods={})


def test_unsupported_null_direction_blocks_confidence_bound_and_incomplete_positive_suppresses_mean():
    config = _config()
    cells = []
    for phase, scenario, seeds in [("positive", "driver", config.positive_seeds),
                                   ("null-test", "null", config.null_seeds)]:
        for seed in seeds:
            cell = _cell(config.mechanism, scenario, seed, {}, Path(phase), "signature")
            cell.update(phase=phase, history_length=1024)
            cells.append(cell)
    cell = cells[-1]["methods"]["known_support"]["metrics"]
    cell["directions"][0] = {"source_state": 0, "status": "unsupported", "n_origins": 0}
    before = deepcopy(cells)
    calibration = {name: {"complete": True, "operational_threshold": .1}
                   for name in ("known_support", "empty_support")}
    result = summarize_replication(cells, list(calibration), config, calibration)
    assert result["summary"]["known_support"]["null_one_sided_95_upper_bound"] is None
    assert not result["known_support_control_passed"]
    assert cells == before
    result = summarize_replication(cells[1:], list(calibration), config, calibration)
    assert result["summary"]["known_support"]["positive"]["topk_recall"]["mean"] is None


@pytest.mark.parametrize("changes", [
    {"positive_seeds": (10000,)}, {"calibration_seeds": (20,)},
    {"null_seeds": (400,)}, {"positive_seeds": (20000, 20000)},
    {"positive_seeds": (-1,)}, {"calibration_seeds": tuple(range(10000, 10018))},
    {"comparison_history": 1024}, {"primary_history": 2048},
    {"nomination_floor": .01}, {"calibration_alpha": float("nan")},
])
def test_disjoint_fresh_seed_pools_and_fixed_procedure_are_validated(changes):
    with pytest.raises(ValueError):
        replace(MechanismReplicationConfig(), **changes)


def test_frozen_full_and_pilot_profiles():
    full = load_mechanism_replication_profile()
    assert full.planned_cells == 119
    assert (len(full.calibration_seeds), len(full.positive_seeds), len(full.null_seeds)) == (19, 20, 60)
    pilot = load_mechanism_replication_profile("pilot")
    assert len(pilot.null_seeds) < pilot.min_null_seeds
    assert not set(pilot.calibration_seeds + pilot.positive_seeds + pilot.null_seeds) & set(
        full.calibration_seeds + full.positive_seeds + full.null_seeds
    )
