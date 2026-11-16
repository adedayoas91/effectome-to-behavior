"""Known support isolates readout recovery; estimated lanes never receive truth."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig
from effectome.experiments.mechanism_controls import (
    MechanismControlConfig,
    make_mechanism_control,
    mechanism_control_trials,
    oracle_mechanism,
)
from effectome.experiments.mechanism_metrics import _tie_aware_recall, summarize_mechanism_cells
from effectome.experiments.mechanism_reporting import render_mechanism_report
from effectome.experiments.mechanism_validation import run_mechanism_validation
from effectome.experiments.mechanism_validation_config import (
    MechanismValidationConfig,
    load_mechanism_validation_profile,
)
from effectome.experiments.structural_mechanism import MechanismRolloutConfig


def _config():
    return MechanismValidationConfig(
        generator=MechanismControlConfig(n_trials=24),
        rollout=MechanismRolloutConfig(n_rollouts=128),
    )


def test_control_is_fully_observed_reproducible_and_retains_independent_trial_boundaries():
    config = MechanismControlConfig(n_trials=24)
    first = make_mechanism_control(config)
    np.testing.assert_array_equal(first.traces, make_mechanism_control(config).traces)
    trials = mechanism_control_trials(first)
    assert trials.shape == (24, 320, 6)
    assert np.isfinite(trials).all()
    assert first.behavior == {}
    assert first.metadata["valid_ranges"] == [(i * 320, (i + 1) * 320) for i in range(24)]
    support = first.metadata["true_support"]
    assert set(zip(*np.where(support), strict=True)) == {(0, 2), (1, 2), (0, 5), (1, 5)}
    model = oracle_mechanism(first)
    previous = np.array([[0.8, 0.2, 0.0, 0.0, 0.0, -1.0]])
    expected = previous * first.metadata["true_self_weights"]
    expected[:, 2] += config.bystander_coupling * np.tanh(previous[:, :2]).sum(axis=1)
    expected[:, 5] += config.coupling * (np.tanh(previous[:, 0]) - np.tanh(previous[:, 1]))
    np.testing.assert_allclose(model.mean(previous), expected)
    null = make_mechanism_control(replace(config, scenario="null"))
    np.testing.assert_array_equal(first.traces[:2], null.traces[:2])
    assert not null.metadata["true_support"][:, -1].any()


def test_known_support_learned_mechanism_recovers_effects_and_empty_support_has_no_effect(tmp_path):
    report = run_mechanism_validation(_config(), tmp_path, oracle_only=True)
    assert report["execution_complete"]
    assert report["available_method_runs"] == report["attempted_method_runs"] == 4
    assert not report["scientific_validation_passed"]
    assert not report["confirmatory_executed"]
    assert not report["nomination_calibrated"]
    for cell in report["cells"]:
        assert set(cell["training_trial_ids"]).isdisjoint(cell["evaluation_trial_ids"])
        known, empty = (cell["methods"][name] for name in ("known_support", "empty_support"))
        assert known["metrics"]["mean_absolute_effect_error"] < 0.06
        for effects in empty["rollout"]["probability_reductions"].values():
            np.testing.assert_array_equal(effects, 0)
        if cell["scenario"] == "driver":
            assert all(row["driver_nomination_recall"] == 1 for row in known["metrics"]["directions"])
            assert all(row["topk_recall"] == 1 for row in known["metrics"]["directions"])
        else:
            assert not known["metrics"]["any_false_nomination"]
    repeated = run_mechanism_validation(_config(), tmp_path, oracle_only=True)
    assert repeated["summary"] == report["summary"]
    rendered = render_mechanism_report(report, tmp_path / "analysis")
    text = rendered.read_text()
    assert "not replace" in text and "provisional development" in text
    assert "known_support" in text and "empty_support" in text
    assert "\r" not in (rendered.parent / "numeric-summary.csv").read_text()


def test_all_zero_scores_do_not_get_favorable_lexical_tie_breaking():
    assert _tie_aware_recall({"neuron:0": 0, "neuron:1": 0, "neuron:2": 0}, {"neuron:0"}) == 1 / 3
    assert _tie_aware_recall({"neuron:0": 1, "neuron:1": 0}, {"neuron:0"}) == 1
    assert _tie_aware_recall({"neuron:0": 0}, set()) is None


def test_native_support_and_mechanism_selection_are_invariant_to_heldout_future_values(tmp_path, monkeypatch):
    import effectome.experiments.mechanism_validation as module

    inputs = []

    class FakeNative:
        def run(self, windows):
            inputs.append(windows.segments.copy())
            assert windows.segments.shape == (9, 256, 6)
            assert "true_support" not in windows.metadata
            graphs = np.zeros((9, 6, 6))
            graphs[:, 0, -1] = 1  # Deliberately incomplete support, never repaired with truth.
            return SimpleNamespace(matrices=graphs, diagnostics={"upstream_provenance": "test-only"})

    monkeypatch.setattr(module, "ConnectivityFactory", lambda _: FakeNative())
    methods = {"fast_cgc": ConnectivityConfig(name="fast_cgc")}
    original = run_mechanism_validation(_config(), tmp_path / "original", methods=methods)
    generator = module.make_mechanism_control

    def changed(config):
        recording = generator(config)
        trials = mechanism_control_trials(recording).copy()
        trials[12:, 1:] += np.random.default_rng(99).normal(size=trials[12:, 1:].shape) * 10
        return replace(recording, traces=trials.reshape(-1, 6).T)

    monkeypatch.setattr(module, "make_mechanism_control", changed)
    altered = run_mechanism_validation(_config(), tmp_path / "altered", methods=methods)
    for first, second in zip(original["cells"], altered["cells"], strict=True):
        for name in first["methods"]:
            a, b = first["methods"][name], second["methods"][name]
            assert a["model"]["metadata"] == b["model"]["metadata"]
            for key in ("intercept", "self_weights", "cross_weights", "noise_scale", "support"):
                np.testing.assert_array_equal(
                    a["model"].get(key, a.get(key)), b["model"].get(key, b.get(key))
                )
        inferred = first["methods"]["fast_cgc"]
        assert inferred["support_recovery"]["fn"] > 0
        assert not inferred["support_metadata"]["uses_ground_truth_support"]
    np.testing.assert_array_equal(inputs[0], inputs[2])
    np.testing.assert_array_equal(inputs[1], inputs[3])
    artifact = next((tmp_path / "original/cells").glob("*.npz"))
    artifact.write_bytes(b"corrupted evidence")
    with pytest.raises(ValueError, match="integrity"):
        run_mechanism_validation(_config(), tmp_path / "original", methods=methods)


def test_missing_native_runtime_is_retained_without_substituting_oracle_support(tmp_path, monkeypatch):
    import effectome.experiments.mechanism_validation as module

    def missing(_):
        raise ImportError("native runtime unavailable")

    monkeypatch.setattr(module, "ConnectivityFactory", missing)
    report = run_mechanism_validation(
        _config(), tmp_path, methods={"fast_cgc": ConnectivityConfig(name="fast_cgc")}
    )
    assert report["execution_complete"]
    assert report["available_method_runs"] == 4 and report["attempted_method_runs"] == 6
    assert all(cell["methods"]["fast_cgc"]["status"] == "unavailable" for cell in report["cells"])
    assert report["summary"]["fast_cgc"]["null_one_sided_95_upper_bound"] is None


def test_profiles_and_confirmation_seed_reservation():
    for name in ("smoke", "development"):
        config = load_mechanism_validation_profile(name)
        assert config.profile == name
    with pytest.raises(ValueError, match="reserved"):
        replace(_config(), seeds=(300,))
    with pytest.raises(ValueError, match="unknown"):
        load_mechanism_validation_profile("confirmatory")
    with pytest.raises(ValueError, match="inside"):
        replace(_config(), discovery_burn_in=100)


def test_configured_rollout_seed_changes_stream_without_changing_training_data(tmp_path):
    config = _config()
    first = run_mechanism_validation(config, tmp_path / "first", oracle_only=True)
    changed = replace(config, rollout=replace(config.rollout, seed=config.rollout.seed + 1))
    second = run_mechanism_validation(changed, tmp_path / "second", oracle_only=True)
    stream_changed = []
    for a, b in zip(first["cells"], second["cells"], strict=True):
        assert a["training_input_fingerprint"] == b["training_input_fingerprint"]
        assert a["rollout_seed_ownership"]["effective_seed"] != b["rollout_seed_ownership"]["effective_seed"]
        assert (a["methods"]["known_support"]["model"]["metadata"]
                == b["methods"]["known_support"]["model"]["metadata"])
        stream_changed.append(not np.array_equal(a["oracle"]["factual_transition_probability"],
                                                 b["oracle"]["factual_transition_probability"]))
    assert any(stream_changed)


def test_ineligible_ranking_seed_is_counted_and_suppresses_aggregate():
    cells = [{"scenario": "driver", "methods": {"known_support": {
        "status": "available", "metrics": {"mean_absolute_effect_error": 0.01,
                                             "directions": [{"topk_recall": 1}, {"topk_recall": None}]},
    }}}]
    result = summarize_mechanism_cells(cells, ["known_support"], (20, 21))["known_support"]
    assert result["positive_seeds_evaluated"] == 1
    assert result["positive_topk_eligible_seeds"] == 0 and result["positive_topk_planned_seeds"] == 2
    assert not result["positive_topk_eligible_seed_set_complete"]
    assert result["positive_mean_topk_recall"] is None
    assert result["null_one_sided_95_upper_bound"] is None
