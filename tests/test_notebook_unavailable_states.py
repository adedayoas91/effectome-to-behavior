"""Short recordings must expose infeasible state evaluation without relaxing purges."""

from __future__ import annotations

import json

import numpy as np
import pytest

from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor
from effectome.dynamics import GraphStateConfig, TransitionConfig
from notebooks._shared import (
    make_run,
    run_graph_states_and_transitions,
    run_linking,
    stage_status_path,
)


@pytest.fixture
def short_run(tmp_path):
    n = 8
    data = np.random.default_rng(7).normal(size=(n, 3, 3))
    data[:, np.arange(3), np.arange(3)] = 0
    anchors = [
        TemporalAnchor(
            dataset_id="dataset", recording_id="short", animal_id="animal",
            session_id=None, segment_id=None,
            context_start=i * 5, context_stop=i * 5 + 10,
            target_start=i * 5 + 8, target_stop=i * 5 + 10,
            anchor_sample=i * 5 + 9, anchor_time_seconds=float(i * 5 + 9),
            sampling_rate_hz=1.0,
        )
        for i in range(n)
    ]
    series = ConnectivitySeries(
        matrices=data, lagged_matrices=data[:, None], window_starts=np.arange(n) * 5,
        method="fixture", directed=True, anchors=anchors,
    )
    run = make_run(tmp_path / "runs", "short", "fixture")
    run.execute("connectivity", lambda: series, config={})
    return run


def _evaluate(run, **kwargs):
    return run_graph_states_and_transitions(
        run,
        graph_cfg=GraphStateConfig(feature_representation="signed_lag_offdiag"),
        transition_cfg=TransitionConfig(n_null=0),
        analysis_scope="prospective",
        **kwargs,
    )


def test_short_state_evaluation_remains_strict_by_default(short_run):
    with pytest.raises(ValueError, match="state evaluation unavailable"):
        _evaluate(short_run)
    evaluation = short_run.load_completed("state_evaluation")
    assert evaluation.status == "unavailable"
    assert evaluation.diagnostics["reason"]
    assert evaluation.selected_model is None


def test_short_state_evaluation_can_return_explicit_unavailable_checkpoints(short_run, capsys):
    assert _evaluate(short_run, on_unavailable="return") == (None, None)
    evaluation = short_run.load_completed("state_evaluation")
    assert evaluation.status == "unavailable"
    assert evaluation.selected_model is None
    assert evaluation.diagnostics["reason"] in capsys.readouterr().out
    for stage in ("graph_states", "transitions"):
        assert short_run.load_completed(stage) is None
        record = json.loads(stage_status_path(short_run, stage).read_text())
        assert record["status"] == "completed"
        assert record["signature_payload"]["config"]["status"] == "unavailable"
        assert record["signature_payload"]["config"]["reason"] == evaluation.diagnostics["reason"]


def test_unavailable_states_replace_previous_successful_checkpoints(short_run):
    for stage in ("graph_states", "transitions"):
        short_run.execute(stage, lambda: "previous fit", config={})
    _evaluate(short_run, on_unavailable="return")
    assert short_run.load_completed("graph_states") is None
    assert short_run.load_completed("transitions") is None
    assert _evaluate(short_run, on_unavailable="return") == (None, None)
    assert short_run.last_record.reused


def test_unavailable_state_policy_rejects_typographical_errors(short_run):
    with pytest.raises(ValueError, match="on_unavailable"):
        _evaluate(short_run, on_unavailable="ignore")


def test_linking_records_original_unavailability_before_other_prerequisites(short_run):
    short_run.execute("linking", lambda: {"decoding": ["previous fit"]}, config={})
    _evaluate(short_run, on_unavailable="return")
    report = run_linking(
        short_run, recording_path="missing-recording.pkl", linking_cfg={"n_folds": 5},
    )
    evaluation = short_run.load_completed("state_evaluation")
    assert report["status"] == "unavailable"
    assert report["fail_safe_reason"] == "state_evaluation_unavailable"
    assert report["reason"] == evaluation.diagnostics["reason"]
    assert report["state_evaluation"] == evaluation.summary()
    assert report["decoding"] == []
    assert report["association"] == {}
    assert short_run.load_completed("linking") == report
    assert run_linking(
        short_run, recording_path="missing-recording.pkl", linking_cfg={"n_folds": 5},
    ) == report
    assert short_run.last_record.reused
