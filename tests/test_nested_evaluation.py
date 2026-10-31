"""Split-owned evaluation skeleton for Appendix-C WP4."""

from __future__ import annotations

import numpy as np

from effectome.data_module.schema import TemporalAnchor, Window, WindowedSegments
from effectome.workflows import (
    EvaluationConfig,
    PredictionRecord,
    build_split_manifest,
    paired_log_loss_delta,
    run_evaluation_skeleton,
    score_classification_records,
)


def _anchor(index: int, *, history: int = 4, recording_id: str = "rec") -> TemporalAnchor:
    start = index * 2
    return TemporalAnchor(
        dataset_id="ds",
        recording_id=recording_id,
        animal_id="animal-0",
        session_id=None,
        segment_id=None,
        context_start=start,
        context_stop=start + history,
        target_start=start + history - 1,
        target_stop=start + history,
        anchor_sample=start + history - 1,
        anchor_time_seconds=float(start + history - 1),
        sampling_rate_hz=1.0,
    )


def _windows(n_windows: int = 10, *, history: int = 4) -> WindowedSegments:
    anchors = [_anchor(index, history=history) for index in range(n_windows)]
    return WindowedSegments(
        segments=np.zeros((n_windows, history, 2), dtype=np.float32),
        windows=[Window(anchor.context_start, anchor.context_stop) for anchor in anchors],
        behavior_per_window={"motif": np.zeros(n_windows, dtype=np.int64)},
        n_neurons=2,
        fps=1.0,
        anchors=anchors,
    )


class SpyFitter:
    name = "spy"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def fit(self, train_indices: np.ndarray, fold):
        train = tuple(int(item) for item in train_indices)
        test = tuple(int(item) for item in fold.test_indices)
        assert set(train).isdisjoint(test)
        self.calls.append((fold.fold_id, train, test))
        return {"fold_id": fold.fold_id, "train": train}

    def predict(self, fitted, test_indices: np.ndarray, fold):
        assert fitted["fold_id"] == fold.fold_id
        return [
            PredictionRecord(
                origin_index=int(index),
                fold_id=fold.fold_id,
                model_name=self.name,
                status="issued",
                issue_sample=int(index),
            )
            for index in test_indices
        ]


def test_evaluation_skeleton_owns_splits_before_fitters_run():
    windows = _windows(12, history=3)
    fitter = SpyFitter()
    result = run_evaluation_skeleton(
        windows,
        EvaluationConfig(n_outer_folds=3, embargo=0, min_train_examples=2),
        fitters=[fitter],
    )

    available = [fold for fold in result.manifest.outer_folds if fold.status == "available"]
    assert available
    assert len(fitter.calls) == len(available)
    assert result.diagnostics["fit_calls"] == len(available)
    assert result.diagnostics["prediction_count"] == len(result.predictions)
    for prediction in result.predictions:
        fold = next(fold for fold in available if fold.fold_id == prediction.fold_id)
        assert prediction.origin_index in fold.test_indices


def test_evaluation_manifest_retains_unavailable_purged_folds():
    windows = _windows(6, history=20)
    manifest = build_split_manifest(
        windows.n_windows,
        EvaluationConfig(n_outer_folds=3, embargo=0, min_train_examples=2),
        anchors=windows.anchors,
    )

    assert any(fold.status == "unavailable" for fold in manifest.outer_folds)
    assert all(fold.test_indices for fold in manifest.outer_folds)
    assert {
        fold.failure_reason for fold in manifest.outer_folds if fold.status == "unavailable"
    } <= {"purging_left_insufficient_training_examples", "fold_size_below_minimum"}


def test_evaluation_skeleton_runs_without_state_or_group_artifacts():
    windows = _windows(10, history=3)
    result = run_evaluation_skeleton(
        windows,
        EvaluationConfig(
            n_outer_folds=2,
            embargo=0,
            states_enabled=False,
            groups_enabled=False,
        ),
    )

    assert result.manifest.states_enabled is False
    assert result.manifest.groups_enabled is False
    assert result.predictions == ()
    assert result.diagnostics["skeleton_status"] == "split_owner_only"


def test_chronological_evaluation_requires_physical_anchors():
    try:
        build_split_manifest(8, EvaluationConfig(protocol="chronological_online"))
    except ValueError as exc:
        assert "anchors" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing physical anchors to fail")


def test_grouped_manifest_keeps_unequal_recording_folds():
    anchors = [_anchor(i, recording_id="a") for i in range(6)]
    anchors += [_anchor(i, recording_id="b") for i in range(10)]
    manifest = build_split_manifest(16, EvaluationConfig(n_outer_folds=2), anchors=anchors)
    assert sorted(len(f.test_indices) for f in manifest.outer_folds) == [6, 10]
    assert all(f.status == "available" for f in manifest.outer_folds)


def test_paired_origins_distinguish_recordings_and_reject_duplicates():
    import pytest

    records = [
        PredictionRecord(
            origin_index=0, fold_id="f", model_name=method, status="issued",
            endpoint_name="e", class_order=(0, 1), probabilities=(p, 1-p),
            observed=0, animal_id="a", recording_id=recording,
        )
        for recording, probabilities in [("r1", (0.8, 0.4)), ("r2", (0.4, 0.8))]
        for method, p in zip(("M0", "M1"), probabilities, strict=True)
    ]
    result = paired_log_loss_delta(records, baseline_model="M0", candidate_model="M1", endpoint_name="e")
    assert result.n_matched_predictions == 2
    assert result.equal_animal_delta_log_loss == 0.0
    with pytest.raises(ValueError, match="duplicate"):
        paired_log_loss_delta(
            [*records, records[0]], baseline_model="M0", candidate_model="M1", endpoint_name="e"
        )


def test_probability_scoring_uses_natural_log_loss():
    records = (
        PredictionRecord(
            origin_index=0,
            fold_id="outer-000",
            model_name="M0",
            endpoint_name="motif",
            endpoint_kind="synthetic",
            class_order=("a", "b"),
            probabilities=(0.25, 0.75),
            observed="b",
            status="issued",
            animal_id="animal-a",
        ),
        PredictionRecord(
            origin_index=1,
            fold_id="outer-000",
            model_name="M0",
            endpoint_name="motif",
            endpoint_kind="synthetic",
            class_order=("a", "b"),
            probabilities=(0.5, 0.5),
            observed="a",
            status="issued",
            animal_id="animal-a",
        ),
        PredictionRecord(
            origin_index=2,
            fold_id="outer-000",
            model_name="M0",
            endpoint_name="motif",
            status="failed",
            failure_reason="unavailable_history",
        ),
    )

    score = score_classification_records(records, model_name="M0", endpoint_name="motif")

    assert score.n_predictions == 2
    assert score.n_failed == 1
    assert score.log_loss_nats == np.mean([-np.log(0.75), -np.log(0.5)])
    assert score.accuracy == 1.0


def test_paired_log_loss_delta_uses_equal_animal_weighting():
    records = []
    for origin, animal, m0, m1 in [
        (0, "animal-a", 0.5, 0.25),
        (1, "animal-a", 0.5, 0.25),
        (2, "animal-b", 0.25, 0.5),
    ]:
        records.extend(
            [
                PredictionRecord(
                    origin_index=origin,
                    fold_id="outer-000",
                    model_name="M0",
                    endpoint_name="motif",
                    class_order=("x", "y"),
                    probabilities=(m0, 1.0 - m0),
                    observed="x",
                    status="issued",
                    animal_id=animal,
                ),
                PredictionRecord(
                    origin_index=origin,
                    fold_id="outer-000",
                    model_name="M1",
                    endpoint_name="motif",
                    class_order=("x", "y"),
                    probabilities=(m1, 1.0 - m1),
                    observed="x",
                    status="issued",
                    animal_id=animal,
                ),
            ]
        )

    delta = paired_log_loss_delta(
        records,
        baseline_model="M0",
        candidate_model="M1",
        endpoint_name="motif",
    )

    animal_a = -np.log(0.5) - -np.log(0.25)
    animal_b = -np.log(0.25) - -np.log(0.5)
    assert delta.n_matched_predictions == 3
    assert delta.per_animal_delta == {
        "animal-a": animal_a,
        "animal-b": animal_b,
    }
    assert delta.equal_animal_delta_log_loss == np.mean([animal_a, animal_b])
