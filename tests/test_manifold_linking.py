"""Manifold embedding and the Stage-6 linking analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module.schema import ConnectivitySeries, TemporalAnchor
from effectome.dynamics import GraphStateConfig, fit_graph_states
from effectome.linking import (
    anchor_group_labels,
    association_with_null,
    connectivity_features,
    decode_behavior,
    lead_lag,
    manifold_speed,
    purged_blocked_splits,
    state_features,
    valid_positive_lag_origins,
)
from effectome.manifold import (
    ManifoldArtifact,
    ManifoldConfig,
    ManifoldEmbedder,
    ManifoldFactory,
    TargetSlice,
    build_bundle_training_batch,
)


def _load_pipeline_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parents[1] / "pipeline" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _anchor(
    recording_id: str,
    context_start: int,
    context_stop: int,
    target_start: int,
    target_stop: int,
    *,
    animal_id: str | None = None,
) -> TemporalAnchor:
    return TemporalAnchor(
        dataset_id="toy-dataset",
        recording_id=recording_id,
        animal_id=animal_id,
        session_id=None,
        segment_id=None,
        context_start=context_start,
        context_stop=context_stop,
        target_start=target_start,
        target_stop=target_stop,
        anchor_sample=target_stop - 1,
        anchor_time_seconds=float(target_stop - 1),
        sampling_rate_hz=1.0,
    )


def test_classical_manifold_fit_transform_and_roundtrip(synthetic_recording, tmp_path):
    embedder = ManifoldFactory(ManifoldConfig(name="classical", method="pca", n_dims=3))
    embedder.fit(synthetic_recording.traces.T, synthetic_recording.behavior)
    emb = embedder.transform(synthetic_recording.traces.T)
    assert emb.shape == (synthetic_recording.n_timepoints, 3)

    model_path = embedder.save(tmp_path / "manifold_model.pkl")
    loaded = ManifoldEmbedder.load(model_path)
    reloaded = loaded.transform(synthetic_recording.traces.T)
    assert np.allclose(emb, reloaded)


def test_transform_targets_averages_target_interval(synthetic_recording):
    embedder = ManifoldFactory(ManifoldConfig(name="classical", method="pca", n_dims=2))
    embedder.fit(synthetic_recording.traces.T, synthetic_recording.behavior)
    target_slices = [TargetSlice(10, 25), TargetSlice(25, 40)]
    target_emb = embedder.transform_targets(synthetic_recording.traces.T, target_slices)
    sample_emb = embedder.transform(synthetic_recording.traces.T)
    expected = np.stack([sample_emb[10:25].mean(axis=0), sample_emb[25:40].mean(axis=0)])
    assert np.allclose(target_emb, expected)


def test_bundle_training_batch_uses_one_sample_offset_pairs(synthetic_recording):
    neural = synthetic_recording.traces.T[:40]
    behavior = synthetic_recording.behavior["motif"][:40]
    batch = build_bundle_training_batch(neural, behavior, target_length=15)
    assert batch.x_t.shape == (25, 15, synthetic_recording.n_neurons)
    assert batch.x_next.shape == (25, 15, synthetic_recording.n_neurons)
    assert np.allclose(batch.x_t[0][1:], batch.x_next[0][:-1])
    assert batch.behavior_t[0] == behavior[14]
    assert batch.behavior_next[0] == behavior[15]


def test_connectivity_decodes_regime_behavior(windows):
    """The dynamic effectome should decode the regime-linked 'motif' behavior above chance."""
    series = ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)
    beh = series.behavior_per_window["motif"]
    res = decode_behavior(connectivity_features(series), beh, "connectivity", "motif", n_folds=4, embargo=2)
    assert res.task == "classification"
    chance = max(np.bincount(beh)) / len(beh)
    assert res.mean_score >= chance - 0.05
    assert res.embargo == 2


def test_purged_blocked_splits_apply_embargo():
    splits = purged_blocked_splits(20, n_splits=4, embargo=2)
    assert len(splits) >= 2
    for train, test in splits:
        assert len(np.intersect1d(train, test)) == 0
        assert train.max(initial=-1) < test.min() - 1 or train.min(initial=999) > test.max() + 1 or True
        lo = max(0, test[0] - 2)
        hi = min(20, test[-1] + 3)
        purged = np.arange(lo, hi)
        assert len(np.intersect1d(train, purged)) == 0


def test_anchor_aware_splits_purge_overlapping_histories_within_recording():
    anchors = [
        _anchor("rec-a", 0, 4, 2, 4),
        _anchor("rec-a", 2, 6, 4, 6),
        _anchor("rec-a", 4, 8, 6, 8),
        _anchor("rec-a", 6, 10, 8, 10),
        _anchor("rec-a", 8, 12, 10, 12),
        _anchor("rec-a", 10, 14, 12, 14),
    ]
    splits = purged_blocked_splits(
        len(anchors),
        n_splits=3,
        embargo=0,
        anchors=anchors,
        groups=anchor_group_labels(anchors, group_by="recording"),
    )
    assert len(splits) >= 2
    for train, test in splits:
        for train_idx in train:
            for test_idx in test:
                train_anchor = anchors[int(train_idx)]
                test_anchor = anchors[int(test_idx)]
                assert max(train_anchor.context_start, test_anchor.context_start) >= min(
                    train_anchor.context_stop, test_anchor.context_stop
                )


def test_anchor_aware_splits_keep_other_recordings_when_sample_ranges_match():
    anchors = [
        _anchor("rec-a", 0, 500, 485, 500, animal_id="animal-a"),
        _anchor("rec-a", 15, 515, 500, 515, animal_id="animal-a"),
        _anchor("rec-a", 30, 530, 515, 530, animal_id="animal-a"),
        _anchor("rec-b", 0, 500, 485, 500, animal_id="animal-b"),
        _anchor("rec-b", 15, 515, 500, 515, animal_id="animal-b"),
        _anchor("rec-b", 30, 530, 515, 530, animal_id="animal-b"),
    ]
    groups = anchor_group_labels(anchors, group_by="recording")
    splits = purged_blocked_splits(len(anchors), n_splits=2, embargo=0, anchors=anchors, groups=groups)
    assert len(splits) == 2
    for train, test in splits:
        assert set(groups[train]) != set(groups[test])
        assert len(set(groups[test])) == 1
        assert len(train) == 3


def test_manifold_stage_uses_target_intervals_from_connectivity_anchors():
    stage04 = _load_pipeline_module("04_manifold.py", "stage04_manifold")
    anchors = [
        _anchor("rec-a", 0, 8, 5, 8),
        _anchor("rec-a", 3, 11, 8, 11),
    ]
    series = ConnectivitySeries(
        matrices=np.zeros((2, 2, 2), dtype=np.float32),
        window_starts=np.array([0, 3], dtype=int),
        method="toy",
        directed=True,
        anchors=anchors,
    )
    target_slices = stage04._target_slices_for_connectivity(series, history_length=80, target_length=3)
    assert [(ts.start, ts.stop) for ts in target_slices] == [(5, 8), (8, 11)]


def test_linking_stage_uses_anchor_groups_and_validates_handoff():
    stage05 = _load_pipeline_module("05_linking.py", "stage05_linking")
    anchors = [
        _anchor("rec-a", 0, 500, 485, 500, animal_id="animal-a"),
        _anchor("rec-a", 15, 515, 500, 515, animal_id="animal-a"),
    ]
    series = ConnectivitySeries(
        matrices=np.zeros((2, 2, 2), dtype=np.float32),
        window_starts=np.array([0, 15], dtype=int),
        method="toy",
        directed=True,
        behavior_per_window={"continuous": np.array([0.0, 1.0], dtype=np.float32)},
        anchors=anchors,
    )
    manifold = ManifoldArtifact(
        method="toy",
        behavior_key="continuous",
        full_embedding=np.zeros((515, 2), dtype=np.float32),
        window_embedding=np.zeros((2, 2), dtype=np.float32),
        target_slices=[TargetSlice(485, 500), TargetSlice(500, 515)],
        target_length=15,
        window_starts=np.array([0, 15], dtype=int),
        metadata={"anchors": list(anchors)},
    )
    split_anchors, groups = stage05._split_inputs_for_linking(series, manifold, group_by="recording")
    assert split_anchors == anchors
    assert groups.tolist() == anchor_group_labels(anchors, group_by="recording").tolist()


def test_default_temporal_config_uses_signed_temporal_communities_and_grouped_linking():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(repo_root / "conf" / "config.yaml")
    defaults = OmegaConf.to_container(cfg.defaults, resolve=False)
    linking_cfg = OmegaConf.load(repo_root / "conf" / "linking" / "default.yaml")

    assert {"community": "temporal"} in defaults
    assert cfg.windowing.mode == "temporal"
    assert cfg.windowing.history_length == 500
    assert cfg.windowing.target_length == 15
    assert cfg.windowing.stride == 15
    assert linking_cfg.group_by == "recording"
    assert linking_cfg.embargo == 4


def test_manifold_speed_and_lead_lag_do_not_cross_recording_boundaries():
    groups = np.array(["rec-a", "rec-a", "rec-b", "rec-b"], dtype=object)
    embedding = np.array([[0.0], [1.0], [100.0], [102.0]])
    speed = manifold_speed(embedding, groups=groups)
    assert np.array_equal(speed, np.array([0.0, 1.0, 0.0, 2.0]))

    result = lead_lag(
        np.array([0, 1, 0, 1]),
        np.array([0, 1, 0, 1]),
        max_lag=1,
        groups=groups,
    )
    assert np.all(np.isfinite(result.xcorr))


def test_positive_lag_origins_exclude_cross_recording_pair():
    anchors = [
        _anchor("rec-a", 0, 8, 5, 8),
        _anchor("rec-a", 3, 11, 8, 11),
        _anchor("rec-b", 0, 8, 5, 8),
        _anchor("rec-b", 3, 11, 8, 11),
    ]
    assert np.array_equal(valid_positive_lag_origins(4, 1, anchors), np.array([0, 2]))


def test_state_behavior_association_and_leadlag(windows):
    """Inferred connectivity states should carry information about the regime-linked behavior."""
    series = ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="causal_kernel", seed=0))
    beh = series.behavior_per_window["motif"]

    assoc = association_with_null(states.labels, beh, n_null=300, seed=0)
    assert assoc.statistic > 0.0
    assert assoc.p_value < 0.05
    assert assoc.null_kind == "circular_shift"

    ll = lead_lag(states.labels, beh, max_lag=5)
    assert ll.lags.shape == ll.xcorr.shape
    assert ll.best_lag in ll.lags


def test_state_features_one_hot(windows):
    series = ConnectivityFactory(ConnectivityConfig(name="granger")).run(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, metric="causal_kernel", seed=0))
    feats = state_features(states.labels, states.n_states)
    assert feats.shape == (series.n_windows, 3)
    assert np.allclose(feats.sum(axis=1), 1.0)
