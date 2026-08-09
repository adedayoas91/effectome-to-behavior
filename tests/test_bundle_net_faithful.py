"""Focused tests for the faithful BundDLe-Net port."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.manifold.base import ManifoldConfig, ManifoldEmbedder, TargetSlice
from effectome.manifold.bundle_net import (
    BundDLeManifold,
    _resolved_hyperparameters,
    build_bundle_training_batch,
)


def test_build_bundle_training_batch_preserves_adjacent_shift_contract() -> None:
    neural = np.arange(16 * 3, dtype=np.float32).reshape(16, 3)
    behavior = np.arange(16, dtype=np.int64) % 8

    batch = build_bundle_training_batch(neural, behavior, target_length=15)

    assert batch.x_t.shape == (1, 15, 3)
    assert batch.x_next.shape == (1, 15, 3)
    assert np.array_equal(batch.x_t[0, 1:], batch.x_next[0, :-1])
    assert batch.behavior_t.tolist() == [behavior[14]]
    assert batch.behavior_next.tolist() == [behavior[15]]


def test_resolved_hyperparameters_use_faithful_defaults_without_schema_changes() -> None:
    default_cfg = ManifoldConfig(name="bunddle", behavior_key="motif")
    explicit_cfg = ManifoldConfig(name="bunddle", behavior_key="motif", max_iter=7)
    override_cfg = ManifoldConfig(
        name="bunddle",
        behavior_key="motif",
        max_iter=7,
        extra={"epochs": 5, "batch_size": 11, "gamma": 0.75, "learning_rate": 2.0e-3},
    )

    assert _resolved_hyperparameters(default_cfg)["epochs"] == 3000
    assert _resolved_hyperparameters(explicit_cfg)["epochs"] == 7
    assert _resolved_hyperparameters(override_cfg) == {
        "epochs": 5,
        "batch_size": 11,
        "gamma": 0.75,
        "learning_rate": 2.0e-3,
        "noise_std": 0.05,
        "behavior_classes": 8,
    }


def test_faithful_bunddle_fit_is_deterministic_and_records_spec(tmp_path) -> None:
    pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    neural = rng.normal(size=(40, 4)).astype(np.float32)
    behavior = {"motif": (np.arange(40, dtype=np.int64) % 8)}
    cfg = ManifoldConfig(
        name="bunddle",
        n_dims=2,
        behavior_key="motif",
        max_iter=2,
        target_length=15,
        seed=7,
        extra={"epochs": 2, "batch_size": 4},
    )

    first = BundDLeManifold(cfg).fit(neural, behavior)
    second = BundDLeManifold(cfg).fit(neural, behavior)

    first_embedding = first.transform(neural)
    second_embedding = second.transform(neural)
    assert first_embedding.shape == (26, 2)
    assert np.allclose(first_embedding, second_embedding)
    assert np.allclose(first_embedding, first.transform(neural))

    target_embedding = first.transform_targets(
        neural,
        [TargetSlice(0, 15), TargetSlice(1, 16)],
    )
    assert target_embedding.shape == (2, 2)

    assert first.model_spec["upstream_commit"] == "cbefad93"
    assert first.model_spec["encoder"]["hidden_layers"] == [50, 30, 25, 10]
    assert first.model_spec["encoder"]["preprocess"] == "flatten"
    assert (
        first.model_spec["encoder"]["output"]
        == "linear_latent -> normalization(axis=-1) -> gaussian_noise(train_only)"
    )
    assert first.model_spec["behavior_head"]["n_logits"] == 8
    assert first.model_spec["behavior_head"]["input"] == "z_next"
    assert first.model_spec["transition"]["type"] == "residual_linear_plus_normalization"
    assert first.model_spec["loss"]["gamma"] == pytest.approx(0.9)
    assert "predictor(z_next)" in first.model_spec["loss"]["formula"]
    assert first.fit_provenance["chronological_batches"] is True
    assert len(first.training_history) == 2
    assert first.training_history[0]["samples"] == 25

    model_path = first.save(tmp_path / "bunddle.pkl")
    loaded = ManifoldEmbedder.load(model_path)
    assert np.allclose(first_embedding, loaded.transform(neural))
