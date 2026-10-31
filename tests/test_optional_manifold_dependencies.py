"""Requested manifold methods must retain their identity when extras are absent."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from effectome.manifold.base import ManifoldConfig
from effectome.manifold.bundle_net import _load_torch
from effectome.manifold.classical import ClassicalManifold


def _block_import(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.split(".")[0] == module:
            raise ModuleNotFoundError(f"No module named '{module}'", name=module)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)


def test_missing_umap_does_not_silently_fit_pca(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_import(monkeypatch, "umap")
    model = ClassicalManifold(ManifoldConfig(method="umap", n_dims=2))
    neural = np.random.default_rng(1).normal(size=(20, 4))

    with pytest.raises(ImportError, match="umap-learn"):
        model.fit(neural, {})

    with pytest.raises(RuntimeError, match="fitted"):
        model.transform(neural)


def test_pca_runs_without_optional_manifold_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_import(monkeypatch, "umap")
    _block_import(monkeypatch, "torch")
    neural = np.random.default_rng(2).normal(size=(20, 4))
    model = ClassicalManifold(ManifoldConfig(method="pca", n_dims=2))

    transformed = model.fit(neural, {}).transform(neural)

    assert transformed.shape == (20, 2)
    assert np.isfinite(transformed).all()


def test_missing_torch_identifies_bundle_extra_and_supported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_import(monkeypatch, "torch")

    with pytest.raises(ImportError, match=r"\.\[bundle\]") as caught:
        _load_torch()

    assert "Intel macOS" in str(caught.value)
    assert "PCA" in str(caught.value)
