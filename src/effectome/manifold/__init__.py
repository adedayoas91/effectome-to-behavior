"""Behavioral-manifold learning (Stage 5): classical, CEBRA, BundDLe-Net."""

from . import bundle_net as _bundle  # noqa: E402,F401
from . import cebra_embed as _cebra  # noqa: E402,F401
from . import classical as _classical  # noqa: E402,F401
from .base import (
    MANIFOLD_REGISTRY,
    ManifoldArtifact,
    ManifoldConfig,
    ManifoldEmbedder,
    ManifoldFactory,
    TargetSlice,
    register_manifold,
)
from .bundle_net import BundleTrainingBatch, build_bundle_training_batch

__all__ = [
    "BundleTrainingBatch",
    "MANIFOLD_REGISTRY",
    "ManifoldArtifact",
    "ManifoldConfig",
    "ManifoldEmbedder",
    "ManifoldFactory",
    "TargetSlice",
    "build_bundle_training_batch",
    "register_manifold",
]
