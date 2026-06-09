"""Behavioral-manifold learning (Stage 5): classical, CEBRA, BundDLe-Net.

Importing this package registers all shipped embedders.
"""

from . import bundle_net as _bundle  # noqa: E402,F401
from . import cebra_embed as _cebra  # noqa: E402,F401

# Import side-effect: register embedders.
from . import classical as _classical  # noqa: E402,F401
from .base import (
    MANIFOLD_REGISTRY,
    ManifoldConfig,
    ManifoldEmbedder,
    ManifoldFactory,
    register_manifold,
)

__all__ = [
    "ManifoldConfig",
    "ManifoldEmbedder",
    "ManifoldFactory",
    "register_manifold",
    "MANIFOLD_REGISTRY",
]
