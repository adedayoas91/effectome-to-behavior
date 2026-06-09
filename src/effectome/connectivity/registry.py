"""Registry + factory for connectivity estimators.

Kept separate from `__init__` so estimator modules can import the registry without a cycle.
"""

from __future__ import annotations

from .base import ConnectivityConfig, ConnectivityEstimator

CONNECTIVITY_REGISTRY: dict[str, type[ConnectivityEstimator]] = {}


def register_connectivity(name: str):
    """Class decorator registering a `ConnectivityEstimator` under `name`."""

    def deco(cls: type[ConnectivityEstimator]) -> type[ConnectivityEstimator]:
        if name in CONNECTIVITY_REGISTRY:
            raise ValueError(f"connectivity '{name}' already registered")
        CONNECTIVITY_REGISTRY[name] = cls
        return cls

    return deco


def ConnectivityFactory(cfg: ConnectivityConfig) -> ConnectivityEstimator:
    """Instantiate the estimator selected by `cfg.name`."""
    if cfg.name not in CONNECTIVITY_REGISTRY:
        raise KeyError(
            f"unknown connectivity '{cfg.name}'; available: {sorted(CONNECTIVITY_REGISTRY)}"
        )
    return CONNECTIVITY_REGISTRY[cfg.name](cfg)
