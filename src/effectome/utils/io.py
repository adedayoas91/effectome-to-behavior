"""Lightweight artifact IO so pipeline stages can run independently and cache results."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _artifact_provenance(obj: Any) -> Any | None:
    from effectome.data_module.schema import ArtifactProvenance

    if isinstance(obj, ArtifactProvenance):
        return obj
    provenance = getattr(obj, "provenance", None)
    return provenance if isinstance(provenance, ArtifactProvenance) else None


def validate_loaded_artifact(obj: Any, *, for_confirmation: bool = False) -> Any:
    """Apply schema/claim checks after unpickling, where constructors do not run."""
    from effectome.data_module.schema import (
        CURRENT_ARTIFACT_SCHEMA_VERSION,
        LEGACY_ARTIFACT_SCHEMA_VERSIONS,
        SUPPORTED_ARTIFACT_SCHEMA_VERSIONS,
    )

    provenance = _artifact_provenance(obj)
    if provenance is None:
        if for_confirmation:
            raise ValueError("confirmatory artifacts require typed provenance")
        return obj
    schema_version = str(getattr(provenance, "schema_version", "unknown"))
    if schema_version not in SUPPORTED_ARTIFACT_SCHEMA_VERSIONS:
        raise ValueError(f"unsupported artifact schema version: {schema_version}")
    if schema_version in LEGACY_ARTIFACT_SCHEMA_VERSIONS:
        metadata = dict(getattr(provenance, "metadata", {}))
        metadata["migration_status"] = "legacy_unverified"
        object.__setattr__(provenance, "metadata", metadata)
        object.__setattr__(provenance, "claim_eligibility", "legacy_unverified")
        if for_confirmation:
            raise ValueError("legacy_unverified artifacts are ineligible for confirmation")
    elif schema_version != CURRENT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"artifact schema is not current: {schema_version}")
    if for_confirmation and getattr(provenance, "claim_eligibility", None) != "confirmatory":
        raise ValueError("artifact provenance is not confirmatory-eligible")
    return obj


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_artifact(obj: Any, path: str | Path) -> Path:
    """Persist a Python object (dataclass, dict, arrays) via pickle."""
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved artifact -> %s", p)
    return p


def load_artifact(path: str | Path, *, for_confirmation: bool = False) -> Any:
    """Load an object saved by `save_artifact`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")
    with open(p, "rb") as f:
        return validate_loaded_artifact(pickle.load(f), for_confirmation=for_confirmation)


def save_matrices(matrices: np.ndarray, path: str | Path) -> Path:
    """Persist a stack of connectivity matrices as a compressed .npz."""
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(p, matrices=matrices)
    logger.info("Saved %d matrices -> %s", len(matrices), p)
    return p
