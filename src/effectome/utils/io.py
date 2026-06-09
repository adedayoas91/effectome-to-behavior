"""Lightweight artifact IO so pipeline stages can run independently and cache results."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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


def load_artifact(path: str | Path) -> Any:
    """Load an object saved by `save_artifact`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)


def save_matrices(matrices: np.ndarray, path: str | Path) -> Path:
    """Persist a stack of connectivity matrices as a compressed .npz."""
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(p, matrices=matrices)
    logger.info("Saved %d matrices -> %s", len(matrices), p)
    return p
