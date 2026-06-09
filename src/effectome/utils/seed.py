"""Reproducibility helpers."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and (if available) Torch for reproducible runs.

    Args:
        seed: Random seed shared across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not available; skipping torch seeding")

    logger.info("Global seed set to %d", seed)
