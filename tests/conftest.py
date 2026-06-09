"""Shared pytest fixtures built on the synthetic ground-truth generator."""

from __future__ import annotations

import numpy as np
import pytest

from effectome.data_module import WindowConfig, make_windows
from effectome.utils import SyntheticConfig, make_synthetic_recording, set_seed


@pytest.fixture(scope="session")
def synthetic_recording():
    set_seed(0)
    cfg = SyntheticConfig(
        n_neurons=15, n_timepoints=1800, n_states=3, regime_dwell=120,
        coupling=0.5, noise_std=0.6, behavior_drivers=3, seed=0,
    )
    rec = make_synthetic_recording(cfg)
    rec.validate()
    return rec


@pytest.fixture(scope="session")
def windows(synthetic_recording):
    return make_windows(synthetic_recording, WindowConfig(length=80, stride=20, mode="sliding"))


@pytest.fixture(scope="session")
def true_graph(synthetic_recording):
    # Union of per-regime ground-truth edges (source->target convention, matching estimators).
    graphs = synthetic_recording.metadata["true_graphs"]  # (S, N, N), [source, target]
    adj = (np.abs(graphs).max(axis=0) > 1e-6).astype(int)
    np.fill_diagonal(adj, 0)
    return adj
