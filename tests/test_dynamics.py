"""Connectivity-state clustering and Markov transition modeling."""

from __future__ import annotations

import numpy as np

from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.dynamics import (
    GraphStateConfig,
    TransitionConfig,
    fit_graph_states,
    fit_transitions,
)


def _series(windows):
    return ConnectivityFactory(ConnectivityConfig(name="granger", absolute=True)).run(windows)


def test_graph_states_recover_regimes(windows):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, seed=0))
    assert states.labels.shape[0] == series.n_windows
    assert states.centroids.shape == (3, series.n_neurons, series.n_neurons)
    # The synthetic data has contiguous regimes -> states should be temporally blocky,
    # i.e. far fewer switches than windows.
    switches = int(np.sum(np.diff(states.labels) != 0))
    assert switches < series.n_windows // 2


def test_transitions_beat_shuffle_null(windows):
    series = _series(windows)
    states = fit_graph_states(series, GraphStateConfig(n_states=3, seed=0))
    tm = fit_transitions(states.labels, states.n_states, TransitionConfig(n_null=200, seed=0))
    assert tm.transition_matrix.shape == (3, 3)
    assert np.allclose(tm.transition_matrix.sum(axis=1), 1.0)
    # Observed sequence is more structured (higher LL) than time-shuffled surrogates.
    assert tm.p_value < 0.05
