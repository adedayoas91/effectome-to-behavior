"""Native binary decisions must not be scored as dynamical coefficients."""

from types import SimpleNamespace

import numpy as np

from effectome.experiments.framework_validation import _recovery


def test_binary_recovery_has_support_metrics_without_coefficient_error():
    weights = np.array([[0., -.6], [.4, 0.]])
    recording = SimpleNamespace(metadata={
        "true_regime": np.zeros(4, int), "true_coefficients": weights[None],
    })
    windows = SimpleNamespace(n_neurons=2, anchors=[SimpleNamespace(context_start=0, context_stop=4,
                                                                   anchor_sample=3)] * 2)
    comparison = {"events": {"evaluation_rows": [0, 1], "origins": [0, 1]},
                  "reference": {"scaler_scale": np.ones(2)}}
    result = {"native_graph_states": {"status": "unavailable"}, "directed": True,
              "lagged_graph_matrices": np.repeat((weights != 0)[None, None], 2, axis=0),
              "weight_semantics": "binary_upstream_causal_support"}
    row = {}
    _recovery(row, result, comparison, recording, windows, {"calcium_decay": 0, "observation_noise_std": 0})
    assert row["support_precision"] == row["support_recall"] == 1
    assert "graph_relative_frobenius_error" not in row
    assert row["graph_recovery_scope"] == "pure_history_structural_lag_support"
