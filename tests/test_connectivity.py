"""Connectivity estimators: interface + ground-truth recovery."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from effectome.connectivity import (
    CONNECTIVITY_REGISTRY,
    ConnectivityConfig,
    ConnectivityEstimator,
    ConnectivityFactory,
)
from effectome.core import CausalisedGC
from effectome.data_module.schema import TemporalAnchor, Window, WindowedSegments
from effectome.experiments import analyze_with_cgc, analyze_with_cgc_star


def test_registry_has_methods():
    assert {"correlation", "granger", "granger_star", "pcmci", "jpcmci", "time_varying"} <= set(
        CONNECTIVITY_REGISTRY
    )


def test_correlation_runs(windows):
    est = ConnectivityFactory(ConnectivityConfig(name="correlation", partial=True))
    series = est.run(windows)
    assert series.matrices.shape == (windows.n_windows, windows.n_neurons, windows.n_neurons)
    assert not series.directed
    assert series.weighted
    assert series.signed
    assert series.weight_semantics == "functional_association"
    assert len(series.anchors) == windows.n_windows
    assert series.diagnostics["window_mode"] == windows.metadata.get("window_mode")


def test_granger_recovers_ground_truth(windows, true_graph):
    """Multivariate Granger should rank true edges above non-edges (AUROC > 0.6)."""
    est = ConnectivityFactory(ConnectivityConfig(name="granger", max_lag=1, ridge=1.0))
    series = est.run(windows)
    mean_influence = series.matrices.mean(axis=0)

    n = mean_influence.shape[0]
    off = ~np.eye(n, dtype=bool)
    scores = np.abs(mean_influence[off])
    labels = true_graph[off]
    auroc = roc_auc_score(labels, scores)
    assert auroc > 0.6, f"Granger AUROC too low: {auroc:.3f}"
    assert series.weight_semantics == "signed_regularized_var_coefficient"
    assert series.diagnostics["estimation_mode"] == "rolling_window"
    assert series.diagnostics["support_kind"] == "logical_and_of_marginal_and_conditional_evidence"
    assert series.diagnostics["support_description"] == (
        "intersection_of_unconditional_and_conditional_evidence"
    )
    assert series.diagnostics["support_variant"] == "cgc"
    assert series.diagnostics["primary_tau_policy"] == "lagged_only_tau_ge_1"
    assert len(series.diagnostics["lag_resolved"]) == windows.n_windows
    assert series.lagged_matrices is not None
    assert series.lagged_matrices.shape[:2] == (windows.n_windows, 1)


def test_granger_preserves_sign(windows):
    est = ConnectivityFactory(ConnectivityConfig(name="granger", max_lag=1, ridge=1.0))
    series = est.run(windows)
    mats = series.matrices[:, ~np.eye(series.n_neurons, dtype=bool)]
    assert np.any(mats > 0.0)
    assert np.any(mats < 0.0)


def test_sequence_level_estimator_hook_runs_once_for_full_series(windows):
    class DummySequenceEstimator(ConnectivityEstimator):
        directed = True
        estimation_mode = "sequence_regularized"

        def estimate(self, segment: np.ndarray) -> np.ndarray:  # pragma: no cover - should not run
            raise AssertionError("estimate should not be called when estimate_sequence is overridden")

        def estimate_sequence(self, segments):
            n = segments.n_neurons
            base = np.eye(n, dtype=np.float32)
            return np.repeat(base[None, :, :], segments.n_windows, axis=0)

    est = DummySequenceEstimator(ConnectivityConfig(name="dummy"))
    series = est.run(windows)

    assert series.matrices.shape == (windows.n_windows, windows.n_neurons, windows.n_neurons)
    assert series.diagnostics["estimation_mode"] == "sequence_regularized"
    assert np.allclose(series.matrices[:, np.arange(windows.n_neurons), np.arange(windows.n_neurons)], 0.0)


def test_time_varying_candidate_resets_regularization_at_recording_boundary():
    x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    positive = np.column_stack([x, x])
    negative = np.column_stack([x, -x])
    segments = np.stack([positive, positive, negative, negative])
    anchors = [
        TemporalAnchor(
            dataset_id="ds",
            recording_id="rec-a" if idx < 2 else "rec-b",
            animal_id=None,
            session_id=None,
            segment_id=None,
            context_start=(idx % 2) * 2,
            context_stop=(idx % 2) * 2 + 8,
            target_start=(idx % 2) * 2 + 7,
            target_stop=(idx % 2) * 2 + 8,
            anchor_sample=(idx % 2) * 2 + 7,
            anchor_time_seconds=float(idx),
            sampling_rate_hz=1.0,
        )
        for idx in range(4)
    ]
    windows = WindowedSegments(
        segments=segments,
        windows=[Window(anchor.context_start, anchor.context_stop) for anchor in anchors],
        behavior_per_window={},
        n_neurons=2,
        fps=1.0,
        anchors=anchors,
    )
    estimator = ConnectivityFactory(
        ConnectivityConfig(
            name="time_varying",
            extra={
                "base_estimator": "correlation",
                "temporal_lambda": 100.0,
            },
        )
    )
    series = estimator.run(windows)

    assert np.all(series.matrices[:2, 0, 1] > 0.99)
    assert np.all(series.matrices[2:, 0, 1] < -0.99)
    assert series.diagnostics["continuous_segments"] == 2
    assert series.diagnostics["candidate_status"] == "candidate_not_promoted"


def test_notebook_cgc_adapters_return_weighted_matrices(synthetic_recording):
    X = synthetic_recording.traces.T
    cgc = analyze_with_cgc(X, [1, 2])
    cgc_star = analyze_with_cgc_star(X, [1])
    assert set(cgc) == {1, 2}
    assert set(cgc_star) == {1}
    assert cgc[1].shape == (X.shape[1], X.shape[1])
    assert cgc_star[1].shape == (X.shape[1], X.shape[1])


def test_granger_analytic_support_can_disable_permutations(windows):
    estimator = ConnectivityFactory(
        ConnectivityConfig(name="granger", max_lag=1, extra={"n_perm": 0, "support_test": "analytic"})
    )
    series = estimator.run(windows)
    assert series.diagnostics["lag_resolved"][0]["support_test"] == "analytic"


def test_granger_permutation_support_requires_positive_n_perm(windows):
    estimator = ConnectivityFactory(
        ConnectivityConfig(
            name="granger",
            max_lag=1,
            extra={"n_perm": 0, "support_test": "circular_shift"},
        )
    )
    with np.testing.assert_raises_regex(ValueError, "n_perm must be positive"):
        estimator.run(windows)


def test_circular_shift_threshold_respects_monte_carlo_resolution():
    data = np.random.default_rng(0).normal(size=(3, 40))
    estimator = CausalisedGC(
        n_perm=9,
        n_pasts=1,
        support_test="circular_shift",
        seed=0,
    ).fit(data)
    with np.testing.assert_raises_regex(ValueError, "p-value resolution"):
        estimator.get_connectivity_matrix(alpha=0.05, beta=0.05)
