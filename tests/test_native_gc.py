"""Native GC support transport is separate from optional predictive refits."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from effectome.connectivity import (
    CONNECTIVITY_REGISTRY,
    ConnectivityConfig,
    ConnectivityFactory,
    UnsupportedForecastError,
    native_gc,
)
from effectome.connectivity import _native_gc_worker as worker
from effectome.linking.transition_synthesis import graph_transition_features


@pytest.fixture
def upstream_fast(monkeypatch):
    calls = []

    class ImportedFastGcStar:
        def __init__(self, **kwargs):
            self.options = kwargs
            calls.append(self)

        def fit(self, data, **kwargs):
            self.data = data.copy()
            self.thresholds = kwargs
            n = data.shape[0]
            self.lag_adjacency_ = np.zeros((2, n, n), dtype=bool)
            self.lag_adjacency_[0, 1, 0] = True  # tau0 must never leak into export
            self.lag_adjacency_[1, 0, 1] = True
            self.lag_adjacency_[1, 0, 0] = True  # self-edges are outside effectome export
            self.marginal_pvalue_ = np.full((2 * n, n), 0.02)
            self.conditional_pvalue_ = np.full((2 * n, n), 0.03)
            self.backend_metadata_ = {"test": "external-test-double"}
            return self

        def get_connectivity_matrix(self, **kwargs):
            self.export_options = kwargs
            # Scores can be signed: native decisions, rather than score>0, are exported.
            return -0.7 * self.lag_adjacency_[1]

    module = SimpleNamespace(
        __name__="fastCGC", __file__="/external/fastCGC/__init__.py",
        FastGcStar=ImportedFastGcStar,
    )
    monkeypatch.setattr(native_gc.importlib, "import_module", lambda name: module)
    return calls


def test_four_native_methods_are_registered():
    assert {
        "fast_cgc", "fast_cgc_star", "conventional_cgc", "conventional_cgc_star",
    } <= set(CONNECTIVITY_REGISTRY)


@pytest.mark.parametrize(("name", "method"), [("fast_cgc", "cgc"), ("fast_cgc_star", "fcgc")])
def test_fast_delegates_and_exports_only_native_positive_lag_support(name, method, windows, upstream_fast):
    estimator = ConnectivityFactory(ConnectivityConfig(
        name=name, alpha=0.1, extra={"n_perm": 0, "beta": 0.05, "seed": 7},
    ))
    series = estimator.run(windows)
    assert len(upstream_fast) == windows.n_windows
    first = upstream_fast[0]
    assert first.options["method"] == method
    assert first.options["n_pasts"] == first.options["n_lags"] == 1
    assert first.options["evaluation_policy"] == "full"
    assert first.options["random_state"] == 7
    assert first.options["dependence_profile"] == "linear"
    np.testing.assert_array_equal(first.data, windows.segments[0].T)
    assert first.thresholds["alpha"] == first.export_options["alpha"] == 0.1
    assert first.thresholds["beta"] == first.export_options["beta"] == 0.05
    assert first.export_options["simulation"] is True
    assert np.all(series.matrices[:, 0, 1] == 1)
    assert np.all(series.matrices[:, 1, 0] == 0)
    assert np.isin(series.matrices, [0, 1]).all()
    assert np.all(np.diagonal(series.matrices, axis1=1, axis2=2) == 0)
    np.testing.assert_array_equal(series.lagged_matrices[:, 0], series.matrices)
    assert not series.weighted and not series.signed
    assert series.weight_semantics == "binary_upstream_causal_support"
    assert series.diagnostics["lag_resolved"][0]["native_evidence"]["conditional_pvalue_"][0][0] == 0.03
    assert series.diagnostics["upstream_provenance"]["module"] == "fastCGC"


def test_native_support_has_no_neural_forecast_or_projected_drive(windows, upstream_fast):
    estimator = ConnectivityFactory(ConnectivityConfig(name="fast_cgc", alpha=0.1, extra={"beta": 0.05}))
    series = estimator.run(windows)
    assert estimator.capabilities.independent_window
    assert not estimator.capabilities.neural_forecast
    with pytest.raises(UnsupportedForecastError):
        estimator.forecast(None, horizon_samples=1)
    features, _, diagnostics = graph_transition_features(
        series, windows, scaled_segments=windows.segments,
        pca_components=np.eye(windows.n_neurons),
        neuron_ids=[f"n{i}" for i in range(windows.n_neurons)],
        fit_indices=np.arange(windows.n_windows), evaluation_indices=np.array([], dtype=int),
    )
    assert np.isfinite(features).all()
    assert diagnostics["drive_supported"] is False
    assert diagnostics["projected_cross_neuron_drive"] is None


@pytest.mark.parametrize("name", ["fast_cgc", "conventional_cgc"])
def test_native_common_lag_contract_is_explicit(name):
    estimator = ConnectivityFactory(ConnectivityConfig(name=name, max_lag=2))
    with pytest.raises(ValueError, match="max_lag=1"):
        estimator.estimate(np.ones((40, 2)))


def test_missing_conventional_runtime_is_an_explicit_failure(tmp_path):
    estimator = ConnectivityFactory(ConnectivityConfig(
        name="conventional_cgc", extra={"runtime_python": str(tmp_path / "missing-python")},
    ))
    with pytest.raises(FileNotFoundError, match="no fast-method fallback"):
        estimator.estimate(np.ones((40, 2)))


def test_conventional_requires_actual_upstream_permutations():
    estimator = ConnectivityFactory(ConnectivityConfig(name="conventional_cgc", extra={"n_perm": 0}))
    with pytest.raises(ValueError, match="requires n_perm>=1"):
        estimator.estimate(np.ones((256, 2)))


def test_fast_rejects_unattainable_complete_shift_threshold_before_fitting(upstream_fast):
    estimator = ConnectivityFactory(ConnectivityConfig(
        name="fast_cgc", alpha=0.01, extra={"beta": 0.005},
    ))
    with pytest.raises(ValueError, match="attainable complete-shift p-value floor"):
        estimator.estimate(np.ones((128, 2)))
    assert not upstream_fast


@pytest.mark.parametrize("persistent", [False, True])
def test_conventional_sequence_uses_one_subprocess_and_preserves_evidence(
    tmp_path, monkeypatch, windows, persistent,
):
    runtime = tmp_path / "python"
    runtime.touch()
    calls = []
    inline_backend = "module://matplotlib_inline.backend_inline"
    monkeypatch.setenv("MPLBACKEND", inline_backend)

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["MPLBACKEND"] == "Agg"
        assert kwargs["env"]["MPLCONFIGDIR"].startswith(str(Path(command[2]).parent))
        with np.load(command[2], allow_pickle=False) as payload:
            segments = payload["segments"]
            options = json.loads(str(payload["options"]))
        assert options["method"] == "cgc_star"
        assert options["upstream_module"] == "causalized_gc"
        matrices = np.zeros((len(segments), segments.shape[2], segments.shape[2]))
        matrices[:, 0, 1] = 1
        np.savez(command[3], matrices=matrices, evidence=np.array([[0.01]]))
        Path(command[4]).write_text(json.dumps({
            "upstream_provenance": {"module": "causalized_gc", "version": "official"},
            "windows": [{"evidence_keys": {"pVal_corr_": "evidence"}} for _ in segments],
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(native_gc.subprocess, "run", run)
    if persistent:
        class Runtime:
            def run(self, executable, script, source, output, metadata_path, environment):
                run([str(p) for p in (executable, script, source, output, metadata_path)], env=environment)

        monkeypatch.setattr(native_gc, "current_conventional_runtime", lambda: Runtime())
    estimator = ConnectivityFactory(ConnectivityConfig(
        name="conventional_cgc_star", extra={"runtime_python": str(runtime), "n_perm": 100},
    ))
    series = estimator.run(windows)
    assert len(calls) == 1
    assert os.environ["MPLBACKEND"] == inline_backend
    assert series.diagnostics["lag_resolved"][0]["native_evidence"]["pVal_corr_"] == [[0.01]]
    assert series.diagnostics["upstream_provenance"]["module"] == "causalized_gc"
    assert np.all(series.matrices[:, 0, 1] == 1)


def test_standalone_worker_delegates_to_official_cpu_graph(tmp_path, monkeypatch):
    instances = []

    class ImportedGcStar:
        def __init__(self, **kwargs):
            self.options = kwargs
            instances.append(self)

        def fit(self, data, verbose=0):
            self.data = data
            self.pVal_corr_ = np.ones((4, 2))
            return self

        def get_connectivity_matrix(self, **kwargs):
            self.export_options = kwargs
            return np.array([[0, 0.7], [0, 0]])

    monkeypatch.setattr(worker.importlib, "import_module", lambda name: SimpleNamespace(
        __file__="/official/causalized_gc/__init__.py", GcStar=ImportedGcStar,
    ))
    monkeypatch.setattr(worker.metadata, "distribution", lambda name: SimpleNamespace(
        version="0.1.0", read_text=lambda name: '{"url":"https://official.example/source"}',
    ))
    source, output, metadata_path = [tmp_path / name for name in ("in.npz", "out.npz", "out.json")]
    segments = np.arange(160).reshape(2, 40, 2)
    np.savez(source, segments=segments, options=np.asarray(json.dumps({
        "upstream_module": "causalized_gc", "seed": 5, "n_perm": 100,
        "method": "cgc_star", "verbose": 0, "alpha": 0.01, "beta": 0.005,
    })))
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(source), str(output), str(metadata_path)])
    worker.main()
    assert len(instances) == 2
    assert instances[0].options["backend"] == "cpu"
    assert instances[0].options["method"] == "cgc_star"
    assert instances[0].export_options["simulation"] is True
    np.testing.assert_array_equal(instances[0].data, segments[0].T)
    with np.load(output, allow_pickle=False) as payload:
        assert np.all(payload["matrices"][:, 0, 1] == 1)
        assert np.all(payload["matrices"][:, 1, 0] == 0)
        assert "window_0__pVal_corr_" in payload
    assert "not_controlled" in json.loads(metadata_path.read_text())["windows"][0]["seed_policy"]
