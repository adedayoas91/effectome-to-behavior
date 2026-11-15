"""Cross-method comparisons must share targets, preserve ownership and report failures."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from effectome.connectivity import ConnectivityConfig
from effectome.data_module import PreprocessConfig, WindowConfig, make_windows, preprocess
from effectome.data_module.schema import ConnectivitySeries, NeuralRecording
from effectome.linking.transition_synthesis import family_rank_synthesis, graph_transition_features
from effectome.workflows.cross_method import (
    CrossMethodConfig,
    load_cross_method_profile,
    run_cross_method_comparison,
    write_cross_method_report,
)
from effectome.workflows.feature_ownership import chronological_feature_split


def _windows():
    rng = np.random.default_rng(184)
    values = rng.normal(size=(3, 720))
    values[1, 1:] += 0.5 * values[0, :-1]
    recording = NeuralRecording(values, np.arange(720, dtype=float), None, np.arange(3), {}, 1.0)
    return make_windows(
        preprocess(recording, PreprocessConfig()),
        WindowConfig(
            mode="temporal",
            history_length=32,
            target_length=4,
            stride=8,
            standardize_per_window=False,
        ),
    )


def _methods():
    return {
        "correlation_partial": ConnectivityConfig(name="correlation", partial=True),
        "sparse_var": ConnectivityConfig(name="sparse_var", max_lag=1, extra={"n_lags": 1, "lambda2": 0.1}),
    }


def test_methods_share_events_baseline_and_empirical_transitions(tmp_path):
    report = run_cross_method_comparison(
        _windows(),
        CrossMethodConfig(n_dims=2),
        method_configs=_methods(),
        neuron_ids=["a", "b", "c"],
        groups={"pair": ["a", "b"]},
    )
    assert all(value["status"] == "available" for value in report["methods"].values())
    common = report["common_baseline"]
    for result in report["methods"].values():
        np.testing.assert_allclose(
            result["prediction"]["baseline_probabilities"], common["baseline_probabilities"]
        )
        assert (
            result["prediction"]["transition_matrices"]["empirical"]
            == common["transition_matrices"]["empirical"]
        )
    constant = report["methods"]["sparse_var"]["native_graph_states"]
    assert constant["n_states"] == 1
    assert constant["constant_graph_one_state"]
    assert not report["methods"]["correlation_partial"]["features"]["drive_supported"]
    assert "group:pair" in {row["unit"] for row in report["family_rank_synthesis"]}
    paths = write_cross_method_report(report, tmp_path)
    assert all(path.exists() for path in paths)
    payload = json.loads(paths[0].read_text())
    assert payload["status"] == "development_only"
    assert len(payload["events"]["recording_ids"]) == len(payload["events"]["origins"])
    assert "origin_time_seconds" in paths[2].read_text().splitlines()[0]
    assert payload["reference"]["metadata"]["region_definition"] == "training_kmeans_geometric"
    assert not payload["scientific_claims"]["biological_necessity_identified"]


def test_heldout_values_cannot_change_shared_fit_or_training_graph():
    windows = _windows()
    fit, evaluation = chronological_feature_split(windows, evaluation_fraction=0.25)
    first = run_cross_method_comparison(
        windows,
        CrossMethodConfig(n_dims=2),
        method_configs=_methods(),
        fit_indices=fit,
        evaluation_indices=evaluation,
    )
    changed = deepcopy(windows)
    changed.segments[evaluation] *= 40
    second = run_cross_method_comparison(
        changed,
        CrossMethodConfig(n_dims=2),
        method_configs=_methods(),
        fit_indices=fit,
        evaluation_indices=evaluation,
    )
    for field in ("scaler_mean", "scaler_scale", "pca_components", "state_centers"):
        np.testing.assert_allclose(first["reference"][field], second["reference"][field])
    for method in first["methods"]:
        assert (
            first["methods"][method]["training_graph_fingerprint"]
            == second["methods"][method]["training_graph_fingerprint"]
        )


def test_method_failure_is_explicit_without_changing_shared_events():
    configs = {**_methods(), "missing_backend": ConnectivityConfig(name="does_not_exist")}
    report = run_cross_method_comparison(_windows(), CrossMethodConfig(n_dims=2), method_configs=configs)
    assert report["methods"]["missing_backend"]["status"] == "unavailable"
    assert report["methods"]["sparse_var"]["status"] == "available"
    assert len(report["events"]["evaluation_rows"]) > 0


def test_historical_profiles_keep_eight_and_declare_smoke_settings():
    cfg, methods = load_cross_method_profile("smoke")
    assert len(methods) == 8
    assert methods["correlation_partial"].partial
    assert methods["correlation_partial"].name == "correlation"
    assert methods["cgc"].extra["beta"] == 0.05
    assert cfg.n_dims == 2
    with pytest.raises(ValueError, match="profile"):
        load_cross_method_profile("unknown")
    assert len(load_cross_method_profile("development")[1]) == 8


def test_csl_profile_selects_four_upstream_support_methods_in_one_family():
    from effectome.linking.transition_synthesis import METHOD_FAMILIES

    _, methods = load_cross_method_profile("csl_sanity")
    assert set(methods) == {"conventional_cgc", "conventional_cgc_star", "fast_cgc", "fast_cgc_star"}
    assert {METHOD_FAMILIES[name] for name in methods} == {"gc"}
    assert all(cfg.max_lag == 1 and cfg.max_lag_seconds is None for cfg in methods.values())
    assert all(cfg.alpha == 0.01 and cfg.extra["beta"] == 0.005 for cfg in methods.values())
    assert methods["fast_cgc"].extra["n_perm"] == 0
    assert methods["conventional_cgc"].extra["n_perm"] == 1000


def test_shared_baseline_receives_all_method_drive_lags():
    methods = _methods()
    methods["sparse_var"] = replace(methods["sparse_var"], max_lag=4, extra={"n_lags": 4})
    report = run_cross_method_comparison(_windows(), CrossMethodConfig(n_dims=2), method_configs=methods)
    names = report["reference"]["metadata"]["baseline_feature_names"]
    assert "neural_history_lag_3_0" in names
    assert report["reference"]["metadata"]["baseline_neural_history_samples"] == 4


@pytest.mark.parametrize(
    ("semantics", "signed", "representation"),
    [
        ("dependence_score", True, "signed_lag_edges_and_causal_changes"),
        ("binary_upstream_causal_support", False, "unsigned_lag_support_and_changes"),
    ],
)
def test_directed_dependence_statistics_never_become_dynamical_drives(semantics, signed, representation):
    windows = _windows()
    n = windows.n_windows
    matrices = np.ones((n, 3, 3))
    matrices[:, np.arange(3), np.arange(3)] = 0
    series = ConnectivitySeries(
        matrices,
        np.asarray([w.start for w in windows.windows]),
        "future_csl",
        True,
        anchors=windows.anchors,
        lagged_matrices=matrices[:, None],
        weight_semantics=semantics,
        signed=signed,
    )
    fit, evaluation = chronological_feature_split(windows)
    _, _, metadata = graph_transition_features(
        series,
        windows,
        scaled_segments=windows.segments,
        pca_components=np.eye(3)[:2],
        neuron_ids=["a", "b", "c"],
        fit_indices=fit,
        evaluation_indices=evaluation,
    )
    assert not metadata["drive_supported"]
    assert metadata["projected_cross_neuron_drive"] is None
    assert metadata["representation"] == representation


def test_outgoing_omission_and_undirected_incident_blocks_preserve_sign():
    windows = _windows()
    n = windows.n_windows
    matrix = np.tile(np.array([[0, -2, 0], [-2, 0, 3], [0, 3, 0]], dtype=float), (n, 1, 1))
    series = ConnectivitySeries(
        matrix, np.asarray([w.start for w in windows.windows]), "correlation", False, anchors=windows.anchors
    )
    fit, evaluation = chronological_feature_split(windows)
    features, blocks, metadata = graph_transition_features(
        series,
        windows,
        scaled_segments=windows.segments,
        pca_components=np.eye(3)[:2],
        neuron_ids=["a", "b", "c"],
        fit_indices=fit,
        evaluation_indices=evaluation,
        groups={"ab": ["a", "b"]},
    )
    assert not metadata["drive_supported"]
    assert features[0, 0] == -2
    assert blocks["neuron:b"] == [0, 2, 3, 5]
    assert set(blocks["group:ab"]) == set(blocks["neuron:a"]) | set(blocks["neuron:b"])
    with pytest.raises(ValueError, match="anchors"):
        graph_transition_features(
            replace(series, anchors=[]),
            windows,
            scaled_segments=windows.segments,
            pca_components=np.eye(3),
            neuron_ids=["a", "b", "c"],
            fit_indices=fit,
            evaluation_indices=evaluation,
        )


def test_family_ranks_weight_related_methods_as_one_family():
    def result(a, b):
        return {
            "status": "available",
            "prediction": {
                "omission_scores": [
                    {
                        "unit": "neuron:a",
                        "source_state": 0,
                        "destination_state": 1,
                        "omission_delta_log_loss": a,
                    },
                    {
                        "unit": "neuron:b",
                        "source_state": 0,
                        "destination_state": 1,
                        "omission_delta_log_loss": b,
                    },
                ]
            },
        }

    rows = family_rank_synthesis(
        {"cgc": result(1, 0), "cgc_star": result(1, 0), "tvvar_smooth": result(0, 1)}
    )
    a, b = rows
    assert a["n_methods"] == 3 and a["n_families"] == 2
    assert a["family_mean_percentile"] == b["family_mean_percentile"]
    assert len(a["leave_family_out"]) == 2


@pytest.mark.parametrize("kwargs", [{"decoder_c": 0}, {"evaluation_fraction": 1}, {"positive_lag": 0}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        CrossMethodConfig(**kwargs)
