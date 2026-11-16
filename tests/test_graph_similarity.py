"""Identity-aware support comparison and topology comparison answer different questions."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from effectome.dynamics.graph_similarity import (
    GRAPH_SIMILARITY_METRICS,
    as_lagged_support,
    directed_graph_characteristics,
    exact_isomorphic,
    graph_similarity_distances,
    lagged_edge_jaccard,
    wl_features,
)


def _graph(edges, n=4, n_lags=1):
    graph = np.zeros((n_lags, n, n), dtype=int)
    for edge in edges:
        if len(edge) == 2:
            graph[0, edge[0], edge[1]] = 1
        else:
            graph[edge[0], edge[1], edge[2]] = 1
    return graph


def test_identity_overlap_separates_isomorphic_graphs_with_different_drivers():
    first = _graph([(0, 2), (2, 3)])
    other = _graph([(1, 2), (2, 3)])
    assert lagged_edge_jaccard(first, other) == pytest.approx(1 / 3)
    assert exact_isomorphic(first, other)
    assert not exact_isomorphic(first, other, node_labels=["a", "b", "c", "d"])
    assert wl_features(first) == wl_features(other)
    assert graph_similarity_distances(np.stack([first, other]), metric="directed_wl")[0, 1] == 0.0


def test_direction_distinguishes_in_star_and_out_star_at_equal_density():
    outward = _graph([(0, 1), (0, 2)])
    inward = outward.transpose(0, 2, 1)
    assert lagged_edge_jaccard(outward, inward) == 0.0
    assert not exact_isomorphic(outward, inward)
    assert wl_features(outward) != wl_features(inward)
    assert graph_similarity_distances(np.stack([outward, inward]), metric="directed_wl")[0, 1] > 0
    first, other = (directed_graph_characteristics(graph)["per_lag"][0] for graph in [outward, inward])
    assert first["density"] == other["density"]
    assert first["out_degree"] == other["in_degree"]
    assert first["triad_census"]["021D"] == other["triad_census"]["021U"] == 1


def test_lag_swaps_survive_even_when_collapsed_graph_is_identical():
    first = _graph([(0, 0, 1), (1, 1, 2)], n_lags=2)
    other = first[::-1].copy()
    np.testing.assert_array_equal(first.sum(axis=0), other.sum(axis=0))
    assert lagged_edge_jaccard(first, other) == 0.0
    assert wl_features(first, lag_labels=[0.1, 0.2]) != wl_features(other, lag_labels=[0.1, 0.2])
    assert not exact_isomorphic(first, other, lag_labels=[0.1, 0.2])
    assert graph_similarity_distances(
        np.stack([first, other]), metric="directed_wl", lag_labels=[0.1, 0.2]
    )[0, 1] > 0


def test_exact_matching_preserves_parallel_lag_edges():
    parallel = _graph([(0, 0, 1), (1, 0, 1)], n_lags=2)
    reciprocal = _graph([(0, 0, 1), (1, 1, 0)], n_lags=2)
    assert not exact_isomorphic(parallel, reciprocal, lag_labels=["fast", "slow"])
    assert exact_isomorphic(parallel, parallel.transpose(0, 2, 1), lag_labels=["fast", "slow"])


def test_wl_collision_does_not_become_an_isomorphism_certificate():
    cycle = _graph([(i, (i + 1) % 6) for i in range(6)], n=6)
    triangles = _graph([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)], n=6)
    assert wl_features(cycle, iterations=5) == wl_features(triangles, iterations=5)
    assert graph_similarity_distances(np.stack([cycle, triangles]), metric="directed_wl")[0, 1] == 0.0
    assert not exact_isomorphic(cycle, triangles)
    assert directed_graph_characteristics(cycle)["per_lag"][0]["triad_census"]["030C"] == 0
    assert directed_graph_characteristics(triangles)["per_lag"][0]["triad_census"]["030C"] == 2


def test_support_excludes_diagonal_without_changing_input():
    original = np.eye(3, dtype=float)
    original[0, 1] = 1
    saved = original.copy()
    support = as_lagged_support(original)
    assert support.shape == (1, 3, 3)
    assert support.dtype == bool
    assert support.sum() == 1
    support[0, 0, 1] = False
    np.testing.assert_array_equal(original, saved)
    assert lagged_edge_jaccard(original, np.eye(3)) == 0.0
    assert wl_features(original) == wl_features(_graph([(0, 1)], n=3))


@pytest.mark.parametrize("metric", GRAPH_SIMILARITY_METRICS)
def test_empty_graph_convention_is_separate_from_undefined_overlap(metric):
    empty = _graph([])
    loop_only = np.eye(4, dtype=int)[None]
    edge = _graph([(0, 1)])
    assert lagged_edge_jaccard(empty, loop_only) is None
    graphs = np.stack([empty, loop_only, edge])
    distances = graph_similarity_distances(graphs, metric=metric)
    assert distances[0, 1] == 0.0
    assert distances[0, 2] > 0
    np.testing.assert_array_equal(np.diag(distances), 0.0)
    assert exact_isomorphic(empty, loop_only)
    row = directed_graph_characteristics(empty)["per_lag"][0]
    assert row["reciprocity"] is None
    assert row["density"] == 0.0


def test_permutations_keep_topology_and_relabelled_node_attributes_consistent():
    graph = _graph([(0, 0, 1), (0, 0, 2), (1, 2, 3), (1, 3, 1)], n_lags=2)
    permutation = np.array([3, 1, 0, 2])
    changed = graph[:, permutation][:, :, permutation]
    assert lagged_edge_jaccard(graph, changed) < 1
    assert wl_features(graph, iterations=3) == wl_features(changed, iterations=3)
    labels = np.array(["motor", "sensory", "motor", "interneuron"])
    assert wl_features(graph, node_labels=labels) == wl_features(changed, node_labels=labels[permutation])
    assert wl_features(graph) == wl_features(graph.copy())
    assert isinstance(wl_features(graph), Counter)
    assert sum(wl_features(graph, iterations=3).values()) == 4 * 4


@pytest.mark.parametrize("metric", GRAPH_SIMILARITY_METRICS)
def test_pairwise_and_reference_assignment_are_symmetric_finite_and_batch_independent(metric):
    graphs = np.stack([
        _graph([]), _graph([(0, 1)]), _graph([(0, 1), (0, 2)]),
        _graph([(1, 0), (2, 0)]), _graph([(0, 1), (1, 2), (2, 0)]),
    ])
    distances = graph_similarity_distances(graphs, metric=metric)
    assert np.isfinite(distances).all()
    assert np.all(distances >= 0)
    np.testing.assert_allclose(distances, distances.T, atol=1e-12)
    np.testing.assert_array_equal(np.diag(distances), 0.0)
    subset = graph_similarity_distances(graphs[3:], graphs[:2], metric=metric)
    np.testing.assert_allclose(subset, distances[3:, :2], atol=1e-12)
    reordered = graph_similarity_distances(graphs[::-1], metric=metric)
    np.testing.assert_allclose(reordered, distances[::-1, ::-1], atol=1e-12)
    if metric == "directed_wl":
        kernel = 1 - distances**2 / 2
        np.testing.assert_allclose(np.diag(kernel), 1)
        assert np.linalg.eigvalsh(kernel).min() > -1e-10
    else:
        assert np.max(distances) <= 1


def test_characteristics_are_json_safe_directed_and_per_lag():
    graph = _graph([(0, 0, 1), (0, 1, 0), (1, 0, 2)], n_lags=2)
    description = directed_graph_characteristics(graph, lag_labels=[np.float64(0.1), np.float64(0.2)])
    json.dumps(description, allow_nan=False)
    assert description["n_edges"] == 3
    assert description["n_nodes"] == 4
    assert description["n_lags"] == 2
    assert description["per_lag"][0]["lag_label"] == 0.1
    assert description["per_lag"][0]["reciprocity"] == 1.0
    assert description["per_lag"][1]["reciprocity"] == 0.0
    assert description["per_lag"][1]["out_degree"] == [1, 0, 0, 0]
    assert description["per_lag"][1]["in_degree"] == [0, 0, 1, 0]
    assert "lag_union" not in description
    assert directed_graph_characteristics(np.ones((1, 1)))["per_lag"][0]["density"] == 0


def test_empty_collections_keep_the_expected_distance_shape():
    graphs = np.empty((0, 1, 4, 4))
    reference = _graph([(0, 1)])[None]
    for metric in GRAPH_SIMILARITY_METRICS:
        assert graph_similarity_distances(graphs, reference, metric=metric).shape == (0, 1)
        assert graph_similarity_distances(reference, graphs, metric=metric).shape == (1, 0)


@pytest.mark.parametrize("graph", [
    np.zeros(3), np.zeros((2, 3)), np.zeros((1, 2, 3)), np.zeros((0, 3, 3)),
    np.zeros((1, 0, 0)), np.zeros((1, 1, 2, 2)), np.full((2, 2), 0.1),
    np.full((2, 2), -1), np.full((2, 2), 2), np.full((2, 2), np.nan),
    np.full((2, 2), np.inf), np.full((2, 2), "1"), np.ones((2, 2), dtype=complex),
])
def test_invalid_or_nonbinary_support_is_rejected(graph):
    with pytest.raises(ValueError):
        as_lagged_support(graph)


@pytest.mark.parametrize("labels", [[0.1], [0.1, 0.1], [0.1, np.nan], [0.1, []], "ab"])
def test_invalid_lag_labels_are_rejected_consistently(labels):
    graph = _graph([(0, 0, 1)], n_lags=2)
    for function in [wl_features, exact_isomorphic, directed_graph_characteristics]:
        with pytest.raises(ValueError):
            if function is exact_isomorphic:
                function(graph, graph, lag_labels=labels)
            else:
                function(graph, lag_labels=labels)
    with pytest.raises(ValueError):
        graph_similarity_distances(graph[None], metric="edge_jaccard", lag_labels=labels)


@pytest.mark.parametrize("iterations", [-1, 0.5, True, np.nan, "2"])
def test_invalid_iteration_counts_are_rejected(iterations):
    graph = _graph([(0, 1)])
    with pytest.raises(ValueError):
        wl_features(graph, iterations=iterations)
    with pytest.raises(ValueError):
        graph_similarity_distances(graph[None], metric="directed_wl", wl_iterations=iterations)


@pytest.mark.parametrize("labels", [["a"], ["a", "b", "c", np.nan], "abcd", ["a", "b", "c", {}]])
def test_invalid_node_labels_are_rejected(labels):
    graph = _graph([(0, 1)])
    with pytest.raises(ValueError):
        wl_features(graph, node_labels=labels)
    with pytest.raises(ValueError):
        exact_isomorphic(graph, graph, node_labels=labels)


def test_identity_distance_rejects_mismatched_axes_and_unknown_metrics():
    graph = _graph([(0, 1)])
    other = _graph([(0, 1)], n=3)
    with pytest.raises(ValueError, match="matching"):
        lagged_edge_jaccard(graph, other)
    with pytest.raises(ValueError, match="matching"):
        graph_similarity_distances(graph[None], other[None], metric="edge_jaccard")
    with pytest.raises(ValueError, match="unknown"):
        graph_similarity_distances(graph[None], metric="made_up")
    with pytest.raises(ValueError, match="collection"):
        graph_similarity_distances(graph, metric="edge_jaccard")
    # The topology kernel can compare different node counts without implying
    # correspondence between the underlying biological cells.
    assert graph_similarity_distances(graph[None], other[None], metric="directed_wl").shape == (1, 1)
