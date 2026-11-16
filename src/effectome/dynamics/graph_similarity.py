"""Compare binary directed effectomes while keeping identity and topology distinct.

The Jaccard view preserves source, target and lag coordinates. The topology view
uses a directed, edge-labelled adaptation of the Weisfeiler--Lehman subtree
kernel of Shervashidze et al. (2011), https://jmlr.org/papers/v12/shervashidze11a.html.
Incoming and outgoing neighbourhoods are separate; lag labels are edge labels.
This adaptation is a structural comparison, not a causal-identification result.

WL feature keys use deterministic SHA256 colours, independent of other graphs
or a fitted vocabulary. A matching feature vector is not proof of isomorphism:
WL has structural collisions, in addition to the theoretical hash-collision
possibility. Exact diagnostic matching uses NetworkX's directed multigraph VF2:
https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html.

Self links are excluded throughout. No threshold, sign or weight is inferred.
Only finite binary support is accepted. Node registration, physical units, data
ownership and training-only group selection belong to the calling workflow.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from math import sqrt
from typing import Any

import networkx as nx
import numpy as np

GRAPH_SIMILARITY_METRICS = ("edge_jaccard", "directed_wl")


def as_lagged_support(graph: np.ndarray) -> np.ndarray:
    """Return a copied Boolean ``(lag, source, target)`` off-diagonal support.

    A single ``(source, target)`` graph receives a length-one lag axis. Values
    must already be zero or one, including diagonal values before exclusion.
    """
    array = np.asarray(graph)
    if array.ndim == 2:
        array = array[None, :, :]
    if (array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0
            or array.shape[1] != array.shape[2]):
        raise ValueError("support must have shape (node,node) or (lag>=1,node>=1,node)")
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("support must contain finite binary zero/one values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("support must contain binary zero/one values; threshold weights explicitly")
    support = np.array(array, dtype=bool, copy=True)
    diagonal = np.arange(support.shape[-1])
    support[:, diagonal, diagonal] = False
    return support


def lagged_edge_jaccard(a: np.ndarray, b: np.ndarray) -> float | None:
    """Directed, lag-specific edge overlap; two empty supports return ``None``."""
    first, second = as_lagged_support(a), as_lagged_support(b)
    if first.shape != second.shape:
        raise ValueError("identity-preserving Jaccard requires matching lag and node axes")
    union = int(np.count_nonzero(first | second))
    return float(np.count_nonzero(first & second) / union) if union else None


def _iterations(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("WL iterations must be a nonnegative integer")
    return int(value)


def _primitive_label(value: Any) -> str | int | float | bool:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        if not value:
            raise ValueError("graph labels must be nonempty strings or finite numeric categories")
        return value
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and np.isfinite(value):
        return value
    raise ValueError("graph labels must be nonempty strings or finite numeric categories")


def _label_keys(
    labels: Sequence[Any] | None,
    count: int,
    *,
    name: str,
    unique: bool = False,
) -> tuple[list[str], list[str | int | float | bool]]:
    if labels is None:
        values: list[str | int | float | bool] = list(range(count))
    else:
        if isinstance(labels, (str, bytes)):
            raise ValueError(f"{name} must have one label per coordinate")
        try:
            if len(labels) != count:
                raise ValueError(f"{name} must have one label per coordinate")
            values = [_primitive_label(value) for value in labels]
        except TypeError as exc:
            raise ValueError(f"{name} must have one label per coordinate") from exc
    if unique and len(set(values)) != count:
        raise ValueError(f"{name} must be unique")
    keys = [
        json.dumps([type(value).__name__, value], ensure_ascii=True, separators=(",", ":"))
        for value in values
    ]
    return keys, values


def _colour(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def wl_features(
    graph: np.ndarray,
    *,
    iterations: int = 2,
    lag_labels: Sequence[Any] | None = None,
    node_labels: Sequence[Any] | None = None,
) -> Counter[str]:
    """Count directed WL colours at levels zero through ``iterations``.

    Without node labels all neurons start with one colour, making this view
    invariant to simultaneous neuron permutations. Optional labels may be
    repeated cell types or unique neuron identities; callers choose the policy.
    Lag labels default to lag-axis indices. Labels are typed finite categorical
    values and are never fitted jointly across the compared graphs.
    """
    support = as_lagged_support(graph)
    count = _iterations(iterations)
    n_lags, n_nodes, _ = support.shape
    edge_labels, _ = _label_keys(lag_labels, n_lags, name="lag_labels", unique=True)
    if node_labels is None:
        colours = [_colour(["unlabelled_node"])] * n_nodes
    else:
        labels, _ = _label_keys(node_labels, n_nodes, name="node_labels")
        colours = [_colour(["node_label", label]) for label in labels]
    incoming: list[list[tuple[str, int]]] = [[] for _ in range(n_nodes)]
    outgoing: list[list[tuple[str, int]]] = [[] for _ in range(n_nodes)]
    for lag, source, target in np.argwhere(support):
        outgoing[int(source)].append((edge_labels[int(lag)], int(target)))
        incoming[int(target)].append((edge_labels[int(lag)], int(source)))
    features: Counter[str] = Counter()
    for level in range(count + 1):
        features.update(f"{level}:{colour}" for colour in colours)
        if level < count:
            colours = [
                _colour([
                    colours[node],
                    sorted((label, colours[other]) for label, other in incoming[node]),
                    sorted((label, colours[other]) for label, other in outgoing[node]),
                ])
                for node in range(n_nodes)
            ]
    return features


def _support_collection(graphs: np.ndarray) -> np.ndarray:
    array = np.asarray(graphs)
    if (array.ndim != 4 or array.shape[1] == 0 or array.shape[2] == 0
            or array.shape[2] != array.shape[3]):
        raise ValueError("graph collection must have shape (graph,lag>=1,node>=1,node)")
    if len(array) == 0:
        return np.empty(array.shape, dtype=bool)
    return np.stack([as_lagged_support(graph) for graph in array])


def _wl_distance(first: Counter[str], second: Counter[str]) -> float:
    if first == second:
        return 0.0
    dot = sum(value * second.get(key, 0) for key, value in first.items())
    first_norm = sum(value * value for value in first.values())
    second_norm = sum(value * value for value in second.values())
    similarity = dot / sqrt(first_norm * second_norm)
    return sqrt(max(0.0, min(2.0, 2.0 - 2.0 * similarity)))


def graph_similarity_distances(
    graphs: np.ndarray,
    references: np.ndarray | None = None,
    *,
    metric: str,
    wl_iterations: int = 2,
    lag_labels: Sequence[Any] | None = None,
) -> np.ndarray:
    """Compare ``(graph,lag,node,node)`` collections without fitting coordinates.

    Returns ``(n_graphs,n_references)`` distances; omitted references compare a
    collection with itself. Jaccard uses ``1-J`` and declares empty/empty
    distance zero solely as a grouping convention. The corresponding overlap
    remains undefined in :func:`lagged_edge_jaccard`.

    WL uses ``sqrt(2 - 2 * normalized_kernel)``. Different node counts are
    permitted for WL comparisons, but do not confer biological registration.
    """
    if metric not in GRAPH_SIMILARITY_METRICS:
        raise ValueError(f"unknown graph similarity metric {metric!r}; choices: {GRAPH_SIMILARITY_METRICS}")
    _iterations(wl_iterations)
    first = _support_collection(graphs)
    second = first if references is None else _support_collection(references)
    _label_keys(lag_labels, first.shape[1], name="lag_labels", unique=True)
    _label_keys(lag_labels, second.shape[1], name="lag_labels", unique=True)
    if metric == "edge_jaccard":
        if first.shape[1:] != second.shape[1:]:
            raise ValueError("identity-preserving Jaccard requires matching lag and node axes")
        n_features = int(np.prod(first.shape[1:]))
        flat = first.reshape(len(first), n_features).astype(np.int64)
        other = second.reshape(len(second), n_features).astype(np.int64)
        intersection = flat @ other.T
        union = flat.sum(axis=1)[:, None] + other.sum(axis=1)[None, :] - intersection
        similarity = np.divide(
            intersection, union, out=np.ones(union.shape, dtype=np.float64), where=union != 0
        )
        return 1.0 - similarity
    first_features = [wl_features(graph, iterations=wl_iterations, lag_labels=lag_labels) for graph in first]
    second_features = first_features if references is None else [
        wl_features(graph, iterations=wl_iterations, lag_labels=lag_labels) for graph in second
    ]
    distance = np.empty((len(first), len(second)), dtype=np.float64)
    for row, features in enumerate(first_features):
        for column, reference in enumerate(second_features):
            distance[row, column] = _wl_distance(features, reference)
    return distance


def _networkx_graph(
    support: np.ndarray,
    *,
    lag_labels: Sequence[Any] | None,
    node_labels: Sequence[Any] | None,
) -> nx.MultiDiGraph:
    labels, _ = _label_keys(lag_labels, len(support), name="lag_labels", unique=True)
    graph = nx.MultiDiGraph()
    node_keys, _ = _label_keys(node_labels, support.shape[1], name="node_labels")
    for node in range(support.shape[1]):
        graph.add_node(node, label=node_keys[node] if node_labels is not None else "unlabelled")
    for lag, source, target in np.argwhere(support):
        graph.add_edge(int(source), int(target), lag_label=labels[int(lag)])
    return graph


def exact_isomorphic(
    a: np.ndarray,
    b: np.ndarray,
    *,
    lag_labels: Sequence[Any] | None = None,
    node_labels: Sequence[Any] | None = None,
) -> bool:
    """Test directed lag-labelled isomorphism, optionally constraining neuron labels.

    A supplied label sequence describes the node-axis policy in both inputs;
    unique labels therefore require neuron-identity-preserving matching.
    """
    first, second = as_lagged_support(a), as_lagged_support(b)
    graph = _networkx_graph(first, lag_labels=lag_labels, node_labels=node_labels)
    reference = _networkx_graph(second, lag_labels=lag_labels, node_labels=node_labels)
    return bool(nx.is_isomorphic(
        graph,
        reference,
        node_match=nx.algorithms.isomorphism.categorical_node_match("label", "unlabelled"),
        edge_match=nx.algorithms.isomorphism.categorical_multiedge_match("lag_label", None),
    ))


def directed_graph_characteristics(
    graph: np.ndarray,
    *,
    lag_labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """JSON-safe directed descriptors for each lag, without collapsing lag axes.

    Reciprocity is the fraction of directed edges with a reverse edge at the
    same lag, and is undefined for an empty lag graph. Degree sequences retain
    the supplied node order. Motif counts are induced directed triad censuses.
    """
    support = as_lagged_support(graph)
    n_lags, n_nodes, _ = support.shape
    _, labels = _label_keys(lag_labels, n_lags, name="lag_labels", unique=True)
    rows = []
    for label, matrix in zip(labels, support, strict=True):
        n_edges = int(matrix.sum())
        directed = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        rows.append({
            "lag_label": label,
            "n_edges": n_edges,
            "density": float(n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0,
            "in_degree": matrix.sum(axis=0).astype(int).tolist(),
            "out_degree": matrix.sum(axis=1).astype(int).tolist(),
            "reciprocity": float((matrix & matrix.T).sum() / n_edges) if n_edges else None,
            "triad_census": {key: int(value) for key, value in nx.triadic_census(directed).items()},
        })
    return {"n_nodes": n_nodes, "n_lags": n_lags, "n_edges": int(support.sum()), "per_lag": rows}
