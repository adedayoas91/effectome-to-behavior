"""Comparable graph features and descriptive synthesis on shared transition events."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import rankdata

from effectome.data_module.schema import ConnectivitySeries, WindowedSegments
from effectome.linking.decoding import anchor_continuity_labels

METHOD_FAMILIES = {
    "cgc": "gc",
    "cgc_star": "gc",
    "correlation_partial": "association",
    "sparse_var": "sparse_var",
    "windowed_sparse_var": "sparse_var",
    "switching_var": "switching_var",
    "tvvar_smooth": "tvvar",
    "tvvar_fused": "tvvar",
    "fast_cgc": "gc",
    "fast_cgc_star": "gc",
    "conventional_cgc": "gc",
    "conventional_cgc_star": "gc",
}


def graph_transition_features(
    series: ConnectivitySeries,
    windows: WindowedSegments,
    *,
    scaled_segments: np.ndarray,
    pca_components: np.ndarray,
    neuron_ids: Sequence[str],
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    groups: Mapping[str, Sequence[str]] | None = None,
) -> tuple[np.ndarray, dict[str, list[int]], dict[str, Any]]:
    """Preserve signed lag profiles, causal changes, and supported outgoing drives.

    Directed node omissions remove outgoing features. Undirected omissions remove
    all incident edges, represented once. Group omissions are unions of these
    blocks; overlap is recorded and is not interpreted as unique causal credit.
    """
    n_windows, _, n_neurons = windows.segments.shape
    if len(neuron_ids) != n_neurons or len(set(neuron_ids)) != n_neurons:
        raise ValueError("neuron IDs must be unique and match the neuron axis")
    if series.matrices.shape != (n_windows, n_neurons, n_neurons):
        raise ValueError("graph series must align with raw windows")
    if series.anchors != windows.anchors:
        raise ValueError("graph series must carry the same temporal anchors")
    if np.asarray(scaled_segments).shape != windows.segments.shape:
        raise ValueError("scaled histories must align with raw windows")
    lagged = (
        series.matrices[:, None].astype(float)
        if series.lagged_matrices is None
        else np.asarray(series.lagged_matrices, dtype=float)
    )
    mask = (
        ~np.eye(n_neurons, dtype=bool)
        if series.directed
        else np.triu(np.ones((n_neurons, n_neurons), dtype=bool), 1)
    )
    sources, targets = np.where(mask)
    edges = lagged[:, :, sources, targets].reshape(n_windows, -1)
    edge_sources = np.tile(sources, lagged.shape[1])
    edge_targets = np.tile(targets, lagged.shape[1])
    ownership = np.zeros(n_windows, dtype=int)
    ownership[fit_indices] = 1
    ownership[evaluation_indices] = 2
    continuity = anchor_continuity_labels(windows.anchors, group_by="segment")
    changes = np.zeros_like(edges)
    for index in range(1, n_windows):
        if (
            ownership[index] != 0
            and ownership[index] == ownership[index - 1]
            and continuity[index] == continuity[index - 1]
            and windows.anchors[index].valid
            and windows.anchors[index - 1].valid
        ):
            changes[index] = edges[index] - edges[index - 1]
    features = np.concatenate([edges, changes], axis=1)
    edge_width = edges.shape[1]
    blocks: dict[str, list[int]] = {}
    for index, neuron in enumerate(neuron_ids):
        selected = edge_sources == index
        if not series.directed:
            selected |= edge_targets == index
        columns = np.flatnonzero(selected)
        blocks[f"neuron:{neuron}"] = np.concatenate([columns, columns + edge_width]).tolist()

    # Coefficients parameterize a one-step population update in common scaled
    # input units. This cross-neuron contribution excludes self-history/intercepts
    # and is not a complete neural forecast or an intervention.
    coefficient_semantics = {
        "signed_regularized_var_coefficient",
        "signed_sparse_var_coefficient",
        "signed_tvvar_endpoint_coefficient",
        "switching_var_preobservation_mixture_coefficient",
    }
    drive_supported = (
        series.directed
        and series.lagged_matrices is not None
        and series.weight_semantics in coefficient_semantics
    )
    projected: np.ndarray | None = None
    if drive_supported:
        if lagged.shape[1] > scaled_segments.shape[1]:
            raise ValueError("lag count exceeds history length")
        components = np.asarray(pca_components, dtype=float)
        if components.ndim != 2 or components.shape[1] != n_neurons:
            raise ValueError("PCA components must align with the neuron axis")
        weights = lagged.copy()
        weights[:, :, np.arange(n_neurons), np.arange(n_neurons)] = 0
        activities = np.stack([scaled_segments[:, -lag, :] for lag in range(1, lagged.shape[1] + 1)], axis=1)
        drive = np.einsum("kli,klij->kij", activities, weights)
        projected = np.einsum("kij,dj->kid", drive, components)
        offset = features.shape[1]
        features = np.concatenate([features, projected.reshape(n_windows, -1)], axis=1)
        for index, neuron in enumerate(neuron_ids):
            start = offset + index * components.shape[0]
            blocks[f"neuron:{neuron}"].extend(range(start, start + components.shape[0]))
    for name, members in (groups or {}).items():
        if not members or any(str(member) not in neuron_ids for member in members):
            raise ValueError(f"group {name!r} must contain known neuron IDs")
        blocks[f"group:{name}"] = sorted(
            {column for member in members for column in blocks[f"neuron:{member}"]}
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("graph features contain nonfinite values")
    return (
        features,
        blocks,
        {
            "representation": (
                "signed_lag_edges_and_causal_changes" if series.signed else "unsigned_lag_support_and_changes"
            ),
            "neuron_block_semantics": "outgoing" if series.directed else "incident_undirected",
            "drive_supported": drive_supported,
            "projected_cross_neuron_drive": projected,
            "drive_scope": "one_step_cross_neuron_update_not_full_forecast_or_intervention",
            "n_features": features.shape[1],
        },
    )


def conditional_outcome_diversity(
    graph_states: np.ndarray,
    manifold_states: np.ndarray,
    destinations: np.ndarray,
    displacements: np.ndarray,
) -> list[dict[str, Any]]:
    """Describe future diversity within the SAME graph AND current manifold state.

    Conditioning on the current region avoids attributing different starting
    population states to ambiguous circuit function. No significance is implied.
    """
    rows = []
    for graph in np.unique(graph_states):
        for manifold in np.unique(manifold_states):
            selected = (graph_states == graph) & (manifold_states == manifold)
            count = int(selected.sum())
            if count == 0:
                continue
            labels, counts = np.unique(destinations[selected], return_counts=True)
            probabilities = counts / count
            rows.append(
                {
                    "graph_state": int(graph),
                    "current_manifold_state": int(manifold),
                    "n_origins": count,
                    "destinations": labels.tolist(),
                    "counts": counts.tolist(),
                    "destination_entropy_nats": float(-np.sum(probabilities * np.log(probabilities))),
                    "displacement_variance": np.var(displacements[selected], axis=0).tolist(),
                    "interpretation": "descriptive_conditional_diversity_not_circuit_degeneracy_proof",
                }
            )
    return rows


def family_rank_synthesis(
    method_results: Mapping[str, Mapping[str, Any]],
    families: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Average within-method ranks within families first, with leave-family-out checks.

    Rankings are descriptive. They do not combine p-values or treat methods as
    biological replicates; negative predictive contributions remain visible.
    """
    family_names = {**METHOD_FAMILIES, **dict(families or {})}
    per_transition: dict[tuple[int, int], dict[str, list[tuple[str, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for method, result in method_results.items():
        if result.get("status") != "available":
            continue
        records: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
        for row in result["prediction"]["omission_scores"]:
            records[(row["source_state"], row["destination_state"])].append(row)
        for transition, values in records.items():
            gains = np.asarray([row["omission_delta_log_loss"] for row in values], dtype=float)
            # Equal scores receive the same midrank; neuron and group ranks are
            # kept separate so differing group counts cannot change node ranks.
            for kind in ("neuron:", "group:"):
                selected = [i for i, row in enumerate(values) if row["unit"].startswith(kind)]
                if not selected:
                    continue
                ranks = rankdata(gains[selected], method="average") / len(selected)
                for index, rank in zip(selected, ranks, strict=True):
                    row = values[index]
                    per_transition[transition][row["unit"]].append((method, float(rank), float(gains[index])))
    output = []
    for (source, target), units in sorted(per_transition.items()):
        for unit, contributions in sorted(units.items()):
            by_family: dict[str, list[float]] = defaultdict(list)
            for method, rank, _ in contributions:
                by_family[family_names.get(method, method)].append(rank)
            scores = {family: float(np.mean(ranks)) for family, ranks in by_family.items()}
            output.append(
                {
                    "unit": unit,
                    "source_state": source,
                    "destination_state": target,
                    "family_mean_percentile": float(np.mean(list(scores.values()))),
                    "family_scores": scores,
                    "leave_family_out": {
                        family: float(np.mean([score for other, score in scores.items() if other != family]))
                        for family in scores
                        if len(scores) > 1
                    },
                    "n_methods": len(contributions),
                    "n_families": len(scores),
                    "n_positive_contributions": sum(gain > 0 for _, _, gain in contributions),
                    "method_contributions": {method: gain for method, _, gain in contributions},
                    "biological_necessity_identified": False,
                }
            )
    return output
