"""Evaluate two meanings of graph similarity on controlled directed graphs.

These are planted graph archetypes, not estimated neural effectomes. Quantizers
see training graphs only; labels enter held-out scoring only. Identity-sensitive
and topology-sensitive targets intentionally differ. Noise preserves edge count,
so graph density alone cannot solve the four-archetype task.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score

from effectome.data_module.schema import ConnectivitySeries
from effectome.dynamics.graph_similarity import (
    GRAPH_SIMILARITY_METRICS,
    directed_graph_characteristics,
    exact_isomorphic,
    graph_similarity_distances,
)
from effectome.dynamics.graph_states import GraphStateConfig, fit_graph_states
from effectome.workflows.cross_method import _json_safe
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

logger = logging.getLogger(__name__)
_ARCHETYPES = ("chain_A", "chain_B", "outward_star", "inward_star")
_TOPOLOGY_CLASSES = np.array([0, 0, 1, 2], dtype=int)


@dataclass(frozen=True)
class GraphSimilarityValidationConfig:
    """Declared graph-sampling and fixed-codebook settings; no label tuning."""

    profile: str = "smoke"
    seeds: tuple[int, ...] = (0, 1, 2)
    train_repeats: int = 6
    evaluation_repeats: int = 6
    rewiring_probabilities: tuple[float, ...] = (0.0, 0.25)
    candidate_counts: tuple[int, ...] = (2, 3, 4)
    metrics: tuple[str, ...] = ("edge_jaccard", "directed_wl")
    wl_iterations: int = 2
    physical_lags: tuple[float, ...] = (0.1, 0.2)

    def __post_init__(self) -> None:
        if self.profile not in {"smoke", "development"}:
            raise ValueError("graph similarity profile must be smoke or development")
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(seed) is not int or seed < 0 for seed in self.seeds)):
            raise ValueError("seeds must be distinct nonnegative integers")
        if any(type(value) is not int or value < 2
               for value in (self.train_repeats, self.evaluation_repeats)):
            raise ValueError("each archetype needs at least two training and held-out copies")
        if (not self.rewiring_probabilities
                or len(set(self.rewiring_probabilities)) != len(self.rewiring_probabilities)
                or any(isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1
                       for value in self.rewiring_probabilities)):
            raise ValueError("rewiring probabilities must be distinct finite values in [0,1]")
        if (not self.candidate_counts or len(set(self.candidate_counts)) != len(self.candidate_counts)
                or any(type(count) is not int or not 1 <= count <= 4 for count in self.candidate_counts)):
            raise ValueError("candidate counts must be distinct integers in [1,4]")
        if (not self.metrics or len(set(self.metrics)) != len(self.metrics)
                or set(self.metrics) - set(GRAPH_SIMILARITY_METRICS)):
            raise ValueError("metrics must be a distinct nonempty subset of supported graph similarities")
        if type(self.wl_iterations) is not int or self.wl_iterations < 1:
            raise ValueError("wl_iterations must be a positive integer")
        lags = np.asarray(self.physical_lags, dtype=float)
        if (lags.shape != (2,) or not np.isfinite(lags).all()
                or np.any(lags <= 0) or np.any(np.diff(lags) <= 0)):
            raise ValueError("the lag diagnostic needs exactly two increasing positive physical lags")


def load_graph_similarity_profile(profile: str) -> GraphSimilarityValidationConfig:
    """Load the versioned graph-only workload from its YAML profile."""
    from omegaconf import OmegaConf

    if profile not in {"smoke", "development"}:
        raise ValueError("unknown graph similarity profile")
    path = Path(__file__).resolve().parents[3] / "conf/graph_similarity" / f"{profile}.yaml"
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("graph similarity profile must be a mapping")
    values = {str(key): value for key, value in raw.items()}
    for field in ("seeds", "rewiring_probabilities", "candidate_counts", "metrics", "physical_lags"):
        values[field] = tuple(values[field])
    return GraphSimilarityValidationConfig(**values)


@dataclass(frozen=True)
class PlantedGraphPanel:
    """Graph input and separately held evaluation labels."""

    series: ConnectivitySeries
    identity_labels: np.ndarray
    topology_labels: np.ndarray
    fit_indices: np.ndarray
    evaluation_indices: np.ndarray


def planted_graph_archetypes() -> np.ndarray:
    """Four six-node, two-edge directed graphs, all at the first physical lag."""
    graphs = np.zeros((4, 2, 6, 6), dtype=bool)
    for index, edges in enumerate((
        ((0, 1), (1, 2)), ((3, 4), (4, 5)),
        ((0, 1), (0, 2)), ((1, 0), (2, 0)),
    )):
        for source, target in edges:
            graphs[index, 0, source, target] = True
    return graphs


def _rewire_one_edge(graph: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rewired = graph.copy()
    lag, source, target = np.argwhere(rewired)[rng.integers(int(rewired.sum()))]
    absent = np.argwhere(~rewired[lag] & ~np.eye(rewired.shape[-1], dtype=bool))
    new_source, new_target = absent[rng.integers(len(absent))]
    rewired[lag, source, target] = False
    rewired[lag, new_source, new_target] = True
    return rewired


def make_planted_graph_panel(
    config: GraphSimilarityValidationConfig,
    *,
    seed: int,
    rewiring_probability: float,
) -> PlantedGraphPanel:
    """Draw train/holdout corruption independently without changing graph density."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not np.isfinite(rewiring_probability) or not 0 <= rewiring_probability <= 1:
        raise ValueError("rewiring_probability must lie in [0,1]")
    templates = planted_graph_archetypes()
    parts, labels = [], []
    for part, repeats in enumerate((config.train_repeats, config.evaluation_repeats)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, part]))
        identities = np.repeat(np.arange(len(templates)), repeats)
        rng.shuffle(identities)
        replace_edge = rng.random(len(identities)) < rewiring_probability
        candidates = [_rewire_one_edge(templates[index], rng) for index in identities]
        graphs = np.stack([
            candidate if corrupted else templates[index].copy()
            for index, candidate, corrupted in zip(identities, candidates, replace_edge, strict=True)
        ])
        parts.append(graphs)
        labels.append(identities)
    tensor = np.concatenate(parts)
    identity = np.concatenate(labels)
    split = len(parts[0])
    series = ConnectivitySeries(
        matrices=tensor.max(axis=1).astype(float), lagged_matrices=tensor.astype(float),
        window_starts=np.arange(len(tensor)), method="planted_graph_archetypes_not_neural_estimation",
        directed=True, signed=False, weighted=False, weight_semantics="binary_support",
        diagnostics={
            "node_ids": [f"planted_node_{index}" for index in range(6)],
            "physical_lags": list(config.physical_lags), "representation_units": "binary_support",
            "scope": "planted_graph_structure_only_no_activity_no_CSL_refitting",
        },
    )
    return PlantedGraphPanel(
        series=series, identity_labels=identity, topology_labels=_TOPOLOGY_CLASSES[identity],
        fit_indices=np.arange(split), evaluation_indices=np.arange(split, len(tensor)),
    )


def graph_similarity_inventory(config: GraphSimilarityValidationConfig) -> dict[str, Any]:
    """Return exact bounded workload counts without fitting models."""
    panels = len(config.seeds) * len(config.rewiring_probabilities)
    return {
        "profile": config.profile, "graph_sampling_seeds": list(config.seeds), "panels": panels,
        "graphs_per_panel": 4 * (config.train_repeats + config.evaluation_repeats),
        "training_graphs_per_panel": 4 * config.train_repeats,
        "heldout_graphs_per_panel": 4 * config.evaluation_repeats,
        "planned_quantizer_fits": panels * len(config.metrics) * len(config.candidate_counts),
        "metrics": list(config.metrics), "fixed_state_counts": list(config.candidate_counts),
        "neural_connectivity_fits": 0,
        "scope": "planted_graph_structure_diagnostic_not_driver_or_biological_validation",
    }


def graph_similarity_probes(config: GraphSimilarityValidationConfig) -> list[dict[str, Any]]:
    """Retain known invariances, lag sensitivity, and an explicit WL collision."""
    graphs = planted_graph_archetypes()
    lag_two = np.zeros_like(graphs[0])
    lag_two[1] = graphs[0, 0]
    cycle = np.zeros_like(graphs[0])
    triangles = np.zeros_like(graphs[0])
    for node in range(6):
        cycle[0, node, (node + 1) % 6] = True
    for offset in (0, 3):
        for node in range(3):
            triangles[0, node + offset, (node + 1) % 3 + offset] = True
    empty = np.zeros_like(graphs[0])
    pairs = (
        ("same_topology_different_neurons", graphs[0], graphs[1]),
        ("outward_versus_inward_star", graphs[2], graphs[3]),
        ("same_edges_different_physical_lag", graphs[0], lag_two),
        ("six_cycle_versus_two_triangles_WL_collision", cycle, triangles),
        ("both_empty_no_positive_recovery_evidence", empty, empty),
    )
    rows = []
    for name, first, second in pairs:
        row: dict[str, Any] = {
            "probe": name,
            "exact_lag_preserving_isomorphic": exact_isomorphic(
                first, second, lag_labels=config.physical_lags
            ),
            "first_edges": int(first.sum()), "second_edges": int(second.sum()),
            "both_empty": not first.any() and not second.any(),
        }
        for metric in config.metrics:
            row[f"{metric}_distance"] = float(graph_similarity_distances(
                first[None], second[None], metric=metric, wl_iterations=config.wl_iterations,
                lag_labels=config.physical_lags,
            )[0, 0])
        rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _fit_panel(
    panel: PlantedGraphPanel, config: GraphSimilarityValidationConfig,
    *, seed: int, rewiring_probability: float,
) -> list[dict[str, Any]]:
    rows = []
    for metric in config.metrics:
        for count in config.candidate_counts:
            logger.info("Graph similarity: seed=%s rewiring=%s metric=%s K=%s",
                        seed, rewiring_probability, metric, count)
            row: dict[str, Any] = {
                "seed": seed, "rewiring_probability": rewiring_probability,
                "metric": metric, "n_states": count,
                "n_training_graphs": len(panel.fit_indices),
                "n_heldout_graphs": len(panel.evaluation_indices),
                "heldout_identity_ari": None, "heldout_topology_ari": None,
            }
            try:
                graph_config = GraphStateConfig(
                    n_states=count, metric=metric, feature_representation="binary_lag_offdiag",
                    assignment="kmedoids", standardize=False, pca_components=None,
                    wl_iterations=config.wl_iterations, analysis_scope="prospective", seed=seed,
                )
                model = fit_graph_states(panel.series, graph_config, fit_indices=panel.fit_indices)
                assigned = model.transform(panel.series)
                heldout = assigned[panel.evaluation_indices]
                row.update({
                    "status": "available", "effective_algorithm": model.effective_algorithm,
                    "effective_distance": model.effective_distance,
                    "heldout_identity_ari": float(adjusted_rand_score(
                        panel.identity_labels[panel.evaluation_indices], heldout
                    )),
                    "heldout_topology_ari": float(adjusted_rand_score(
                        panel.topology_labels[panel.evaluation_indices], heldout
                    )),
                    "n_assigned_training_states": len(np.unique(assigned[panel.fit_indices])),
                    "n_assigned_heldout_states": len(np.unique(heldout)),
                    "fit_indices": model.fit_indices,
                    "training_assignments": assigned[panel.fit_indices], "heldout_assignments": heldout,
                    "medoid_fingerprint": content_fingerprint({
                        "lagged_medoids": model.lagged_medoid_matrices,
                        "schema": model.graph_schema, "metric": model.metric,
                        "wl_iterations": model.wl_iterations,
                    }),
                })
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                row.update({"status": "unavailable", "reason": str(error)})
            rows.append(row)
    return rows


def run_graph_similarity_validation(
    config: GraphSimilarityValidationConfig, output_dir: str | Path,
) -> dict[str, Any]:
    """Fit frozen training quantizers and report all requested held-out contrasts."""
    root = Path(output_dir)
    protocol = {
        "config": asdict(config), "implementation_fingerprint": implementation_fingerprint(),
        "identity_target": "four_generating_neuron_specific_archetypes",
        "topology_target": "three_lag_preserving_isomorphism_classes_before_corruption",
        "labels_enter_fitting": False, "codebook_selection": "fixed_K_no_heldout_selection",
        "density_control": "six_nodes_two_cross_edges_per_graph_all_panels",
        "noise": "one_edge_replaced_at_same_lag_with_declared_probability",
    }
    frozen = root / "graph-similarity-protocol.json"
    if frozen.is_file() and json.loads(frozen.read_text()) != _json_safe(protocol):
        raise ValueError("output directory belongs to another graph similarity protocol; use a new directory")
    _write_json(frozen, protocol)
    report: dict[str, Any] = {
        "schema_version": "planted_graph_similarity_v1", "protocol": protocol,
        "protocol_fingerprint": content_fingerprint(protocol),
        "inventory": graph_similarity_inventory(config),
        "scope": "development_structural_graph_diagnostic_not_neural_driver_validation",
        "independent_unit": "graph_sampling_seed_not_neural_process_or_biological_replicate",
        "scientific_validation_passed": False, "biological_necessity_validated": False,
        "native_CSL_refitted": False, "execution_complete": False, "panels": [], "rows": [],
        "probes": graph_similarity_probes(config),
        "characteristics": [
            {"archetype": name, **directed_graph_characteristics(graph, lag_labels=config.physical_lags)}
            for name, graph in zip(_ARCHETYPES, planted_graph_archetypes(), strict=True)
        ],
    }
    for seed in config.seeds:
        for rewiring in config.rewiring_probabilities:
            panel = make_planted_graph_panel(config, seed=seed, rewiring_probability=rewiring)
            tensor = panel.series.lagged_matrices
            assert tensor is not None  # The planted panel always provides lag-resolved support.
            report["panels"].append({
                "seed": seed, "rewiring_probability": rewiring,
                "training_input_fingerprint": content_fingerprint(
                    tensor[panel.fit_indices]
                ),
                "evaluation_input_fingerprint": content_fingerprint(
                    tensor[panel.evaluation_indices]
                ),
                "all_edge_counts": sorted(set(tensor.sum(axis=(1, 2, 3)).tolist())),
                "identity_labels": panel.identity_labels,
                "topology_labels": panel.topology_labels,
                "fit_indices": panel.fit_indices, "evaluation_indices": panel.evaluation_indices,
            })
            report["rows"].extend(_fit_panel(panel, config, seed=seed, rewiring_probability=rewiring))
            report["attempted_fits"] = len(report["rows"])
            report["available_fits"] = sum(row["status"] == "available" for row in report["rows"])
            report["execution_complete"] = (
                report["attempted_fits"] == report["inventory"]["planned_quantizer_fits"]
            )
            _write_json(root / "graph-similarity.json", report)
    return _json_safe(report)


def render_graph_similarity_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    """Export seed rows and descriptive plots, without selecting a winning metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(report["rows"])
    scalar = rows.drop(columns=[
        "fit_indices", "training_assignments", "heldout_assignments", "medoid_fingerprint",
    ], errors="ignore")
    scalar.to_csv(root / "seed-grouping.csv", index=False, lineterminator="\n")
    pd.DataFrame(report["probes"]).to_csv(root / "distance-probes.csv", index=False, lineterminator="\n")
    pd.DataFrame(report["characteristics"]).to_csv(
        root / "graph-characteristics.csv", index=False, lineterminator="\n"
    )
    config = report["protocol"]["config"]
    settings = [(metric, count, probability) for metric in config["metrics"]
                for count in config["candidate_counts"] for probability in config["rewiring_probabilities"]]
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(settings) * 0.85), 7), constrained_layout=True)
    for axis, field, title in zip(axes, ("heldout_identity_ari", "heldout_topology_ari"), (
        "Neuron-specific archetype labels (four classes)", "Topology labels (three classes)"
    ), strict=True):
        for position, (metric, count, probability) in enumerate(settings):
            subset = rows[(rows.metric == metric) & (rows.n_states == count)
                          & (rows.rewiring_probability == probability)]
            available = subset[subset.status == "available"]
            if available.empty:
                axis.text(position, 0.03, "unavailable", rotation=90, ha="center", fontsize=7)
                continue
            values = available[field].to_numpy(dtype=float)
            axis.bar(position, values.mean(), color="steelblue", alpha=0.6)
            axis.scatter(np.full(len(values), position), values, c="black", s=16, zorder=3)
        axis.axhline(0, color="black", linewidth=0.6)
        axis.set(title=title, ylabel="Held-out adjusted Rand index", ylim=(-0.25, 1.08))
        axis.set_xticks(np.arange(len(settings)), [
            f"{metric}\nK={count}, p={probability:g}" for metric, count, probability in settings
        ], rotation=45, ha="right", fontsize=7)
    fig.suptitle("Planted structural diagnostic; dots are graph-sampling seeds, not neural replicates")
    for suffix in ("pdf", "png"):
        fig.savefig(root / f"figure-01-heldout-grouping.{suffix}", dpi=160)
    plt.close(fig)
    probes = report["probes"]
    values = np.array([[row[f"{metric}_distance"] for metric in config["metrics"]] for row in probes])
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = axis.imshow(values, cmap="viridis", aspect="auto", vmin=0)
    axis.set_xticks(np.arange(len(config["metrics"])), config["metrics"])
    axis.set_yticks(np.arange(len(probes)), [row["probe"].replace("_", " ") for row in probes], fontsize=8)
    axis.set_title("Known comparisons and explicit WL collision; compare within each metric")
    for row in range(len(probes)):
        for column in range(len(config["metrics"])):
            colour = "black" if values[row, column] > 0.6 * values.max() else "white"
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", color=colour)
    fig.colorbar(image, ax=axis, label="Distance (metric-specific geometry)")
    for suffix in ("pdf", "png"):
        fig.savefig(root / f"figure-02-distance-probes.{suffix}", dpi=160)
    plt.close(fig)
    available = scalar[scalar.status == "available"]
    aggregate = available.groupby(["metric", "n_states", "rewiring_probability"])[
        ["heldout_identity_ari", "heldout_topology_ari"]
    ].agg(["mean", "std", "count"])
    aggregate.to_csv(root / "grouping-summary.csv", lineterminator="\n")
    path = root / "analysis-report.md"
    path.write_text(
        "# Planted graph similarity diagnostic\n\n"
        f"Attempted fits: {report['attempted_fits']}; available: {report['available_fits']}. "
        f"Execution complete: {report['execution_complete']}.\n\n"
        "Identity-sensitive edge overlap and topology-sensitive directed WL answer different questions. "
        "All four archetypes have six nodes and two edges. Corruption replaces one edge while preserving "
        "density. Labels score held-out assignments only; fixed K and metrics are not selected "
        "on these scores.\n\n"
        "Clean WL has three distinguishable topology classes, so a requested four-state partition may be "
        "unavailable. The six-cycle/two-triangle comparison demonstrates that equal WL features do not "
        "prove isomorphism. Empty-graph agreement does not establish recovery.\n\n"
        "CSV files retain each graph-sampling seed, unavailable fits and reasons. Bars describe seed means; "
        "dots are individual graph-sampling repeats. No inferential confidence intervals are claimed. "
        "The identity and topology targets are generating-archetype labels before corruption.\n\n"
        "This diagnostic does not rerun native discovery, predict behavior, recover causal drivers, or "
        "validate biological regimes. Graph estimation and downstream intervention validation remain "
        "separate. See docs/graph_similarity_methodology.md for citations and assumptions.\n"
    )
    return path
