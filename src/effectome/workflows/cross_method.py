"""A shared, training-owned transition comparison across connectivity estimators.

The first delivery is a chronological development holdout. All methods predict
the same geometric manifold regions on the same events. No biological regime,
necessity, or independent-animal confirmation claim is inferred automatically.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from effectome.attribution.transition_prediction import evaluate_transition_predictors
from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module.schema import ArtifactProvenance, ConnectivitySeries, RawInterval, WindowedSegments
from effectome.dynamics.graph_states import GraphStateConfig, fit_graph_states
from effectome.linking.shared_transitions import (
    SharedTransitionConfig,
    fit_shared_transition_reference,
    state_correspondence,
)
from effectome.linking.transition_diagnostics import (
    frozen_markov_diagnostics,
    group_control_comparisons,
    matched_group_controls,
    matched_transition_contrasts,
)
from effectome.linking.transition_synthesis import (
    METHOD_FAMILIES,
    conditional_outcome_diversity,
    family_rank_synthesis,
    graph_transition_features,
)
from effectome.workflows.feature_ownership import chronological_feature_split
from effectome.workflows.resume import content_fingerprint

logger = logging.getLogger(__name__)

LEGACY_CROSS_METHOD_LANES = (
    "cgc", "cgc_star", "correlation_partial", "sparse_var", "windowed_sparse_var",
    "switching_var", "tvvar_smooth", "tvvar_fused",
)


@dataclass(frozen=True)
class CrossMethodConfig:
    n_dims: int = 3
    n_manifold_states: int = 3
    positive_lag: int = 1
    native_states: int = 3
    evaluation_fraction: float = 0.25
    decoder_c: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        SharedTransitionConfig(self.n_dims, self.n_manifold_states, self.positive_lag, self.seed)
        if not isinstance(self.native_states, int) or self.native_states < 1:
            raise ValueError("native_states must be a positive integer")
        if not 0 < self.evaluation_fraction < 1:
            raise ValueError("evaluation_fraction must lie in (0, 1)")
        if not np.isfinite(self.decoder_c) or self.decoder_c <= 0:
            raise ValueError("decoder_c must be finite and positive")


def load_cross_method_profile(
    profile: str = "development", *, config_root: str | Path | None = None
) -> tuple[CrossMethodConfig, dict[str, ConnectivityConfig]]:
    """Load declared settings, preserving historical eight-lane profiles."""
    from omegaconf import OmegaConf

    if profile not in {"smoke", "development", "csl_sanity"}:
        raise ValueError("profile must be smoke, development or csl_sanity")
    root = Path(config_root) if config_root else Path(__file__).resolve().parents[3] / "conf"
    values = OmegaConf.to_container(OmegaConf.load(root / "cross_method" / f"{profile}.yaml"), resolve=True)
    required = {"comparison", "connectivity_overrides"}
    if not isinstance(values, dict) or not required <= set(values) <= required | {"methods"}:
        raise ValueError("cross-method profile requires comparison and connectivity_overrides")
    config = CrossMethodConfig(**values["comparison"])
    lanes = values.get("methods", list(LEGACY_CROSS_METHOD_LANES))
    if (
        not isinstance(lanes, list)
        or not lanes
        or any(not isinstance(lane, str) for lane in lanes)
        or len(set(lanes)) != len(lanes)
        or set(lanes) - set(METHOD_FAMILIES)
    ):
        raise ValueError("profile methods must be a nonempty unique list of known lanes")
    methods = {}
    for lane in lanes:
        name = "correlation" if lane == "correlation_partial" else lane
        base = OmegaConf.load(root / "connectivity" / f"{name}.yaml")
        merged = OmegaConf.merge(base, values["connectivity_overrides"].get(lane, {}))
        raw = OmegaConf.to_container(merged, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError("connectivity profile must contain a mapping")
        methods[lane] = ConnectivityConfig(**{str(key): value for key, value in raw.items()})
    return config, methods


def _subset(windows: WindowedSegments, indices: np.ndarray) -> WindowedSegments:
    return replace(
        windows,
        segments=windows.segments[indices],
        windows=[windows.windows[int(i)] for i in indices],
        anchors=[windows.anchors[int(i)] for i in indices],
        behavior_per_window={key: values[indices] for key, values in windows.behavior_per_window.items()},
    )


def _estimate_owned_graphs(
    windows: WindowedSegments,
    cfg: ConnectivityConfig,
    fit: np.ndarray,
    evaluation: np.ndarray,
    scope: str,
) -> ConnectivitySeries:
    resolved = cfg.resolve(windows.fps)
    if resolved.absolute:
        raise ValueError("shared signed comparisons reject absolute-value graph transformations")
    estimator = ConnectivityFactory(resolved)
    if estimator.capabilities.retrospective_sequence:
        raise ValueError("retrospective graph estimators are ineligible for prospective comparisons")
    if resolved.name == "sparse_var":
        fitted = estimator.run(_subset(windows, fit))
        if fitted.diagnostics.get("adaptation_policy") != "constant_fit_repeated_for_windows":
            raise ValueError("constant VAR must reconstruct and fit one training-owned graph")
        series = replace(
            fitted,
            matrices=np.repeat(fitted.matrices[:1], windows.n_windows, axis=0),
            lagged_matrices=None
            if fitted.lagged_matrices is None
            else np.repeat(fitted.lagged_matrices[:1], windows.n_windows, axis=0),
            window_starts=np.asarray([window.start for window in windows.windows]),
            anchors=list(windows.anchors),
            behavior_per_window=windows.behavior_per_window,
        )
    else:
        # Stateful temporal penalties restart at ownership and invalid-data
        # boundaries. No embargo-window warm start enters held-out fitting.
        ownership = np.zeros(windows.n_windows, dtype=int)
        ownership[fit], ownership[evaluation] = 1, 2
        anchors = list(windows.anchors)
        for index in range(1, len(anchors)):
            previous, current = anchors[index - 1], anchors[index]
            if (
                ownership[index] != ownership[index - 1]
                or current.bad_frame_before
                or previous.bad_frame_after
                or not current.valid
                or not previous.valid
            ):
                anchors[index] = replace(current, gap_before=True)
        series = estimator.run(replace(windows, anchors=anchors))
        series.anchors = list(windows.anchors)
    used = np.union1d(fit, evaluation)
    statuses = series.diagnostics.get("fit_statuses", [])
    if statuses and len(statuses) == windows.n_windows:
        failures = [int(i) for i in used if statuses[int(i)] not in {"converged", "accepted"}]
        if failures:
            raise ValueError(f"graph fitting was not accepted at required anchors {failures[:10]}")
    if series.diagnostics.get("status") == "nonconverged":
        raise ValueError("constant graph fit did not converge")
    if not np.all(np.isfinite(series.matrices)):
        raise ValueError("graph estimation returned nonfinite coefficients")
    series.provenance = ArtifactProvenance(
        identity=windows.provenance.identity,
        stage="cross_method_connectivity",
        fit_scope_id=scope,
        split_manifest_hash=content_fingerprint((fit, evaluation)),
        fitted_raw_intervals=tuple(
            RawInterval(
                windows.anchors[int(i)].recording_id,
                windows.anchors[int(i)].context_start,
                windows.anchors[int(i)].context_stop,
            )
            for i in fit
        ),
        analysis_mode="prospective",
        claim_eligibility="development_only",
        metadata={"resolved_config": asdict(resolved), "ownership_producer": "run_cross_method_comparison"},
    )
    return series


def _native_diagnostics(
    series: ConnectivitySeries,
    fit: np.ndarray,
    evaluation: np.ndarray,
    manifold_states: np.ndarray,
    origins: np.ndarray,
    targets: np.ndarray,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    displacement: np.ndarray,
    config: CrossMethodConfig,
) -> dict[str, Any]:
    tensor = series.matrices if series.lagged_matrices is None else series.lagged_matrices
    constant = bool(np.all(tensor[fit] == tensor[fit[0]]))
    count = 1 if constant else config.native_states
    model = fit_graph_states(
        series,
        GraphStateConfig(
            n_states=count,
            feature_representation="signed_lag_offdiag",
            assignment="kmeans",
            metric="frobenius",
            analysis_scope="prospective",
            seed=config.seed,
        ),
        fit_indices=fit,
    )
    future_labels = np.full(len(manifold_states), -1, dtype=int)
    future_labels[origins] = manifold_states[targets]
    correspondence = state_correspondence(
        model.labels,
        manifold_states,
        fit_indices=origins[train_rows],
        evaluation_indices=origins[evaluation_rows],
        future_targets=future_labels,
    )
    native_counts = np.zeros((count, count), dtype=int)
    np.add.at(native_counts, (model.labels[origins[train_rows]], model.labels[targets[train_rows]]), 1)
    totals = native_counts.sum(axis=1, keepdims=True)
    native_prob = np.divide(
        native_counts, totals, out=np.full_like(native_counts, np.nan, dtype=float), where=totals > 0
    )
    return {
        "status": "available",
        "labels": model.labels,
        "n_states": model.n_states,
        "requested_n_states": config.native_states,
        "constant_graph_one_state": constant,
        "state_centers": model.feature_centroids,
        "training_transition_counts": native_counts,
        "training_transition_matrix": native_prob,
        "correspondence": correspondence,
        "conditional_future_diversity": conditional_outcome_diversity(
            model.labels[origins[evaluation_rows]],
            manifold_states[origins[evaluation_rows]],
            manifold_states[targets[evaluation_rows]],
            displacement[evaluation_rows],
        ),
        "regime_claim_eligible": False,
    }


def run_cross_method_comparison(
    windows: WindowedSegments,
    config: CrossMethodConfig | None = None,
    *,
    method_configs: Mapping[str, ConnectivityConfig] | None = None,
    fit_indices: np.ndarray | None = None,
    evaluation_indices: np.ndarray | None = None,
    neuron_ids: Sequence[str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    retain_graphs: bool = False,
    graph_origin_only: bool = False,
) -> dict[str, Any]:
    """Fit once, issue paired predictions, and retain every method failure explicitly."""
    cfg = config or CrossMethodConfig()
    if method_configs is None:
        _, method_configs = load_cross_method_profile()
    if not method_configs:
        raise ValueError("at least one method is required")
    if (fit_indices is None) != (evaluation_indices is None):
        raise ValueError("supply both fit and evaluation indices or neither")
    if fit_indices is None:
        fit_indices, evaluation_indices = chronological_feature_split(
            windows, evaluation_fraction=cfg.evaluation_fraction
        )
    assert evaluation_indices is not None
    # Every method receives the same raw autoregressive baseline, including all
    # activity lags used by ANY requested graph-drive feature. Otherwise an
    # apparent graph benefit could simply expose extra neural history to M1.
    neural_history_samples = max(
        2,
        *(
            int(method_cfg.extra.get("n_lags", method_cfg.resolve(windows.fps).max_lag))
            for method_cfg in method_configs.values()
        ),
    )
    reference = fit_shared_transition_reference(
        windows,
        fit_indices=np.asarray(fit_indices),
        evaluation_indices=np.asarray(evaluation_indices),
        config=SharedTransitionConfig(
            cfg.n_dims,
            cfg.n_manifold_states,
            cfg.positive_lag,
            cfg.seed,
            neural_history_samples=neural_history_samples,
        ),
    )
    ids = neuron_ids
    id_source = "explicit"
    if ids is None:
        ids = windows.metadata.get("raw_evaluation_contract", {}).get("canonical_neuron_ids")
        id_source = "raw_manifest" if ids is not None else "column_index_unmapped"
    ids = [str(value) for value in (ids if ids is not None else range(windows.n_neurons))]
    if len(ids) != windows.n_neurons or len(set(ids)) != len(ids):
        raise ValueError("neuron_ids must uniquely identify the neuron axis")
    if groups and any(
        not members or any(str(member) not in ids for member in members) for members in groups.values()
    ):
        raise ValueError("all group members must reference known neuron IDs")
    scaled = (np.asarray(windows.segments, dtype=float) - reference.scaler_mean) / reference.scaler_scale
    transformed = replace(windows, segments=scaled)
    controlled_groups, group_control_manifest = matched_group_controls(
        groups or {},
        ids,
        windows.segments,
        reference.fit_indices,
        seed=cfg.seed,
    )
    events = reference.events
    origins, targets = events.origins, events.targets
    baseline = reference.baseline_features[origins]
    displacement = reference.embeddings[targets] - reference.embeddings[origins]
    signature = content_fingerprint(
        (windows, asdict(cfg), reference.fit_indices, reference.evaluation_indices, graph_origin_only)
    )
    scope = f"shared-transition-development-{signature[:16]}"
    common_prediction = evaluate_transition_predictors(
        baseline,
        np.empty((len(origins), 0)),
        events.source_states,
        events.target_states,
        train_rows=events.train_rows,
        evaluation_rows=events.evaluation_rows,
        feature_blocks={},
        n_states=cfg.n_manifold_states,
        seed=cfg.seed,
        c=cfg.decoder_c,
        future_displacement=displacement,
    )
    methods: dict[str, dict[str, Any]] = {}
    graph_windows = _subset(transformed, origins) if graph_origin_only else transformed
    graph_fit = events.train_rows if graph_origin_only else reference.fit_indices
    graph_evaluation = events.evaluation_rows if graph_origin_only else reference.evaluation_indices
    for lane, method_cfg in method_configs.items():
        logger.info("Shared transition comparison: %s", lane)
        try:
            if graph_origin_only and not ConnectivityFactory(method_cfg).capabilities.independent_window:
                raise ValueError("origin-only discovery requires an independent-window estimator")
            series = _estimate_owned_graphs(
                graph_windows, method_cfg, graph_fit, graph_evaluation, scope
            )
            features, blocks, feature_metadata = graph_transition_features(
                series,
                graph_windows,
                scaled_segments=graph_windows.segments,
                pca_components=reference.pca_components,
                neuron_ids=ids,
                fit_indices=graph_fit,
                evaluation_indices=graph_evaluation,
                groups=controlled_groups,
            )
            blocks = {key.replace("group:__control__", "control:"): value for key, value in blocks.items()}
            event_features = features if graph_origin_only else features[origins]
            prediction = evaluate_transition_predictors(
                baseline,
                event_features,
                events.source_states,
                events.target_states,
                train_rows=events.train_rows,
                evaluation_rows=events.evaluation_rows,
                feature_blocks=blocks,
                n_states=cfg.n_manifold_states,
                seed=cfg.seed,
                c=cfg.decoder_c,
                future_displacement=displacement,
            )
            try:
                if graph_origin_only:
                    raise ValueError(
                        "future graphs were not estimated in the origin-only prediction protocol"
                    )
                native = _native_diagnostics(
                    series,
                    reference.fit_indices,
                    reference.evaluation_indices,
                    reference.manifold_states,
                    origins,
                    targets,
                    events.train_rows,
                    events.evaluation_rows,
                    displacement,
                    cfg,
                )
            except (ValueError, RuntimeError) as error:
                native = {"status": "unavailable", "reason": str(error)}
            projected = feature_metadata.pop("projected_cross_neuron_drive")
            if projected is not None:
                selected_drive = projected[
                    events.evaluation_rows if graph_origin_only else origins[events.evaluation_rows]
                ]
                motion = displacement[events.evaluation_rows]
                unit_motion = motion / np.maximum(np.linalg.norm(motion, axis=1, keepdims=True), 1e-12)
                feature_metadata["heldout_drive_along_observed_displacement"] = np.einsum(
                    "kid,kd->ki", selected_drive, unit_motion
                )
            methods[lane] = {
                "status": "available",
                "family": METHOD_FAMILIES.get(lane, lane),
                "directed": series.directed,
                "weight_semantics": series.weight_semantics,
                "connectivity_config": asdict(method_cfg),
                "diagnostics": series.diagnostics,
                "graph_fingerprint": content_fingerprint((series.matrices, series.lagged_matrices)),
                "training_graph_fingerprint": content_fingerprint(series.matrices[graph_fit]),
                "graph_anchor_indices": origins if graph_origin_only else np.arange(windows.n_windows),
                "graph_scope": "prediction_origins_only" if graph_origin_only else "all_anchors",
                "graph_provenance": asdict(series.provenance),
                "features": feature_metadata,
                "prediction": prediction,
                "group_control_comparisons": group_control_comparisons(
                    prediction["omission_scores"],
                    group_control_manifest,
                ),
                "native_graph_states": native,
                "matched_transition_contrasts": matched_transition_contrasts(
                    baseline,
                    event_features,
                    events.source_states,
                    events.target_states,
                    events.train_rows,
                    events.evaluation_rows,
                    blocks,
                ),
            }
            if retain_graphs:
                methods[lane]["graph_matrices"] = series.matrices
                methods[lane]["lagged_graph_matrices"] = series.lagged_matrices
        except (ValueError, RuntimeError, ImportError, KeyError, np.linalg.LinAlgError) as error:
            logger.warning("Method %s unavailable: %s", lane, error)
            methods[lane] = {
                "status": "unavailable",
                "reason": str(error),
                "connectivity_config": asdict(method_cfg),
            }
    event_manifest = {
        **asdict(events),
        "recording_ids": [windows.anchors[int(i)].recording_id for i in origins],
        "dataset_ids": [windows.anchors[int(i)].dataset_id for i in origins],
        "animal_ids": [windows.anchors[int(i)].animal_id for i in origins],
        "origin_samples": [windows.anchors[int(i)].anchor_sample for i in origins],
        "origin_time_seconds": [windows.anchors[int(i)].anchor_time_seconds for i in origins],
        "target_time_seconds": [windows.anchors[int(i)].anchor_time_seconds for i in targets],
    }
    return {
        "schema_version": "shared_transitions_v1",
        "status": "development_only",
        "config": asdict(cfg),
        "fit_scope_id": scope,
        "input_fingerprint": signature,
        "neuron_ids": ids,
        "neuron_id_source": id_source,
        "groups": dict(groups or {}),
        "matched_group_controls": group_control_manifest,
        "reference": asdict(reference),
        "common_baseline": common_prediction,
        "events": event_manifest,
        "methods": methods,
        "family_rank_synthesis": family_rank_synthesis(methods),
        "markov_adequacy": frozen_markov_diagnostics(
            windows,
            reference.manifold_states,
            fit_indices=reference.fit_indices,
            evaluation_indices=reference.evaluation_indices,
            n_states=cfg.n_manifold_states,
            positive_lag=cfg.positive_lag,
        ),
        "scientific_claims": {
            "validated_neural_regimes": False,
            "circuit_degeneracy_identified": False,
            "biological_necessity_identified": False,
            "independent_animal_confirmation": False,
            "null_calibration": "not_run",
            "hyperparameter_selection": "declared_fixed_defaults",
            "missing_data_policy": "finite_raw_histories_required",
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_cross_method_report(report: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, ...]:
    """Write one shared artifact, paired event predictions, and candidate tables."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "comparison.json"
    manifest.write_text(json.dumps(_json_safe(dict(report)), indent=2, allow_nan=False) + "\n")
    rows = [
        dict(method=method, **row)
        for method, result in report["methods"].items()
        if result["status"] == "available"
        for row in result["prediction"]["omission_scores"]
    ]
    attribution = destination / "neuron_group_contributions.csv"
    with attribution.open("w", newline="") as handle:
        fields = [
            "method",
            "unit",
            "source_state",
            "destination_state",
            "n_origins",
            "n_events",
            "omission_delta_log_loss",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    event_path = destination / "transition_events.csv"
    events = report["events"]
    evaluation = set(np.asarray(events["evaluation_rows"]).tolist())
    with event_path.open("w", newline="") as handle:
        event_writer = csv.writer(handle)
        event_writer.writerow(
            [
                "row",
                "recording_id",
                "dataset_id",
                "animal_id",
                "origin",
                "target",
                "origin_sample",
                "origin_time_seconds",
                "target_time_seconds",
                "source_state",
                "target_state",
                "horizon_seconds",
                "split",
            ]
        )
        for row, origin in enumerate(events["origins"]):
            event_writer.writerow(
                [
                    row,
                    events["recording_ids"][row],
                    events["dataset_ids"][row],
                    events["animal_ids"][row],
                    origin,
                    events["targets"][row],
                    events["origin_samples"][row],
                    events["origin_time_seconds"][row],
                    events["target_time_seconds"][row],
                    events["source_states"][row],
                    events["target_states"][row],
                    events["horizon_seconds"][row],
                    "evaluation" if row in evaluation else "training",
                ]
            )
    summary = destination / "README.md"
    lines = [
        "# Shared transition development report",
        "",
        "One manifold and one event table serve every method.",
        "Region codes are geometric summaries; correspondence diversity and feature omission",
        "scores are descriptive development evidence, without identified biological necessity.",
        "",
        "| Method | Status | Held-out log-loss gain |",
        "|---|---|---|",
    ]
    for method, result in report["methods"].items():
        gain = result.get("prediction", {}).get("log_loss_gain", "unavailable")
        lines.append(f"| {method} | {result['status']} | {gain} |")
        if result["status"] != "available":
            lines.extend(["", f"{method}: {result['reason']}", ""])
    summary.write_text("\n".join(lines) + "\n")
    return manifest, attribution, event_path, summary
