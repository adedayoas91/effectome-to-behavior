"""Training-owned feature artifacts for the exploratory sensitivity lane.

The outer split is chronological and purged in raw sample space. Scaling, PCA,
and role centroids are frozen from training histories. Graph refits use only each
issued history; the constant VAR instead fits the training histories once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from effectome.community import CommunityConfig, fit_temporal_role_model
from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module.schema import (
    ArtifactProvenance,
    CommunitySeries,
    ConnectivitySeries,
    RawInterval,
    WindowedSegments,
)
from effectome.manifold import ManifoldArtifact, ManifoldConfig, ManifoldFactory, TargetSlice
from effectome.utils.io import load_artifact, save_artifact

from .resume import content_fingerprint, implementation_fingerprint


@dataclass
class OwnedFeatureArtifacts:
    connectivity: ConnectivitySeries
    community: CommunitySeries
    manifold: ManifoldArtifact
    fit_indices: np.ndarray
    evaluation_indices: np.ndarray
    fit_scope_id: str
    training_parameters: dict[str, object]


def chronological_feature_split(
    windows: WindowedSegments, *, evaluation_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve a trailing block and purge whole training histories against it."""
    if not windows.anchors or not 0 < evaluation_fraction < 1:
        raise ValueError("a chronological split requires anchors and evaluation_fraction in (0, 1)")
    first = int(np.floor(windows.n_windows * (1.0 - evaluation_fraction)))
    if not 0 < first < windows.n_windows:
        raise ValueError("not enough windows for a chronological split")
    evaluation = np.arange(first, windows.n_windows, dtype=int)
    first_evaluation = min(windows.anchors[index].context_start for index in evaluation)
    fit = np.asarray(
        [index for index in range(first) if windows.anchors[index].context_stop <= first_evaluation],
        dtype=int,
    )
    if fit.size < 5 or evaluation.size < 3:
        raise ValueError("raw-history purging leaves too few windows; use a longer recording")
    return fit, evaluation


def _validate_partitions(
    windows: WindowedSegments, fit_indices: np.ndarray, evaluation_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(windows.anchors) != windows.n_windows:
        raise ValueError("owned feature fitting requires typed raw temporal anchors")
    arrays = []
    for name, values in (("fit_indices", fit_indices), ("evaluation_indices", evaluation_indices)):
        indices = np.asarray(values)
        if (
            indices.ndim != 1
            or not np.issubdtype(indices.dtype, np.integer)
            or indices.size == 0
            or np.any(indices < 0)
            or np.any(indices >= windows.n_windows)
            or np.any(np.diff(indices) <= 0)
        ):
            raise ValueError(f"{name} must be nonempty, strictly increasing integer window indices")
        arrays.append(indices.astype(int))
    fit, evaluation = arrays
    if fit[-1] >= evaluation[0]:
        raise ValueError("owned sensitivity features require chronological disjoint outer ownership")
    dependency = windows.metadata.get("preprocess_dependency", {})
    if (
        not isinstance(dependency, dict)
        or dependency.get("kind") not in {"identity", "finite"}
        or dependency.get("past_support", 0) != 0
        or dependency.get("future_support", 0) != 0
    ):
        raise ValueError("owned feature fitting requires identity-preprocessed raw windows")
    standardization = windows.metadata.get("neural_standardization", {})
    if not isinstance(standardization, dict) or standardization.get("enabled", True):
        raise ValueError("owned feature fitting rejects per-window standardization")
    for train_index in fit:
        training = windows.anchors[int(train_index)]
        for test_index in evaluation:
            testing = windows.anchors[int(test_index)]
            if training.recording_id == testing.recording_id and max(
                training.context_start, testing.context_start
            ) < min(training.context_stop, testing.context_stop):
                raise ValueError("outer fit and evaluation histories overlap in raw sample space")
    return fit, evaluation


def fit_owned_feature_artifacts(
    windows: WindowedSegments,
    *,
    connectivity_cfg: ConnectivityConfig,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    fit_scope_id: str,
    n_dims: int = 3,
    n_roles: int = 3,
    seed: int = 42,
) -> OwnedFeatureArtifacts:
    """Construct new owned artifacts; never relabel globally fitted artifacts."""
    fit, evaluation = _validate_partitions(windows, fit_indices, evaluation_indices)
    if not fit_scope_id:
        raise ValueError("fit_scope_id must be explicit and nonempty")
    training = np.asarray(windows.segments[fit], dtype=float).reshape(-1, windows.n_neurons)
    training = training[np.all(np.isfinite(training), axis=1)]
    if training.size == 0:
        raise ValueError("training histories contain no complete neural samples")
    if not 1 <= n_dims <= min(training.shape) or not 1 <= n_roles <= fit.size * windows.n_neurons:
        raise ValueError("invalid PCA or role dimensions for the owned training data")
    # Repeated samples have the declared window-weighted interpretation; no
    # held-out history contributes to either fitted moment.
    scaler = StandardScaler().fit(training)
    scaled = (np.asarray(windows.segments, dtype=float) - scaler.mean_) / scaler.scale_
    transformed = replace(windows, segments=scaled)
    config = connectivity_cfg.resolve(windows.fps)
    if config.name not in {
        "cgc",
        "cgc_star",
        "sparse_var",
        "windowed_sparse_var",
        "switching_var",
        "tvvar_smooth",
        "tvvar_fused",
    }:
        raise ValueError("owned sensitivity supports only audited directed causal-history estimators")
    estimator = ConnectivityFactory(config)
    if config.name == "sparse_var":
        training_windows = replace(
            transformed,
            segments=scaled[fit],
            windows=[windows.windows[int(index)] for index in fit],
            anchors=[windows.anchors[int(index)] for index in fit],
            behavior_per_window={key: values[fit] for key, values in windows.behavior_per_window.items()},
        )
        fitted = estimator.run(training_windows)
        connectivity = replace(
            fitted,
            matrices=np.repeat(fitted.matrices[:1], windows.n_windows, axis=0),
            lagged_matrices=(
                None
                if fitted.lagged_matrices is None
                else np.repeat(fitted.lagged_matrices[:1], windows.n_windows, axis=0)
            ),
            window_starts=np.asarray([window.start for window in windows.windows]),
            anchors=list(windows.anchors),
            behavior_per_window=windows.behavior_per_window,
        )
    else:
        if estimator.capabilities.retrospective_sequence:
            raise ValueError("retrospective sequence estimators cannot produce prospective owned features")
        connectivity = estimator.run(transformed)
    split_hash = content_fingerprint((fit, evaluation))
    raw_intervals = tuple(
        RawInterval(
            windows.anchors[int(index)].recording_id,
            windows.anchors[int(index)].context_start,
            windows.anchors[int(index)].context_stop,
        )
        for index in fit
    )
    parameters: dict[str, object] = {
        "scale_mean": scaler.mean_,
        "scale_scale": scaler.scale_,
        "connectivity_config": asdict(config),
        "fit_indices": fit.tolist(),
        "evaluation_indices": evaluation.tolist(),
        "fit_scope_id": fit_scope_id,
        "split_manifest_hash": split_hash,
        "graph_update_policy": "training_constant" if config.name == "sparse_var" else "causal_history_refit",
        "scaling_weighting": "training_history_window_weighted",
        "n_dims": n_dims,
        "n_roles": n_roles,
        "seed": seed,
        "implementation_hash": implementation_fingerprint(),
    }
    provenance = ArtifactProvenance(
        identity=windows.provenance.identity,
        stage="owned_features",
        fit_scope_id=fit_scope_id,
        split_manifest_hash=split_hash,
        fitted_raw_intervals=raw_intervals,
        analysis_mode="prospective",
        claim_eligibility="development_only",
        scale_id=content_fingerprint((scaler.mean_, scaler.scale_)),
        metadata={"ownership_producer": "fit_owned_feature_artifacts", **parameters},
    )
    connectivity.provenance = replace(provenance, stage="connectivity")
    connectivity.diagnostics = {**connectivity.diagnostics, **parameters}
    role_model = fit_temporal_role_model(
        connectivity,
        CommunityConfig(
            name="temporal",
            seed=seed,
            extra={"n_communities": n_roles, "online_learning_rate": 0.0, "temporal_penalty": 0.0},
        ),
        fit_indices=fit,
    )
    # Frozen, independent role assignment avoids held-out label state entering a
    # later training window or introducing an undeclared history dependency.
    labels = role_model.filter(connectivity).labels
    community = CommunitySeries(
        labels=labels,
        method="frozen_training_roles",
        n_communities_per_window=np.asarray([np.unique(row).size for row in labels]),
        anchors=list(windows.anchors),
        window_starts=connectivity.window_starts,
        provenance=replace(provenance, stage="community"),
        metadata={"fit_scope_id": fit_scope_id, "grouping_target": "roles", "update_policy": "frozen"},
    )
    embedder = ManifoldFactory(ManifoldConfig(name="classical", method="pca", n_dims=n_dims, seed=seed))
    embedder.fit(scaler.transform(training), {})
    target_lengths = {anchor.target_length for anchor in windows.anchors}
    if len(target_lengths) != 1:
        raise ValueError("owned manifold requires equal target lengths")
    target_length = target_lengths.pop()
    from effectome.manifold import causal_forward_fill_nonfinite

    embeddings = np.stack(
        [
            embedder.transform(causal_forward_fill_nonfinite(segment)[0][-target_length:]).mean(axis=0)
            for segment in scaled
        ]
    )
    manifold = ManifoldArtifact(
        method="training_pca",
        behavior_key="unsupervised",
        full_embedding=embeddings,
        window_embedding=embeddings,
        target_slices=[TargetSlice(anchor.target_start, anchor.target_stop) for anchor in windows.anchors],
        target_length=target_length,
        window_starts=connectivity.window_starts,
        anchors=list(windows.anchors),
        provenance=replace(provenance, stage="manifold"),
        metadata={
            "full_embedding_indexing": "target_window_end",
            "fit_scope_id": fit_scope_id,
            "missing_value_policy": "within_history_causal_forward_fill",
        },
    )
    parameters["role_centroids"] = role_model.initial_centroids
    return OwnedFeatureArtifacts(connectivity, community, manifold, fit, evaluation, fit_scope_id, parameters)


def load_or_fit_owned_features(
    windows_path: Path,
    bundle_path: Path,
    connectivity_cfg: ConnectivityConfig,
    ownership: dict[str, Any],
) -> OwnedFeatureArtifacts:
    """CLI handoff with content-checked reuse, separate from descriptive artifacts."""
    windows = load_artifact(windows_path)
    fit, evaluation = ownership.get("fit_indices"), ownership.get("evaluation_indices")
    if fit is None and evaluation is None:
        fit, evaluation = chronological_feature_split(windows)
    if fit is None or evaluation is None:
        raise ValueError("both outer fit/evaluation partitions must be supplied together")
    fit, evaluation = np.asarray(fit), np.asarray(evaluation)
    settings = {
        "n_dims": int(ownership.get("n_dims", 3)),
        "n_roles": int(ownership.get("n_roles", 3)),
        "seed": int(ownership.get("seed", 42)),
    }
    signature = content_fingerprint(
        (windows, asdict(connectivity_cfg), fit, evaluation, settings, implementation_fingerprint())
    )
    scope = ownership.get("fit_scope_id") or f"development-outer-{signature[:16]}"
    if bundle_path.is_file():
        previous = load_artifact(bundle_path)
        if (
            isinstance(previous, OwnedFeatureArtifacts)
            and previous.fit_scope_id == scope
            and previous.training_parameters.get("input_signature") == signature
        ):
            return previous
    result = fit_owned_feature_artifacts(
        windows,
        connectivity_cfg=connectivity_cfg,
        fit_indices=fit,
        evaluation_indices=evaluation,
        fit_scope_id=str(scope),
        **settings,
    )
    result.training_parameters["input_signature"] = signature
    save_artifact(result, bundle_path)
    return result


__all__ = [
    "OwnedFeatureArtifacts",
    "chronological_feature_split",
    "fit_owned_feature_artifacts",
    "load_or_fit_owned_features",
]
