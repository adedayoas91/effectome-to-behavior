"""Shared helpers for method-split resumable analysis notebooks."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

import numpy as np
from tqdm.auto import tqdm

from effectome.attribution import qualify_candidate_drivers
from effectome.community import CommunityConfig, CommunityFactory
from effectome.connectivity import ConnectivityConfig, ConnectivityFactory
from effectome.data_module import (
    ArtifactProvenance,
    NeuralRecording,
    PreprocessConfig,
    WindowConfig,
    WindowedSegments,
    exclude_named_neurons,
    get_loader,
    make_windows,
    preprocess,
)
from effectome.dynamics import (
    GraphStateConfig,
    ProbabilisticStateConfig,
    TransitionConfig,
    fit_graph_states,
    fit_probabilistic_states,
    fit_transitions,
)
from effectome.intervention.surrogate import (
    PerturbationResult,
    fit_linear_surrogate,
    run_predictor_feature_sensitivity,
)
from effectome.linking import (
    DependencySupport,
    activity_magnitude_features,
    anchor_continuity_labels,
    anchor_group_labels,
    association_with_null,
    combined_continuity_groups,
    community_features,
    community_reconfiguration_features,
    connectivity_features,
    decode_behavior,
    future_manifold_displacement,
    incremental_decode_behavior,
    lead_lag,
    manifold_speed,
    purged_blocked_splits,
    state_features,
    valid_positive_lag_origins,
)
from effectome.manifold import (
    BundleTrainingBatch,
    ManifoldArtifact,
    ManifoldConfig,
    ManifoldFactory,
    TargetSlice,
    build_bundle_training_batch,
    causal_forward_fill_nonfinite,
)
from effectome.utils.io import load_artifact
from effectome.workflows import (
    EvaluationConfig,
    ResumableRun,
    combine_primary_evaluation_results,
    estimate_connectivity_resumable,
    primary_evaluation_fold_ids,
    run_evaluation_profile,
    run_primary_evaluation,
)
from effectome.workflows.feature_ownership import (
    OwnedFeatureArtifacts,
    chronological_feature_split,
    fit_owned_feature_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
T = TypeVar("T")


class _ProgressReporter:
    """Create a notebook-friendly progress bar only when work is actually performed."""

    def __init__(self, description: str, unit: str) -> None:
        self.description = description
        self.unit = unit
        self._bar: Any | None = None
        self._reused = 0

    def __call__(self, completed: int, total: int, reused: bool = False) -> None:
        if self._bar is None:
            self._bar = tqdm(
                total=total,
                desc=self.description,
                unit=self.unit,
                dynamic_ncols=True,
                leave=True,
            )
        if reused:
            self._reused += 1
            self._bar.set_postfix(reused=self._reused, refresh=False)
        self._bar.update(max(0, completed - self._bar.n))

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def _run_with_status(
    run: ResumableRun,
    stage: str,
    operation: Callable[[], T],
) -> T:
    """Report checkpoint-aware stage timing without changing stage signatures."""
    prefix = f"[{run.run_id}/{run.method}] {stage}"
    print(f"{prefix}: checking checkpoint; computing if needed...")
    started = perf_counter()
    try:
        value = operation()
    except Exception:
        print(f"{prefix}: failed after {perf_counter() - started:.1f}s")
        raise
    record = getattr(run, "last_record", None)
    reused = record is not None and record.stage == stage and record.reused
    action = "reused checkpoint" if reused else "completed"
    print(f"{prefix}: {action} in {perf_counter() - started:.1f}s")
    return value


def _execute_stage(
    run: ResumableRun,
    stage: str,
    compute: Callable[[], T],
    *,
    config: Any,
    dependencies: Mapping[str, str | Path] | None = None,
    force: bool = False,
) -> T:
    return _run_with_status(
        run,
        stage,
        lambda: run.execute(
            stage,
            compute,
            config=config,
            dependencies=dependencies,
            force=force,
        ),
    )


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def make_run(artifact_root: str | Path, run_id: str, method: str) -> ResumableRun:
    return ResumableRun(Path(artifact_root), run_id, method)


def stage_artifact_path(run: ResumableRun, stage: str) -> Path:
    return run.run_dir / "stages" / stage / "artifact.pkl"


def stage_status_path(run: ResumableRun, stage: str) -> Path:
    return run.run_dir / "stages" / stage / "status.json"


def require_stage(run: ResumableRun, stage: str) -> tuple[Any, Path]:
    path = stage_artifact_path(run, stage)
    try:
        artifact = run.load_completed(stage)
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"Missing stage '{stage}' for run_id={run.run_id!r}, method={run.method!r}. "
            f"Expected {path}. Run the prerequisite notebook first."
        ) from exc
    return artifact, path


def completed_stage_records(run: ResumableRun) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in run.completed_stages():
        status_path = stage_status_path(run, stage)
        rows.append(load_artifact_json(status_path))
    return rows


def stage_status_records(run: ResumableRun) -> list[dict[str, Any]]:
    """Return every visible stage status record, including failed or running stages."""
    stages_dir = run.run_dir / "stages"
    if not stages_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for status_path in sorted(stages_dir.glob("*/status.json")):
        record = load_artifact_json(status_path)
        record.setdefault("stage", status_path.parent.name)
        record["status_path"] = str(status_path)
        rows.append(record)
    return rows


def load_artifact_json(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _stage_lock_state(run: ResumableRun, stage: str) -> str:
    lock_path = run.run_dir / "stages" / stage / ".lock"
    if not lock_path.exists():
        return ""
    try:
        lock_text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        lock_text = "present"
    return f"; lock={lock_text or 'present'}"


def print_stage_status(
    run: ResumableRun,
    expected_stages: Iterable[str] | None = None,
) -> None:
    records = {str(record.get("stage")): record for record in stage_status_records(run)}
    expected = list(expected_stages or [])
    stages = sorted(set(records) | set(expected))
    if not stages:
        print(f"No stage records for {run.run_id}/{run.method}")
        return
    print(f"Stage status for {run.run_id}/{run.method}:")
    for stage in stages:
        record = records.get(stage)
        if record is None:
            print(f"  - {stage}: not started")
            continue
        status = str(record.get("status", "unknown"))
        signature = str(record.get("signature", ""))[:10]
        artifact = Path(str(record.get("artifact_path", stage_artifact_path(run, stage))))
        artifact_state = "artifact=yes" if artifact.is_file() else "artifact=no"
        extra = f"signature={signature}, {artifact_state}{_stage_lock_state(run, stage)}"
        if status == "failed" and record.get("error"):
            extra = f"{extra}, error={record['error']}"
        print(f"  - {stage}: {status} ({extra})")


def print_notebook_header(
    run: ResumableRun,
    *,
    title: str,
    expected_stages: Iterable[str],
    force: bool,
) -> None:
    """Print the resume contract at the top of a notebook."""
    print("=" * 72)
    print(title)
    print(f"run_id={run.run_id} method={run.method}")
    print(f"run_dir={run.run_dir}")
    print(
        "resume mode: "
        + (
            "FORCE=True, matching checkpoints will be recomputed"
            if force
            else "FORCE=False, matching completed checkpoints will be reused"
        )
    )
    print("Rerun interrupted notebooks from the top; completed stages resume from checkpoints.")
    print_stage_status(run, expected_stages=expected_stages)
    print("=" * 72)


def _checkpoint_input(
    run: ResumableRun,
    *,
    stage: str,
    path: str | Path,
    force: bool = False,
) -> Any:
    artifact_path = Path(path).resolve()
    dependency_name = artifact_path.stem.replace(".", "_")
    return _execute_stage(
        run,
        stage,
        lambda: load_artifact(artifact_path),
        config={"source_name": artifact_path.name},
        dependencies={dependency_name: artifact_path},
        force=force,
    )


def run_preprocessing(
    run: ResumableRun,
    *,
    data_cfg: dict[str, Any],
    preprocess_cfg: PreprocessConfig,
    window_cfg: WindowConfig,
    force: bool = False,
) -> tuple[NeuralRecording, WindowedSegments]:
    """Load, preprocess, and window one dataset into resumable shared stages."""
    source_dependencies: dict[str, Path] = {}
    source_path = data_cfg.get("path")
    if source_path is not None:
        resolved_source = Path(source_path).resolve()
        if resolved_source.is_dir():
            configured_files = data_cfg.get("files", {})
            if not isinstance(configured_files, dict) or not configured_files:
                raise ValueError(
                    f"Directory dataset {resolved_source} must declare a non-empty files mapping"
                )
            for name, relative_path in sorted(configured_files.items()):
                dependency_path = (resolved_source / str(relative_path)).resolve()
                if not dependency_path.is_file():
                    raise FileNotFoundError(
                        f"Configured raw dataset file not found: {dependency_path}"
                    )
                source_dependencies[f"raw_{name}"] = dependency_path
        elif resolved_source.is_file():
            source_dependencies["raw_data"] = resolved_source
        else:
            raise FileNotFoundError(
                f"Raw dataset path not found: {resolved_source}. "
                "Update the dataset config before running this notebook."
            )

    raw_recording = _execute_stage(
        run,
        "raw_recording",
        lambda: get_loader(str(data_cfg["name"]))(data_cfg),
        config=data_cfg,
        dependencies=source_dependencies,
        force=force,
    )
    recording = _execute_stage(
        run,
        "recording",
        lambda: preprocess(raw_recording, preprocess_cfg),
        config=asdict(preprocess_cfg),
        dependencies={"raw_recording": stage_artifact_path(run, "raw_recording")},
        force=force,
    )
    windows = _execute_stage(
        run,
        "windows",
        lambda: make_windows(recording, window_cfg),
        config=asdict(window_cfg),
        dependencies={"recording": stage_artifact_path(run, "recording")},
        force=force,
    )
    return recording, windows


def run_bundle_net_reference_preprocessing(
    run: ResumableRun,
    *,
    preprocess_cfg: PreprocessConfig,
    exclude_neuron_names: list[str] | tuple[str, ...] = (),
    exclude_neuron_name_source: str = "raw",
    behavior_key: str = "motif",
    target_length: int = 15,
    force: bool = False,
) -> tuple[NeuralRecording, BundleTrainingBatch]:
    """Checkpoint Bundle-Net's exact retrospective filter and paired inputs."""
    raw_recording, raw_path = require_stage(run, "raw_recording")
    reference_config = {
        "preprocessing": asdict(preprocess_cfg),
        "exclude_neuron_names": list(exclude_neuron_names),
        "exclude_neuron_name_source": exclude_neuron_name_source,
    }
    reference_recording = _execute_stage(
        run,
        "bundle_net_reference_recording",
        lambda: preprocess(
            exclude_named_neurons(
                raw_recording,
                exclude_neuron_names,
                name_source=exclude_neuron_name_source,
            ),
            preprocess_cfg,
        ),
        config=reference_config,
        dependencies={"raw_recording": raw_path},
        force=force,
    )
    if behavior_key not in reference_recording.behavior:
        raise KeyError(f"behavior key {behavior_key!r} is absent from the reference recording")
    training_pairs = _execute_stage(
        run,
        "bundle_net_training_pairs",
        lambda: build_bundle_training_batch(
            reference_recording.traces.T,
            reference_recording.behavior[behavior_key],
            target_length=target_length,
        ),
        config={"behavior_key": behavior_key, "target_length": target_length},
        dependencies={
            "bundle_net_reference_recording": stage_artifact_path(
                run, "bundle_net_reference_recording"
            )
        },
        force=force,
    )
    return reference_recording, training_pairs


def run_connectivity(
    run: ResumableRun,
    *,
    windows_path: str | Path,
    connectivity_cfg: ConnectivityConfig,
    chunk_size: int = 16,
    force: bool = False,
):
    windows = _checkpoint_input(run, stage="windows_input", path=windows_path, force=False)
    estimator = ConnectivityFactory(connectivity_cfg.resolve(windows.fps))
    progress = _ProgressReporter("Connectivity", "chunk")
    try:
        return _run_with_status(
            run,
            "connectivity",
            lambda: estimate_connectivity_resumable(
                run,
                estimator,
                windows,
                chunk_size=chunk_size,
                input_artifacts={"windows": Path(windows_path).resolve()},
                force=force,
                progress_callback=progress,
            ),
        )
    finally:
        progress.close()


def run_primary_probability_evaluation(
    run: ResumableRun,
    *,
    windows_path: str | Path,
    evaluation_cfg: EvaluationConfig,
    force: bool = False,
):
    """Checkpoint the outer-owned M0/M1 evaluation used by primary notebooks."""
    windows = _checkpoint_input(run, stage="windows_input", path=windows_path, force=False)
    if evaluation_cfg.mode == "confirmatory":
        return _execute_stage(
            run, "primary_evaluation",
            lambda: run_evaluation_profile(windows, evaluation_cfg),
            config=asdict(evaluation_cfg),
            dependencies={"windows": Path(windows_path).resolve()}, force=force,
        )
    fold_ids = primary_evaluation_fold_ids(windows, evaluation_cfg)
    if not fold_ids:
        return _execute_stage(
            run,
            "primary_evaluation",
            lambda: run_primary_evaluation(windows, evaluation_cfg),
            config=asdict(evaluation_cfg),
            dependencies={"windows": Path(windows_path).resolve()},
            force=force,
        )
    progress = _ProgressReporter("Primary evaluation", "fold")
    try:
        result = run.execute_chunks(
            "primary_evaluation",
            fold_ids,
            lambda fold_id: run_primary_evaluation(
                windows,
                evaluation_cfg,
                selected_fold_ids=frozenset({fold_id}),
            ),
            lambda results: combine_primary_evaluation_results(results, cfg=evaluation_cfg),
            config=asdict(evaluation_cfg),
            dependencies={"windows": Path(windows_path).resolve()},
            force=force,
            progress_callback=progress,
        )
    finally:
        progress.close()
    for fold_id, diagnostics in result.diagnostics.get("folds", {}).items():
        print(
            f"{fold_id}: lambda1={diagnostics['lambda1']}, "
            f"M0 {diagnostics['decoder_parameter']}={diagnostics['m0_decoder_value']}, "
            f"M1 {diagnostics['decoder_parameter']}={diagnostics['m1_decoder_value']}, "
            f"train={diagnostics['training_examples']}, test={diagnostics['test_examples']}"
        )
    return result


def run_graph_states_and_transitions(
    run: ResumableRun,
    *,
    graph_cfg: GraphStateConfig,
    transition_cfg: TransitionConfig,
    force: bool = False,
):
    connectivity, connectivity_path = require_stage(run, "connectivity")
    states = _execute_stage(
        run,
        "graph_states",
        lambda: fit_graph_states(connectivity, graph_cfg),
        config=asdict(graph_cfg),
        dependencies={"connectivity": connectivity_path},
        force=force,
    )
    progress = _ProgressReporter("Transition null", "draw")
    try:
        transitions = _execute_stage(
            run,
            "transitions",
            lambda: fit_transitions(
                states.labels,
                states.n_states,
                transition_cfg,
                window_starts=states.window_starts,
                boundary_indices=states.boundary_indices,
                progress_callback=progress,
            ),
            config=asdict(transition_cfg),
            dependencies={"graph_states": stage_artifact_path(run, "graph_states")},
            force=force,
        )
    finally:
        progress.close()
    return states, transitions


def run_probabilistic_states(
    run: ResumableRun,
    *,
    probabilistic_cfg: ProbabilisticStateConfig,
    force: bool = False,
):
    connectivity, connectivity_path = require_stage(run, "connectivity")
    progress = _ProgressReporter("HMM expectation-maximization", "iteration")
    try:
        return _execute_stage(
            run,
            "probabilistic_states",
            lambda: fit_probabilistic_states(
                connectivity,
                probabilistic_cfg,
                progress_callback=progress,
            ),
            config=asdict(probabilistic_cfg),
            dependencies={"connectivity": connectivity_path},
            force=force,
        )
    finally:
        progress.close()


def run_community(
    run: ResumableRun,
    *,
    community_cfg: CommunityConfig,
    force: bool = False,
):
    connectivity, connectivity_path = require_stage(run, "connectivity")
    detector = CommunityFactory(community_cfg)
    unit = "run" if community_cfg.name == "temporal" else "window"
    progress = _ProgressReporter("Community detection", unit)
    try:
        return _execute_stage(
            run,
            "community",
            lambda: detector.run(connectivity, progress_callback=progress),
            config=asdict(community_cfg),
            dependencies={"connectivity": connectivity_path},
            force=force,
        )
    finally:
        progress.close()


def _target_slices(window_starts: np.ndarray, history_length: int, target_length: int) -> list[TargetSlice]:
    return [
        TargetSlice(int(start + history_length - target_length), int(start + history_length))
        for start in window_starts
    ]


def _target_slices_for_connectivity(
    connectivity,
    history_length: int,
    target_length: int,
) -> list[TargetSlice]:
    if connectivity.anchors:
        target_slices = [
            TargetSlice(int(anchor.target_start), int(anchor.target_stop)) for anchor in connectivity.anchors
        ]
        if any(ts.length != target_length for ts in target_slices):
            raise ValueError("connectivity anchors do not match the requested target_length")
        return target_slices
    return _target_slices(connectivity.window_starts, history_length, target_length)


def _cross_fitted_window_embedding(
    recording,
    connectivity,
    target_slices: list[TargetSlice],
    manifold_cfg: ManifoldConfig,
    linking_cfg: dict[str, Any],
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if not connectivity.anchors:
        raise ValueError("cross-fitted manifold learning requires typed temporal anchors")
    if manifold_cfg.name != "classical" or manifold_cfg.method != "pca":
        raise ValueError(
            "prospective cross-fitting currently requires classical PCA; fold-specific UMAP "
            "or learned coordinates are retrospective until an identified alignment is added"
        )
    preprocess_dependency = recording.metadata.get("preprocess_dependency", {})
    if preprocess_dependency.get("kind") == "global":
        raise ValueError(
            "recording-global preprocessing is incompatible with prospective manifold cross-fitting"
        )
    dependency_support = DependencySupport(
        lag_extension=int(linking_cfg.get("lag_extension", 0)),
        preprocessing_past=max(
            int(linking_cfg.get("preprocessing_past_support", 0)),
            int(preprocess_dependency.get("past_support", 0)),
        ),
        preprocessing_future=max(
            int(linking_cfg.get("preprocessing_future_support", 0)),
            int(preprocess_dependency.get("future_support", 0)),
        ),
    )
    groups = anchor_group_labels(
        connectivity.anchors,
        group_by=str(linking_cfg.get("group_by", "recording")),
    ).tolist()
    splits = purged_blocked_splits(
        connectivity.n_windows,
        manifold_cfg.n_folds,
        embargo=int(linking_cfg.get("embargo", 0)),
        anchors=connectivity.anchors,
        groups=groups,
        dependency_support=dependency_support,
    )
    embeddings = np.full((connectivity.n_windows, manifold_cfg.n_dims), np.nan, dtype=np.float32)
    fold_ids = np.full(connectivity.n_windows, -1, dtype=int)
    fit_sample_counts: list[int] = []
    neural, _ = causal_forward_fill_nonfinite(recording.traces.T)
    for fold_id, (train, test) in enumerate(splits):
        fit_mask = np.zeros(recording.n_timepoints, dtype=bool)
        for idx in train:
            anchor = connectivity.anchors[int(idx)]
            start = max(0, anchor.context_start - dependency_support.total_past)
            stop = min(recording.n_timepoints, anchor.context_stop + dependency_support.preprocessing_future)
            fit_mask[start:stop] = True
        fit_indices = np.flatnonzero(fit_mask)
        if fit_indices.size <= manifold_cfg.n_dims:
            raise ValueError("too few purged samples to fit the held-out manifold fold")
        fold_behavior = {
            key: np.asarray(values)[fit_indices] for key, values in recording.behavior.items()
        }
        embedder = ManifoldFactory(manifold_cfg)
        embedder.fit(neural[fit_indices], fold_behavior)
        embeddings[test] = embedder.transform_targets(
            neural,
            [target_slices[int(idx)] for idx in test],
            recording.behavior,
        )
        fold_ids[test] = fold_id
        fit_sample_counts.append(int(fit_indices.size))
        if progress_callback is not None:
            progress_callback(fold_id + 1, len(splits))
    if np.any(fold_ids < 0) or not np.all(np.isfinite(embeddings)):
        raise RuntimeError("cross-fitted manifold embedding is incomplete")
    return embeddings, fold_ids, fit_sample_counts


def run_manifold(
    run: ResumableRun,
    *,
    recording_path: str | Path,
    manifold_cfg: ManifoldConfig,
    linking_cfg: dict[str, Any],
    history_length: int | None = None,
    force: bool = False,
) -> ManifoldArtifact:
    recording = _checkpoint_input(run, stage="recording_input", path=recording_path, force=False)
    connectivity, connectivity_path = require_stage(run, "connectivity")

    if history_length is None:
        if connectivity.anchors:
            history_length = int(connectivity.anchors[0].history_length)
        else:
            raise ValueError("history_length is required when connectivity anchors are unavailable")
    manifold_cfg = manifold_cfg.resolve(recording.fps)
    if connectivity.anchors:
        anchor_target_lengths = {anchor.target_length for anchor in connectivity.anchors}
        if len(anchor_target_lengths) != 1:
            raise ValueError("all connectivity anchors must share one target length")
        manifold_cfg = replace(
            manifold_cfg,
            target_length=anchor_target_lengths.pop(),
        )
    target_slices = _target_slices_for_connectivity(
        connectivity, history_length, manifold_cfg.target_length
    )

    training_progress = _ProgressReporter("Manifold training", "iteration")
    fold_progress = _ProgressReporter("Cross-fitted manifold", "fold")

    def _compute() -> ManifoldArtifact:
        neural, missing_value_contract = causal_forward_fill_nonfinite(recording.traces.T)
        embedder = ManifoldFactory(manifold_cfg)
        embedder.fit(neural, recording.behavior, progress_callback=training_progress)
        full_embedding = embedder.transform(neural)
        fold_ids = None
        fit_sample_counts: list[int] = []
        if manifold_cfg.cross_fit:
            window_embedding, fold_ids, fit_sample_counts = _cross_fitted_window_embedding(
                recording,
                connectivity,
                target_slices,
                manifold_cfg,
                linking_cfg,
                progress_callback=fold_progress,
            )
            mode = "cross_fitted"
        else:
            window_embedding = embedder.transform_targets(neural, target_slices, recording.behavior)
            mode = "retrospective_full_fit"
        model_path = run.run_dir / "models" / f"manifold-{manifold_cfg.name}.pkl"
        embedder.save(model_path)
        return ManifoldArtifact(
            method=manifold_cfg.name,
            behavior_key=manifold_cfg.behavior_key,
            full_embedding=full_embedding,
            window_embedding=window_embedding,
            target_slices=target_slices,
            target_length=manifold_cfg.target_length,
            model_path=str(model_path),
            window_starts=np.asarray(connectivity.window_starts, dtype=np.int64),
            anchors=list(connectivity.anchors),
            provenance=ArtifactProvenance(
                identity=connectivity.provenance.identity,
                stage="manifold",
                source=connectivity.provenance.source,
                configuration_id=manifold_cfg.name,
                random_seed=manifold_cfg.seed,
                code_version=connectivity.provenance.code_version,
                fit_data_ids=(recording.identity.recording_id,),
                transform_data_ids=(recording.identity.recording_id,),
                metadata={"upstream_stage": connectivity.provenance.stage},
            ),
            metadata={
                "history_length": int(history_length),
                "window_embedding_mode": mode,
                "cross_fit_fold_ids": fold_ids,
                "cross_fit_fit_sample_counts": fit_sample_counts,
                "prospective_eligible": bool(manifold_cfg.cross_fit),
                "missing_value_contract": missing_value_contract,
                "fold_coordinate_contract": (
                    "ordered deterministic PCA coordinates; derivatives must break at fold boundaries"
                    if manifold_cfg.cross_fit
                    else "single full-fit coordinate system for retrospective use"
                ),
                "full_embedding_claim_boundary": "retrospective_visualization_only",
                "anchors": list(connectivity.anchors),
                "target_alignment": (
                    "anchor_target_interval"
                    if connectivity.anchors
                    else "window_start_fallback"
                ),
                "target_length_source": (
                    "connectivity_anchor" if connectivity.anchors else "manifold_config"
                ),
                "full_embedding_indexing": (
                    "target_window_end" if manifold_cfg.name == "bunddle" else "sample"
                ),
                "full_embedding_sample_offset": (
                    manifold_cfg.target_length - 1 if manifold_cfg.name == "bunddle" else 0
                ),
            },
        )

    try:
        return _execute_stage(
            run,
            "manifold",
            _compute,
            config={
                "manifold": asdict(manifold_cfg),
                "linking": linking_cfg,
                "history_length": history_length,
            },
            dependencies={
                "recording": Path(recording_path).resolve(),
                "connectivity": connectivity_path,
            },
            force=force,
        )
    finally:
        training_progress.close()
        fold_progress.close()


def run_linking(
    run: ResumableRun,
    *,
    recording_path: str | Path,
    linking_cfg: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    connectivity, connectivity_path = require_stage(run, "connectivity")
    states, states_path = require_stage(run, "graph_states")
    community, community_path = require_stage(run, "community")
    manifold, manifold_path = require_stage(run, "manifold")
    recording = _checkpoint_input(run, stage="recording_input", path=recording_path, force=False)

    def _compute() -> dict[str, Any]:
        def association_with_progress(label: str, **kwargs: Any) -> Any:
            progress = _ProgressReporter(label, "draw")
            try:
                return association_with_null(**kwargs, progress_callback=progress)
            finally:
                progress.close()

        anchors = list(connectivity.anchors) if connectivity.anchors else None
        groups = (
            anchor_group_labels(
                anchors,
                group_by=str(linking_cfg.get("group_by", "recording")),
            ).tolist()
            if anchors
            else None
        )
        continuity_groups = (
            anchor_continuity_labels(
                anchors,
                group_by=str(linking_cfg.get("group_by", "recording")),
            )
            if anchors
            else None
        )
        preprocess_dependency = recording.metadata.get("preprocess_dependency", {})
        if str(preprocess_dependency.get("kind", "")).startswith("global") and not bool(
            linking_cfg.get("allow_global_preprocessing", False)
        ):
            raise ValueError(
                "recording-global preprocessing cannot support prospective linking"
            )
        dependency_support = DependencySupport(
            lag_extension=int(linking_cfg.get("lag_extension", 0)),
            preprocessing_past=max(
                int(linking_cfg.get("preprocessing_past_support", 0)),
                int(preprocess_dependency.get("past_support", 0)),
            ),
            preprocessing_future=max(
                int(linking_cfg.get("preprocessing_future_support", 0)),
                int(preprocess_dependency.get("future_support", 0)),
            ),
        )
        conn_feat = connectivity_features(connectivity)
        state_feat = state_features(states.labels, states.n_states)
        community_feat = community_features(community)
        reconfiguration_feat = community_reconfiguration_features(community)
        manifold_prospective = bool(manifold.metadata.get("prospective_eligible", False))
        fold_ids = manifold.metadata.get("cross_fit_fold_ids") if manifold_prospective else None
        manifold_groups = combined_continuity_groups(
            continuity_groups,
            None if fold_ids is None else np.asarray(fold_ids),
            connectivity.n_windows,
        )
        manifold_dyn = manifold_speed(
            manifold.window_embedding,
            groups=manifold_groups,
        ).reshape(-1, 1)
        activity_feat = (
            activity_magnitude_features(recording.traces.T, anchors)
            if anchors
            else np.empty((connectivity.n_windows, 0), dtype=float)
        )
        report: dict[str, Any] = {
            "decoding": [],
            "association": {},
            "lead_lag": {},
            "lead_lag_activity_adjusted": {},
            "incremental": {},
            "positive_lag_incremental": {},
            "effectome_to_future_manifold": {},
            "splitter": {
                "mode": "anchor_aware_grouped_purged" if anchors is not None else "purged_blocked",
                "group_by": str(linking_cfg.get("group_by", "recording")) if anchors else None,
                "continuity_boundaries": "recording_hard_gap_and_sparse_bad_frame",
                "embargo": int(linking_cfg["embargo"]),
                "raw_dependency_support": {
                    "lag_extension": dependency_support.lag_extension,
                    "preprocessing_past": dependency_support.preprocessing_past,
                    "preprocessing_future": dependency_support.preprocessing_future,
                    "preprocess_kind": preprocess_dependency.get("kind", "undeclared"),
                },
            },
            "manifold_feature_mode": manifold.metadata.get("window_embedding_mode", "undeclared"),
            "manifold_derivative_boundaries": (
                "recording_hard_gap_sparse_bad_frame_and_cross_fit_fold"
            ),
        }
        switching_count = np.rint(
            reconfiguration_feat[:, 0] * float(community.labels.shape[1])
        ).astype(int)
        report["manifold_alignment"] = {
            "state_to_manifold_speed_lead_lag": lead_lag(
                states.labels,
                manifold_dyn[:, 0],
                max_lag=int(linking_cfg["max_lag"]),
                n_bins=int(linking_cfg["n_bins"]),
                target_name="manifold_speed",
                groups=manifold_groups,
            ),
            "state_to_manifold_speed_association": association_with_progress(
                "State-manifold null",
                states=states.labels,
                behavior=manifold_dyn[:, 0],
                n_null=int(linking_cfg["n_null"]),
                seed=int(linking_cfg["seed"]),
                n_bins=int(linking_cfg["n_bins"]),
                null_kind=str(linking_cfg.get("null_kind", "circular_shift")),
                block_length=int(linking_cfg.get("block_length", 8)),
                groups=manifold_groups,
            ),
            "switching_count_to_manifold_speed_association": association_with_progress(
                "Switching-manifold null",
                states=switching_count,
                behavior=manifold_dyn[:, 0],
                n_null=int(linking_cfg["n_null"]),
                seed=int(linking_cfg["seed"]) + 1,
                n_bins=int(linking_cfg["n_bins"]),
                null_kind=str(linking_cfg.get("null_kind", "circular_shift")),
                block_length=int(linking_cfg.get("block_length", 8)),
                groups=manifold_groups,
            ),
            "mode": (
                "prospective_cross_fitted"
                if manifold_prospective
                else "retrospective_descriptive"
            ),
            "claim_boundary": (
                "cross_fitted_current_state_alignment"
                if manifold_prospective
                else "full_fit_manifold_correspondence_only_not_prospective_or_causal"
            ),
        }
        positive_lag = int(linking_cfg.get("positive_lag", 1))
        if not manifold_prospective:
            report["effectome_to_future_manifold"] = {
                "lag": positive_lag,
                "status": "invalid_non_cross_fitted_manifold",
                "claim_boundary": (
                    "retrospective_manifold_not_eligible_for_prospective_prediction"
                ),
            }
        else:
            manifold_origins = valid_positive_lag_origins(
                connectivity.n_windows,
                positive_lag,
                anchors=anchors,
            )
            if manifold_groups is not None and manifold_origins.size:
                manifold_group_arr = np.asarray(manifold_groups, dtype=object)
                manifold_origins = manifold_origins[
                    manifold_group_arr[manifold_origins]
                    == manifold_group_arr[manifold_origins + positive_lag]
                ]
            if manifold_origins.size < 4:
                report["effectome_to_future_manifold"] = {
                    "lag": positive_lag,
                    "status": "too_few_within_segment_within_chart_transitions",
                }
            else:
                _, future_manifold_distance = future_manifold_displacement(
                    manifold.window_embedding,
                    manifold_origins,
                    lag=positive_lag,
                    groups=manifold_groups,
                )
                manifold_lag_anchors = (
                    [anchors[int(idx)] for idx in manifold_origins] if anchors else None
                )
                manifold_future_anchors = (
                    [anchors[int(idx + positive_lag)] for idx in manifold_origins]
                    if anchors
                    else None
                )
                manifold_lag_groups = (
                    np.asarray(manifold_groups, dtype=object)[manifold_origins].tolist()
                    if manifold_groups is not None
                    else None
                )
                manifold_report: dict[str, Any] = {
                    "lag": positive_lag,
                    "n_transitions": int(manifold_origins.size),
                    "outcome": "euclidean_norm_of_z_future_minus_z_current",
                    "raw_effectome_decode": decode_behavior(
                        conn_feat[manifold_origins],
                        future_manifold_distance,
                        "connectivity",
                        "future_manifold_displacement",
                        n_folds=int(linking_cfg["n_folds"]),
                        seed=int(linking_cfg["seed"]),
                        embargo=int(linking_cfg["embargo"]),
                        anchors=manifold_lag_anchors,
                        groups=manifold_lag_groups,
                        dependency_support=dependency_support,
                        outcome_anchors=manifold_future_anchors,
                    ),
                    "coordinate_contract": (
                        "magnitude_is_rotation_reflection_invariant; transitions never cross "
                        "folds, recordings, or declared gaps"
                    ),
                    "claim_boundary": "prospective_prediction_not_interventional_causation",
                }
                if getattr(community, "mode", None) == "prospective":
                    manifold_baseline = np.concatenate(
                        [
                            manifold_dyn[manifold_origins],
                            activity_feat[manifold_origins],
                            conn_feat[manifold_origins],
                        ],
                        axis=1,
                    )
                    manifold_report["community_reconfiguration_increment"] = (
                        incremental_decode_behavior(
                            manifold_baseline,
                            reconfiguration_feat[manifold_origins],
                            future_manifold_distance,
                            n_folds=int(linking_cfg["n_folds"]),
                            seed=int(linking_cfg["seed"]),
                            embargo=int(linking_cfg["embargo"]),
                            anchors=manifold_lag_anchors,
                            groups=manifold_lag_groups,
                            dependency_support=dependency_support,
                            outcome_anchors=manifold_future_anchors,
                        )
                    )
                    manifold_report["community_feature_contract"] = (
                        "switching_fraction_plus_neuron_resolved_allegiance_changes"
                    )
                else:
                    manifold_report["community_reconfiguration_status"] = (
                        "invalid_future_aware_community_features"
                    )
                report["effectome_to_future_manifold"] = manifold_report
        if manifold_prospective:
            report["lead_lag"]["manifold_speed"] = lead_lag(
                states.labels,
                manifold_dyn[:, 0],
                max_lag=int(linking_cfg["max_lag"]),
                n_bins=int(linking_cfg["n_bins"]),
                target_name="manifold_speed",
                groups=manifold_groups,
            )
            report["lead_lag_activity_adjusted"]["manifold_speed"] = lead_lag(
                states.labels,
                manifold_dyn[:, 0],
                max_lag=int(linking_cfg["max_lag"]),
                n_bins=int(linking_cfg["n_bins"]),
                target_name="manifold_speed",
                groups=manifold_groups,
                controls=activity_feat if activity_feat.shape[1] else None,
            )
        for behavior_key in linking_cfg["behavior_keys"]:
            if behavior_key not in connectivity.behavior_per_window:
                print(f"[linking] skipping absent behavior: {behavior_key}")
                continue
            behavior = np.asarray(connectivity.behavior_per_window[behavior_key])
            behavior = (
                behavior.astype(np.int64)
                if np.allclose(behavior, np.round(behavior))
                else behavior.astype(float)
            )
            feature_sets = [
                ("connectivity", conn_feat),
                ("state", state_feat),
                ("community", community_feat),
            ]
            if manifold_prospective:
                feature_sets.append(("manifold_dynamics", manifold_dyn))
            if activity_feat.shape[1]:
                feature_sets.append(("activity_magnitude", activity_feat))
            print(
                f"[linking] {behavior_key}: decoding {len(feature_sets)} feature sets "
                "with blocked cross-validation"
            )
            for name, features in feature_sets:
                print(f"[linking] {behavior_key}: decoding {name}")
                report["decoding"].append(
                    decode_behavior(
                        features,
                        behavior,
                        name,
                        behavior_key,
                        n_folds=int(linking_cfg["n_folds"]),
                        seed=int(linking_cfg["seed"]),
                        embargo=int(linking_cfg["embargo"]),
                        anchors=anchors,
                        groups=groups,
                        dependency_support=dependency_support,
                    )
                )
            report["association"][behavior_key] = association_with_progress(
                f"{behavior_key} association null",
                states=states.labels,
                behavior=behavior,
                n_null=int(linking_cfg["n_null"]),
                seed=int(linking_cfg["seed"]),
                n_bins=int(linking_cfg["n_bins"]),
                null_kind=str(linking_cfg.get("null_kind", "circular_shift")),
                block_length=int(linking_cfg.get("block_length", 8)),
                groups=continuity_groups,
            )
            report["lead_lag"][behavior_key] = lead_lag(
                states.labels,
                behavior,
                max_lag=int(linking_cfg["max_lag"]),
                n_bins=int(linking_cfg["n_bins"]),
                target_name=behavior_key,
                groups=continuity_groups,
            )
            report["lead_lag_activity_adjusted"][behavior_key] = lead_lag(
                states.labels,
                behavior,
                max_lag=int(linking_cfg["max_lag"]),
                n_bins=int(linking_cfg["n_bins"]),
                target_name=behavior_key,
                groups=continuity_groups,
                controls=activity_feat if activity_feat.shape[1] else None,
            )
            baseline_parts = [state_feat, activity_feat]
            if manifold_prospective:
                baseline_parts.append(manifold_dyn)
            baseline = np.concatenate(baseline_parts, axis=1)
            report["incremental"][behavior_key] = incremental_decode_behavior(
                baseline,
                community_feat,
                behavior,
                n_folds=int(linking_cfg["n_folds"]),
                seed=int(linking_cfg["seed"]),
                embargo=int(linking_cfg["embargo"]),
                anchors=anchors,
                groups=groups,
                dependency_support=dependency_support,
            )

            community_mode = getattr(community, "mode", None)
            if not manifold_prospective or community_mode != "prospective":
                report["positive_lag_incremental"][behavior_key] = {
                    "lag": positive_lag,
                    "status": (
                        "invalid_non_cross_fitted_manifold"
                        if not manifold_prospective
                        else "invalid_future_aware_community_features"
                    ),
                    "community_mode": community_mode,
                }
                continue
            origins = valid_positive_lag_origins(
                connectivity.n_windows,
                positive_lag,
                anchors=anchors,
            )
            if origins.size < 4:
                report["positive_lag_incremental"][behavior_key] = {
                    "lag": positive_lag,
                    "status": "insufficient_samples",
                    "community_mode": community_mode,
                }
                continue
            lag_anchors = [anchors[int(idx)] for idx in origins] if anchors else None
            future_anchors = [anchors[int(idx + positive_lag)] for idx in origins] if anchors else None
            lag_groups = (
                anchor_group_labels(
                    lag_anchors,
                    group_by=str(linking_cfg.get("group_by", "recording")),
                ).tolist()
                if lag_anchors
                else None
            )
            future_behavior = behavior[origins + positive_lag]
            current_behavior = behavior[origins].astype(float).reshape(-1, 1)
            predictive_baseline = np.concatenate(
                [
                    current_behavior,
                    manifold_dyn[origins],
                    activity_feat[origins],
                    conn_feat[origins],
                ],
                axis=1,
            )
            report["positive_lag_incremental"][behavior_key] = incremental_decode_behavior(
                predictive_baseline,
                community_feat[origins],
                future_behavior,
                n_folds=int(linking_cfg["n_folds"]),
                seed=int(linking_cfg["seed"]),
                embargo=int(linking_cfg["embargo"]),
                anchors=lag_anchors,
                groups=lag_groups,
                dependency_support=dependency_support,
                outcome_anchors=future_anchors,
            )
        return report

    return _execute_stage(
        run,
        "linking",
        _compute,
        config=linking_cfg,
        dependencies={
            "recording": Path(recording_path).resolve(),
            "connectivity": connectivity_path,
            "graph_states": states_path,
            "community": community_path,
            "manifold": manifold_path,
        },
        force=force,
    )


def run_owned_feature_artifacts(
    run: ResumableRun,
    *,
    windows_path: str | Path,
    connectivity_cfg: ConnectivityConfig,
    fit_scope_id: str,
    fit_indices: np.ndarray | None = None,
    evaluation_indices: np.ndarray | None = None,
    evaluation_fraction: float = 0.2,
    n_dims: int = 3,
    n_roles: int = 3,
    seed: int = 42,
    force: bool = False,
) -> OwnedFeatureArtifacts:
    """Fit a separate outer-owned bundle without reusing descriptive features."""
    windows = _checkpoint_input(run, stage="windows_input", path=windows_path, force=False)
    if fit_indices is None and evaluation_indices is None:
        fit_indices, evaluation_indices = chronological_feature_split(
            windows, evaluation_fraction=evaluation_fraction
        )
    if fit_indices is None or evaluation_indices is None:
        raise ValueError("supply both fit_indices and evaluation_indices, or derive both")
    return _execute_stage(
        run, "outer_owned_features",
        lambda: fit_owned_feature_artifacts(
            windows, connectivity_cfg=connectivity_cfg, fit_indices=fit_indices,
            evaluation_indices=evaluation_indices, fit_scope_id=fit_scope_id,
            n_dims=n_dims, n_roles=n_roles, seed=seed,
        ),
        config={"connectivity": asdict(connectivity_cfg), "fit_indices": fit_indices,
                "evaluation_indices": evaluation_indices, "fit_scope_id": fit_scope_id,
                "n_dims": n_dims, "n_roles": n_roles, "seed": seed},
        dependencies={"windows": Path(windows_path).resolve()}, force=force,
    )


def _owned_notebook_features(
    run: ResumableRun, fit_indices: np.ndarray | None,
    evaluation_indices: np.ndarray | None, fit_scope_id: str | None,
) -> tuple[OwnedFeatureArtifacts, Path]:
    if fit_indices is None or evaluation_indices is None or fit_scope_id is None:
        raise ValueError(
            "revised notebooks require explicit outer ownership; run_owned_feature_artifacts first"
        )
    features, path = require_stage(run, "outer_owned_features")
    if (
        features.fit_scope_id != fit_scope_id
        or not np.array_equal(features.fit_indices, fit_indices)
        or not np.array_equal(features.evaluation_indices, evaluation_indices)
    ):
        raise ValueError("requested ownership differs from the frozen outer-owned feature bundle")
    return features, path


def run_candidate_drivers(
    run: ResumableRun,
    *,
    recording_path: str | Path,
    behavior_key: str,
    lag: int = 1,
    n_folds: int = 5,
    embargo: int = 0,
    seed: int = 42,
    matched_control_percentile: float = 95.0,
    fit_indices: np.ndarray | None = None,
    evaluation_indices: np.ndarray | None = None,
    fit_scope_id: str | None = None,
    force: bool = False,
):
    features, features_path = _owned_notebook_features(run, fit_indices, evaluation_indices, fit_scope_id)
    connectivity, community, manifold = features.connectivity, features.community, features.manifold
    if not connectivity.directed:
        raise ValueError("candidate-driver attribution requires a directed connectivity estimator")
    _checkpoint_input(run, stage="recording_input", path=recording_path, force=False)
    if behavior_key not in connectivity.behavior_per_window:
        raise KeyError(f"behavior key {behavior_key!r} is absent from connectivity windows")
    behavior = np.asarray(connectivity.behavior_per_window[behavior_key])
    return _execute_stage(
        run,
        "candidate_drivers",
        lambda: qualify_candidate_drivers(
            connectivity,
            community,
            manifold,
            behavior,
            behavior_key,
            lag=lag,
            n_folds=n_folds,
            embargo=embargo,
            seed=seed,
            matched_control_percentile=matched_control_percentile,
            fit_indices=fit_indices,
            evaluation_indices=evaluation_indices,
            fit_scope_id=fit_scope_id,
            require_feature_ownership=True,
        ),
        config={
            "behavior_key": behavior_key,
            "lag": lag,
            "n_folds": n_folds,
            "embargo": embargo,
            "seed": seed,
            "matched_control_percentile": matched_control_percentile,
            "fit_indices": fit_indices,
            "evaluation_indices": evaluation_indices,
            "fit_scope_id": fit_scope_id,
        },
        dependencies={
            "recording": Path(recording_path).resolve(),
            "outer_owned_features": features_path,
        },
        force=force,
    )


def run_feature_sensitivity(
    run: ResumableRun,
    *,
    recording_path: str | Path,
    behavior_key: str,
    lag: int = 1,
    ridge_alpha: float = 1.0,
    n_folds: int = 5,
    embargo: int = 0,
    min_skill: float = 0.0,
    scale: float = 0.0,
    dose_scales: list[float] | None = None,
    node_indices: list[int] | None = None,
    top_k: int = 1,
    endpoint: str = "behavior",
    seed: int = 42,
    fit_indices: np.ndarray | None = None,
    evaluation_indices: np.ndarray | None = None,
    fit_scope_id: str | None = None,
    multiplicity_family: tuple[str, ...] | None = None,
    hypothesis_id: str | None = None,
    min_absolute_skill: float = 0.0,
    max_distribution_shift: float = 5.0,
    n_controls: int = 64,
    control_alpha: float = 0.05,
    control_resolution_policy: str = "unavailable",
    force: bool = False,
):
    features, features_path = _owned_notebook_features(run, fit_indices, evaluation_indices, fit_scope_id)
    connectivity, manifold = features.connectivity, features.manifold
    if not connectivity.directed:
        raise ValueError("directional feature-sensitivity analysis requires directed connectivity")
    _checkpoint_input(run, stage="recording_input", path=recording_path, force=False)
    if behavior_key not in connectivity.behavior_per_window:
        raise KeyError(f"behavior key {behavior_key!r} is absent from connectivity windows")
    behavior = np.asarray(connectivity.behavior_per_window[behavior_key])
    candidates, candidates_path = require_stage(run, "candidate_drivers")
    if candidates.fit_scope_id != fit_scope_id:
        raise ValueError("candidate screen is not owned by the requested outer scope")
    if not multiplicity_family or hypothesis_id not in multiplicity_family:
        raise ValueError("declare the complete frozen multiplicity_family and hypothesis_id before fitting")

    if node_indices is None:
        node_indices = [score.node for score in candidates.scores if score.is_candidate][:top_k]
    if not set(node_indices).issubset(candidates.candidate_nodes):
        raise ValueError("feature edits must use candidates frozen by the owned training screen")
    ownership_config = {
        "fit_indices": fit_indices, "evaluation_indices": evaluation_indices,
        "fit_scope_id": fit_scope_id, "candidate_nodes": node_indices,
        "multiplicity_family": multiplicity_family, "hypothesis_id": hypothesis_id,
        "min_absolute_skill": min_absolute_skill, "max_distribution_shift": max_distribution_shift,
        "n_controls": n_controls, "control_alpha": control_alpha,
        "control_resolution_policy": control_resolution_policy,
        "dose_scales": dose_scales, "interval_seed": seed,
    }
    if not node_indices:
        return _execute_stage(
            run, "feature_sensitivity",
            lambda: PerturbationResult(
                status="unavailable", node_indices=[], scale=scale, effect_size=None,
                control_effects=[], control_pvalue=None, endpoint=endpoint,
                validation_status="unavailable", fail_safe_reason="no_preliminary_candidate_nodes",
                candidate_stage="screened_out", effect_status="unavailable",
                control_pvalue_status="unavailable", sham_effect_status="unavailable",
                provenance={"fit_scope_id": fit_scope_id, "multiplicity_family": multiplicity_family,
                            "hypothesis_id": hypothesis_id},
            ),
            config={"behavior_key": behavior_key, "endpoint": endpoint, **ownership_config},
            dependencies={"outer_owned_features": features_path, "candidate_drivers": candidates_path},
            force=force,
        )

    surrogate = _execute_stage(
        run,
        "surrogate_model",
        lambda: fit_linear_surrogate(
            connectivity,
            manifold,
            behavior,
            lag=lag,
            ridge_alpha=ridge_alpha,
            n_folds=n_folds,
            embargo=embargo,
            min_skill=min_skill,
            endpoint=endpoint,
            fit_indices=fit_indices, evaluation_indices=evaluation_indices,
            fit_scope_id=fit_scope_id, candidate_nodes=node_indices,
            strict_outer_ownership=True, require_feature_ownership=True,
            multiplicity_family=multiplicity_family, hypothesis_id=hypothesis_id,
            min_absolute_skill=min_absolute_skill, max_shift=max_distribution_shift,
            n_controls=n_controls, control_alpha=control_alpha,
            control_resolution_policy=control_resolution_policy,
            dose_scales=dose_scales, interval_seed=seed,
        ),
        config={
            "behavior_key": behavior_key,
            "lag": lag,
            "ridge_alpha": ridge_alpha,
            "n_folds": n_folds,
            "embargo": embargo,
            "min_skill": min_skill,
            "endpoint": endpoint,
            "analysis_kind": "predictor_feature_sensitivity",
            "biological_intervention_identified": False,
            **ownership_config,
        },
        dependencies={
            "recording": Path(recording_path).resolve(),
            "outer_owned_features": features_path,
            "candidate_drivers": candidates_path,
        },
        force=force,
    )
    return _execute_stage(
        run,
        "feature_sensitivity",
        lambda: run_predictor_feature_sensitivity(
            surrogate,
            connectivity,
            manifold,
            behavior,
            node_indices=node_indices,
            scale=scale,
            endpoint=endpoint,
            dose_scales=dose_scales,
            seed=seed,
            n_controls=n_controls,
            hypothesis_id=hypothesis_id,
        ),
        config={
            "behavior_key": behavior_key,
            "lag": lag,
            "node_indices": node_indices,
            "scale": scale,
            "dose_scales": dose_scales,
            "endpoint": endpoint,
            "seed": seed,
            "analysis_kind": "predictor_feature_sensitivity",
            "biological_intervention_identified": False,
            **ownership_config,
        },
        dependencies={
            "recording": Path(recording_path).resolve(),
            "outer_owned_features": features_path,
            "candidate_drivers": candidates_path,
            "surrogate_model": stage_artifact_path(run, "surrogate_model"),
        },
        force=force,
    )


def run_counterfactual_perturbation(
    run: ResumableRun,
    *,
    recording_path: str | Path,
    behavior_key: str,
    lag: int = 1,
    ridge_alpha: float = 1.0,
    n_folds: int = 5,
    embargo: int = 0,
    min_skill: float = 0.0,
    scale: float = 0.0,
    dose_scales: list[float] | None = None,
    node_indices: list[int] | None = None,
    top_k: int = 1,
    endpoint: str = "behavior",
    seed: int = 42,
    force: bool = False,
):
    """Backward-compatible wrapper for notebooks created before WP10."""
    return run_feature_sensitivity(
        run,
        recording_path=recording_path,
        behavior_key=behavior_key,
        lag=lag,
        ridge_alpha=ridge_alpha,
        n_folds=n_folds,
        embargo=embargo,
        min_skill=min_skill,
        scale=scale,
        dose_scales=dose_scales,
        node_indices=node_indices,
        top_k=top_k,
        endpoint=endpoint,
        seed=seed,
        force=force,
    )
