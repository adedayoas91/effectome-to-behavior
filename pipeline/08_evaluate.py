"""Stage 8: run the split-owned probability-level M0/M1 evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from effectome.linking import valid_positive_lag_origins
from effectome.utils import hydra, set_seed
from effectome.utils.io import load_artifact, save_artifact
from effectome.workflows.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    build_primary_inputs_from_manifest,
    build_split_manifest,
    combine_primary_evaluation_results,
    primary_evaluation_fold_ids,
    run_evaluation_profile,
    run_primary_evaluation,
)
from effectome.workflows.resume import ResumableRun, content_fingerprint


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def preflight_manifest(manifest_data: dict[str, Any]) -> dict[str, Any]:
    """Validate raw inputs and split feasibility without fitting a model."""
    windows, evaluation_cfg = build_primary_inputs_from_manifest(manifest_data)
    origins = valid_positive_lag_origins(
        windows.n_windows, evaluation_cfg.positive_lag, windows.anchors, strict=True
    )
    if origins.size < 4:
        raise ValueError("too few strictly future endpoint origins for primary evaluation")
    split = build_split_manifest(
        len(origins), evaluation_cfg,
        anchors=[windows.anchors[int(index)] for index in origins],
    )
    endpoint = windows.behavior_per_window.get(evaluation_cfg.endpoint_name)
    if endpoint is None:
        raise ValueError(f"endpoint {evaluation_cfg.endpoint_name!r} is absent from raw inputs")
    if evaluation_cfg.endpoint_type == "categorical" and evaluation_cfg.class_vocabulary:
        unknown = set(np.asarray(endpoint).tolist()) - set(evaluation_cfg.class_vocabulary)
        if unknown:
            raise ValueError(f"raw endpoint values are absent from the declared vocabulary: {unknown}")
    confirmation_plan = None
    if evaluation_cfg.mode == "confirmatory":
        from effectome.workflows.confirmation import preflight_confirmation

        confirmation_plan = preflight_confirmation(evaluation_cfg)
    available = sum(fold.status == "available" for fold in split.outer_folds)
    return {
        "status": "ready_for_model_fitting" if available else "unavailable",
        "models_fitted": False,
        "profile": evaluation_cfg.profile,
        "mode": evaluation_cfg.mode,
        "endpoint_name": evaluation_cfg.endpoint_name,
        "endpoint_kind": evaluation_cfg.endpoint_kind,
        "endpoint_verified": evaluation_cfg.endpoint_verified,
        "calibration_status": evaluation_cfg.calibration_status,
        "confirmation_plan": confirmation_plan,
        "neurons": windows.n_neurons,
        "sampling_rate_hz": windows.fps,
        "windows": windows.n_windows,
        "strictly_future_origins": len(origins),
        "recording_ids": sorted({anchor.recording_id for anchor in windows.anchors}),
        "animal_ids": sorted({str(anchor.animal_id) for anchor in windows.anchors}),
        "available_folds": available,
        "unavailable_folds": len(split.outer_folds) - available,
        "split_manifest": asdict(split),
    }


def run_manifest_checkpointed(
    manifest_data: dict[str, Any], output_dir: Path, *, force: bool = False
) -> EvaluationResult:
    """Resume development folds; preserve a single confirmatory cohort fit."""
    windows, evaluation_cfg = build_primary_inputs_from_manifest(manifest_data)
    run = ResumableRun(output_dir / "checkpoints", "raw-evaluation", evaluation_cfg.profile)
    signature = {
        "evaluation": asdict(evaluation_cfg),
        "inputs": content_fingerprint(windows),
    }
    if evaluation_cfg.profile == "revised_primary" and evaluation_cfg.mode == "development":
        fold_ids = primary_evaluation_fold_ids(windows, evaluation_cfg)
        if fold_ids:
            return run.execute_chunks(
                "evaluation", fold_ids,
                lambda fold_id: run_primary_evaluation(
                    windows, evaluation_cfg, selected_fold_ids=frozenset({fold_id})
                ),
                lambda results: combine_primary_evaluation_results(results, cfg=evaluation_cfg),
                config=signature, force=force,
                progress_callback=lambda completed, total, reused: print(
                    f"Evaluation fold {completed}/{total}: "
                    + ("reused checkpoint" if reused else "completed")
                ),
            )
    return run.execute(
        "evaluation", lambda: run_evaluation_profile(windows, evaluation_cfg),
        config=signature, force=force,
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    art = Path(cfg.paths.artifacts)
    manifest_path = cfg.paths.get("evaluation_manifest")
    if manifest_path:
        raw_manifest = OmegaConf.to_container(
            OmegaConf.load(Path(str(manifest_path))), resolve=True
        )
        if not isinstance(raw_manifest, dict) or not all(isinstance(key, str) for key in raw_manifest):
            raise TypeError("evaluation manifest must contain a mapping")
        manifest_data = cast(dict[str, Any], raw_manifest)
        art.mkdir(parents=True, exist_ok=True)
        if bool(cfg.get("preflight", False)):
            readiness = preflight_manifest(manifest_data)
            (art / "preflight.json").write_text(
                json.dumps(_json_value(readiness), indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(_json_value({
                key: value for key, value in readiness.items() if key != "split_manifest"
            }), indent=2, allow_nan=False))
            if readiness["status"] != "ready_for_model_fitting":
                raise ValueError("preflight found no usable folds; see preflight.json")
            return
        OmegaConf.save(OmegaConf.create(manifest_data), art / "evaluation-manifest.yaml")
        result = run_manifest_checkpointed(manifest_data, art, force=bool(cfg.get("force", False)))
    else:
        if bool(cfg.get("preflight", False)):
            raise ValueError("preflight requires paths.evaluation_manifest")
        eval_cfg_data = cast(
            dict[str, Any], OmegaConf.to_container(cfg.evaluation, resolve=True)
        )
        evaluation_cfg = EvaluationConfig.from_mapping(eval_cfg_data)
        if evaluation_cfg.profile == "revised_primary":
            raise ValueError(
                "revised_primary must start from paths.evaluation_manifest so raw-data "
                "preprocessing ownership can be validated"
            )
        windows = load_artifact(art / "windows.pkl")
        result = run_evaluation_profile(windows, evaluation_cfg)
    save_artifact(result, art / "evaluation.pkl")
    (art / "evaluation-report.json").write_text(
        json.dumps(_json_value({
            "diagnostics": result.diagnostics,
            "provenance": result.provenance,
            "split_manifest": result.manifest,
            "prediction_count": len(result.predictions),
        }), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Primary evaluation: "
        f"{result.diagnostics['available_folds']} available folds, "
        f"{result.diagnostics['unavailable_folds']} unavailable folds"
    )
    for fold in result.manifest.outer_folds:
        print(
            f"  {fold.fold_id}: {fold.status} "
            f"train={len(fold.train_indices)} test={len(fold.test_indices)}"
            + (f" reason={fold.failure_reason}" if fold.failure_reason else "")
        )


if __name__ == "__main__":
    main()
