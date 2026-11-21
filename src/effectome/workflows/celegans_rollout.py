"""Chronological motif forecasts and fitted disconnections for the NoStim worms."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from importlib.metadata import version
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from effectome.data_module.schema import NeuralRecording, Window
from effectome.experiments.behavior_rollout_online_policy import fast_refit_loss_decisions
from effectome.experiments.behavior_rollout_timing_kernels import fit_timing_mechanisms, timing_forecasts
from effectome.experiments.behavior_supervised import _run_lock
from effectome.experiments.behavior_transition_graphs import (
    discover_transition_supports,
    fit_transition_mechanism,
)
from effectome.experiments.celegans_motif_readout import MotifReadout, fit_motif_readout
from effectome.experiments.celegans_rollout_data import WormPlan, load_worm, plan_worm
from effectome.experiments.manifold_validation import _digest, _verify_artifacts
from effectome.experiments.mechanism_validation import _write_json
from effectome.experiments.structural_mechanism import AdditiveMechanism, MechanismRolloutConfig
from effectome.workflows.cross_method import load_cross_method_profile
from effectome.workflows.resume import content_fingerprint, implementation_fingerprint

CONFIG = Path("conf/behavior_rollout_pipeline/celegans.yaml")
PREFLIGHT = "preflight.json"
logger = logging.getLogger(__name__)


def _read(path: Path, prerequisite: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run {prerequisite} for this RUN_ID first.")
    return json.loads(path.read_text())


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _protocol(repo: Path) -> dict[str, Any]:
    settings = OmegaConf.to_container(OmegaConf.load(repo / CONFIG), resolve=True)
    if not isinstance(settings, dict):
        raise ValueError("C. elegans configuration must be a mapping")
    for name in ("draws", "batch_size"):
        _positive_integer(settings["rollout"][name], name)
    _positive_integer(settings["effect_stride_origins"], "effect_stride_origins")
    frequency = settings["rollout"]["support_frequency"]
    alpha = settings["rollout"]["ridge_alpha"]
    if not 0 < frequency <= 1 or not isfinite(alpha) or alpha <= 0:
        raise ValueError("support_frequency must be in (0, 1] and ridge_alpha must be finite and positive")
    _, methods = load_cross_method_profile("csl_sanity", config_root=repo / "conf")
    return {
        "settings": settings,
        "method": asdict(methods["fast_cgc"]),
        "dataset_config_sha256": _digest(repo / "conf/data/c_elegans.yaml"),
        "implementation": implementation_fingerprint(repo),
        "packages": {name: version(name) for name in ("numpy", "scikit-learn", "scipy", "fastcgc")},
    }


def _inventory(recording: NeuralRecording, worm_index: int, plan: WormPlan) -> dict[str, Any]:
    names = recording.metadata["state_names"]
    label = recording.behavior["motif"]
    counts = np.bincount(label, minlength=len(names))
    trained = set(label[: plan.readout_stop].tolist())
    return {
        "worm_index": worm_index,
        "recording_id": recording.identity.recording_id,
        "neurons": recording.n_neurons,
        "frames": recording.n_timepoints,
        "fps": recording.fps,
        "timing": plan.summary(),
        "fictive_motif_counts": dict(zip(names, counts.tolist(), strict=True)),
        "absent_training_motifs": [name for code, name in enumerate(names) if code not in trained],
        "roi_ids": recording.neuron_ids.tolist(),
        "endpoint": asdict(recording.endpoint_metadata["motif"]),
        "source_sha256": recording.metadata["verified_source_sha256"],
    }


def preflight(
    repo: Path,
    output_root: Path,
    *,
    worm_indices: tuple[int, ...] = (0,),
    data_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Verify the pinned dataset and freeze each recording's causal forecast grid."""
    if (
        not worm_indices
        or any(type(index) is not int or index < 0 for index in worm_indices)
        or len(set(worm_indices)) != len(worm_indices)
    ):
        raise ValueError("worm_indices must contain distinct nonnegative integer indices")
    source = data_root.resolve() if data_root is not None else None
    protocol = _protocol(repo)
    settings = protocol["settings"]
    worms = []
    for worm_index in worm_indices:
        recording = load_worm(repo, worm_index, data_root=source)
        plan = plan_worm(recording, **settings["timing"])
        worms.append(_inventory(recording, worm_index, plan))
    frozen = {"protocol": protocol, "data_root": str(source) if source is not None else None, "worms": worms}
    signature = content_fingerprint(frozen)
    output_root.mkdir(parents=True, exist_ok=True)
    with _run_lock(output_root):
        path = output_root / PREFLIGHT
        if path.is_file() and _read(path, "00_data_and_timing.ipynb")["signature"] != signature:
            raise ValueError("This RUN_ID belongs to a different dataset or protocol. Choose a new RUN_ID.")
        _write_json(path, {"signature": signature, **frozen})
    return worms


def _load_preflight(repo: Path, output_root: Path) -> dict[str, Any]:
    frozen = _read(output_root / PREFLIGHT, "00_data_and_timing.ipynb")
    content = {key: value for key, value in frozen.items() if key != "signature"}
    if frozen["signature"] != content_fingerprint(content):
        raise ValueError("The saved preflight does not match its content fingerprint.")
    if content_fingerprint(frozen["protocol"]) != content_fingerprint(_protocol(repo)):
        raise ValueError("The implementation, configuration or runtime changed. Use a new RUN_ID and run 00.")
    return frozen


def _collect_worm(
    repo: Path, inventory: dict[str, Any], frozen: dict[str, Any]
) -> tuple[NeuralRecording, WormPlan]:
    source = Path(frozen["data_root"]) if frozen["data_root"] is not None else None
    recording = load_worm(repo, inventory["worm_index"], data_root=source)
    plan = plan_worm(recording, **frozen["protocol"]["settings"]["timing"])
    current = _inventory(recording, inventory["worm_index"], plan)
    if content_fingerprint(current) != content_fingerprint(inventory):
        raise ValueError(f"Worm {inventory['worm_index']} no longer matches the preflight.")
    return recording, plan


def _rollout_config(plan: WormPlan, settings: dict[str, Any], worm_index: int) -> MechanismRolloutConfig:
    rollout = settings["rollout"]
    return MechanismRolloutConfig(
        plan.horizon,
        plan.target_length,
        rollout["draws"],
        rollout["seed"] + worm_index,
    )


def _histories(neural: np.ndarray, origins: np.ndarray, length: int) -> np.ndarray:
    return neural[origins[:, None] - np.arange(length - 1, -1, -1)]


def _brier(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.square(predicted - target).sum(axis=-1)


def _probabilities(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if (
        probabilities.shape != shape
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(probabilities.sum(axis=-1), 1.0, rtol=0, atol=1e-10)
    ):
        raise ValueError(f"motif forecasts must be finite probability vectors with shape {shape}")
    return probabilities


def _forecast_worm(
    repo: Path,
    output_root: Path,
    inventory: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    index = inventory["worm_index"]
    folder = output_root / f"worm-{index}"
    path = folder / "forecasts.json"
    recording, plan = _collect_worm(repo, inventory, frozen)
    case_signature = content_fingerprint((frozen["signature"], inventory))
    if path.is_file():
        saved = _read(path, "01_fit_and_forecast.ipynb")
        if saved["signature"] != case_signature:
            raise ValueError(f"Worm {index} forecasts belong to a different preflight.")
        _verify_artifacts(folder, saved["artifacts"])
        return saved["fit_summary"]
    settings = frozen["protocol"]["settings"]
    rollout = settings["rollout"]
    _, methods = load_cross_method_profile("csl_sanity", config_root=repo / "conf")
    n_classes = len(recording.metadata["state_names"])
    labels = recording.behavior["motif"]
    raw = recording.traces.T.astype(float)
    training = slice(plan.readout_stop, plan.training_stop)
    scaler = StandardScaler().fit(raw[training])
    neural = scaler.transform(raw)
    readout = fit_motif_readout(
        neural[: plan.readout_stop],
        labels[: plan.readout_stop],
        n_classes,
        **settings["readout"],
    )
    discovery = neural[plan.discovery_starts[:, None] + np.arange(plan.discovery_history)]
    windows = [Window(int(start), int(start + plan.discovery_history)) for start in plan.discovery_starts]
    evidence = folder / "native-support.json"
    votes = discover_transition_supports(
        discovery,
        windows,
        methods["fast_cgc"],
        recording.fps,
        evidence,
        content_fingerprint((case_signature, discovery, windows)),
    )
    support = votes.mean(axis=0) >= rollout["support_frequency"]
    pooled = fit_transition_mechanism(neural[training][None], support, rollout["ridge_alpha"])
    config = _rollout_config(plan, settings, index)
    shape = (len(plan.origins), n_classes)
    base, local = np.empty(shape), np.empty(shape)
    for start in range(0, len(plan.origins), rollout["batch_size"]):
        batch = slice(start, start + rollout["batch_size"])
        origins = plan.origins[batch]
        histories = _histories(neural, origins, plan.history)
        states = histories[:, -1]
        base[batch] = timing_forecasts(
            [pooled] * len(states),
            states,
            origins,
            readout,
            config,
            output_width=n_classes,
        )
        local_models = fit_timing_mechanisms(histories, support, rollout["ridge_alpha"])
        local[batch] = timing_forecasts(
            local_models,
            states,
            origins,
            readout,
            config,
            output_width=n_classes,
        )
    offsets = np.arange(plan.horizon - plan.target_length + 1, plan.horizon + 1)
    target_indices = plan.origins[:, None] + offsets
    decoded = _probabilities(readout.predict(neural[target_indices]).mean(axis=1), shape)
    choice = fast_refit_loss_decisions(
        plan.origins[None],
        _brier(_probabilities(base, shape), decoded)[None],
        _brier(_probabilities(local, shape), decoded)[None],
        horizon=plan.horizon,
        **settings["selector"],
    )[0]
    prevalence = np.bincount(labels[: plan.readout_stop], minlength=n_classes) / plan.readout_stop
    forecast = {
        "pooled": base,
        "local": local,
        "selected": np.where(choice[:, None], local, base),
        "decoded_persistence": _probabilities(readout.predict(neural[plan.origins]), shape),
        "training_prevalence": np.broadcast_to(prevalence, shape),
    }
    fit_summary = {
        "worm_index": index,
        "recording_id": inventory["recording_id"],
        "forecast_origins": len(plan.origins),
        "support_edges": int(support.sum()),
        "local_fraction": float(choice.mean()),
        "readout_iterations": readout.metadata["iterations"],
        "readout_max_iter": readout.metadata["max_iter"],
        "forecast_path": str(path),
    }
    _write_json(
        path,
        {
            "signature": case_signature,
            "artifacts": {evidence.name: _digest(evidence)},
            "fit_summary": fit_summary,
            "origins": plan.origins,
            "origin_seconds": recording.time[plan.origins],
            "target_indices": target_indices,
            "motif_names": recording.metadata["state_names"],
            "actual_occupancy": np.eye(n_classes)[labels[target_indices]].mean(axis=1),
            "decoded_target": decoded,
            "selected_local": choice,
            "forecasts": forecast,
            "neural_mean": scaler.mean_,
            "neural_scale": scaler.scale_,
            "readout": readout.to_dict(),
            "pooled_fit": asdict(pooled),
        },
    )
    logger.info(
        "Worm %s saved %s forecast origins on %s learned edges",
        index,
        len(plan.origins),
        support.sum(),
    )
    return fit_summary


def run_forecasts(repo: Path, output_root: Path) -> list[dict[str, Any]]:
    """Fit each worm independently, with future-free forecast origins and selection."""
    frozen = _load_preflight(repo, output_root)
    with _run_lock(output_root), threadpool_limits(limits=1):
        return [_forecast_worm(repo, output_root, inventory, frozen) for inventory in frozen["worms"]]


def _disconnect(
    recording: NeuralRecording,
    plan: WormPlan,
    settings: dict[str, Any],
    case: dict[str, Any],
    rois: tuple[int, ...],
    output: Path,
) -> list[dict[str, Any]]:
    readout = MotifReadout.from_dict(case["readout"])
    pooled = AdditiveMechanism(**case["pooled_fit"])
    raw = recording.traces.T.astype(float)
    neural = (raw - np.asarray(case["neural_mean"])) / np.asarray(case["neural_scale"])
    sampled = np.arange(0, len(plan.origins), settings["effect_stride_origins"])
    origins = plan.origins[sampled]
    histories = _histories(neural, origins, plan.history)
    choice = np.asarray(case["selected_local"])[sampled]
    models = [pooled] * len(origins)
    if np.any(choice):
        local = fit_timing_mechanisms(histories[choice], pooled.support, settings["rollout"]["ridge_alpha"])
        for selected, model in zip(np.flatnonzero(choice), local, strict=True):
            models[selected] = model
    config = _rollout_config(plan, settings, case["fit_summary"]["worm_index"])
    states = histories[:, -1]
    factual = timing_forecasts(models, states, origins, readout, config, output_width=readout.n_classes)
    np.testing.assert_allclose(
        factual,
        np.asarray(case["forecasts"]["selected"])[sampled],
        rtol=0,
        atol=1e-10,
    )
    results = []
    summary = []
    for roi in rois:
        source = int(np.flatnonzero(recording.neuron_ids == roi)[0])
        shifted = timing_forecasts(
            models,
            states,
            origins,
            readout,
            config,
            output_width=readout.n_classes,
            disconnected=(source,),
        )
        delta = factual - shifted
        result = {
            "roi_id": roi,
            "recorded_name": recording.metadata["raw_neuron_names"][source],
            "outgoing_support_edges": int(pooled.support[source].sum()),
            "effect_origins": len(origins),
            "mean_abs_probability_shift": float(np.abs(delta).mean()),
            "mean_shift_by_motif": dict(zip(case["motif_names"], delta.mean(axis=0).tolist(), strict=True)),
        }
        summary.append(result)
        results.append({**result, "disconnected": shifted, "factual_minus_disconnected": delta})
    _write_json(
        output,
        {
            "estimand": "fitted_outgoing_disconnection_effect_on_fictive_motif_probabilities",
            "forecast_signature": case["signature"],
            "paired_innovations": True,
            "origins": origins,
            "motif_names": case["motif_names"],
            "factual": factual,
            "effects": results,
        },
    )
    return summary


def _write_scores(path: Path, case: dict[str, Any], losses: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["origin", "seconds", "selected_local", *losses])
        writer.writerows(
            zip(
                case["origins"], case["origin_seconds"], case["selected_local"], *losses.values(), strict=True
            ),
        )
    temporary.replace(path)


def summarize(
    repo: Path, output_root: Path, *, disconnect_rois: tuple[int, ...] = ()
) -> list[dict[str, Any]]:
    """Score recorded motif labels and optionally repeat matched fitted interventions."""
    if any(isinstance(roi, bool) or not isinstance(roi, Integral) or roi < 0 for roi in disconnect_rois):
        raise ValueError("disconnect_rois must contain nonnegative recording-local ROI IDs")
    rois = tuple(map(int, disconnect_rois))
    if len(set(rois)) != len(rois):
        raise ValueError("disconnect_rois must contain distinct ROI IDs")
    frozen = _load_preflight(repo, output_root)
    settings = frozen["protocol"]["settings"]
    pending = []
    with _run_lock(output_root), threadpool_limits(limits=1):
        for inventory in frozen["worms"]:
            index = inventory["worm_index"]
            folder = output_root / f"worm-{index}"
            case = _read(folder / "forecasts.json", "01_fit_and_forecast.ipynb")
            if case["signature"] != content_fingerprint((frozen["signature"], inventory)):
                raise ValueError(f"Worm {index} forecasts belong to a different preflight.")
            _verify_artifacts(folder, case["artifacts"])
            missing = set(rois) - set(inventory["roi_ids"])
            if missing:
                raise ValueError(f"Worm {index} lacks recording-local ROI IDs {sorted(missing)}")
            pending.append((inventory, folder, case))
        rows = []
        for inventory, folder, case in pending:
            target = np.asarray(case["actual_occupancy"])
            losses = {
                name: _brier(_probabilities(np.asarray(prediction), target.shape), target)
                for name, prediction in case["forecasts"].items()
            }
            _write_scores(folder / "scores.csv", case, losses)
            readout = MotifReadout.from_dict(case["readout"])
            unseen = sorted(set(range(readout.n_classes)) - set(readout.classes))
            disconnections = []
            if rois:
                recording, plan = _collect_worm(repo, inventory, frozen)
                disconnections = _disconnect(
                    recording,
                    plan,
                    settings,
                    case,
                    rois,
                    folder / "disconnections.json",
                )
            rows.append(
                {
                    **case["fit_summary"],
                    "endpoint_kind": "fictive_motif",
                    "brier": {name: float(loss.mean()) for name, loss in losses.items()},
                    "trained_motif_ids": list(readout.classes),
                    "unseen_motif_target_origins": int(
                        np.count_nonzero(np.any(target[:, unseen] > 0, axis=1))
                    ),
                    "disconnections": disconnections,
                }
            )
        _write_json(output_root / "summary.json", {"signature": frozen["signature"], "worms": rows})
    return rows
