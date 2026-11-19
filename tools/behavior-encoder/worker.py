"""Standalone continuous behavior/dynamics encoder worker; no effectome imports.

This pointwise continuous-output adaptation is distinct from the published
categorical temporal-window BunDLe port. The latent residual transition is a
training regularizer, not the mechanism used for neural interventions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _thread_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_name, "1")

import numpy as np  # noqa: E402


def runtime_info(torch: Any, device: str = "cpu") -> dict[str, Any]:
    return {
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "python_version": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "device": device,
    }


def _configure(torch: Any, seed: int, n_threads: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(n_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable in the encoder runtime")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("encoder device must be cpu, cuda or auto")
    return requested


def _network(torch: Any, n_inputs: int, config: dict[str, Any], device: str):
    nn = torch.nn
    layers = []
    width = n_inputs
    for hidden in config["hidden_widths"]:
        layers.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    layers.append(nn.Linear(width, config["latent_dim"]))
    encoder = nn.Sequential(*layers).to(device)
    head = nn.Linear(config["latent_dim"], 2).to(device)
    transition = nn.Linear(config["latent_dim"], config["latent_dim"]).to(device)
    return encoder, head, transition


def _scaling(neural: np.ndarray, current: np.ndarray, following: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.concatenate((neural[current], neural[following]))
    return rows.mean(axis=0), np.maximum(rows.std(axis=0), 1e-6)


def _tensors(
    torch: Any,
    neural: np.ndarray,
    behavior: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: str,
):
    return tuple(
        torch.tensor(values, dtype=torch.float32, device=device)
        for values in (
            (neural[current] - mean) / scale,
            (neural[following] - mean) / scale,
            behavior[current],
            behavior[following],
        )
    )


def _validation(torch: Any, encoder: Any, head: Any, tensors: tuple) -> float:
    encoder.eval()
    head.eval()
    x, following, y, y_next = tensors
    with torch.no_grad():
        error = 0.5 * (
            (head(encoder(x)) - y).square().mean() + (head(encoder(following)) - y_next).square().mean()
        )
    value = float(error.cpu())
    if not math.isfinite(value):
        raise ValueError("nonfinite behavioral validation loss")
    return value


def _train(
    torch: Any,
    neural: np.ndarray,
    behavior: np.ndarray,
    current: np.ndarray,
    following: np.ndarray,
    config: dict[str, Any],
    dynamic_weight: float,
    epochs: int,
    *,
    validation_indices: tuple[np.ndarray, np.ndarray] | None,
    phase: str,
    device: str,
) -> dict[str, Any]:
    _configure(torch, int(config["seed"]), int(config["n_threads"]))
    mean, scale = _scaling(neural, current, following)
    encoder, head, transition = _network(torch, neural.shape[1], config, device)
    parameters = list(encoder.parameters()) + list(head.parameters()) + list(transition.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    tensors = _tensors(torch, neural, behavior, current, following, mean, scale, device)
    validation = (
        None
        if validation_indices is None
        else _tensors(
            torch,
            neural,
            behavior,
            *validation_indices,
            mean,
            scale,
            device,
        )
    )
    history = []
    best_score, best_epoch = float("inf"), 0
    n_pairs = len(current)
    for epoch in range(1, epochs + 1):
        encoder.train()
        head.train()
        transition.train()
        order = torch.randperm(n_pairs, device=device)
        total_behavior = total_dynamic = total_loss = 0.0
        for start in range(0, n_pairs, config["batch_size"]):
            selected = order[start : start + config["batch_size"]]
            x, x_next, y, y_next = (values[selected] for values in tensors)
            optimizer.zero_grad()
            z, z_next = encoder(x), encoder(x_next)
            behavior_loss = 0.5 * ((head(z) - y).square().mean() + (head(z_next) - y_next).square().mean())
            dynamic_loss = (z + transition(z) - z_next).square().mean()
            loss = behavior_loss + dynamic_weight * dynamic_loss
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite behavior/dynamics training loss")
            loss.backward()
            optimizer.step()
            count = len(selected)
            total_behavior += float(behavior_loss.detach().cpu()) * count
            total_dynamic += float(dynamic_loss.detach().cpu()) * count
            total_loss += float(loss.detach().cpu()) * count
        record = {
            "epoch": epoch,
            "behavior_loss": total_behavior / n_pairs,
            "dynamic_loss": total_dynamic / n_pairs,
            "loss": total_loss / n_pairs,
        }
        if validation is not None:
            score = _validation(torch, encoder, head, validation)
            record["validation_behavior_mse"] = score
            if epoch >= config["min_epochs"] and score < best_score - 1e-12:
                best_score, best_epoch = score, epoch
        history.append(record)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(
                json.dumps(
                    {
                        "event": "epoch",
                        "phase": phase,
                        "dynamic_weight": dynamic_weight,
                        "epoch": epoch,
                        "epochs": epochs,
                        **record,
                    }
                ),
                flush=True,
            )
        if (
            validation is not None
            and epoch >= config["min_epochs"]
            and best_epoch > 0
            and epoch - best_epoch >= config["patience"]
        ):
            break
    encoder.eval()
    head.eval()
    transition.eval()
    return {
        "encoder": encoder,
        "head": head,
        "transition": transition,
        "input_mean": mean,
        "input_scale": scale,
        "history": history,
        "best_validation_behavior_mse": best_score if validation is not None else None,
        "best_epoch": best_epoch if validation is not None else epochs,
    }


def _validate_inputs(data: Any, config: dict[str, Any]) -> tuple:
    neural, behavior = np.asarray(data["neural"], dtype=float), np.asarray(data["behavior"], dtype=float)
    lengths = np.asarray(config["trial_lengths"])
    if (
        neural.ndim != 2
        or not neural.size
        or behavior.shape != (len(neural), 2)
        or not np.isfinite(neural).all()
        or not np.isfinite(behavior).all()
    ):
        raise ValueError("worker inputs must be finite neural/continuous behavior matrices")
    if (
        lengths.ndim != 1
        or len(lengths) < 2
        or not np.issubdtype(lengths.dtype, np.integer)
        or np.any(lengths < 3)
        or lengths.sum() != len(neural)
    ):
        raise ValueError("worker trial lengths do not partition its supplied training data")
    current, following, owners, times = (
        np.asarray(data[key]) for key in ("current_indices", "next_indices", "trial_ids", "time_indices")
    )
    if any(
        array.ndim != 1 or not np.issubdtype(array.dtype, np.integer)
        for array in (current, following, owners, times)
    ):
        raise ValueError("worker pair indices must be integer vectors")
    if not (current.shape == following.shape == owners.shape == times.shape) or not current.size:
        raise ValueError("worker pair axes must match and be nonempty")
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    if (
        np.any(owners < 0)
        or np.any(owners >= len(lengths))
        or np.any(current < 0)
        or np.any(following >= len(neural))
        or not np.array_equal(following, current + 1)
        or not np.array_equal(current, offsets[owners] + times)
        or np.any(times < 0)
        or np.any(times >= lengths[owners] - 1)
    ):
        raise ValueError("worker pairs cross a trial boundary or do not match their declared ownership")
    return neural, behavior, current, following, owners, lengths


def train(data: Any, config: dict[str, Any], torch: Any) -> dict[str, Any]:
    neural, behavior, current, following, owners, lengths = _validate_inputs(data, config)
    device = _device(torch, config["device"])
    n_validation = max(1, int(np.ceil(len(lengths) * config["validation_fraction"])))
    split = len(lengths) - n_validation
    if split < 1:
        raise ValueError("inner validation must leave complete training trials")
    inner, validation = owners < split, owners >= split
    if not inner.any() or not validation.any():
        raise ValueError("both training and validation trials need within-trial pairs")
    candidates = []
    for weight in config["dynamics_weights"]:
        fitted = _train(
            torch,
            neural,
            behavior,
            current[inner],
            following[inner],
            config,
            float(weight),
            config["max_epochs"],
            validation_indices=(current[validation], following[validation]),
            phase="selection",
            device=device,
        )
        candidates.append(
            {
                "dynamic_weight": float(weight),
                "best_epoch": fitted["best_epoch"],
                "validation_behavior_mse": fitted["best_validation_behavior_mse"],
                "history": fitted["history"],
                "scaler_mean": fitted["input_mean"].tolist(),
                "scaler_scale": fitted["input_scale"].tolist(),
            }
        )
    selected = min(range(len(candidates)), key=lambda index: candidates[index]["validation_behavior_mse"])
    choice = candidates[selected]
    final = _train(
        torch,
        neural,
        behavior,
        current,
        following,
        config,
        choice["dynamic_weight"],
        choice["best_epoch"],
        validation_indices=None,
        phase="refit",
        device=device,
    )
    layers = []
    for module in final["encoder"]:
        if isinstance(module, torch.nn.Linear):
            layers.append(
                {
                    "weights": module.weight.detach().cpu().numpy().tolist(),
                    "bias": module.bias.detach().cpu().numpy().tolist(),
                    "activation": "linear",
                }
            )
        elif isinstance(module, torch.nn.ReLU):
            layers[-1]["activation"] = "relu"
        else:
            raise TypeError("unsupported encoder layer in numerical export")
    latent_layer = len(layers) - 1
    head = final["head"]
    layers.append(
        {
            "weights": head.weight.detach().cpu().numpy().tolist(),
            "bias": head.bias.detach().cpu().numpy().tolist(),
            "activation": "linear",
        }
    )
    training = {
        "candidates": candidates,
        "selected_candidate": selected,
        "selected_dynamic_weight": choice["dynamic_weight"],
        "selected_epochs": choice["best_epoch"],
        "inner_training_trial_ids": list(range(split)),
        "inner_validation_trial_ids": list(range(split, len(lengths))),
        "final_training_trial_ids": list(range(len(lengths))),
        "inner_training_pair_count": int(inner.sum()),
        "inner_validation_pair_count": int(validation.sum()),
        "final_training_pair_count": len(current),
        "refit_history": final["history"],
        "selection_metric": "whole_validation_trial_balanced_pair_behavior_mse",
        "refit_policy": "all_outer_training_trials_fixed_selected_epoch_and_weight",
        "fit_scope": "supplied_outer_training_trials_only",
        "trial_boundary_crossings": 0,
    }
    artifact = {
        "name": "bunddle_behavior",
        "input_mean": final["input_mean"].tolist(),
        "input_scale": final["input_scale"].tolist(),
        "layers": layers,
        "latent_layer": latent_layer,
        "metadata": {
            "variant": "BunDLe_inspired_continuous_pointwise_not_faithful_categorical_window_port",
            "encoder_hidden_widths": config["hidden_widths"],
            "latent_dimension": config["latent_dim"],
            "behavior_dimension": 2,
            "behavior_head": "linear",
            "source_support": "instantaneous_observed_state",
            "future_aggregation": "decode_each_point_before_future_window_mean",
            "loss": "MSE_behavior_current_next + dynamic_weight*MSE_latent_residual_next",
            "transition": {
                "formula": "z_pred=z+Linear(z)",
                "weights": final["transition"].weight.detach().cpu().numpy().tolist(),
                "bias": final["transition"].bias.detach().cpu().numpy().tolist(),
                "causal_simulation_uses_latent_transition": False,
            },
            "training": training,
            "settings": config,
            "runtime": runtime_info(torch, device),
            "latent_markov_closure_validated": False,
        },
    }
    positions = np.unique(np.linspace(0, len(neural) - 1, min(64, len(neural))).astype(int))
    probe = neural[positions]
    tensor = torch.tensor(
        (probe - final["input_mean"]) / final["input_scale"], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        latent = final["encoder"](tensor)
        prediction = head(latent)
    return {
        "artifact": artifact,
        "training": training,
        "runtime": runtime_info(torch, device),
        "parity": {
            "neural": probe.tolist(),
            "latent": latent.cpu().numpy().tolist(),
            "prediction": prediction.cpu().numpy().tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import torch

    if args.probe:
        print(json.dumps(runtime_info(torch)), flush=True)
        return
    if args.input is None or args.config is None or args.output is None:
        parser.error("training requires --input, --config and --output")
    config = json.loads(args.config.read_text())
    with np.load(args.input, allow_pickle=False) as data:
        result = train(data, config, torch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
