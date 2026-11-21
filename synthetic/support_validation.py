"""Audit a finished synthetic support benchmark and export its validation views."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from effectome.experiments.behavior_rollout_online_decisions import online_decision
from effectome.experiments.behavior_rollout_online_policy import fast_refit_decisions
from effectome.experiments.behavior_rollout_support_decisions import support_decision
from effectome.experiments.manifold_validation import _verify_artifacts
from effectome.experiments.mechanism_validation import _write_json
from effectome.workflows.resume import content_fingerprint
from synthetic._support_intervention_view import write_intervention_view
from synthetic._support_validation_plots import write_validation_figures
from synthetic.behavior_rollout_support import Profile, _replicate_mean, _support_masks
from synthetic.effectome_rollout import GAIN_MARGIN


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Complete notebook 01 for the same profile and RUN_ID.")
    return json.loads(path.read_text())


def _array(values: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have shape {shape} with finite values")
    return result


def _same_rows(saved: Any, expected: list[dict[str, Any]], name: str) -> None:
    keys = ["scenario", "seed", "arm"]
    try:
        left = pd.DataFrame(saved).sort_values(keys).reset_index(drop=True)
        right = pd.DataFrame(expected).sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_like=True, atol=1e-12, rtol=0)
    except (AssertionError, KeyError, ValueError) as error:
        raise ValueError(f"{name} disagrees with replayed metrics: {error}") from error


def _replay_case(
    case: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scenario, seed = case["scenario"], case["seed"]
    points = np.asarray(case["origin_samples"])
    trial_count = protocol["generator"]["n_trials"] - protocol["partitions"]["training_stop"]
    shape = (trial_count, len(points), 2)
    if points.ndim != 1 or not len(points) or points.dtype.kind not in "iu" or np.any(np.diff(points) <= 0):
        raise ValueError("invalid forecast origin samples")
    observed = _array(case["behavior_targets"], shape, "realized future behavior")
    decoded = _array(case["decoded_targets"], shape, "decoded target")
    switches = np.asarray(case["scoring_switch_samples"])
    origins = np.broadcast_to(points, shape[:2])
    age = origins[..., None] - switches[:, None, :]
    early = ((age >= 0) & (age < protocol["early"])).any(axis=-1)
    if not early.any():
        raise ValueError("early scoring band is empty")

    effect_index = np.asarray(case["effect_index"])
    if (
        effect_index.ndim != 1
        or not len(effect_index)
        or effect_index.dtype.kind not in "iu"
        or np.any(effect_index < 0)
        or np.any(effect_index >= origins.size)
        or np.any(np.diff(effect_index) <= 0)
    ):
        raise ValueError("invalid matched intervention origins")
    horizon = protocol["generator"]["horizon"]
    trial, column = np.divmod(effect_index, len(points))
    sample = points[column]
    age_effect = sample[:, None] - switches[trial]
    if np.any((age_effect < 0) & (age_effect + horizon >= 0)):
        raise ValueError("intervention score includes a regime-crossing origin")
    if np.any(column % protocol["policy"]["effect_stride_origins"] != 0):
        raise ValueError("intervention origins differ from the frozen sampling cadence")

    units = case["effect_units"]
    effect_shape = (len(effect_index), len(units), 2)
    oracle = _array(case["oracle_effects"], effect_shape, "simulator disconnection effects")
    active = np.linalg.norm(oracle, axis=-1) / np.sqrt(2) >= protocol["effect_floor"]
    np.testing.assert_array_equal(active, case["oracle_active"])
    variant = case["generative"]
    learned = np.asarray(variant["learned"]["pooled_fit"]["support"])
    masks = _support_masks(learned, seed, protocol["settings"]["shuffled_masks"])
    if set(variant) != set(masks):
        raise ValueError("saved generative comparisons differ from the frozen support grid")
    settings = {key: value for key, value in protocol["policy"].items() if key != "effect_stride_origins"}
    common = {"scenario": scenario, "seed": seed, "origins": origins.size}

    def errors(prediction: np.ndarray) -> tuple[float, float]:
        mae = np.abs(prediction - observed).mean(axis=-1)
        return float(mae.mean()), float(mae[early].mean())

    rows = []
    estimates = {}
    selector = {}
    for name, mask in masks.items():
        arm = variant[name]
        np.testing.assert_array_equal(arm["pooled_fit"]["support"], mask)
        pooled = _array(arm["pooled_forecast"], shape, "pooled forecast")
        local = _array(arm["local_forecast"], shape, "local forecast")
        choice = fast_refit_decisions(origins, pooled, local, decoded, horizon=horizon, **settings)
        np.testing.assert_array_equal(choice, arm["selected_local"])
        prediction = np.where(choice[..., None], local, pooled)
        full_mae, early_mae = errors(prediction)
        effect = _array(arm["effects"], effect_shape, "fitted disconnection effects")
        estimates[name] = effect
        effect_error = np.abs(effect - oracle).mean(axis=-1)
        row = {
            "arm": name,
            **common,
            "forecast_mae": full_mae,
            "early_forecast_mae": early_mae,
            "all_effect_mae": float(effect_error.mean()),
            "active_effect_mae": float(effect_error[active].mean()) if active.any() else None,
            "effect_origins": len(effect_index),
            "support_edges": int(mask.sum()),
            "learned_overlap_edges": int((mask & learned).sum()),
            "local_fraction": float(choice.mean()),
        }
        _same_rows([arm["metrics"]], [row], f"{name} case metrics")
        rows.append(row)
        if name == "learned":
            pool_error, local_error, selected_error = (
                np.abs(values - observed).mean(axis=-1) for values in (pooled, local, prediction)
            )
            selector = {
                **common,
                "early_origins": int(early.sum()),
                "gated_mae": full_mae,
                "pooled_mae": float(pool_error.mean()),
                "local_mae": float(local_error.mean()),
                "pooled_gain": float((pool_error - selected_error).mean()),
                "local_gain": float((local_error - selected_error).mean()),
                "early_local_gain": float((local_error - selected_error)[early].mean()),
                "local_fraction": row["local_fraction"],
                "early_local_fraction": float(choice[early].mean()),
            }

    direct = _array(case["direct_predictions"], shape, "direct neural-history forecast")
    full_mae, early_mae = errors(direct)
    aggregate = [
        rows[0],
        rows[1],
        _replicate_mean(rows[2:], "shuffled"),
        {
            "arm": "direct_history",
            **common,
            "forecast_mae": full_mae,
            "early_forecast_mae": early_mae,
            "all_effect_mae": None,
            "active_effect_mae": None,
            "effect_origins": None,
            "support_edges": None,
            "learned_overlap_edges": None,
            "local_fraction": None,
        },
    ]
    _same_rows(case["rows"], aggregate, "the per-case seed rows")
    for key, value in selector.items():
        if key == "scenario":
            if value != case["learned_selector_metrics"][key]:
                raise ValueError("learned selector scenario differs from replay")
        else:
            np.testing.assert_allclose(value, case["learned_selector_metrics"][key], rtol=0, atol=1e-12)
    fps = protocol["generator"]["fps"]
    effect_record = {
        "scenario": scenario,
        "seed": seed,
        "units": units,
        "oracle": oracle,
        "arms": estimates,
        "active": active,
        "origins": [
            {
                "trial": int(t) + protocol["partitions"]["training_stop"],
                "sample": int(s),
                "seconds": float(s / fps),
            }
            for t, s in zip(trial, sample, strict=True)
        ],
    }
    return aggregate, selector, effect_record


def validate_and_visualize(run_root: Path) -> dict[str, Any]:
    """Recompute seed scores without fitting and write matched validation figures."""
    root = run_root.resolve()
    protocol = _read(root / "support-protocol.json")
    report = _read(root / "support-report.json")
    signature = protocol.pop("signature")
    if signature != content_fingerprint(protocol) or signature != report["signature"]:
        raise ValueError("the recorded protocol and report disagree")
    profile = protocol["profile"]
    seeds = Profile(**protocol["settings"]).seeds
    expected = tuple(product(seeds, protocol["scenarios"]))
    if (
        report["profile"] != profile
        or not report["execution_complete"]
        or report["completed_cases"] != len(expected)
        or report["planned_cases"] != len(expected)
    ):
        raise ValueError("the selected benchmark is incomplete; finish notebook 01 before validation")

    metrics = []
    selector_rows = []
    cases = []
    oracle_active = []
    estimated_active: dict[str, list[np.ndarray]] = {"learned": [], "dense": [], "shuffled": []}
    for seed, scenario in sorted(expected):
        folder = root / "cases" / f"{scenario}-{seed}"
        case = _read(folder / "support-case.json")
        if (case["seed"], case["scenario"], case["signature"]) != (seed, scenario, signature):
            raise ValueError(f"wrong seed, condition or protocol in {folder}")
        _verify_artifacts(folder, case["artifacts"])
        rows, selector, effect = _replay_case(case, protocol)
        metrics.extend(rows)
        selector_rows.append(selector)
        active, oracle, arms = effect["active"], effect["oracle"], effect["arms"]
        if scenario == "driver_switch":
            oracle_active.append(oracle[active])
            for name in ("learned", "dense"):
                estimated_active[name].append(arms[name][active])
            shuffles = [values[active] for name, values in arms.items() if name.startswith("shuffle_")]
            estimated_active["shuffled"].append(np.mean(shuffles, axis=0))
        # Compact the standalone display data. Scoring and figures use the original precision.
        cases.append(
            {
                **effect,
                "oracle": oracle.round(7).tolist(),
                "active": active.tolist(),
                "arms": {name: values.round(7).tolist() for name, values in arms.items()},
            }
        )

    seed_scores = pd.read_csv(root / "seed-metrics.csv")
    _same_rows(seed_scores, metrics, "seed-metrics.csv")
    rules = protocol["decision"]
    support = None
    learned_selector = None
    if profile == "confirmation":
        support = support_decision(metrics, seeds, **rules)
        learned_selector = online_decision(selector_rows, seeds, margin=GAIN_MARGIN, level=rules["level"])
    if support != report["support"] or learned_selector != report["learned_selector"]:
        raise ValueError("saved decision differs from the paired seed-level replay")

    output = root / "validation"
    figures = write_validation_figures(
        seed_scores,
        np.concatenate(oracle_active),
        {name: np.concatenate(values) for name, values in estimated_active.items()},
        output,
        profile=profile,
        forecast_margin=rules["forecast_margin"],
        effect_margin=rules["effect_margin"],
    )
    interactive = write_intervention_view(output / "interventions.html", cases, profile=profile)
    result = {
        "signature": signature,
        "profile": profile,
        "case_count": len(expected),
        "seed_count": len(seeds),
        "support": support,
        "learned_selector": learned_selector,
        "figures": figures,
        "interactive_view": str(interactive),
        "seed_metrics": str(root / "seed-metrics.csv"),
    }
    _write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_and_visualize(args.run_root), indent=2))


if __name__ == "__main__":
    main()
