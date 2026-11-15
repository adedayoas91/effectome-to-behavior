"""Training-owned controls and descriptive adequacy checks for shared transitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from effectome.data_module.schema import WindowedSegments
from effectome.linking.decoding import valid_positive_lag_origins


def matched_group_controls(
    groups: Mapping[str, Sequence[str]],
    neuron_ids: Sequence[str],
    scaled: np.ndarray,
    fit_indices: np.ndarray,
    *,
    seed: int,
    n_controls: int = 3,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Match group size and training-only activity RMS; never use held-out outcomes."""
    output = {name: [str(member) for member in members] for name, members in groups.items()}
    activity = np.sqrt(np.mean(scaled[fit_indices] ** 2, axis=(0, 1)))
    rng = np.random.default_rng(seed)
    records: dict[str, Any] = {}
    for name, raw_members in groups.items():
        members = [str(member) for member in raw_members]
        target = np.asarray([neuron_ids.index(member) for member in members])
        available = np.asarray([i for i, neuron in enumerate(neuron_ids) if neuron not in members])
        if len(available) < len(target):
            records[name] = {"status": "unavailable", "reason": "insufficient disjoint neurons"}
            continue
        candidates: dict[tuple[int, ...], float] = {}
        for _ in range(max(100, n_controls * 40)):
            selected = tuple(sorted(rng.choice(available, size=len(target), replace=False).tolist()))
            candidates[selected] = float(abs(np.mean(activity[list(selected)]) - np.mean(activity[target])))
        controls = sorted(candidates, key=lambda key: (candidates[key], key))[:n_controls]
        keys = []
        for index, selected in enumerate(controls):
            key = f"__control__{name}:{index}"
            output[key] = [neuron_ids[i] for i in selected]
            keys.append(key)
        records[name] = {
            "status": "available",
            "control_groups": keys,
            "n_requested": n_controls,
            "n_available": len(controls),
            "target_size": len(target),
            "activity_rms_difference": [candidates[selected] for selected in controls],
            "matching_uses_evaluation": False,
            "matching_variables": ["group_size", "training_activity_rms"],
        }
    return output, records


def matched_transition_contrasts(
    baseline: np.ndarray,
    features: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    blocks: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    """Match held-out A→B events to A→A stays by baseline distance.

    Outcomes define retrospective contrast sets only. This does not produce a
    prospective predictor, a null p-value, or independent transition replicates.
    """
    coordinates = StandardScaler().fit(baseline[train_rows]).transform(baseline)
    output = []
    for source in np.unique(sources[evaluation_rows]):
        stays = evaluation_rows[(sources[evaluation_rows] == source) & (targets[evaluation_rows] == source)]
        for destination in np.unique(targets[evaluation_rows]):
            if source == destination:
                continue
            events = evaluation_rows[
                (sources[evaluation_rows] == source) & (targets[evaluation_rows] == destination)
            ]
            unused = list(stays)
            matched = []
            for event in events:
                if not unused:
                    break
                distances = np.linalg.norm(coordinates[unused] - coordinates[event], axis=1)
                control = unused.pop(int(np.argmin(distances)))
                matched.append((int(event), int(control)))
            for unit, columns in blocks.items():
                if unit.startswith("control:"):
                    continue
                if matched and len(columns):
                    left, right = np.asarray(matched).T
                    difference = features[left][:, columns] - features[right][:, columns]
                    signed_mean = difference.mean(axis=0)
                else:
                    signed_mean = None
                output.append(
                    {
                        "unit": unit,
                        "source_state": int(source),
                        "destination_state": int(destination),
                        "n_transition_origins": len(events),
                        "n_stay_origins": len(stays),
                        "n_matched_pairs": len(matched),
                        "signed_feature_mean_difference": signed_mean,
                        "matched_event_rows": matched,
                        "analysis_scope": "descriptive_heldout_matched_contrast",
                        "independent_event_inference": False,
                    }
                )
    return output


def frozen_markov_diagnostics(
    windows: WindowedSegments,
    states: np.ndarray,
    *,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    n_states: int,
    positive_lag: int,
) -> dict[str, Any]:
    """Compare powers of a training matrix with held-out longer-horizon occupancy."""
    origins = valid_positive_lag_origins(windows.n_windows, positive_lag, windows.anchors, strict=True)
    train = origins[np.isin(origins, fit_indices) & np.isin(origins + positive_lag, fit_indices)]
    train = np.asarray(
        [i for i in train if all(j in fit_indices for j in range(int(i), int(i) + positive_lag + 1))],
        dtype=int,
    )
    counts = np.zeros((n_states, n_states), dtype=int)
    np.add.at(counts, (states[train], states[train + positive_lag]), 1)
    totals = counts.sum(1)
    transition = np.divide(
        counts, totals[:, None], out=np.zeros_like(counts, dtype=float), where=totals[:, None] > 0
    )
    checks: list[dict[str, Any]] = []
    for factor in (2, 3):
        lag = factor * positive_lag
        candidates = valid_positive_lag_origins(windows.n_windows, lag, windows.anchors, strict=True)
        selected = candidates[
            np.isin(candidates, evaluation_indices) & np.isin(candidates + lag, evaluation_indices)
        ]
        # Require every intermediate ownership as well as the temporal path.
        selected = np.asarray(
            [i for i in selected if all(j in evaluation_indices for j in range(int(i), int(i) + lag + 1))],
            dtype=int,
        )
        if not len(selected) or np.any(totals == 0):
            checks.append(
                {
                    "factor": factor,
                    "status": "unavailable",
                    "reason": "insufficient held-out events or unsupported training transition rows",
                }
            )
            continue
        predicted = np.linalg.matrix_power(transition, factor)[states[selected]]
        observed = np.eye(n_states)[states[selected + lag]]
        checks.append(
            {
                "factor": factor,
                "status": "available",
                "n_origins": len(selected),
                "brier": float(np.mean(np.sum((predicted - observed) ** 2, axis=1))),
                "observed_counts": np.bincount(states[selected + lag], minlength=n_states),
                "predicted_occupancy": predicted.mean(0),
            }
        )
    return {
        "training_counts": counts,
        "training_matrix": transition,
        "unsupported_training_rows": np.flatnonzero(totals == 0),
        "longer_horizon_checks": checks,
        "markov_property_established": False,
        "test_scope": "frozen_training_matrix_predictive_adequacy",
    }


def group_control_comparisons(
    omission_scores: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Compare joint omission gains with declared size/activity matched controls."""
    output = []
    lookup = {(row["unit"], row["source_state"], row["destination_state"]): row for row in omission_scores}
    for name, manifest in controls.items():
        for row in omission_scores:
            if row["unit"] != f"group:{name}":
                continue
            values = [
                lookup[
                    (
                        f"control:{key.removeprefix('__control__')}",
                        row["source_state"],
                        row["destination_state"],
                    )
                ]["omission_delta_log_loss"]
                for key in manifest.get("control_groups", [])
                if (
                    f"control:{key.removeprefix('__control__')}",
                    row["source_state"],
                    row["destination_state"],
                )
                in lookup
            ]
            mean = float(np.mean(values)) if values else None
            output.append(
                {
                    "group": name,
                    "source_state": row["source_state"],
                    "destination_state": row["destination_state"],
                    "n_controls": len(values),
                    "group_omission_gain": row["omission_delta_log_loss"],
                    "mean_control_omission_gain": mean,
                    "gain_minus_matched_controls": None
                    if mean is None
                    else row["omission_delta_log_loss"] - mean,
                    "inferential_scope": "descriptive_not_neuron_significance",
                }
            )
    return output
