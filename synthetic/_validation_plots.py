"""Scientific panels for the frozen synthetic validation evidence."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

METHOD_LABELS = {
    "conventional_cgc": "c-GC",
    "conventional_cgc_star": "c-GC*",
    "fast_cgc": "fastC-GC",
    "fast_cgc_star": "fastC-GC*",
}


SCENARIO_LABELS = {
    "one_to_one": "Separated means",
    "matched_marginals": "Matched marginals",
    "fixed_circuit": "Fixed circuit",
    "stationary": "Stationary",
}


def _save(figure: Any, destination: Path, name: str) -> None:
    import matplotlib.pyplot as plt

    for extension in ("pdf", "png"):
        figure.savefig(destination / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _predictive_overview(report: dict, statistics: dict, destination: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(c["scenario"] for c in report["planned_cells"]))
    n_rows = (len(scenarios) + 1) // 2
    figure, axes = plt.subplots(n_rows, 2, figsize=(9.5, 3.2 * n_rows), squeeze=False, layout="constrained")
    methods = report["planned_methods"]
    for axis, scenario in zip(axes.ravel(), scenarios, strict=False):
        spec = next(c["spec_id"] for c in report["planned_cells"] if c["scenario"] == scenario)
        summary = {r["method"]: r for r in statistics["summaries"] if r["spec_id"] == spec}
        for index, method in enumerate(methods):
            row = summary[method]
            mean, interval = row["mean_gain_nats"], row["mean_gain_95ci"]
            if mean is None:
                axis.text(0, index, "NA", va="center")
                continue
            if interval is not None:
                axis.plot(interval, [index, index], color="#0072B2", linewidth=1.5)
            axis.plot(mean, index, "o", color="#0072B2", markersize=4)
        axis.axvline(0, color="0.4", linestyle="--", linewidth=0.8)
        axis.set_yticks(range(len(methods)), [METHOD_LABELS.get(m, m) for m in methods])
        axis.invert_yaxis()
        axis.set(xlabel="Held-out log-loss gain (nats)", title=SCENARIO_LABELS.get(scenario, scenario))
    for axis in axes.ravel()[len(scenarios) :]:
        axis.axis("off")
    _save(figure, destination, "figure-01-predictive-gain")

    attempted = [sum(r["method"] == m for r in report["rows"]) for m in methods]
    available = [
        sum(r["method"] == m and r["status"] == "available" for r in report["rows"]) for m in methods
    ]
    figure, axis = plt.subplots(figsize=(8, 3.5), layout="constrained")
    axis.barh(range(len(methods)), attempted, color="white", edgecolor="0.3", hatch="//", label="Attempted")
    axis.barh(range(len(methods)), available, color="#56B4E9", edgecolor="0.3", label="Available")
    maximum = max(attempted, default=0)
    for index, (count, total) in enumerate(zip(available, attempted, strict=True)):
        axis.text(total + max(0.15, maximum * 0.02), index, f"{count}/{total}", va="center")
    axis.set_yticks(range(len(methods)), [METHOD_LABELS.get(m, m) for m in methods])
    axis.invert_yaxis()
    axis.set(xlabel="Method runs", xlim=(0, max(1, maximum * 1.18)))
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    _save(figure, destination, "figure-02-availability")


def _designed_controls(report: dict[str, Any], destination: Path, recording_factory: Callable) -> None:
    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(c["scenario"] for c in report["planned_cells"]))
    figure, axes = plt.subplots(len(scenarios), 3, figsize=(9, 2.25 * len(scenarios)), squeeze=False)
    colors, markers = ("#0072B2", "#D55E00"), ("o", "^")
    for row, scenario in enumerate(scenarios):
        cell = next(c for c in report["planned_cells"] if c["scenario"] == scenario)
        recording = recording_factory(report, cell)
        truth, values = recording.metadata, recording.traces
        matrices = np.asarray(truth["true_coefficients"])
        coefficient = (
            matrices[0]
            if len(matrices) == 1
            else np.concatenate((matrices[0], np.full((recording.n_neurons, 1), np.nan), matrices[1]), axis=1)
        )
        limit = float(np.max(np.abs(matrices)))
        mesh = axes[row, 0].imshow(coefficient, vmin=-limit, vmax=limit, cmap="RdBu_r", aspect="auto")
        axes[row, 0].set(
            ylabel=f"{SCENARIO_LABELS.get(scenario, scenario)}\nSource neuron",
            xlabel="Target neuron (W0 | W1)" if len(matrices) > 1 else "Target neuron (one W)",
        )
        axes[row, 0].set_yticks(range(recording.n_neurons))
        ticks = list(range(recording.n_neurons))
        labels = [str(i) for i in ticks]
        if len(matrices) > 1:
            ticks += list(range(recording.n_neurons + 1, 2 * recording.n_neurons + 1))
            labels += labels.copy()
        axes[row, 0].set_xticks(ticks, labels)
        figure.colorbar(mesh, ax=axes[row, 0], fraction=0.04, pad=0.03)
        codes = np.asarray(
            truth["true_initial_condition_class"] if scenario == "fixed_circuit" else truth["true_regime"]
        )
        origins = np.concatenate([np.arange(a, b - 1) for a, b in truth["valid_ranges"]])
        for code in np.unique(codes):
            selected = codes == code
            settings = {
                "color": colors[int(code) % 2],
                "marker": markers[int(code) % 2],
                "s": 5,
                "alpha": 0.35,
            }
            axes[row, 1].scatter(values[0, selected], values[1, selected], **settings)
            lagged = origins[codes[origins] == code]
            axes[row, 2].scatter(values[0, lagged], values[1, lagged + 1], **settings)
        axes[row, 1].set(xlabel="Neuron 0 activity", ylabel="Neuron 1 activity")
        axes[row, 2].set(xlabel="Neuron 0 at t", ylabel="Neuron 1 at t + 1 sample")
    for column, label in enumerate(
        ("a  Generating coefficients", "b  Snapshot occupancy", "c  Lagged relationship")
    ):
        axes[0, column].set_title(label, fontsize=12, loc="left")
    figure.tight_layout()
    _save(figure, destination, "figure-03-designed-controls")


def _matrix(axis: Any, values: Any, unsupported: list[int], label: str) -> Any:
    matrix = np.asarray(values, dtype=float)
    mask = ~np.isfinite(matrix)
    mask[unsupported] = True
    mesh = axis.imshow(np.ma.array(matrix, mask=mask), vmin=0, vmax=1, cmap="Blues")
    axis.set_title(label, fontsize=12, loc="left")
    axis.set(xticks=range(len(matrix)), yticks=range(len(matrix)), xlabel="Destination", ylabel="Source")
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(
            j,
            i,
            "NA" if mask[i, j] else f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=12,
            color="white" if value > 0.65 and not mask[i, j] else "black",
        )
    return mesh


def _shared_transitions(comparison: dict, cell: dict, destination: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(10, 5.8), layout="constrained")
    flat = axes.ravel()
    reference, events = comparison["reference"], comparison["events"]
    embedding = np.asarray(reference["embeddings"])
    evaluation = np.asarray(events["evaluation_rows"], dtype=int)
    origins = np.asarray(events["origins"], dtype=int)[evaluation]
    targets = np.asarray(events["targets"], dtype=int)[evaluation]
    states = np.asarray(reference["manifold_states"])
    y = embedding[:, 1] if embedding.shape[1] > 1 else np.zeros(len(embedding))
    flat[0].scatter(embedding[:, 0], y, s=8, color="0.8", label="All eligible anchors")
    for source, target in zip(origins, targets, strict=True):
        flat[0].plot(embedding[[source, target], 0], y[[source, target]], color="0.6", linewidth=0.6)
    for state in np.unique(states[origins]):
        color = ("#0072B2", "#D55E00", "#009E73")[state % 3]
        selected = origins[states[origins] == state]
        flat[0].scatter(
            embedding[selected, 0], y[selected], s=20, color=color, marker=("o", "^", "s")[state % 3]
        )
    flat[0].set(xlabel="Shared PC1", ylabel="Shared PC2", title="a  Shared chart")
    matrices = comparison["common_baseline"]["transition_matrices"]
    _matrix(flat[1], matrices["empirical"], matrices["unsupported_rows"], "b  Observed")
    _matrix(
        flat[2], matrices["baseline_predicted"], matrices["unsupported_rows"], "c  Baseline"
    )
    flat[3].axis("off")
    flat[3].text(
        0,
        0.9,
        f"{cell['scenario']}\nFirst declared seed: {cell['seed']}\n"
        f"Held-out origins: {len(origins)}\nSource counts: {matrices['source_counts']}\n"
        f"Horizon: {reference['metadata']['physical_horizon_seconds']:.1f} s\n"
        "Same events and scale\nGeometric states; local labels",
        va="top",
        fontsize=12,
        linespacing=1.6,
    )
    mesh = None
    for axis, (method, result) in zip(flat[4:], comparison["methods"].items(), strict=True):
        if result["status"] != "available":
            axis.axis("off")
            axis.text(0.1, 0.5, f"{METHOD_LABELS.get(method, method)}\nUnavailable")
            continue
        item = result["prediction"]["transition_matrices"]
        mesh = _matrix(
            axis, item["augmented_predicted"], item["unsupported_rows"], METHOD_LABELS.get(method, method)
        )
    if mesh is not None:
        figure.colorbar(
            mesh, ax=axes[:, 1:].ravel().tolist(), shrink=0.7, label="Transition probability", pad=0.02
        )
    _save(figure, destination, "figure-04-shared-transitions")


def _attribution(comparison: dict, destination: Path) -> None:
    import matplotlib.pyplot as plt

    methods = comparison["methods"]
    all_rows = [
        r for m in methods.values() if m["status"] == "available" for r in m["prediction"]["omission_scores"]
    ]
    units = [f"neuron:{n}" for n in comparison["neuron_ids"]] + ["group:rotating_pair"]
    units += sorted({r["unit"] for r in all_rows if r["unit"].startswith("control:")})
    transitions = sorted({(r["source_state"], r["destination_state"]) for r in all_rows})
    limit = max((abs(r["omission_delta_log_loss"]) for r in all_rows), default=1.0) or 1.0
    figure, axes = plt.subplots(2, 2, figsize=(8.5, 5.8), layout="constrained")
    mesh = None
    for axis, (method, result) in zip(axes.ravel(), methods.items(), strict=True):
        axis.set_title(METHOD_LABELS.get(method, method), fontsize=12, loc="left")
        if result["status"] != "available":
            axis.axis("off")
            axis.text(0.1, 0.5, "Unavailable")
            continue
        rows = result["prediction"]["omission_scores"]
        lookup = {(r["unit"], r["source_state"], r["destination_state"]): r for r in rows}
        values = np.asarray(
            [
                [lookup.get((u, *t), {}).get("omission_delta_log_loss", np.nan) for t in transitions]
                for u in units
            ]
        )
        mesh = axis.imshow(
            np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
        )
        labels = []
        for source, target in transitions:
            example = next(
                (r for r in rows if (r["source_state"], r["destination_state"]) == (source, target)), None
            )
            labels.append(
                f"{source}→{target}\n{example['n_events']}/{example['n_origins']}"
                if example
                else f"{source}→{target}\nNA"
            )
        axis.set_xticks(range(len(transitions)), labels, fontsize=11)
        axis.set_yticks(
            range(len(units)),
            [
                u.replace("neuron:", "N")
                .replace("group:rotating_pair", "G01")
                .replace("control:rotating_pair:", "Control ")
                for u in units
            ],
            fontsize=11,
        )
        for (i, j), value in np.ndenumerate(values):
            axis.text(
                j,
                i,
                "NA" if not np.isfinite(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=11,
                color="white" if abs(value) > 0.65 * limit else "black",
            )
    if mesh is not None:
        figure.colorbar(
            mesh, ax=axes.ravel().tolist(), shrink=0.8, label="Omitted − full log loss (nats)", pad=0.02
        )
    _save(figure, destination, "figure-05-transition-attribution")
