"""Publication-ready diagnostics for the paired synthetic benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

ARMS = {
    "learned": ("Learned", "#167267"),
    "dense": ("Dense", "#3d5e9a"),
    "shuffled": ("Shuffled", "#b17036"),
    "direct_history": ("Direct history", "#666981"),
}
SCENARIOS = {"driver_switch": "Driver switch", "constant": "Constant", "nuisance_switch": "Nuisance switch"}


def _save(figure: Figure, directory: Path, name: str) -> dict[str, str]:
    exports = {extension: directory / f"{name}.{extension}" for extension in ("png", "pdf")}
    for path in exports.values():
        figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {suffix: str(path) for suffix, path in exports.items()}


def _forecast_panels(metrics: pd.DataFrame, profile: str) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), layout="constrained", sharey=True)
    panels = [
        ("driver_switch", "forecast_mae", "Driver switch · full trajectory"),
        ("driver_switch", "early_forecast_mae", "Driver switch · early forecasts"),
        ("constant", "forecast_mae", "Constant circuit"),
        ("nuisance_switch", "forecast_mae", "Nuisance switch"),
    ]
    for ax, (scenario, metric, title) in zip(axes.ravel(), panels, strict=True):
        for index, (arm, (_, color)) in enumerate(ARMS.items()):
            values = metrics.loc[(metrics.scenario == scenario) & (metrics.arm == arm), metric].dropna()
            if values.empty:
                continue
            spread = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else [0.0]
            ax.scatter(index + np.asarray(spread), values, color=color, s=18, alpha=0.65, edgecolors="none")
            ax.scatter(index, values.mean(), marker="D", color=color, s=38, edgecolors="white", zorder=3)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(range(len(ARMS)), [label for label, _ in ARMS.values()], fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    fig.supylabel("Forecast MAE in behavioral units")
    fig.suptitle(f"Forecast accuracy · {profile} · dots show seeds and diamonds show means", fontsize=13)
    return fig


def _gain_panels(metrics: pd.DataFrame, profile: str, forecast_margin: float, effect_margin: float) -> Figure:
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), layout="constrained")
    panels = [
        ("driver_switch", "forecast_mae", "Driver switch", forecast_margin),
        ("constant", "forecast_mae", "Constant circuit", -forecast_margin),
        ("nuisance_switch", "forecast_mae", "Nuisance switch", -forecast_margin),
        ("driver_switch", "early_forecast_mae", "Early driver forecasts", None),
        ("driver_switch", "active_effect_mae", "Active driver effects", effect_margin),
    ]
    for ax, (scenario, metric, title, margin) in zip(axes.ravel(), panels, strict=False):
        subset = metrics.loc[metrics.scenario == scenario].pivot(
            index="seed",
            columns="arm",
            values=metric,
        )
        ax.axvline(0, color="#a3acb0", linewidth=1)
        if margin is not None:
            ax.axvline(margin, color="#191f25", linestyle="--", linewidth=1)
        for index, comparator in enumerate(("dense", "shuffled")):
            values = (subset[comparator] - subset["learned"]).dropna()
            if values.empty:
                continue
            spread = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
            color = ARMS[comparator][1]
            ax.scatter(values, index + np.asarray(spread), color=color, alpha=0.65, s=23, edgecolors="none")
            ax.scatter(values.mean(), index, marker="D", color=color, s=38, edgecolors="white", zorder=3)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_yticks([0, 1], ["Dense", "Shuffled"])
        ax.set_ylim(-0.45, 1.45)
        ax.set_xlabel("Comparator MAE − learned MAE", fontsize=9)
        ax.grid(axis="x", alpha=0.20)
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.text(
        0.03,
        0.65,
        "Positive values favor learned support.\n"
        "Dashed lines show frozen margins.\n"
        "Dots show paired seeds.\n"
        "Diamonds show their means.",
        transform=ax.transAxes,
        va="top",
        linespacing=1.7,
    )
    fig.suptitle(f"Paired support gains · {profile}", fontsize=13)
    return fig


def _effect_panels(oracle: np.ndarray, estimates: dict[str, np.ndarray], profile: str) -> Figure:
    if oracle.ndim != 2 or oracle.shape[1] != 2:
        raise ValueError("oracle effects must have one row per pair and two behavioral coordinates")
    names = ("learned", "dense", "shuffled")
    if any(estimates[name].shape != oracle.shape for name in names):
        raise ValueError("fitted effects must match the simulator effect grid")
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.3), layout="constrained", sharex=True, sharey=True)
    if not len(oracle):
        for ax in axes:
            ax.axis("off")
        axes[1].text(0.5, 0.5, "No oracle-active driver-switch pairs", ha="center", va="center")
    else:
        limit = max(np.abs(values).max() for values in (oracle, *(estimates[name] for name in names)))
        limit = max(float(limit) * 1.04, 0.01)
        points = np.linspace(0, len(oracle) - 1, min(2500, len(oracle)), dtype=int)
        colors = ("#167267", "#8e5480")
        for ax, arm in zip(axes, names, strict=True):
            ax.plot([-limit, limit], [-limit, limit], color="#58666c", linestyle="--", linewidth=1)
            for coordinate, color in enumerate(colors):
                ax.scatter(
                    oracle[points, coordinate],
                    estimates[arm][points, coordinate],
                    color=color,
                    label=f"Behavior {coordinate + 1}",
                    alpha=0.45,
                    s=12,
                    linewidths=0,
                    rasterized=True,
                )
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), title=ARMS[arm][0])
            ax.axhline(0, color="#d7dddd", linewidth=1, zorder=0)
            ax.axvline(0, color="#d7dddd", linewidth=1, zorder=0)
            ax.set_aspect("equal")
        axes[-1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.supxlabel("Simulator disconnection effect in behavioral units")
    fig.supylabel("Fitted disconnection effect")
    fig.suptitle(f"Oracle-active driver-switch pairs · {profile} · shuffled vectors averaged", fontsize=13)
    return fig


def write_validation_figures(
    metrics: pd.DataFrame,
    oracle: np.ndarray,
    estimates: dict[str, np.ndarray],
    output: Path,
    *,
    profile: str,
    forecast_margin: float,
    effect_margin: float,
) -> dict[str, dict[str, str]]:
    """Plot per-seed scores, matched gains and signed simulator effects."""
    output.mkdir(parents=True, exist_ok=True)
    return {
        "Forecast accuracy": _save(_forecast_panels(metrics, profile), output, "forecast_accuracy"),
        "Paired support gains": _save(
            _gain_panels(metrics, profile, forecast_margin, effect_margin),
            output,
            "paired_support_gains",
        ),
        "Simulator versus fitted effects": _save(
            _effect_panels(oracle, estimates, profile),
            output,
            "disconnection_calibration",
        ),
    }
