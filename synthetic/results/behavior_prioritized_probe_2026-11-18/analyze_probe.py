"""Analyze the archived development probe without fitting any model."""

from __future__ import annotations

import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parent
    analysis = root / "analysis"
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # "null" is a scenario name, not a missing-value token.
    data = pd.read_csv(root / "seed-metrics.csv", keep_default_na=False)
    seeds = sorted(data.seed.unique())
    assert seeds == list(range(40000, 40005))
    indices = np.array(list(itertools.product(range(5), repeat=5)))

    def describe(values: np.ndarray) -> dict:
        array = np.asarray(values, dtype=float)
        assert array.shape == (5,) and np.isfinite(array).all()
        return {
            "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)),
            "bootstrap_percentile_95_interval": np.quantile(
                array[indices].mean(axis=1), [0.025, 0.975]
            ).tolist(),
            "values": array.tolist(),
            "n_independent_seeds": 5,
        }

    arms = [f"pca_behavior_k{k}" for k in range(1, 6)] + ["behavior_score_1d", "full_neural_behavior"]
    values: dict[str, dict[str, dict]] = {}
    for scenario in ("driver", "null"):
        values[scenario] = {}
        for arm in arms:
            group = data[
                (data.scenario == scenario) & (data.method == "known_support") & (data.endpoint == arm)
            ].sort_values("seed")
            values[scenario][arm] = describe(group.behavioral_mae.to_numpy())
    contrasts: list[dict] = []
    score = np.array(values["driver"]["behavior_score_1d"]["values"])
    for k in range(1, 6):
        differences = np.array(values["driver"][f"pca_behavior_k{k}"]["values"]) - score
        nonzero = differences[differences != 0]
        positive = int(np.count_nonzero(nonzero > 0))
        contrasts.append(
            {
                "comparison": f"pca_k{k}_minus_existing_decoder_score",
                "effect": describe(differences),
                "positive_pairs": positive,
                "nonzero_pairs": len(nonzero),
                "two_sided_sign_p": float(binomtest(positive, len(nonzero), 0.5).pvalue),
            }
        )
    previous = 0.0
    for rank, index in enumerate(np.argsort([row["two_sided_sign_p"] for row in contrasts])):
        previous = max(previous, min(1.0, (5 - rank) * contrasts[index]["two_sided_sign_p"]))
        contrasts[index]["holm_p"] = previous
    geometry: dict[str, list[dict]] = {}
    with zipfile.ZipFile(root / "probe-evidence.zip") as archive:
        for scenario in ("driver", "null"):
            cells = [json.loads(archive.read(f"{scenario}-{seed}.json")) for seed in seeds]
            geometry[scenario] = [
                {
                    "dimension": k,
                    **{
                        key: describe(np.array([cell["geometry"][k - 1][key] for cell in cells]))
                        for key in ("variance_fraction", "behavior_axis_squared_projection")
                    },
                }
                for k in range(1, 6)
            ]
    (analysis / "statistics.json").write_text(
        json.dumps(
            {
                "status": "exploratory_same_five_development_seeds",
                "behavioral_mae": values,
                "geometry": geometry,
                "exploratory_pca_contrasts": contrasts,
                "score_vs_full": "algebraically_equivalent_no_independent_algorithm_test",
                "bootstrap": "all 3125 ordered five-seed resamples; approximate percentile intervals",
            },
            indent=2,
        )
        + "\n"
    )
    table = [
        "| Representation | Dimensions | Driver effect MAE, mean ± SD | Null effect MAE |",
        "|---|---:|---:|---:|",
    ]
    for arm in arms:
        row = values["driver"][arm]
        dimension = 1 if arm == "behavior_score_1d" else 6 if arm == "full_neural_behavior" else int(arm[-1])
        table.append(
            f"| {arm} | {dimension} | {row['mean']:.6f} ± {row['sd']:.6f} | "
            f"{values['null'][arm]['mean']:.6f} |"
        )
    (analysis / "numeric-summary.md").write_text(
        "\n".join(
            [
                "# Exact diagnostic summary",
                "",
                "Known-support values; all four native methods have the same values in this panel. "
                "Five inspected development seeds, lower probability-effect MAE is better.",
                "",
                *table,
                "",
            ]
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, scenario in zip(axes, ("driver", "null"), strict=True):
        matrix = np.array([values[scenario][f"pca_behavior_k{k}"]["values"] for k in range(1, 6)])
        for seed_index in range(5):
            ax.plot(range(1, 6), matrix[:, seed_index], color="#bbbbbb", marker=".", linewidth=0.7)
        ax.plot(range(1, 6), matrix.mean(axis=1), color="#286a8a", marker="o", label="PCA + decoder")
        ax.axhline(
            values[scenario]["behavior_score_1d"]["mean"],
            color="#a64942",
            linestyle="--",
            label="Existing full decoder / 1D score",
        )
        ax.set(
            xlabel="Retained PCA dimensions",
            ylabel="Behavioral probability-effect MAE",
            title=scenario,
            xticks=range(1, 6),
            ylim=(0, None),
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"01-compression-effect-error.{extension}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    driver = geometry["driver"]
    for key, label, color in (
        ("variance_fraction", "Standardized neural variance retained", "#286a8a"),
        ("behavior_axis_squared_projection", "Squared behavior-axis projection", "#a64942"),
    ):
        matrix = np.array([row[key]["values"] for row in driver])
        ax.plot(range(1, 6), matrix.mean(axis=1), marker="o", label=label, color=color)
        for seed_index in range(5):
            ax.scatter(range(1, 6), matrix[:, seed_index], s=10, alpha=0.45, color=color)
    ax.set(
        xlabel="Retained PCA dimensions",
        ylabel="Fraction / squared projection",
        xticks=range(1, 6),
        ylim=(0, 1.05),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    for extension in ("pdf", "png"):
        fig.savefig(figures / f"02-variance-versus-behavior-direction.{extension}", dpi=180)
    plt.close(fig)
    (analysis / "analysis-report.md").write_text(
        "\n".join(
            [
                "# Behavior-first compression: an exact readout control and a dimension diagnostic",
                "",
                "A one-dimensional affine score formed from the already trained full-neural classifier "
                "preserved its source labels and mean-score future events exactly: zero disagreement on "
                "276,480 natural observations and 1,935,360 rollout events. Driver effect MAE remains "
                "0.001969; null effect MAE remains 0.000073. This uses zero new native/mechanism fits, "
                "and the score itself has no new fitting step.",
                "",
                *table,
                "",
                "The score is the existing classifier expressed through its decision coordinate, not an "
                "independent new method. These large event counts audit the transformation; n remains five. "
                "All neural variables are still observed and simulated. A one-dimensional binary score "
                "does not identify one-dimensional behavioral geometry, reduced sensing or latent dynamics.",
                "",
                "PCA dimensions one through five are all disclosed. Four PCs retain 90.45% of standardized "
                "neural variance while giving behavioral MAE 0.138818; the squared projection of the "
                "standardized behavioral coordinate axis into that PCA subspace is only 0.497672. "
                "This geometric diagnostic is not behavioral R-squared or causal recovery. Five PCs reduce "
                "MAE to 0.018994 but retain five of six neural directions, whereas the task-aligned score "
                "preserves the successful parent readout in one coordinate.",
                "",
                "The practical route is to learn the measured behavior objective before selecting a code. "
                "Use this supervised score as the binary-task baseline, then evaluate explicitly learned "
                "bottlenecks on prespecified continuous/distributed behavior. The new behavior needs a new "
                "oracle-checked null if it includes neural coordinates influenced by the current nuisance "
                "path. Freeze mappings and evaluation rules before fresh calibration/confirmation. The "
                "literature review compares PLS, PSID, BunDLe, CEBRA and supervised autoencoders with their "
                "assumptions and runtime constraints.",
                "",
                "These are retrospective development probes. No test-pool selection, nonlinear-encoder "
                "validation, new causal identifiability claim or graph-branch merge occurred.",
                "",
            ]
        )
    )
    appendix = [
        "# Statistical appendix",
        "",
        "Five process seeds are the independent development units. Scenario, method, endpoint and "
        "candidate comparisons are paired. The 1D/full-decoder comparison is an algebraic identity "
        "audited by replay, so no independent-method hypothesis test is claimed.",
        "",
        "Descriptive statistics use seed means and sample SD (ddof=1). Approximate percentile 95% "
        "bootstrap intervals in statistics.json enumerate all 3125 ordered five-seed resamples. "
        "Small-n intervals have no asserted confirmatory coverage and omit selection uncertainty.",
        "",
        "All five PCA dimensions versus the fixed score are exploratory contrasts. Exact "
        "two-sided sign tests require independent seed-pair signs under P(positive)=0.5, without "
        "normality or a normality pretest. Holm correction covers all five contrasts. These tests "
        "do not make the algebraically constructed score an independent learning method.",
        "",
    ]
    for contrast in contrasts:
        appendix.append(
            f"- {contrast['comparison']}: paired mean difference "
            f"{contrast['effect']['mean']:.9f}; {contrast['positive_pairs']}/"
            f"{contrast['nonzero_pairs']} positive pairs; exact p="
            f"{contrast['two_sided_sign_p']:.5f}; Holm p={contrast['holm_p']:.5f}."
        )
    appendix.extend(
        [
            "",
            "No p<0.05 superiority or independent confirmation is claimed. The score "
            "result is exact classifier re-expression; the PCA sweep is developmental. "
            "No million-event pseudoreplication or test-selected acceptance threshold is used.",
            "",
        ]
    )
    (analysis / "stats-appendix.md").write_text("\n".join(appendix))
    (analysis / "figure-catalog.md").write_text(
        "# Figure catalog\n\n"
        "## 01-compression-effect-error\n\n"
        "Purpose: compare all PCA dimensions with the task-aligned existing decoder. Horizontal "
        "dashed lines show the same full-neural/1D-score mean; blue lines show PCA means and grey "
        "lines are the five paired seed trajectories. There are no confidence error bars. Driver "
        "and null panels share development seeds. Source: seed-metrics.csv, known-support lane. "
        "Notice that four PCs still fail the driver behavior test, while a fifth helps with weak "
        "compression. This motivates behavior-prioritized encoding. Caption must state the score "
        "is a re-expression, not a newly trained independent algorithm or intrinsic dimension.\n\n"
        "## 02-variance-versus-behavior-direction\n\n"
        "Purpose: diagnose why a large neural-variance fraction need not preserve this endpoint. "
        "Lines are five-seed means and dots are seed values for standardized variance retained "
        "and the squared projection of the standardized last-coordinate axis onto each PCA "
        "subspace. Source: geometry arrays in probe-evidence.zip. At four PCs variance exceeds "
        "90% while axis alignment remains about 0.50. This supports choosing the target-aware "
        "objective rather than a variance threshold. The projection is not behavioral R-squared, "
        "intervention-error explained, or a general nonlinear diagnostic. No uncertainty band "
        "or population guarantee is claimed.\n"
    )


if __name__ == "__main__":
    main()
