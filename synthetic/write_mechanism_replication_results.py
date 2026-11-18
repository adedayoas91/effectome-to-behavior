"""Write a cited results/discussion document from a completed, frozen replication."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from synthetic.export_mechanism_replication import METHODS, export_replication_figures

LABELS = dict(
    zip(METHODS, ["Known support", "Empty support", "c-GC", "c-GC*", "fastC-GC", "fastC-GC*"], strict=True)
)


def _number(value, digits=6):
    return "NA" if value is None else f"{value:.{digits}f}"


def write_replication_results(root: Path) -> Path:
    report = json.loads((root / "replication-report.json").read_text())
    if not report["execution_complete"]:
        raise ValueError("results documents require the complete declared study")
    export_replication_figures(root)
    config = report["protocol"]["config"]
    destination = root / "analysis"
    methods = [name for name in METHODS if name in report["summary"]]
    primary, nulls, paired, eligibility = [], [], [], []
    for name in methods:
        summary = report["summary"][name]
        positive = summary["positive"]
        mae, brier, rank = [positive[key] for key in ("effect_mae", "factual_brier", "topk_recall")]
        primary.append(
            " & ".join(
                [
                    LABELS[name],
                    f"${_number(mae['mean'])} \\pm {_number(mae['sd'])}$",
                    _number(brier["mean"]),
                    f"{rank['n_seeds']}/{rank['planned_seeds']}",
                    _number(rank["available_mean_descriptive"], 3),
                    _number(positive["nomination_recall"]["available_mean_descriptive"], 3),
                    _number(positive["nomination_precision"]["available_mean_descriptive"], 3),
                ]
            )
            + r" \\"
        )
        frozen = report["calibration"][name]
        nulls.append(
            " & ".join(
                [
                    LABELS[name],
                    _number(frozen["operational_threshold"], 3),
                    f"{summary['null_false_nomination_seeds']}/{summary['null_available_seeds']}",
                    _number(summary["null_one_sided_95_upper_bound"]),
                    "pass" if summary["gates"]["null_bound"] else "not passed",
                ]
            )
            + r" \\"
        )
        change = report["paired_history"][name]
        metric = change["delta_effect_mae"]
        interval = metric["mean_95ci"]
        interval_text = f"$[{interval[0]:+.6f},\\ {interval[1]:+.6f}]$" if interval else "NA"
        pvalue = change.get("holm_paired_mae_p")
        paired.append(
            " & ".join(
                [
                    LABELS[name],
                    str(change["n_paired_seeds"]),
                    _number(metric["mean"]),
                    interval_text,
                    f"{pvalue:.3g}" if pvalue is not None else "--",
                ]
            )
            + r" \\"
        )

    exceptions = []
    for cell in report["cells"]:
        if cell["phase"] != "positive":
            continue
        known = cell["methods"]["known_support"]
        if known["status"] != "available":
            continue
        for direction in known["metrics"]["directions"]:
            if direction.get("topk_recall") is not None:
                continue
            gold = direction.get("true_probability_effects", {})
            neurons = [v for k, v in gold.items() if k.startswith("neuron:")]
            groups = {k: v for k, v in gold.items() if k.startswith("group:")}
            unit = max(groups, key=lambda key: groups[key]) if groups else None
            nominees = []
            estimates = []
            for name in methods:
                if (
                    name in {"known_support", "empty_support"}
                    or cell["methods"][name]["status"] != "available"
                ):
                    continue
                target = next(
                    d
                    for d in cell["methods"][name]["metrics"]["directions"]
                    if d["source_state"] == direction["source_state"]
                )
                score = target.get("estimated_probability_effects", {}).get(unit)
                if score is not None:
                    estimates.append(score)
                    nominees.append(score > report["calibration"][name]["operational_threshold"])
            record = {
                "seed": cell["seed"],
                "source_state": direction["source_state"],
                "n_origins": direction["n_origins"],
                "largest_individual_effect": max(neurons, default=0),
                "group": unit,
                "group_true_effect": groups.get(unit),
                "native_estimated_effects": estimates,
                "native_nominating_group": sum(nominees),
            }
            exceptions.append(record)
            estimate_range = f"$[{min(estimates):.6f},\\ {max(estimates):.6f}]$" if estimates else "NA"
            eligibility.append(
                " & ".join(
                    [
                        str(cell["seed"]),
                        str(direction["source_state"]),
                        str(direction["n_origins"]),
                        _number(record["largest_individual_effect"]),
                        _number(record["group_true_effect"]),
                        estimate_range,
                        f"{sum(nominees)}/{len(nominees)}",
                    ]
                )
                + r" \\"
            )
    (destination / "eligibility-diagnostics.json").write_text(json.dumps(exceptions, indent=2) + "\n")
    native = [name for name in methods if name not in {"known_support", "empty_support"}]
    improvements = [
        100
        * (
            1
            - report["summary"][name]["positive"]["effect_mae"]["mean"]
            / report["summary"][name]["comparison"]["effect_mae"]["mean"]
        )
        for name in native
        if report["summary"][name]["positive"]["effect_mae"]["mean"] is not None
        and report["summary"][name]["comparison"]["effect_mae"]["mean"]
    ]
    reduction = f"{min(improvements):.2f}--{max(improvements):.2f}" if improvements else "NA"
    substitution = {
        "@@PRIMARY@@": "\n".join(primary),
        "@@NULL@@": "\n".join(nulls),
        "@@PAIRED@@": "\n".join(paired),
        "@@ELIGIBILITY@@": "\n".join(eligibility),
        "@@DECISION@@": str(report["bounded_control_replication_passed"]),
        "@@CELLS@@": str(report["completed_cells"]),
        "@@LANES@@": str(report["available_method_runs"]),
        "@@CALN@@": str(len(config["calibration_seeds"])),
        "@@POSN@@": str(len(config["positive_seeds"])),
        "@@NULLN@@": str(len(config["null_seeds"])),
        "@@SIGNATURE@@": report["manifest_fingerprint"],
        "@@REDUCTION@@": reduction,
    }
    text = TEMPLATE
    for marker, value in substitution.items():
        text = text.replace(marker, value)
    assert "@@" not in text
    path = destination / "replication-results.tex"
    path.write_text(text)
    repository = Path(__file__).resolve().parents[1]
    shutil.copyfile(repository / "docs/mechanism_replication_references.bib", destination / "references.bib")
    stats = """# Statistical appendix

This document exports frozen runner summaries and decisions without new fitting,
cutoff selection or alternative acceptance rules. The protocol and all exact
settings are in ../study-protocol.json; source and candidate ownership are recorded
in ../source-manifest.json. Raw independent-seed values are in seed-metrics.csv.

Primary analysis uses 20 positive seeds and 60 untouched null seeds, after 19
separate calibration-null seeds. Directions are first averaged within seed;
windows, candidates, trials and related method variants do not multiply n.
Known/empty-support controls and all unavailable/ineligible evidence remain explicit.
Means/SDs and 2,000-resample percentile intervals use seed means. Incomplete planned
sets receive no complete-set mean or interval. The separate available means are
conditional descriptive summaries, not replacements for the frozen complete-set gate.

Calibration orders global-null whole-seed maxima, with rank ceil((n+1)*.95), uses
strict exceedance and retains the prespecified 0.10 effect-size floor. It provides
marginal protection under exchangeability of the randomized scoring procedure,
not a 95% conditional guarantee for the realized cutoff. Fresh null-test exact
Clopper–Pearson bounds are per-method whole-seed events, not simultaneous coverage
or strong non-null FDR. Incomplete null sets receive no bound.

The 256/1024 sensitivity comparison pairs full fitting data, origins and oracle
probabilities within each positive seed. Native MAE sign tests exclude exact ties
and use one fixed four-method Holm family; missing contrasts remain p=1. They test
sign tendency, not mean magnitude, and cannot override primary acceptance gates.
Candidate group diagnostics in eligibility-diagnostics.json describe prespecified
interventions; they do not retrospectively replace the individual-ranking criterion.
Verified references appear in references.bib and the project citation audit.
"""
    (destination / "stats-appendix.md").write_text(stats)
    return path


TEMPLATE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern,microtype}
\usepackage{amsmath,booktabs,graphicx,xurl}
\usepackage[numbers,sort&compress]{natbib}\usepackage[hidelinks]{hyperref}
\usepackage{placeins}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.35em}
\setlength{\emergencystretch}{2em}
\title{Fresh-seed mechanism replication\\\large Results, eligibility and discussion}
\author{Effectome-to-behavior project}\date{17 November 2026}
\begin{document}\maketitle
\section{Completed study and interpretation boundary}
All @@CELLS@@ declared cells completed, with @@LANES@@ available method/control
entries. This study evaluates the revised outgoing-disconnection readout in the
fully observed, model-matched control. It does not replace the old 560-run
latent-bistable benchmark, validate a learned behavioral manifold, or include
graph-similarity experiments. The imported native estimators remain c-GC/c-GC*
and fastC-GC/fastC-GC* \citep{adedayo2025cgc,adedayo2026fastcgc}.

The paired native comparisons show @@REDUCTION@@\% lower mean effect MAE at the
longer history across @@POSN@@ fresh process seeds. Exact null outcomes and the
individual-ranking eligibility limit are reported separately below. These provide
positive evidence for effect estimation within this control while preserving the
original acceptance decision.

The frozen all-four bounded-control acceptance flag is \textbf{@@DECISION@@}.
Complete execution, accurate effect estimation, applicable individual-neuron
ranking and null specificity are different outcomes. Ineligible evidence is
retained, not silently dropped to obtain a passing decision.

\section{Frozen design and calibration}
The primary discovery history is 1,024 samples; a 256-sample history is paired on
the same positive-test recordings. Each seed has 24 trials of length 1,152 and six
observed neurons. Twelve trials supply mechanism fitting, including three reserved
for penalty/noise selection; native discovery sees only the nine inner-training
trials. Twelve whole trials are held out. This ownership prevents evaluation from
selecting fit parameters \citep{cawley2010overfitting}. Known support supplies only
adjacency to its explicit oracle lane; native lanes learn support and all lanes
fit coefficients/noise. Ridge fitting uses the declared additive-tanh basis
\citep{hoerl1970ridge,peters2014additive}; this favorable match does not establish
general causal identifiability.

The disjoint pools contain @@CALN@@ calibration-null seeds, @@POSN@@ positive-test
seeds and @@NULLN@@ null-test seeds. For each method, one calibration score is the
maximum estimated probability effect across eight candidates and both directions
within a seed. Its rank is $r=\lceil(n+1)(1-0.05)\rceil$. The operational cutoff
is the larger of that ordered calibration score and the existing 0.10 minimum
effect size, with strict exceedance for nomination. Thresholds were locked before
test jobs were submitted. The finite-sample rank principle gives marginal global-null
protection under exchangeability \citep{angelopoulos2021conformal}; it does not
certify a realized cutoff's conditional false-event rate at 95\% confidence.

Conventional CPU permutation randomness is not controlled by the public seed API
in the pinned upstream package. Its randomization is part of the fixed scoring
procedure. First-completed native evidence is retained, and favorable reruns are
not selected. Exchangeability of randomized scores is an assumption, not a claim
of controlled or byte-identical conventional random streams.

\section{Effect estimation and individual ranking}
Effect MAE compares estimated with simulator probability reductions after outgoing
disconnection. This structural operation differs from refitting a predictive
model without graph features \citep{pearl2010causalinference,williamson2023importance}.
Brier scores concern observed factual binary events. Top-k and nomination recall
concern intervention-positive neurons; nomination precision includes all declared
candidate units/groups. Table~\ref{tab:primary} preserves the planned denominators.

\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{Primary history. MAE and Brier use the complete planned positive-seed set
when available. Ranking and nomination columns are explicitly conditional
descriptions on eligible seeds; they do not replace suppressed complete-set means.
SD uses independent seed means.}
\label{tab:primary}
\begin{tabular}{@{}lrrrrrr@{}}\toprule
Method & Effect MAE $\pm$ SD & Brier & Rank eligible & Top-k & Nom. recall & Nom. precision\\\midrule
@@PRIMARY@@
\bottomrule\end{tabular}\end{table}

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/01-fresh-seed-history.pdf}
\caption{Paired fresh-seed history comparison. Filled points and bars summarize
complete planned seed sets with descriptive seed-bootstrap 95\% intervals.
Hollow points show eligible-subset descriptions without an interval or a
complete-set acceptance claim. Dashed lines are the frozen ranking/MAE gates.}
\end{figure}\FloatBarrier

\section{Eligibility and the predeclared group intervention}
The following rows lack an intervention-positive individual neuron under the
0.10 simulator-effect threshold. They remain in the study. The group candidate
list was fixed before evaluation, including the pair of neurons 0/1; its recovery
is a descriptive diagnostic, not a new retrospective acceptance criterion.

\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}
\caption{Directions ineligible for individual-neuron top-k under simulator truth.
The group column uses the strongest predeclared group. The final column counts
native methods nominating that group at their already frozen cutoffs.}
\begin{tabular}{@{}rrrrrrr@{}}\toprule
Seed & Source & Origins & Max neuron effect & Group truth & Native estimate range & Nominate\\\midrule
@@ELIGIBILITY@@
\bottomrule\end{tabular}\end{table}

A positive joint effect with individually subthreshold effects makes individual
ranking inapplicable; an estimated graph cannot change that simulator gold set.
The complete-set individual-ranking criterion therefore remains unmet when a
planned direction has no gold neuron, even if the known-support and native
readouts recover the joint intervention accurately. This is an eligibility and
causal-unit issue, rather than evidence that all native discovery failed.
Any future group-aware acceptance rule must be frozen on fresh seeds. Candidate
group discovery itself is not tested by providing the designed pairs here.

\section{Untouched null-test specificity}
One independent null seed contributes one event: any false nominee across the
candidates and supported directions. Table~\ref{tab:null} reports per-method
one-sided exact 95\% upper bounds \citep{clopper1934confidence}. Zero events in 60
seeds gives approximately 0.04870. Calibration and null validation are separate;
neither implies simultaneous coverage across methods or strong non-null FDR.

\begin{table}[htbp]\centering\small
\caption{Frozen operational cutoffs and complete null-test audit.}\label{tab:null}
\begin{tabular}{@{}lrrrr@{}}\toprule
Method & Cutoff & False-event seeds & Exact upper & Null gate\\\midrule
@@NULL@@
\bottomrule\end{tabular}\end{table}

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/02-nomination-and-null.pdf}
\caption{Positive nomination descriptions and untouched-null uncertainty.
Hollow positive points use eligible subsets; null caps are per-method one-sided
Clopper--Pearson 95\% upper bounds. Passing the null gate does not override failed
positive eligibility or establish biological specificity.}
\end{figure}\FloatBarrier

\section{Paired discovery-history effects}
Within each seed, histories share full mechanism-fitting data, evaluation origins,
oracle probabilities and rollout noise. Only discovery context changes. Table~\ref{tab:paired}
reports 1,024-minus-256 MAE changes; negative values favor the longer history.
Intervals resample independent seed differences 2,000 times \citep{efron1979bootstrap}.
The four native sign tests form one prespecified Holm family
\citep{holm1979sequential}. Their p-values concern sign tendency and cannot
override primary gates. The secondary history receives no calibrated null claim.

\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}
\caption{Paired MAE changes, probability units. Controls receive no native-family
Holm p-value. Intervals are descriptive percentile-bootstrap 95\% intervals.}
\label{tab:paired}
\begin{tabular}{@{}lrrrr@{}}\toprule
Method & Paired seeds & Mean change & Descriptive 95\% interval & Holm p\\\midrule
@@PAIRED@@
\bottomrule\end{tabular}\end{table}
\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/03-paired-mae-changes.pdf}
\caption{Every native paired seed difference. Crosses are individual process-seed
changes; dots/bars are complete-set means and descriptive bootstrap 95\% intervals.}
\end{figure}\FloatBarrier

\section{Discussion and next decision}
The useful comparison is effect estimation against the known- and empty-support
controls across fresh seeds, together with a separate fixed-rule null audit.
Accurate effects and favorable conditional rankings can coexist with an unmet
complete-set ranking gate when individual effects are subthreshold but the joint
effect is positive. Reporting the latter as zero recovery would be incorrect; silently
dropping it to declare full acceptance would also change the frozen protocol.

The study changes the strength of evidence for the favorable mechanism control,
not the data-generating assumptions. Full observability, the matched nonlinear
basis, predeclared endpoint and designed candidate pairs remain limitations.
The previous latent/manifold benchmark remains a failed evaluation. Calibration
that retains the old 0.10 floor does not itself improve point estimates; its
contribution is a separate cutoff-selection procedure followed by untouched null
validation. No performance gain should be attributed to graph similarity, which
was not used. A future graph comparison can address representation and group
candidate generation, but cannot make an individual-neuron gold set exist in a
joint-only intervention case.

The frozen all-four acceptance flag remains \textbf{@@DECISION@@}; the exact failed
gates and all individual seed values are preserved in the JSON and CSV evidence.
Prospective group-aware eligibility and robustness to observation/functional
misspecification are distinct next tests; repeated optimization on these inspected
test seeds would risk selection bias \citep{cawley2010overfitting}.

\section{Evidence and reproduction}
Study fingerprint: \path{@@SIGNATURE@@}. The parent directory contains the frozen
protocol, calibration evidence, full cell/native artifacts and implementation
snapshot. This report exports stored summaries without refitting or changing the
decision. See \path{seed-metrics.csv}, \path{statistics.json},
\path{eligibility-diagnostics.json}, \path{stats-appendix.md} and
\path{figure-catalog.md}. The implementation-test log records the regression suite;
the project protocol supplies the reference-verification audit.
{\small\setlength{\bibsep}{1pt}\bibliographystyle{plainnat}\bibliography{references}}
\end{document}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/driver-mechanism-replication/replication/mechanism-replication-01"),
    )
    args = parser.parse_args()
    print(write_replication_results(args.output_root))


if __name__ == "__main__":
    main()
