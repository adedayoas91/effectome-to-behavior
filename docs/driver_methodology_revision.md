# Driver methodology revision after the completed four-CSL study

Decision date: 15 November 2026. Status: the first development diagnostic is
implemented; revised confirmation has not been run. This document records why the method is changing,
the first implementation boundary, and the evidence required before a new claim.
The pre-revision code and completed-manuscript snapshot are preserved at public
checkpoint [`ed543c1`](https://github.com/adedayoas91/effectome-to-behavior/commit/ed543c139cfba89f65a337964802ca86c8323f28).
Bibliographic metadata and the scope of every methodological citation are audited
in [driver_methodology_citation_audit.md](driver_methodology_citation_audit.md);
the retrieved entries are in
[driver_methodology_references.bib](driver_methodology_references.bib).

## Evidence and reason for the change

The frozen scheduled-v2 study completed all 560 method/recording runs and 71,680
native window fits. All four methods failed the declared driver-recovery criteria.
The authoritative numerical record is
`synthetic/results/driver_confirmation_v2/driver-validation.json`; its original
source, graphs and scoring audits remain in the accompanying reproduction bundle.
These results are retained as a failed evaluation of the original pipeline.

The existing score refits a transition predictor after removing a neuron's
outgoing graph-feature block. The same neuron's measured activity, population
readouts and manifold coordinates remain available. This is conditional
predictive importance, whereas the benchmark truth disconnects outgoing influence
inside the generating mechanism. These are different estimands: predictive
importance is a contrast in predictiveness [1], while an intervention changes
structural assignments [2]. A causal influence need not supply additional
predictive information conditional on its proxies and descendants.

Each recording supplies only 64 training origins. The decoder has 48 baseline
columns after repeated source indicators and 112 graph columns. The 56 graph-change
columns are identically zero because trials have separate continuity groups.
Unsigned support also discards interaction strength and is entered independently
of source activity. These implementation facts motivate diagnostics; they do not
establish the cause of the negative result. Removing zero columns alone cannot
create signal or identify intervention effects.

## Adopted revision

1. Preserve graph-feature omission as an incremental predictive diagnostic.
   Do not rename it a causal intervention or use raw activity deletion as a
   substitute for a generative intervention.
2. Add a graph-constrained nonlinear structural transition readout. Imported CSL
   methods still discover support; separately fitted functions parameterize
   transitions and outgoing-disconnection rollouts. An estimated causal graph
   alone does not specify the required intervention distribution [2,3].
3. Compare known support with estimated support while keeping the simulator,
   training/evaluation trials, endpoint and intervention definition identical.
   Known-support recovery evaluates the downstream readout; estimated-support
   recovery adds discovery error. Known support is an explicitly labeled oracle
   diagnostic and is never passed to the estimated-support lane.
4. Start with a fully observed, lag-one, additive nonlinear control with independent
   innovations and a predeclared observed endpoint. This is deliberately a
   model-matched favorable control, not a replacement for the failed bistable
   benchmark or evidence of robustness to hidden state. Additive-noise results
   motivate explicit structural assumptions, but their iid DAG identifiability
   results do not automatically establish identifiability of this time-series
   implementation [3].
5. Tune mechanism regularization using only training trials; evaluate complete
   held-out trials. Calibration, nonlinear feature ablations, sample-size curves,
   redundant groups, observation stresses and independent confirmation are later
   stages. Repeated model selection can overfit the evaluation criterion [4].

## First executable implementation

The new development path uses independent trials of

`X_j(t+1) = b_j + a_j X_j(t) + sum_i w_ij tanh(X_i(t)) + epsilon_j(t+1)`.

Self-history is retained separately; the sum excludes self-edges. Known support
and native CSL estimates constrain which cross-neuron terms may be fitted. Ridge
regression [5] is used on the declared nonlinear features; regularization is
selected with a chronological split of whole training trials [4].
The nonlinear transition readout is not a new discovery algorithm or a VAR
estimator, and its fitted weights are not native CSL weights.
The support estimators remain the imported conventional c-GC/c-GC* and
fastC-GC/fastC-GC* implementations. Their software pins and existing method
attribution are recorded in the [installation/provenance guide](../README.md#install),
the [conventional runtime guide](../tools/conventional-cgc/README.md), and the
[four-CSL migration record](four_csl_migration.md). This revision changes the
downstream mechanism readout; it does not reimplement or claim authorship of
those discovery algorithms.

Outgoing disconnection removes the selected sources' cross-neuron terms in every
future update, retaining their own activity equations and self-history. The
simulator implements exactly the same operation. Factual and disconnected
branches use paired innovations. Their future-target probabilities share one
predeclared observed endpoint and threshold, fixed by the control design.
The new endpoint is a favorable-control coordinate; this stage does not validate
the older learned-manifold alignment problem.

The fixed `tanh` basis is a model-matched design choice for this first control,
not a claim of a generally correct neural update law. It lets the known-support
comparison test parameter learning and rollout attribution before introducing
function-class misspecification. Estimated support, fitted coefficients and
simulator coefficients remain separate objects. Each learned lane estimates its
own innovation scales from inner-validation residuals; it receives neither the
simulator's coefficients nor its innovation scales. The known-support lane alone
receives the simulator's adjacency mask, explicitly as evaluation-side oracle
information.

The first deliverable includes executable known-support, empty-support negative
control, and optional four imported CSL support lanes; trial-safe fitting and
rollouts; paired simulator truth; transparent numerical/recovery summaries;
configuration and source provenance; and tests of invariance to held-out fitting
data, disconnection semantics, null effects and oracle recovery. Native discovery
uses only inner-training-trial histories and never concatenates across trial boundaries.
The last whole training trials are reserved for penalty selection and residual
noise estimation; they are also excluded from support discovery. After selection,
the mechanism is refitted on all outer training trials with that fixed support.

Use a new namespace, `outputs/driver-mechanism-validation/`. Development seeds are
separate from the completed confirmation seeds 300--319 and 400--459. Development
evidence must never be reported as independent confirmation. Any new acceptance
rules must be documented and frozen before fresh confirmation is evaluated.

The implemented entrypoints are `pipeline/15_mechanism_validation.py` and
`synthetic/archive/06_mechanism_driver_validation.ipynb`. A paired history-length launcher
is `python -m synthetic.mechanism_history_diagnostic`; default execution only
inventories settings. Initial measured outcomes, including the short-history
failures and the longer-history operating case, are recorded in
[mechanism_development_findings.md](mechanism_development_findings.md). They are
development evidence under the limits below, not confirmation of the revised method.

## Scope and decision sequence

| Stage | Question answered | Interpretation boundary |
|---|---|---|
| Exact simulator replay | Is the numerical disconnection and endpoint calculation correct? | A software/oracle check, not learned recovery |
| Known support plus fitted mechanism | Can the readout learn intervention effects when adjacency is supplied? | An oracle-support, model-matched development result |
| Native estimated support plus fitted mechanism | What changes when discovery error is added? | The same endpoint and fitting ownership must be retained |
| Empty support | Can the readout invent outgoing effects without outgoing terms? | Negative implementation control, not a calibrated null-rate bound |
| Broader future studies | Do recovery and specificity survive redundancy, observation stress and larger samples? | Requires separately frozen designs and fresh confirmation |

A favorable result in the new control would establish a limited operating case;
it would not overturn the completed failure in the partially observed bistable
benchmark. Conversely, a known-support failure points first to readout,
observation or rollout problems rather than proving that native discovery failed.
The latter is a diagnostic inference to be investigated, not an identified cause
of the earlier negative result.

Prediction quality, intervention-effect error, candidate ranking and false
nomination are separate outcomes. In particular, the old `0.01`-nat omission
threshold cannot be applied to probability reductions. Maximum-score calibration
across candidate units and directions is a planned design change; it requires
separate development/calibration seeds before a frozen independent null study.
The previously declared 60-null-seed false-nomination criterion is not relaxed or
retrospectively reinterpreted.

### Null uncertainty in development summaries

For each method, one independent null seed contributes one Bernoulli outcome:
whether **any** candidate unit in **either** supported direction was falsely
nominated. Neurons, groups, directions and overlapping time samples are not
independent binomial trials. With `k` such outcomes among all `n` planned and
available null seeds, the reported one-sided 95% Clopper--Pearson upper bound [6]
is `BetaQuantile(0.95; k+1, n-k)` when `k<n`, and 1 when `k=n`. This is the
beta-quantile form of inversion of the binomial lower-tail test; it is not a
posterior probability. Incomplete planned null sets do not receive this bound.

Coverage assumes independent seeds and a fixed nomination rule. The current
probability-effect cutoff is a development rule, not a calibrated causal
significance threshold. The bound concerns the per-method probability of the
whole seed-level false-nomination event, not neuron-level false discovery rate
or simultaneous coverage over all methods. A small development run can therefore
have no false nominees and still have a very large upper bound: for zero events
in one seed the bound is 0.95. Calibration and adequate fresh null repetition
remain necessary before a specificity claim.

## Verification and completion criteria

- Known-support fitted rollouts approximate paired simulator effects in the
  favorable control; an empty-support readout cannot invent outgoing effects.
- Simulator and readout both retain self-history during outgoing disconnection.
- Scaling, regularization selection, residual distributions and support discovery
  use training trials only. Adjacent samples across trials are never treated as
  a transition. Test outcomes cannot select parameters or endpoint thresholds.
- Candidate/group recovery, prediction and null calibration are reported
  separately; good prediction is not a sufficient or necessary ranking claim.
- All failures and missing native runtimes remain explicit. No discovery fallback,
  favorable-seed selection, relaxed old gates, or overwritten confirmation files.
- Run focused tests, the full test suite, lint and static typing. Compile the
  manuscript after the cited rationale and completed negative result are synced.

## References

[1] Williamson, B. D., Gilbert, P. B., Simon, N. R., and Carone, M. (2023).
A general framework for inference on algorithm-agnostic variable importance.
*Journal of the American Statistical Association*, 118(543), 1645--1658.
https://doi.org/10.1080/01621459.2021.2003200;
author manuscript: https://arxiv.org/abs/2004.03683.

[2] Pearl, J. (2010). Causal Inference. *Proceedings of Machine Learning Research*,
6, 39--58. https://proceedings.mlr.press/v6/pearl10a.html.

[3] Peters, J., Mooij, J. M., Janzing, D., and Schölkopf, B. (2014). Causal
Discovery with Continuous Additive Noise Models. *Journal of Machine Learning
Research*, 15(58), 2009--2053. https://jmlr.org/papers/v15/peters14a.html.

[4] Cawley, G. C., and Talbot, N. L. C. (2010). On Over-fitting in Model Selection
and Subsequent Selection Bias in Performance Evaluation. *Journal of Machine
Learning Research*, 11(70), 2079--2107.
https://www.jmlr.org/papers/v11/cawley10a.html.

[5] Hoerl, A. E., and Kennard, R. W. (1970). Ridge Regression: Biased Estimation
for Nonorthogonal Problems. *Technometrics*, 12(1), 55--67.
https://doi.org/10.1080/00401706.1970.10488634.

[6] Clopper, C. J., and Pearson, E. S. (1934). The Use of Confidence or Fiducial
Limits Illustrated in the Case of the Binomial. *Biometrika*, 26(4), 404--413.
https://doi.org/10.1093/biomet/26.4.404.

Publisher/arXiv records were inspected on 15 November 2026. A bibliography and
claim-to-source verification record accompany the implementation documentation.
