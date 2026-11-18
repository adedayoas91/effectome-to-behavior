# Validate the learned behavioral manifold before expanding observation stresses

Decision record: 18 November 2026. The initial plan and completed replication were
checkpointed in commit `fac7ea4` before implementation. Status: the linear PCA
bridge is implemented; the distributed/nonlinear study and confirmation remain
planned. Priority is a learned behavioral endpoint. Hidden inputs
and measurement-noise sweeps can remain outside this first claim, with explicit
assumptions of full neural-state observation and the declared observation model.
The completed mechanism replication and its unmet ranking-eligibility gate remain
unchanged. This proposal does not merge graph-similarity work.

## The claim to establish

Demonstrate that a low-dimensional representation learned from training data
continues to measure a separately defined behavior when the neural mechanism is
intervened upon, and that the fitted neural mechanism estimates those behavioral
effects on held-out trials.

There are two distinct requirements:

1. **Mechanism validity at a learned endpoint:** fitted and true neural dynamics
   give similar intervention effects after applying the same frozen chart.
2. **Behavioral validity of that endpoint:** effects measured through the chart
   agree with effects on the independently specified behavioral reference.

The behavioral reference must be defined independently of manifold fitting, not
statistically independent of neural activity. It can be a prespecified synthetic
motor/behavioral readout of the fully observed neural system. It must not be
defined retrospectively as the first learned PC, a selected cluster or the
coordinate giving the best test result.

Agreement on the same chart alone is insufficient: both simulator and fitted
model can agree perfectly about an irrelevant neural partition. A constant chart
can also have zero within-chart effect error while missing every behavioral effect.

## What the current implementation does

The successful mechanism experiment uses an observed neuron index, its initial
sign, and the mean of its last four future samples. The endpoint is not learned:
see `paired_intervention_probabilities` in
`src/effectome/experiments/structural_mechanism.py`.

The shared-transition pipeline already fits training-owned scaling, PCA and
KMeans. PCA is fitted to per-sample rows of training histories; target embeddings
average transformed samples over each target interval. Its metadata explicitly
calls the resulting regions neural geometric bins, without an established
behavioral interpretation. The old training-only region/well assignment validates
natural-observation alignment, not interventional invariance. These distinctions
are recorded in `linking/shared_transitions.py` and
`experiments/driver_validation.py::_alignment`.

Thus the missing experiment is a bridge from the neural-space mechanism to a
frozen, behaviorally validated learned endpoint. It is not another graph-only
sanity check or an observational decoding result alone.

## Keep the causal model in neural space

Fit and intervene on the full observed neural state `X`, as in the current
mechanism implementation. Learn a low-dimensional chart `Z = Phi(X)` and a
state/behavior readout, then use that chart only to measure each generated future.

This avoids additionally assuming that `Z` has closed Markov dynamics. A
projection can discard predictive state even when every neural variable is
observed. Defining meaningful interventions on learned coordinates is a separate
problem from finding a statistical representation
([Schölkopf et al., 2021](https://arxiv.org/abs/2102.11107)). The initial claim
should concern neural interventions and learned behavioral outcomes, rather than
intervening directly on latent coordinate numbers.

Fit each neural model once per seed/method. Reuse the same factual and disconnected
neural trajectories across all endpoint arms, with paired innovations. This
isolates endpoint error and avoids repeating expensive native discovery for each
manifold. Simulator coefficients enter only the oracle dynamics, and true
adjacency enters only the known-support diagnostic.

## Separate neural geometry from behavior-anchored learning

Use the following endpoint arms, with common origins, horizons, interventions and
neural models:

| Endpoint arm | What is learned | Supported interpretation if successful |
|---|---|---|
| True behavior reference | Nothing; reference fixed by generator design | Mechanism control and causal-effect gold standard |
| Neural-only geometry | Encoder and geometric regions from neural training data; only a training label permutation may name regions | Unsupervised neural regions align with the behavior of interest |
| Behavior-anchored learned manifold | Low-dimensional encoder plus behavioral readout using neural/behavior training pairs | A learned representation generalizes as a behavioral measurement |
| Full-neural behavioral readout | A training-owned decoder without dimension reduction | Diagnostic of behavioral information lost by compression |

The last arm diagnoses a failed manifold: good full-neural decoding with poor
low-dimensional decoding indicates information loss or an unsuitable objective.
It is not a replacement for the low-dimensional claim.

A learned behavioral decision boundary changes the supervision claim; it is more
than naming arbitrary clusters. Both approaches are legitimate and must be
labeled separately. Behavior supervision is part of established representation
methods, including [CEBRA](https://www.nature.com/articles/s41586-023-06031-6) and
behavior-prioritized dynamical identification
([Sani et al., 2021](https://www.nature.com/articles/s41593-020-00733-0)). These
papers motivate training and evaluation choices, not automatic validity under our
outgoing-mechanism intervention.

Use behavioral measurements only from training/validation to fit or select the
encoder, semantic map, decoder and event boundaries. Never supply true driver
identities, simulator weights or held-out intervention labels to those fits.
Held-out natural and intervention behavior remains evaluation evidence.

## Crossed diagnostic experiment

Cross each endpoint above with these neural-dynamics arms:

| Neural dynamics | Question answered |
|---|---|
| True simulator mechanism | Does the frozen endpoint measure the intended behavioral effect at all? |
| Fitted mechanism with known support | Does parameter/noise learning preserve that effect? |
| Fitted mechanism with each native support estimate | Does adding support-discovery error change recovery? |
| Fitted empty support | Does removing outgoing neural interactions eliminate learned outgoing effects as designed? |

For a given history and intervention, compute three effects:

- `Delta_B`: true simulator futures measured with the independent behavior reference.
- `Delta_M`: the **same true futures** measured with the frozen learned endpoint.
- `Delta_hat_M`: fitted-model futures measured with that same learned endpoint.

Primary behavioral effect error is `|Delta_hat_M - Delta_B|`. The within-chart
error `|Delta_hat_M - Delta_M|` is a mechanism diagnostic, not a substitute for it.
The gap `|Delta_M - Delta_B|` isolates whether the learned measurement changes the
behavioral question. Check absolute errors before averaging; signed averages can
cancel opposite errors.

All comparisons must use the **same factual histories and conditioning set**.
Do not average the chart effect over chart-defined source states and compare it
with a behavioral effect averaged over different behavior-defined source states.
Use a common history set and fixed event definitions; include any origin-label
disagreement in the full transition-event discrepancy. True-behavior strata may
label evaluation tables, but cannot select different input populations for the
two estimands or become an undeclared model input.

## A precise error bound to test

For one fixed training-fitted endpoint, history `h` and intervention `S`, let
`F_M` and `F_B` be the learned and reference **binary transition events** on the
same future trajectory. Their definitions include the factual source label when
the outcome is a transition. Let `P_0` and `P_S` be the true factual and
intervened future distributions, conditional on that same history. Then:

\[
 |\widehat\Delta_M(S;h)-\Delta_B(S;h)|
 \leq |\widehat\Delta_M(S;h)-\Delta_M(S;h)|
 +\epsilon_0(h)+\epsilon_S(h),
\]

where `epsilon_0 = P_0(F_M != F_B)` and
`epsilon_S = P_S(F_M != F_B)`.

This is an elementary project-specific consequence of the triangle inequality:
for binary events, `|E(F_M)-E(F_B)| <= E|F_M-F_B| = P(F_M != F_B)`; apply it to
both branches. It is not a theorem attributed to a representation-learning paper.
Averaging the pointwise bound over the same origins/candidates yields a matching
mean-absolute-error bound. Coupled simulator futures estimate both event labels
on identical samples. Source-class disagreement must be included, not filtered out.

The bound makes an experimental success interpretable. For illustration,
mechanism error of 0.01 plus factual/intervened event disagreements of 0.02 each
gives a total bound of 0.05 probability units. These are illustrative development
error budgets, not achieved findings or already-frozen acceptance thresholds.
Finite Monte Carlo and seed-level uncertainty must still be reported; the exact
inequality does not establish population guarantees from a small dataset.
For continuous behavior, the analogous argument uses absolute/vector-norm
readout errors instead of event disagreements, in behavioral rather than arbitrary
latent-coordinate units.

## Controls that distinguish genuine behavior from a proxy

1. **Behavior changes, neural proxy remains stable.** A neural source can remain
   active after its outgoing influence is cut while the behavior changes. A
   manifold that represents the source/proxy rather than behavior should fail
   the behavioral endpoint test, even if natural decoding was good.
2. **Neural nuisance changes, behavior does not.** Retain strong driver/bystander
   dependence while removing every path into the behavioral output. An
   intervention can then produce real neural/manifold changes with zero true
   behavioral effect. The current endpoint-null generator already supplies a
   useful first version. Require behavioral fidelity as well as a null nomination
   bound; threshold calibration cannot rescue an invalid behavioral chart.
3. **Constant and irrelevant charts.** They must fail occupancy, class coverage,
   continuous-behavior decoding or intervention-semantic gates even if their
   within-chart effect error is zero. Include a strong coherent nuisance neural
   block in the main control so capturing neural variance is not automatically
   equivalent to recovering behavior.
4. **Consistent coordinate changes.** Rotate/sign-flip PCA codes and transform the
   state centers/readout consistently. Events and intervention effects must be
   unchanged. Coordinate-wise MSE against arbitrary latent axes is not the primary
   validity criterion. For a phase/cycle reference, use circular distances or
   sine/cosine decoding rather than raw unwrapped-angle MSE.
5. **Intervention support.** Record how far factual and intervention samples lie
   from training support. No encoder refit, per-intervention alignment or removal
   of hard/off-support interventions is allowed after inspection. If the chart
   extrapolates incorrectly, that is an endpoint failure to report.

Natural and intervention behavioral agreement are both required. Observational
decoding accuracy alone can remain high for a proxy whose relationship to behavior
breaks after the neural mechanism changes.

## Sequence of work

### 1. Minimal learned-endpoint bridge

Implement a frozen endpoint adapter around the existing neural-space rollouts.
PCA is available in the current runtime and already has a persisted transform.
Start with two states and the current eight-step horizon/four-sample future
aggregation. An instantaneous source most closely matches the current endpoint;
use the same source/target definitions in all comparison arms.

First require an identity/known-coordinate adapter to reproduce the current
probabilities exactly. Then fit PCA on training data only and compare neural-only
regions with a separately declared behavior-anchored readout. This establishes
the adapter and learned-endpoint mechanism bridge. By itself it is not evidence
for general nonlinear behavioral manifold learning.

### 2. Main behavioral-manifold control

Use a fully observed neural system with a distributed or two-dimensional behavior
readout fixed before fitting. A phase/trajectory control is preferable to merely
renaming the final neuron. All causally active neural variables should remain
observed, and the structural update class should remain within the fitted
mechanism family. Otherwise a new function-class mismatch could be mistaken for
manifold error. Develop and check the generator with true/known-support arms first.

Evaluate a nonlinear encoder with frozen out-of-sample inference, using the same
diagnostic matrix and behavior reference. BunDLe-Net is a relevant existing
behavior/dynamics approach; CEBRA is a useful separately labeled comparator.
Choose the primary encoder and supervision mode on development data, then freeze
it. Do not select the successful encoder after viewing confirmation outcomes.

### 3. Independent replication and calibration

Use new pools, for example development 40000–40019, calibration 50000–50018,
positive test 60000–60019 and null test 70000–70059. These are proposed reservations,
not completed runs. Do not reuse the inspected 20000/30000 test pools to claim new
confirmation. The final counts and gates must be frozen after engineering checks.

Fit the complete chart procedure separately within each recording's training
trials if following the current per-recording workflow. Calibration must repeat
that same training procedure. A shared cohort chart instead needs its own training
recordings and cannot fit calibration/test recordings transductively.

Calibration must include the learned endpoint and use the complete declared
candidate/direction maximum. It cannot reuse the old known-coordinate cutoffs.
Keep independent null testing, positive power and behavioral fidelity separate.
The existing [replication protocol](mechanism_replication_protocol.md) specifies
the marginal-rank versus conditional-null-bound distinction and seed-level units.

Evaluate predeclared individual and group interventions prospectively. A joint-only
gold case is valid for group recovery and inapplicable to individual-neuron
ranking; keep those denominators distinct. Do not retrospectively change the
completed study's 19/20 eligibility or its frozen acceptance flag.

## Implementation constraints found in the audit

- `ManifoldEmbedder.fit/transform/transform_targets` already separates fitting
  from inference. Add a frozen event/readout adapter and endpoint fingerprint,
  rather than fitting an embedding inside each intervention call.
- The current mechanism starts at trial sample 0 and supplies only initial
  neural states. A temporal/window encoder needs real factual prehistory or a
  prospectively shifted origin, with a declared cue schedule. It must not borrow
  future samples or silently pad a history to obtain a code.
- The current default BundDLe target length is 15, exceeding the mechanism's
  horizon 8/target 4. Its training-label timestamps and future-window inference
  contract need explicit alignment before use. CEBRA's current wrapper uses
  `offset10-model` without an explicit receptive-field support contract.
- PCA is installed; UMAP, CEBRA and Torch were absent in the audited runtime.
  Nonlinear evaluation requires its declared runtime, not a silent PCA fallback.
- The old generic feature-ownership workflow has an estimator allowlist excluding
  the four newer native names. Reuse its ownership principles, not an unmodified
  end-to-end entrypoint assumed to support them.

Required regression tests include unchanged chart parameters after held-out
perturbations, no fit during inference, exact identity-endpoint replay, consistent
coordinate-change invariance, common origin/noise ownership, strict prehistory and
target support, and rejection/reporting of collapsed or irrelevant endpoints.

## What BunDLe-Net theory can support

[BunDLe-Net v5](https://www.biorxiv.org/content/10.1101/2023.08.08.551978v5) provides
conditional causal-consistency theory for complete state-reset interventions.
Its assumptions include a fixed surjection, neural Markov dynamics and exact
distributional consistency across all states mapping to the same latent state;
the supplied proof is discrete-state. This is not an automatic theorem for our
finite-sample continuous chart or outgoing-mechanism deletions. See the
[theorem-scope audit](learned_manifold_validation_citation_audit.md) for exact
version, assumptions and intervention class.

The first bridge avoids needing an autonomous latent transition law. If a later
claim treats the manifold itself as a causal dynamical state, add latent
sufficiency/fiber tests and commuting-transition checks under the actual
intervention family. A small training consistency loss does not establish those
conditions off training support.

## Verdict criteria

A convincing learned-behavioral-endpoint result requires all of: nontrivial
behavioral information retained at the declared reduced dimension; natural and
intervention semantic agreement; low absolute behavioral-effect error on fresh
seeds; appropriate individual/group recovery; and fixed-rule null control. Report
which component fails rather than treating every failure as a clustering problem.

Hidden-state and observation-noise generalization can remain explicitly outside
this claim. Interventional behavioral validity cannot: it is the central new
property being asserted. Graph similarity can later address effectome grouping
or candidate-group generation; it cannot establish a behavioral meaning for an
unvalidated manifold.

Verified bibliographic records are in
[learned_manifold_validation_references.bib](learned_manifold_validation_references.bib).
The checkpoint preserves the original design and justification. Subsequent
implementation and findings are recorded below without rewriting the completed
replication or its acceptance flags.

## Implemented first bridge

Notebook [08](../synthetic/08_learned_manifold_validation.ipynb) and
`python -m synthetic.manifold_validation` run the independent bridge. The default
development profile uses seeds **40000–40004**, both driver and endpoint-null
scenarios, the same 24-trial/1,152-sample generator, 1,024-sample native discovery,
eight-step futures and four-sample targets. The seven dynamics lanes are the true
simulator, known support, empty support and four native methods. There are 360
native window fits, reused across all seven endpoint arms. No calibration or
confirmation pool is executed. Output is isolated under
`outputs/learned-manifold-validation/development/<RUN_LABEL>/`.

PCA is a deliberate engineering bridge: it offers an explicit frozen affine map,
requires no additional runtime, and permits exact coordinate-change checks.
[Jolliffe and Cadima (2016)](https://doi.org/10.1098/rsta.2015.0202) describe PCA;
this reference does not imply that variance retention establishes behavioral
meaning. Two KMeans regions supply the neural-geometry arm, using the
centroid/partition approach associated with
[Lloyd (1982)](https://doi.org/10.1109/TIT.1982.1056489). PCA, KMeans and regularized
logistic readouts use [scikit-learn](https://www.jmlr.org/papers/v12/pedregosa11a.html).
These are implementation references, not causal-identification guarantees.

The scaler, two-dimensional PCA and KMeans regions see only nine inner-training
trials. A training-only Hungarian permutation names the geometric regions; it
cannot adjust their decision boundary. Behavioral readouts are explicitly
supervised: logistic regularization is selected using the three inner-validation
trials, then the readout alone is refit on all 12 outer-training trials with the
encoder frozen. Each decoder row is either an instantaneous neural observation
or a complete, nonoverlapping four-sample within-trial mean, with equal row weight
(four instantaneous rows per window row). This trains the declared instantaneous
source and averaged future target without constructing cross-trial windows.
Artifacts record these counts, parameters, ownership and fingerprints. The
full-neural supervised arm diagnoses information lost by compression.

All neural-dynamics arms share the same 12 held-out factual origins. Each model's
paired innovations and generated futures are reused across endpoint arms.
Per-origin/candidate errors are made absolute before averaging. Saved binary
events permit independent reconstruction of the semantic discrepancy and
empirical triangle bound, including source-label errors. Both individual neurons
and the predeclared driver/control pairs remain in the evaluation. This bridge
reports effect error rather than claiming calibrated nominations or changing the
old individual-ranking gate.

The constant endpoint is retained with its occupancy failure; source-proxy and
neural-nuisance coordinate endpoints remain visible. Raw neural training-envelope
excursions are recorded per origin and branch. A coordinate box is a diagnostic,
not a support theorem, and no observations are removed. Per-endpoint code boxes
are also stored in frozen learned artifacts. No chart is refit or realigned on
held-out natural or interventional outcomes.

Regression coverage includes exact replay of the old coordinate API, ownership of
inner-trial PCA, artifact roundtrips, sign/rotation invariance with a consistently
transformed readout, origin/candidate ordering invariance, constant/proxy/nuisance
counterexamples, source-label error in the event bound, prevention of signed-error
cancellation, resume integrity and visible unavailable native lanes. This verifies
the bridge; it does not substitute for the planned nonlinear behavioral study.

The first five-seed bridge is complete. Its
[findings and implications](learned_manifold_development_findings.md) retain the
PCA semantic failure and the successful full-neural diagnostic separately.
The next substantive representation experiment remains the distributed,
behavior-prioritized encoder study above. The current notebook defaults to a new
`endpoint-bridge-03` namespace after integrity-guard changes; the first run's
executed source and evidence remain frozen in its curated archive.
