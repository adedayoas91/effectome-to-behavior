# Transition-driver ground-truth validation

Decision: 14 November 2026; four-CSL revision: 15 November 2026.
Historical protocol: **proposed benchmark criteria at the original decision**.
The later scheduled-origin revision completed 560/560 method runs and all four
methods failed its frozen gates; see `driver_confirmation_revision.md` and
`../synthetic/results/driver_confirmation_v2/`. The separate current mechanism
revision is documented in `driver_methodology_revision.md`. Seed/timing proposals
below retain their original historical scope; the frozen manifest is authoritative.
This protocol adds an intervention-defined recovery target to the existing
circuit/activity controls. It is not a preregistration or a report that the
framework has passed scientific validation.

The current panel is conventional c-GC, c-GC*, fastC-GC and fastC-GC*, imported
from pinned upstream packages. It excludes VAR estimators and uses native
unsigned lag support without a VAR coefficient readout. The four variants share
one discovery family; their agreement supplies consistency evidence rather than
independent causal replication. Begin with
[`synthetic/01_four_csl_sanity.ipynb`](../synthetic/01_four_csl_sanity.ipynb)
and the [runtime guide](../tools/conventional-cgc/README.md).

## Question and scope

Can a method's transition-specific neuron/group graph-feature scores distinguish
neurons whose outgoing influence changes a future state-transition probability
from correlated but causally inert neurons? Does a jointly acting group remain
detectable when either member can substitute for the other?

The existing `one_to_one`, `matched_marginals`, `fixed_circuit`, and `stationary`
examples retain their original purpose: they test circuit/activity distinctions
and comparison contracts. Known circuit matrices alone do not establish known
transition-driving neurons. Their previous figures and smoke results are not
retroactively relabelled as driver-recovery validation.

## Designed causal process

The new simulator generates independent trials with a bistable population
coordinate and observed neural activity. Neurons 0 and 1 supply the planted
outgoing drive, neuron 2 can be a correlated proxy, population readouts occupy
neurons 3 and 4, and the final neuron is a neutral control. Initial trial wells
support both semantic directions, A→B and B→A. Cue onset occurs after the first
eligible history so pre-transition origins are available. Trial boundaries are
declared; neither histories nor future events may cross them.
The revised profiles use 512-sample trials, 256-sample histories, cue onset at
288 and cue offset at 480. The first complete history precedes cue onset by 32
samples. Target length 8 and stride 16 preserve the earlier forecast horizon;
the longer history supports the fast complete-shift beta=0.005 resolution.
This geometry is a new protocol, not a relabelling of the older pilot.

| Condition | Causal mechanism | Validation purpose |
|---|---|---|
| `single_driver` | A source-specific actuator supplies outgoing drive | Recover a transition-specific individual contributor |
| `redundant_group` | Either member of the planted pair can supply drive | Compare joint recovery with weaker individual necessity |
| `correlated_bystander` | A proxy tracks the drive without controlling the population coordinate | Distinguish predictive association from intervention-defined contribution |
| `null` | Observed candidate neurons do not drive transitions | Measure false candidate nominations |
| `hidden_confounder` | An unobserved input drives both observed activity and transitions | Expose a causal-identification failure mode; this is a diagnostic challenge |

Observation noise and causal calcium-like filtering are configurable measurement
stresses. Nonlinear bistable dynamics also challenge linear graph estimators.
Simulation labels, latent coordinates, replay state, and intervention results
remain metadata. Inference receives observed traces, their temporal contract,
and predeclared candidate groups.

## Intervention-defined truth

For each factual origin and its eligible future target, replay the future causal
process after disconnecting a candidate neuron's or group's outgoing influence.
The factual and disconnected futures use paired common random numbers. This
changes the simulator's generating mechanism; it does not erase columns from a
fitted predictor or censor the measured population readout.

The source well is defined by the factual trailing target window. The future
event uses the mean population coordinate over the future target window at the
same physical horizon as the prediction. A candidate's simulator effect is the
factual minus disconnected probability of entering the destination well.
Effects are conditional on the factual origin and may reveal inhibition or
stabilization as well as promotion of a transition. Positive effect is not
assumed simply because a neuron was assigned a nominal actuator index.

The first proposed positive-truth cutoff is a probability reduction of **0.10**.
Recovery labels use these measured simulator effects at the evaluated horizon.
Nominal circuit membership alone is not the gold label.

## Shared target and semantic alignment

All four current methods use one training-owned PCA reference, the same
predeclared two-region vocabulary, identical future events, and the same
activity/history baseline. The two regions are geometric bins, not an assertion
that inferred biological regimes have been identified.

A training-only contingency table and assignment map align region labels with
the simulator's two semantic wells. The map is frozen before held-out recovery
evaluation; held-out alignment is audited separately. State numbers are not
pooled across independent seeds. Poor alignment makes a recovery claim
ineligible rather than justifying a favorable relabelling of held-out outcomes.
Each direction uses the minimum of source-well purity and class-conditional
future-target precision/recall agreement. An absent future class is unsupported.
Global accuracy cannot hide rare-transition misalignment or dominant stays.
The simulator's latent-well event and a geometric manifold-region event remain
different objects, and their agreement must be reported.

Native support can contribute graph-feature changes and outgoing omission
blocks. It does not parameterize a dynamical update, so projected coefficient
drives and neural forecasts are unavailable for this panel. The earlier opposite
rotation controls differ in coefficient signs while sharing the same lag
support. Binary support cannot distinguish these planted regimes. A separate
support-changing matched-marginal control is required before claiming the
current panel discovers circuit/manifold degeneracy; the original mathematical
occupancy result remains valid.

## Proposed acceptance criteria

The authoritative numeric settings are in
[`conf/driver_validation/acceptance.yaml`](../conf/driver_validation/acceptance.yaml).
They are benchmark-specific proposals, not universal identifiability thresholds.

| Component | Proposed requirement |
|---|---|
| Event support | At least 20 eligible origins and 5 observed events per evaluated direction |
| Semantic agreement | Alignment accuracy at least 0.80 |
| Positive-control repetition | At least 20 independent raw-process seeds, with planned results retained |
| Individual recovery | Top-k ranking assessed against intervention-positive neurons; proposed recovery target 0.75 |
| Redundant-group recovery | The pair has supported simulator effect and its predictive score exceeds the predeclared size-matched control pair |
| Prediction | Report baseline-minus-augmented held-out log loss, including negative gains; the proposed positive-control mean-gain gate is explicit in configuration |
| Availability | At least 0.90 of the planned fits available in a mature evaluation |
| Null repetition | At least 60 independent null seeds |
| False nomination | One-sided 95% exact upper bound on the per-method probability of any false nomination no greater than 0.05 |

The fixed positive predictive-score nomination cutoff is initially **0.01 nats**.
The benchmark control pair is neuron 2 plus the final neutral neuron. Its size
and identity are predeclared; it is not claimed to be activity-matched. Only
candidate blocks whose outgoing-disconnection effects are actually simulated
enter recovery scoring. The core workflow's additional activity-matched controls
remain complementary diagnostics.

It defines a benchmark nomination rule, not neuron-level causal significance.
For each null seed, any false nomination across that method's evaluated units
and directions counts as one false-nomination outcome. This avoids treating
correlated neurons or directions as independent replicates. With zero such
outcomes in 60 independent seeds, the one-sided exact upper bound is about 0.049.
This is a per-method bound; it does not establish a 0.05 bound across the four
methods or across arbitrary future selection procedures.

Directions are summarized within each simulation seed before repeated-seed
statistics. Bootstrap intervals require the complete planned eligible seed set
and at least three independent seeds; a small pilot remains exploratory. Missing
fits, weak alignment, insufficient event support, and unexecuted seeds remain
explicit in the denominators. They do not disappear from the acceptance report.

The hidden-confounder condition is reported as a diagnostic. Strong predictive
scores in the absence of outgoing-disconnection effects show the limits of
causal interpretation. It is not a required pass condition that could certify
that latent confounding has been solved.

## Execution and stopping rules

From the application root:

```bash
.venv/bin/python pipeline/14_csl_sanity.py --profile smoke
.venv/bin/python pipeline/14_csl_sanity.py --profile smoke --run
.venv/bin/python pipeline/13_driver_validation.py --profile smoke
.venv/bin/python pipeline/13_driver_validation.py --profile confirmatory
```

Without `--run`, inventory the declared cells and methods. Execute a bounded
pilot only after inspecting its scope:

```bash
.venv/bin/python pipeline/13_driver_validation.py --profile smoke --run \
  --output-dir outputs/driver-validation/csl_sanity/smoke
```

`--max-cells` records a partial run, and `--method` can declare a subset. A subset
cannot establish validation across the declared four-method panel.
`synthetic/05_driver_recovery.ipynb` now enables fitting for the requested
faster-machine rerun (16 November 2026). Its shipped profile repeats the existing
560-method-run study; disable `RUN_DRIVER_VALIDATION` explicitly for inspection.

Use the pilot to inspect dynamics, transition counts, runtime, recovery
variability, numerical failures, and measurement stresses. Before confirmation,
freeze the protocol/configuration and use seed ranges disjoint from the pilot.
The proposed confirmatory positive seeds begin at 100 and null seeds at 200;
the resolved run manifest is authoritative. Changing a gate after observing
confirmation outcomes requires a new protocol version and fresh confirmation.

A component or method receives `pass`, `fail`, or `inconclusive`. Insufficient
independent repetition or support remains inconclusive. Mature negative results
remain failures. Passing software tests or executing a complete matrix does not
override these decisions. All four methods and their disagreements remain in
the report; no successful subset silently becomes a consensus claim.

## Evidence artifacts and remaining limits

`driver-validation.json` records the protocol, cells, failures, seed-level
semantic-direction evidence, and acceptance decisions. The `analysis/` bundle
contains `acceptance.json`, `numeric-summary.md`/`.csv`, `analysis-report.md`, and
two vector figures with PNG previews:

- `figure-01-intervention-truth`: actual simulator risk reductions for candidates.
- `figure-02-predictive-omissions`: signed predictive omission scores for neurons,
  the planted pair, and controls for semantic well 0→1. Both directions remain
  in the raw evidence and numeric summaries.

Figures must show negative effects/scores and actual seed counts. Predictive
recovery and simulator intervention recovery are presented together, with
their different meanings preserved.

Current outputs live under `outputs/driver-validation/csl_sanity/<profile>/`.
The native graph sanity check has its own `outputs/csl-sanity/smoke/` artifacts;
its edge-recovery question is separate from this transition-driver question.

The proposed core confirmatory inventory is 140 raw-process cells and 560
method runs: 20 seeds for each positive control and the confounding challenge,
plus 60 null seeds. This supplements the earlier circuit/history stress matrix;
it is not a claim that either full study has run. Smoke uses seeds 0–2,
development uses 10–14, and proposed confirmation uses 100–119 and 200–259.

This study validates the declared synthetic process and nomination rule only.
The oracle uses finite paired Monte Carlo rollouts. Effects close to the 0.10
truth cutoff require a rollout-count sensitivity check before confirmation;
the pilot does not establish precise truth labels at that boundary.
Full graph-estimation uncertainty, adequate lead-time precision, broader
misspecification, realistic observation models, and independent-animal
replication remain further requirements. Biological intervention claims require
biological evidence; `biological_necessity_validated` remains false.

## Historical bounded pilot and next scientific gate

The 14 November 2026 eight-method, short-history pilot completed 120 method runs;
105 produced
usable numerical fits. Fused TVVAR was rejected in all 15 cells. No method has
an eligible semantic direction under the combined minimum event-support and
class-conditional state-alignment criteria. All eight decisions remain
**inconclusive**. This result tests and exposes a limitation; it is not recovery
validation. Larger confirmation was inventoried but not executed.
These results remain at `outputs/driver-validation/smoke/` with their original
figures and acceptance artifacts. They are not results for the revised
four-method, longer-history protocol. The larger current confirmation remains
unexecuted.

The next development question is whether the shared state representation tracks
the intended transition variable with adequate support. Global accuracy can hide
rare-region errors, and assigning a well name to a geometric bin does not make
the two objects equivalent. Examine representations and event coverage using
development simulations, retain the current pilot, then freeze a revised protocol
before fresh confirmation. More repetitions alone do not repair an unsuitable
state definition.

Historical engineering verification: 519 tests passed, one optional skip; Ruff,
mypy and compilation checks passed for that revision. Notebook 04, two vector figures/PNG previews,
numeric tables, and acceptance artifacts are available. Read
[the visual interpretation notes](../synthetic/driver_visual_notes.md) before
using the descriptive figures. Manuscript eligibility tables can be regenerated
with `python synthetic/export_driver_validation.py` from the application root.
