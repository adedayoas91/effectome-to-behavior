# Predicting behavioral transitions from dynamic effectomes

Notebook 09 showed that a nonlinear behavior-supervised readout can preserve
intervention effects in a fully observed circuit with fixed couplings. With
known support, the two-dimensional joint and behavior-only readouts achieved
behavioral-effect MAEs of 0.01023 and 0.00950. Two-dimensional PCA and PLS
achieved 0.04790 and 0.04844. The joint readout passed the frozen confirmation.
The behavior-only ablation had lower overall error under known support and all
four estimated supports. This motivates using it as the shared readout in the
next experiment.
[Notebook 09 confirmation](../outputs/behavior-supervised/confirmation/distributed-behavior-01/review/analysis-report.md)

The next question is whether a change in the effectome improves prediction of a
later behavioral change beyond the information a strong predictor can use from
recent activity, latent trajectory, and behavior. This connects the validated
readout to the project's transition-specific driver objective. Conditional
switching models provide a precedent for letting current observations or
continuous state inform transition predictions.
[Project goal](../../goal.md) · [Linderman et al., 2017](https://proceedings.mlr.press/v54/linderman17a.html)

## First experiment

1. Extend the Notebook 09 generator with two recurrent coupling regimes. Change
   which driver pathways affect the readout block and keep the behavioral
   measurement fixed. Vary switch timing across independent trials. Include
   constant-coupling and nuisance-only reorganization controls. Place enough
   observations after a coupling change and before the forecast origin to test
   whether the effectome can resolve the established regime. Record the elapsed
   time from regime start and the usable origins at each history length.
2. Fit one two-dimensional behavior-only encoder per outer training split. Use
   its recent trajectory as part of the baseline and score predictions against
   the simulator's true future behavioral change at a fixed positive horizon.
   Compare a common set of forecast models with and without trailing graph
   features on identical held-out origins. Include a static-graph model
   and an oracle dynamic-graph model. Match or stratify held-out origins on
   current behavior and recent trajectory as a diagnostic for differences in
   future outcome among similar starting states.
3. Require a useful held-out error reduction over the activity/history and
   static-graph baselines on independent seeds. Set the practical gain and
   uncertainty criteria before confirmation. Check that nuisance-only graph
   changes offer no reproducible gain. If oracle graphs help and estimated
   graphs do poorly, examine graph timing and estimation. If oracle graphs also
   add little, check whether recent histories already predict the endpoint
   nearly as well as the oracle does.
4. Evaluate which neurons or groups account for any gain. Use feature
   attribution on reserved nomination trials to freeze the candidates.
   Then test their modeled effects on the independent evaluation trials using
   paired outgoing disconnections in each pre-transition context. Include
   matched controls and compare the predicted changes in behavioral direction
   and magnitude with simulator intervention truth.

The first analysis can use continuous signed directed graph features. If they
carry held-out predictive value, compare a graph-similarity representation or
discrete states on the same events to see how much of that value they retain.

## Implementation and decision rules

[Notebook 10](../synthetic/10_behavior_transition_experiment.ipynb) runs the
development experiment independently. Each trial visits three episodes in
alternating order with randomized starting regime and timing. The driver-switch
scenario swaps which identically parameterized driver controls the behavioral
block. The nuisance switch changes coupling only within the nuisance block and
preserves the constant control's full behavior trace under paired innovations.
The forecast comes from established regimes with complete trailing histories.
The design measures prediction before future behavior changes, with switch
detection latency left to a later benchmark with origins nearer the switch.

The input histories contain 256 or 1,024 samples at 10 Hz. For each origin the
outcome is the mean of behavior at offsets five through eight minus the
observed origin behavior. Whole trials are assigned to four disjoint roles
for encoder fitting, forecast fitting, nomination and final scoring. The
history model sees recent raw activity, measured behavior, latent states and
long-history summaries including unselected lagged covariances. The graph
models augment these features with signed graph action at the origin and use
the same whole-trial ridge and RBF search. Native c-GC supplies unsigned
support and the additive mechanism estimates signed coefficients. Fast c-GC
is the primary estimator. The three other native lanes are available through
the same runner for sensitivity analyses. The static model pools forecast-fit
histories and retains edges selected in at least a quarter of those windows.
This lets pathways present in roughly half the windows enter the static fit
even if discovery misses some of them.

The primary comparison uses the same forecast family to measure the value of
the graph features. A secondary comparison forecasts behavior directly from
paired mechanism rollouts using the frozen behavioral map. The oracle rollout
shows how well the known mechanisms can predict the same held-out outcome
without a downstream regression. The rollouts also measure the direction and
magnitude of outgoing-disconnection effects. Candidate selection uses forecast
feature omission on the reserved nomination trials, within predicted increase
and decrease of the first behavior coordinate. Controls match group size and
use origin-state magnitude and estimated outgoing degree for nearest matching.
Per-origin effect vectors and the true effects by generating regime are saved.

Development uses five independent seeds and confirmation reserves twenty new
seeds. The confirmatory primary comparison is fast c-GC with 1,024 samples.
At that history, the declared margin is 0.005 behavioral MAE units. The test
requires gains above that margin against both baselines in the driver scenario
and gains below it against both baselines in both controls. Seed-level exact
one-sided sign tests assess whether more than half the gains lie on the
specified side of the margin. All six tests must pass at 0.05, making the
decision an intersection of the predeclared criteria. Perturbation errors and
nominations are descriptive in this profile.

## Development results

All 15 cases and 2,880 fast c-GC window fits completed. At 1,024 samples,
the graph-augmented regression had MAE 0.24204 in the driver-switch scenario,
compared with 0.24405 for history alone and 0.24254 for the static graph. Both
gains were below the declared 0.005 margin, and the static comparison showed
larger gains in both controls. The short-history graph regression showed a
larger average gain in the driver scenario while native support Jaccard was
only 0.185. The short-history mechanism rollout showed no advantage over a
static rollout in that scenario. These diagnostics do not support using the
direct regression gain as evidence of reliable mechanism recovery.

The long-history dynamic mechanism rollout reduced driver-switch forecast
MAE from 0.20395 for the static rollout to 0.17862 across all five seeds. The
oracle rollout reached 0.17838. The corresponding average gains in the
constant and nuisance-switch controls were about 0.002. On oracle-active
interventions, effect MAE in the driver-switch scenario fell from 0.08941 to
0.01671. Long-history native support Jaccard was 0.954 in this scenario.
Mechanism propagation therefore provides a concrete candidate for a new
fresh-seed confirmation with its own primary comparison. These results
concern a model-matched simulation with established regimes and a fully
observed circuit.
[Development results and audit](../outputs/behavior-transition/development/recurrent-coupling-01/analysis-report.md)

## Empirical pilot

The local V2a dataset has four independent ten-minute zebrafish recordings
with calcium traces and finite tail-angle samples at 300 Hz. The accompanying
data description states that tail video was acquired synchronously. Verify the
alignment and recording boundaries, define one prespecified continuous outcome
such as tail-motion intensity over a wholly future interval, and count usable
origins after accounting for masks and temporal overlap. The earlier state
preflight left too few eligible histories in the two evaluated recordings, so
feasibility needs to be checked for this continuous endpoint. Four fish support
a pilot with per-fish estimates. Generalization across fish needs more animals
and a comparison that handles different recorded neuron panels.
[Local data description](../data/v2a-RSNs/README.md) · [V2a recording study](https://doi.org/10.1038/s41593-023-01418-0) · [Manuscript preflight](../../manuscript-260926/sections/validation.tex)

Use training-owned readouts and chronological splits that purge shared raw
dependencies. Hold out a complete fish only where the neuron panel and fitted
transforms can be applied prospectively. Estimate the effectome's incremental
predictive value for measured tail motion. Any empirical driver nomination
would describe predictive evidence and a model-based perturbation hypothesis.

The [tail-motion preflight](../synthetic/preflight_tail_transition.py) inventories
2-second future windows using RMS angular speed over raw tail samples and
chronological splits purged for both calcium and tail dependencies. With
60-second calcium histories, all four fish have a nonempty purged outer split,
and three have usable nested validation. Test segments contain only one to three
origins with mutually disjoint complete raw support per fish. The candidate
120-second history leaves no usable nested split for the strict native estimator.
These counts assume nominal alignment from the reported frame rates. Confirm the
shared trigger and relative offset before fitting the empirical endpoint.
[Preflight output](../outputs/behavior-transition/empirical-preflight/tail-transition-preflight.json)
