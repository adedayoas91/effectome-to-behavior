# Full behavior-supervised experiment: implementation decision record

18 November 2026. The completed PCA/score investigation is preserved in commit
`98f2055`, including its manuscript source/PDF snapshot. This experiment is a new
namespace; it does not revise notebook 08, its inspected seeds, or earlier gates.

## Question and reason for the change

Does an encoder trained to preserve a separately defined, continuous, distributed
behavior retain behavioral intervention effects after compression? Notebook 08
found low within-chart error but large PCA semantic error. The scalar-score
replay preserved an existing binary classifier; it did not establish nonlinear
geometry or a new learning algorithm. The new benchmark therefore uses two
continuous behavioral outputs computed nonlinearly from four observed neural
variables, with a separate responsive nuisance block.

The neural structural model remains fully observed and additive-tanh, matching
the existing mechanism estimator. This isolates the behavioral measurement from
additional hidden-state or neural function-class changes. Outgoing disconnection
retains source evolution and self-history. Behavioral effect vectors, in the
declared behavioral units, replace binary probability-effect scores.

## Prespecified generator and controls

Ten neural variables are observed. Nodes 0/1 are independently initialized
drivers; 2--5 form the behavioral readout block; 6--9 form a responsive nuisance
block. Both drivers have signed outgoing influence on the readout and nuisance
blocks. Nuisance variables have no causal path into the readout block. The fixed
two-row readout mixes all four readout coordinates and applies tanh componentwise.
Its definition is fixed before any encoder fitting.

The behavioral null removes every incoming cross term into nodes 2--5 while
retaining driver/nuisance dependence. This must yield exactly zero oracle
behavioral effects under paired innovations for every declared intervention.
The nuisance representation may still respond. We test all ten individual
outgoing disconnections and the predeclared driver/nuisance groups; these are
design controls, not discovered groups.

## Encoders and baselines

- True behavior reference, used only as the outcome definition/oracle diagnostic.
- PCA with two/four dimensions and training-owned Ridge readouts.
- Two-dimensional partial least squares with a training-owned Ridge readout.
  The [scikit-learn cross-decomposition documentation](https://scikit-learn.org/stable/modules/cross_decomposition.html)
  describes the covariance-based projection used by this baseline.
- Full-neural linear and nonlinear behavioral readouts.
- Two-dimensional nonlinear behavior-only bottleneck, with the same neural
  architecture but zero latent-dynamics loss.
- Two/four-dimensional joint behavior/dynamics encoders, registered separately
  as `bunddle_behavior`.
- Constant, driver-proxy and responsive-nuisance readouts.

The existing `bunddle` component is a categorical temporal-window port. It remains
unchanged. The new component is explicitly a **BunDLe-inspired pointwise,
continuous-output adaptation**, reusing the documented encoder widths and
residual latent transition idea. It is not labeled a faithful reproduction of the
published categorical/window method. A linear behavioral head and a latent
transition auxiliary loss are jointly trained. Latent dynamics are an auxiliary
training objective, not a causal state used for intervention simulation.

Pointwise inference resolves the current source-at-trial-start support contract:
no future frames, history padding or retrospective CEBRA offsets are required.
Predicted behavior is decoded at each future sample **before** averaging over the
last four samples of the eight-step rollout. A nonlinear decoder of an averaged
code would define a different outcome and is not substituted.

The scaler and candidate models fit inner whole training trials, selection uses
separate whole validation trials, and all selected parameters are refit on the
outer training set for the selected fixed hyperparameters/epoch count. No outer
evaluation labels or interventional outcomes enter fitting or selection.
Training pairs stay inside trials. A prospectively balanced sample of early
transients and later samples avoids letting thousands of near-stationary frames
dominate the behavioral objective; the same declared sampling applies to all
trainable arms. Repeated within-trial rows never increase independent sample size.

## Causal evaluation and error decomposition

Cross each frozen readout with true neural dynamics, fitted known support,
fitted empty support and all four imported native support estimates. Fit each
neural model once and reuse its futures for every readout. Initial states,
origin identities and innovations are identical across arms. No neural/behavioral
map is refit, aligned or selected using evaluation rollouts.

For each origin and unit compute true behavioral effect `Delta_B`, the same true
futures through the learned map `Delta_M`, and fitted futures through that map
`Delta_hat_M`. Report per-origin vector errors before averaging:

`behavioral MAE = ||Delta_hat_M - Delta_B||_1 / 2`,
`within-chart MAE = ||Delta_hat_M - Delta_M||_1 / 2`, and
`semantic MAE = ||Delta_M - Delta_B||_1 / 2`.

The empirical bound adds within-chart error and factual/intervened pathwise
behavioral discrepancies, measured on the same oracle futures. Record all
support excursions and prediction collapse; do not remove hard interventions.
Also report natural versus transient decoding, per-unit effect vectors,
singleton/group recovery separately, positive-unit errors and null nominations.

## Profiles and prospective confirmation

Smoke and development use the previously reserved **40005--40009** development
range. They are implementation/development evidence, not confirmation. A full
confirmation profile implements separate calibration-null **50000--50018**,
positive-test **60000--60019**, and null-test **70000--70059** pools. Its source,
runtime, encoder procedure, primary endpoint and decision budgets are frozen
before calibration; calibration thresholds are frozen before test scheduling.
The confirmation runner requires a completed same-source development record.

Null calibration uses a whole-seed maximum over the declared unit magnitudes,
separately for each dynamics/readout combination. It cannot compensate for an
endpoint's semantic failure: behavioral fidelity, effect accuracy, positive
power and independent null specificity are separate gates. Exact binomial null
limits are per combination, not simultaneous guarantees over all combinations.
Prospective gates and their units are recorded in configuration; failed gates
remain failures. Biological validity and hidden-state robustness remain outside
the claim even if these bounded controls pass.

## Runtime and provenance

The main runtime remains Python 3.14. A separate Python 3.12 encoder runtime
exports frozen numerical weights, allowing NumPy inference and shared neural
rollouts in the main process. Intel macOS uses the last officially available
PyTorch 2.2.2 wheel; supported Linux/Apple Silicon platforms use a separately
pinned modern wheel. This is a platform/runtime adaptation, not a PCA fallback.
Package versions, worker source, lock hashes, training curves, exported-weight
parity and native evidence are retained. Missing runtimes fail preflight.

The [official PyTorch deprecation announcement](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690)
documents the Intel restriction. The
[verified representation review](behavior_prioritized_alternatives.md) and
[citation audit](behavior_prioritized_alternatives_citation_audit.md) provide the
PSID, CEBRA, BunDLe and supervised-autoencoder motivations and their claim limits.
BunDLe's conditional state-reset theory is not treated as a guarantee for our
finite-sample outgoing-mechanism deletions.

## Completion evidence

Require generator/null invariance tests; strict whole-trial sample ownership;
artifact roundtrips and frozen inference; nonlinear worker/NumPy parity;
continuous error-bound checks; unchanged origins/noise; changed-input and
source/runtime resume rejection; first-completed native evidence retention;
ready-to-run numbered notebooks; and an actual nonlinear all-native smoke and
development execution. Confirmation is implemented for subsequent independent
execution; development results must not be presented as that execution.

The [development findings](behavior_supervised_development_findings.md) record
the completed all-native panel, the behavior-only ablation and the operational
evidence in notebook 09's output directory.
