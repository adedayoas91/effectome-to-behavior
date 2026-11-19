# Preserve behavior during compression: alternatives after the PCA bridge

The [continuous-behavior study](behavior_supervised_experiment_plan.md) now
implements the proposed comparison with a distributed target, a redesigned
null, and a pointwise BunDLe-inspired encoder. This review records the reasoning
that led to that design.

Research decision record, 18 November 2026. This review addresses the completed
notebook-08 development result; it does not alter that experiment or its gates.
The original plan and completed replication were checkpointed as `fac7ea4`
before the learned-endpoint implementation. Graph comparison remains separate.

## What the failure does and does not imply

Five development seeds gave driver behavioral-effect MAE **0.131144** for PCA
regions and **0.138818** for PCA followed by a behavioral decoder. The full-neural
behavioral decoder gave **0.001969**. Within-chart errors were about 0.002, and all
40 native consensus supports matched known support. The true simulator itself
had large PCA semantic error. These are the measured findings in
[the notebook-08 review](../outputs/learned-manifold-analysis/endpoint-bridge-03/review-2026-11-18/analysis/analysis-report.md).

This localizes the present problem to the measurement/representation, rather
than demonstrating that low dimensionality itself is impossible. A PCA projection
is chosen without the behavioral target. Adding a decoder afterward cannot
change the retained subspace. Official [PLS documentation](https://scikit-learn.org/stable/modules/cross_decomposition.html)
explicitly distinguishes this from target-informed compression. The broader
statistical objective is to retain the relationship between neural observations
and behavior, as in [sufficient dimension reduction](https://www.jmlr.org/papers/v5/fukumizu04a.html).
Those observational objectives alone do not guarantee intervention validity.

## First test: factor the successful decoder into one behavioral coordinate

The current full-neural endpoint has an affine neural transform
`Phi_full(x) = D x + d` and a binary linear readout
`h(Phi_full(x)) = 1[w^T Phi_full(x) + b >= 0]`. Define

\[
z(x)=w^\top\Phi_{\rm full}(x)+b
     =(w^\top D)x+(w^\top d+b).
\]

Then source labels are `1[z(x0) >= 0]`, and the existing future-window endpoint
is `1[mean_t z(xt) >= 0]`. Because the readout is affine,
`mean_t z(xt) = w^T mean_t Phi_full(xt) + b`. Therefore the scalar code preserves
the fitted full-neural classifier and its declared window event algebraically,
apart from floating-point boundary effects that the replay must check.

This is a project-specific identity, not a theorem attributed to the cited
representation papers. Its capacity diagnostic used already fitted endpoint
parameters and cached neural mechanisms, with **zero new native or mechanism
fits**. The scalar factorization gave zero disagreement across **276,480 natural
observations** and **1,935,360 factual/intervened rollout events**, spanning ten
cases and 70 cached neural models. Its driver behavioral MAE was **0.001969** and
null MAE **0.000073**, identical to the parent full-neural endpoint as expected
from the factorization. These sample counts validate replay; they do not increase
the independent sample size beyond the same five inspected development seeds.
The [probe evidence](../synthetic/results/behavior_prioritized_probe_2026-11-18/README.md)
records the numerical diagnostic separately from notebook 08.

A separate supervised-decoder/PCA dimension diagnostic gave driver MAE 0.146908,
0.138818, 0.138875, 0.138818 and 0.018994 for one through five retained PCs.
Four PCs retained about **90.45%** of standardized neural variance while retaining
the large behavioral error. Increasing dimension can help, but preserving a
variance fraction is not the behavioral preservation criterion. This sweep is
exploratory and does not select a confirmation setting.

This score is **a supervised one-dimensional behavioral readout**. Its successful
replay shows that compression can preserve this fitted binary task. It does not
discover a nonlinear manifold, prove the true behavior distribution sufficient,
retain arbitrary behavior trajectories, or establish closed latent dynamics.
The present behavior is a predeclared scalar observed coordinate, making this an
especially favorable control. Do not present a classifier factorization as a
newly identified biological state. Any binary classifier can be represented by
a scalar decision score; this factorization is not evidence that behavior's
intrinsic dimension is one. Computing the score still consumes the full neural
observation, and the structural simulator still evolves the full neural state.
It does not establish reduced sensing or a one-dimensional dynamical manifold.

Preserve the existing mean-logit/mean-code contract. Averaging
`sigmoid(z(xt))` and then thresholding generally defines a different event from
thresholding `mean_t z(xt)`.

## Candidate methods and their fit to the problem

| Candidate | What it changes | Why consider it here | Cost and limitation | Recommended role |
|---|---|---|---|---|
| Supervised scalar score; linear discriminant or reduced-rank behavioral readout | Learn the behavioral direction before discarding neural dimensions | Tests whether a small code can preserve the fitted binary task | Current NumPy/scikit-learn runtime; a binary score does not preserve general geometry or autonomous dynamics | Completed scalar replay diagnostic and mandatory simple baseline |
| PLS / supervised linear compression | Choose latent directions using training neural/behavior covariance | A compact, target-informed alternative to PCA-first regression | Available in scikit-learn; covariance and linear projection may miss nonlinear behavior; component count must use validation | Cheap comparator, particularly for continuous or multivariate behavior |
| [PSID](https://www.nature.com/articles/s41593-020-00733-0) | Prioritize behaviorally relevant neural dynamics during subspace identification | Targets the behavioral/dynamical objective rather than neural variance alone | Linear state-space method; requires a declared causal-history/filter contract; its model assumptions do not match every nonlinear neural mechanism | Temporal linear baseline once origins have real prehistory |
| [BunDLe-Net v5](https://www.biorxiv.org/content/10.1101/2023.08.08.551978v5) | Joint behavior/dynamics representation learning | Closest existing project direction for a behavior-focused neural representation | Existing wrapper needs compatible Torch runtime and aligned window/label support; theory's complete state resets do not certify outgoing-edge deletion | Main nonlinear candidate after generator and history contracts are fixed |
| [CEBRA-Behavior / Hybrid](https://www.nature.com/articles/s41586-023-06031-6) | Behavior-informed contrastive embedding; optional temporal component | Tests whether training labels guide a useful geometry rather than merely naming PCA regions | Optional runtime absent in the audited environment; temporal receptive fields must be explicit; natural decoding is not an intervention theorem | Independently labeled nonlinear comparator |
| [Supervised autoencoder](https://proceedings.neurips.cc/paper/2018/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html) | Optimize behavior prediction and reconstruction jointly | Places behavioral loss at the bottleneck during learning | Reconstruction can preserve nuisance variation if it dominates; paper's linear stability result does not establish nonlinear intervention generalization | Alternative joint-loss architecture with a tuned nuisance/behavior tradeoff |
| [Kernel sufficient dimension reduction](https://www.jmlr.org/papers/v5/fukumizu04a.html) | Estimate a low-dimensional subspace preserving the neural/behavior statistical relationship using kernel conditional-independence criteria | A principled objective beyond variance explained | Numerical kernel optimization and scaling need implementation work; observational sufficiency is not intervention invariance | Research comparator if simpler supervised methods fail |

The PSID motivation is behaviorally prioritized dynamics, not a general theorem
for nonlinear interventions. CEBRA supplies supervised/time/hybrid objectives.
The supervised-autoencoder reference jointly predicts inputs and targets; its
theoretical result concerns linear models and specified assumptions. These
papers inform the alternatives, rather than supplying evidence that one has
already solved this benchmark. Bibliographic records and exact claim limits are
in the [reference audit](behavior_prioritized_alternatives_citation_audit.md).

## A defensible next experiment

1. **Retain the completed capacity check.** The scalar score replay preserves the
   fitted full-neural endpoint; the PCA dimension sweep diagnoses retained
   information. Neither is an independent method win or confirmation. Results on
   seeds 40000–40004 remain development evidence. Carry these simple baselines
   forward when testing a genuinely learned bottleneck.
2. **Define a less trivial behavior before fitting.** Specify a distributed or
   multivariate behavioral readout independently of the learned chart, retaining
   every causally active neural state as observed and keeping the neural update
   family compatible with the mechanism estimator. A linear score on the current
   final-neuron target is a baseline, not the main nonlinear result.
3. **Construct valid new controls.** If the behavioral readout now includes a
   bystander/nuisance coordinate, the old endpoint-null generator may have real
   behavioral effects. Recompute oracle truth and remove every causal path into
   the newly specified behavioral readout for a genuine behavioral null. Do not
   copy the old null label or calibration rule unchanged.
4. **Train a behavior-prioritized encoder.** Start with supervised linear/PLS and
   the full-neural comparator; select one nonlinear joint behavior/dynamics
   candidate on development/validation only. A candidate objective is
   `L_behavior + lambda_dyn L_future_behavior + lambda_rec L_reconstruction`.
   This is a proposed project objective, not the exact published loss of every
   method above. Use continuous/vector behavior when relevant and preserve the
   declared target-window aggregation. Label the supervision explicitly.
5. **Resolve temporal support prospectively.** The present origin is trial sample
   zero. Temporal encoders need actual observed prehistory and a shifted origin,
   or an explicitly pointwise architecture. The current BunDLe default window
   length 15 exceeds horizon 8/target 4; CEBRA's `offset10-model` wrapper does not
   itself state a causal receptive-field contract. Future context cannot be
   borrowed to manufacture source labels. Training future labels may be
   prediction targets; they may not become inference inputs.
6. **Freeze and evaluate semantics as well as fit.** Keep causal simulation and
   disconnection in full observed neural space. Freeze encoder/readout, and use
   the same origins and paired futures in every arm. Report behavioral MAE,
   within-chart MAE, oracle semantic discrepancy, source/target agreement,
   occupancy, group effects and support excursions. Keep collapse, proxy and
   nuisance controls visible, without per-intervention re-alignment or filtering.
7. **Then reserve independent inference.** Use fresh development recordings for
   objective selection, followed by the separately reserved calibration/positive/
   null pools after freezing the procedure. Refit the complete training-owned
   chart procedure within each recording as declared. Do not treat an encoder
   selected on inspected seeds, or repeated execution of them, as confirmation.

No dependency was installed for this review. Existing locked NumPy/scikit-learn
components suffice for the first diagnostics. Selecting a nonlinear backend does
not authorize silently replacing its unavailable runtime with PCA. No graph
branch merge is needed to test the representation hypothesis.

The methodological decision is to **learn what must be preserved before
compressing it**. The remaining scientific test is whether that learned
measurement still matches independently defined behavior after the actual neural
intervention. Behavioral sufficiency, interpretable geometry and closed latent
dynamics remain separate claims.
