# Framework validation and notebook migration audit

Decision: 14 November 2026. This document audits software delivery, not completion
of the scientific validation study.

| Adopted action | Prior delivery | Work required in this revision |
|---|---|---|
| One shared training-owned manifold | Implemented in a separate Python workflow | Canonical recording notebooks using raw checkpoints |
| Comparable observed, predicted and native transition matrices | Implemented for one horizon | Multi-horizon validation and frozen Markov adequacy diagnostics |
| Conditional transition prediction | Paired M0/M1 implementation | Repeated independent seeds, fixed experimental matrix and negative controls |
| Transition-specific neuron/group contributions | Signed features, projection and omission scores | Matched pre-transition contrasts and size/activity-matched group controls |
| Cross-method evidence synthesis | Family ranks and leave-family-out ranks | Seed-level uncertainty, corrected comparisons, retained failure denominators and manuscript exports |

## Execution plan

1. Preserve existing notebook code/results and local metadata. Add canonical
   shared-transition entrypoints per recording and scope notices in all old
   stage notebooks; add a resumable raw-recording helper.
2. Add a reproducible validation runner with explicit smoke, development and
   manuscript matrices, immutable settings, independent seeds, raw-process
   controls, observation stresses and history/dwell/horizon variation.
3. Add seed-level descriptive statistics, bootstrap intervals, exact sign tests,
   full-family Holm correction, failure reporting, real vector figures and
   numeric/LaTeX exports. A completed run is not automatically a passed study.
4. Add the remaining comparison diagnostics without replacing the established
   estimator implementations or inventing intervention claims.
5. Update synthetic notebooks, both execution-order guides, project notes and
   manuscript protocol. Run bounded repeated simulations and record their
   actual artifact lineage; distinguish those results from the larger study.
6. Verify notebook preservation, all new contracts, executable examples,
   complete regression tests, lint, types and manuscript compilation.

## Scientific gates that software cannot declare completed

Publication-scale calibration, independent-animal replication, graph-estimation
uncertainty propagation and biological intervention validation require actual
experiments. Geometric conditional diversity is not a test of biological circuit
equivalence. Existing historical notebooks stay explicitly scoped; they are not
silently relabelled as the new shared analysis.

## Current implementation

The canonical workflow now covers each of the five software areas above:
shared references; comparable transition objects and multi-horizon diagnostics;
conditional prediction; matched neuron/group contrasts and controls; and
family-aware synthesis with repeated-seed analysis. Three canonical recording
notebooks and a repeated-validation notebook were added. Scope notices in all
171 old recording notebooks preserve code, results and local metadata.

This does not mean every scientific requirement has been demonstrated. The
manuscript matrix is a proposed finite study, not an executed or preregistered
publication result. Graph-estimation uncertainty propagation, latent-confounding
calibration, independent-animal generalization, and intervention validation
remain outstanding. Optional learned group definitions and prediction stacking
are not required for the first frozen-group comparison and are not claimed as
implemented. The default comparison is a chronological development holdout;
arbitrary nonlinear manifold equivalence is not identified.

Seed-level intervals currently describe the primary method-level predictive
gain. The exported transition-specific omission scores retain local state codes
and seed IDs; they are not automatically pooled into neuron-level confidence
intervals. Candidate lead-time uncertainty and cross-animal rank stability still
need dedicated analysis with supported event alignment and independent recordings.

## Final verification — 14 November 2026

- Complete regression suite: **477 passed, one optional skip**. All four synthetic
  notebooks execute their default code cells; the repeated-study default is
  inventory/inspection, not automatic execution of the full matrix.
- Ruff over the repository, mypy over 73 source files, Python compilation and
  diff checks pass. The suite retains two nonfatal backend/environment warnings.
- All 176 recording/diagnostic notebooks and 555 code cells parse. Bootstrap and
  configuration cells for all three new canonical recording notebooks execute
  from their own directories with temporary artifact roots.
- Repeated smoke: four scenarios × three independent seeds × eight methods =
  **96 attempted method runs; 95 available**. Fused TVVAR on `fixed_circuit`, seed
  2, was rejected at anchor 43. It remains in the failure denominator; the
  incomplete contrast receives no confidence interval or inferential p-value.
- The full planned family contains 32 contrasts. No Holm-adjusted result is
  significant in this three-seed engineering run. This is not a powered study.
- Simulation, statistical and manuscript provenance fingerprints agree with the
  current implementation. Resolved comparison, method and generator settings
  are retained in `outputs/framework-validation/smoke/validation.json`.
- Generated vector figures, numeric tables and bounded-run text are exported to
  the active manuscript's `validation_artifacts/`. `tectonic --keep-logs main.tex`
  compiles successfully, with nonfatal style/underfull-box warnings.
- Manuscript preflight declares **7,680 cells / 61,440 method runs**, with 20
  independent seeds. That full matrix and biological fits were **not executed**.

The execution path is simplified to raw preprocessing followed by one shared
comparison per recording; separate method manifolds are optional analyses.
No dependencies were added. Existing notebook code/results and user artifacts
were preserved.

## Visual delivery verification

The manuscript now includes five vector figures and two captioned tables:
repeated predictive gain, availability, generating controls, shared transition
matrices and transition-specific omission maps; tables cover control design and
mean/SD gains with available/planned seed counts. The renderer and panel helpers
live under `synthetic/`; presentation provenance is separate from the frozen
simulation manifest. Notebook 03 can regenerate and display the visual bundle.

The fixed first matched-marginal specification at seed 0 reproduces all saved
method availability and gain values before its matrices are plotted. The raw
simulation and statistical artifacts retain their original hashes. Figures keep
negative scores and single-origin risk sets visible; the latter do not support
robust neuron claims. Full regression verification after this addition:
**480 passed, one optional skip**; Ruff, mypy (75 source files), compilation and
diff checks pass. The manuscript compiles, and rendered figure/table pages were
visually inspected. No dependencies were added.

## Visual evidence added to the synthetic workflow

The manuscript visual bundle extends the predictive-gain and availability
plots with designed-circuit/activity controls, a worked shared-manifold
transition comparison, and transition-specific neuron/group omission heatmaps.
The bundle contains five figures and two tables: compact four-panel
predictive-gain and availability figures, the three worked/control figures,
and numeric and control-design tables. The original analysis figures remain
available beside these manuscript versions. The worked
comparison is the first declared matched-marginal specification at seed 0;
numeric state labels are not pooled across seeds, and no best seed is selected.

The reproducible entrypoint is `synthetic/validation_figures.py`. Notebook 03
can render the bundle with `RENDER_VISUALS=True`, or display existing figures
and tables with its default settings. Presentation artifacts have a separate
`visual-provenance.json` linked to the frozen validation report; plotting does
not silently replace completed experimental results. Captions and the figure
catalog distinguish raw activity projections from the shared manifold and
predictive omission scores from biological intervention effects.

## Intervention-defined driver benchmark

The driver-ground-truth extension now supplies single actuators, redundant
groups, inert correlated proxies, no-driver nulls, and a hidden-confounder
challenge. Actual same-origin/horizon simulator interventions are compared with
predictive omission scores. The separate proposed acceptance protocol requires
independent seeds, event support, and direction-specific semantic alignment;
geometric bins are not automatically the simulator's wells.

The three-seed pilot completed 120 method runs, with 105 numerical fits and
15 fused-TVVAR rejections. No method has a direction eligible under all support
and alignment gates. All recovery decisions remain inconclusive. Confirmation
was inventoried (140 cells / 1,120 method runs), not executed. This result identifies
state correspondence and event coverage as development questions before scaling.
See [the driver protocol](driver_ground_truth_protocol.md) and notebook 04.
Full suite: 519 passed, one optional skip; lint/types/compilation checks pass.
