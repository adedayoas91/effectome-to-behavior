# Four imported CSL methods: simulation sanity-check plan

Decision: 15 November 2026. The simulation sanity check now targets conventional
c-GC, conventional c-GC*, fastC-GC and fastC-GC*. VAR estimator lanes and partial
correlation are excluded from this comparison. Existing historical results keep
their original method identities and are not relabelled.

## Executed evidence

The three-seed chain/null study completed all 24 fits. Every method recovered
exactly both planted links (0→1 and 1→2), without extra links, in all three chain
recordings. All null graphs were empty. Positive pairwise support Jaccard was 1;
empty-null Jaccard is undefined. This supports native graph-recovery sanity under
the declared direct linear Gaussian observations. Three null seeds do not
establish false-positive calibration: the per-method two-sided exact 95% interval
for any false edge is [0, .708].

Frozen evidence, two vector figures, seed tables and statistical notes are in
[`synthetic/results/four_csl_sanity_2026-11-15/`](../synthetic/results/four_csl_sanity_2026-11-15/analysis/analysis-report.md).
The full shared-manifold workflow also completed a bounded, single-seed
four-method integration with all four predictions available. Its output is
`outputs/framework-validation/csl_sanity/integration/`; it is not the full study.

Engineering verification: 541 tests passed, one optional skip; Ruff, mypy (81
source files), notebook parsing and diff checks passed. The synthetic collection
contains six output-free notebooks and 34 parseable code cells. Upstream CPU
permutation randomness is imperfectly seeded; the frozen realization is the
authoritative record of this pilot.

The current entrypoint is
[`synthetic/archive/01_four_csl_sanity.ipynb`](../synthetic/archive/01_four_csl_sanity.ipynb),
after notebook 00 and before notebooks 02–06. The numeric prefixes now match
the recommended execution order; historical evidence archives retain old paths.
It compares the four native implementations on identical planted-chain and
independent-null recordings. CLI execution is
`python pipeline/14_csl_sanity.py --run`, with artifacts under
`outputs/csl-sanity/smoke/`. The online packages supply the estimators; project
adapters transport their outputs and provenance.

## Implementation and cleanup plan

1. Verify authoritative online package revisions, native input/output conventions,
   supported Python runtimes and dependence-test thresholds. Install official
   code in a pinned environment. Do not copy or rewrite its discovery algorithms.
2. Protect existing forecast behavior with its regression tests. Rename the old
   connectivity adapter to identify its optional predictive readout role; remove
   the misleading `connectivity/causalised_gc.py` path and update imports. The
   historical readout is outside the new native-graph comparison.
3. Add four thin, explicitly named registry adapters using upstream graph outputs.
   Do not reinterpret native unsigned support or significance scores as signed
   dynamical coefficients. Unsupported forecasts stay unsupported.
4. Add a native graph-recovery sanity benchmark on identical planted recordings
   and independent seeds. Report graph truth, sensitivity/precision, false links,
   pairwise agreement and support counts; empty-graph agreement is uninformative.
5. Switch the current synthetic comparison profiles/notebooks to the four-method
   panel, preserve old outputs and record upstream versions. Keep their common
   future-event ownership and state-alignment requirements.
6. Run adapter and forecast regressions, sanity fits, notebook checks, full tests,
   lint and types; update execution order and project notes with actual results.

## Interpretation requirements

The four variants share a discovery family. Their agreement is consistency
evidence, not four independent causal confirmations. A successful graph-recovery
sanity check does not establish transition-driver recovery. Replacing estimators
does not resolve insufficient transition counts or poor manifold/state alignment.
The purpose is to learn which comparisons support a conclusion, retaining
negative, unavailable and degenerate outputs without weakening acceptance gates.

## Current protocol and historical evidence

The four registry names are `conventional_cgc`, `conventional_cgc_star`,
`fast_cgc` and `fast_cgc_star`, selected by `estimator_profile: csl_sanity`.
Fast estimators import `fastCGC.FastGcStar` from the pinned project environment.
Conventional estimators import `causalized_gc.GcStar` from the separately locked
Python 3.13 CPU runtime; see [its installation guide](../tools/conventional-cgc/README.md).
The conventional upstream CPU permutation RNG is not controlled by its public
`random_state` argument. Data seeds are repeatable, but exact conventional
permutation p-values are not guaranteed repeatable; retain the evidence artifacts.

The native graphs are unsigned positive-lag support with no neural forecast or
coefficient-drive interpretation. All four use one lag, alpha=0.01 and
beta=0.005. Current framework smoke uses history 256, dwell 512 and 4,096 samples;
development uses histories 256/512, dwell 512 and 8,192 samples. The manuscript
geometry is unchanged. Current driver profiles use history 256 and trial length
512, with cue samples 288–480, target length 8 and stride 16. Histories therefore
support the fast complete-shift threshold while retaining pre-cue origins.

Current output roots include `csl_sanity` before the profile name:
`outputs/framework-validation/csl_sanity/<profile>/` and
`outputs/driver-validation/csl_sanity/<profile>/`. Notebook 01 uses
`outputs/synthetic-transition-examples/<scenario>/seed-<seed>/four-csl-sanity/`.
Historical eight-method outputs, figures and manuscript artifacts retain their
original identities. The historical driver pilot completed 120 fits, 105
available, with all eight methods inconclusive; it is not a result for this revision.

The current proposed framework manuscript matrix has 7,680 cells and 30,720
method fits; driver confirmation has 140 cells and 560 fits. These larger
studies remain unexecuted. A small graph sanity run must not be reported as
completion of either study.

The opposite rotations in `matched_marginals` have different signs but identical
support. Native binary graphs cannot distinguish them. A support-changing
matched-marginal control is required before using this panel to claim discovery
of circuit/manifold degeneracy. The existing covariance identity remains a
valid mathematical illustration, and the planted-chain sanity check tests a
separate, recoverable support target.
