# Appendix C implementation review and repair handoff

Review date: 2026-10-01.
Reviewed repository: the nested `effectome-to-behavior/` package.
Branch: `astra-revise`.
Reviewed HEAD: `f1cacace164967aa84a3f4e6a5a3c01c38c3e0e4`.
Specification: parent `../report.md`, especially Appendix C, WP0-WP11.

> Remediation note (2026-10-02): this document preserves the pre-fix audit evidence that drove
> the implementation pass. The mandatory findings below were subsequently addressed across the
> amended WP commits on `astra-revise`. Optional scientific extensions remain explicitly deferred,
> and the biological notebooks plus publication-scale calibration grids were not executed.
Baseline: `d58be179ee3b1499532ec4b2da3fc2f8a3957d3b`.

## Verdict

**The implementation does not yet fulfill Appendix C.** There are useful repairs and tested numerical components, but the revised end-to-end analysis is not implemented. The current notebooks still run the older analysis workflow, and the newly generated sparse-VAR/TVVAR notebooks fail their default chunking contract on recording-sized inputs. Passing the existing tests is not evidence that the complete methodology or its caveats are fulfilled.

This is a review and implementation handoff, not a repair commit or an experimental result. No analysis notebooks or biological runs were executed for this review. Tests may import notebook helpers and execute setup-only cells. Production code, configs, notebooks, existing outputs, `outputs 2/`, and `outputs.zip` were not edited.

Severity: **P0** blocks the validity of the intended prospective scientific workflow; **P1** blocks a required method, analysis run, or correctness guarantee; **P2** is a narrower reporting/reproducibility issue. Missing capabilities are identified separately from defects in implemented code. Explicitly optional scientific extensions are not counted as mandatory omissions.

## Commit coverage

Commit subjects alone do not establish WP completion. The history has bundled early work, incomplete later slices, and no dedicated WP11 implementation commit.

| Commit | Intended scope | Actual review assessment |
| --- | --- | --- |
| `fc9ac0b` | Preserve notebooks before branching | Preservation only; explicitly not methodology validation. |
| `0e4d677` | Initial runnable repairs | Corrected signed cGC self-history, prospective role packaging, decoding edge cases and notebook wiring; not WP0/WP1 completion. |
| `e17f4d9` | WP3 | Inductive signed-lag states and explicit HMM outputs; prospective ownership and state-carry requirements remain incomplete. |
| `8842c87` | WP4 | Split-owner interfaces, dummy fitter execution and scoring helpers; no real nested M0/M1 evaluator. |
| `9d33f65` | WP5 | Sparse VAR numerical fits and prediction snapshots; integration, fit ownership and selection remain incomplete. |
| `2368ff5` | WP6 | One-sided smooth/fused numerical fitting; integration and method-contract gaps remain. |
| `b20b5a0` | WP7 | Observed switching VAR component; objective/filter/fitting limitations need correction. |
| `c01bf5a` | WP8 | Separate transition fit/score helpers and backend metadata; optional-analysis diagnostics/ablations are incomplete. |
| `e61d211` | WP9 | One-state repair, seed/manifest/statistics helpers; commit explicitly calls this a scaffold, not the raw-process benchmark. |
| `2cce0e6` | WP10 | Feature-sensitivity language and unavailable outputs repaired; fold ownership and inferential safeguards remain incomplete. |
| `f1cacac` | Notebook resume/progress | Status headers added; existing checkpoint correctness and method-routing gaps remain. |

The review examined each commit's changed-file scope and decision record, and the final integrated source. Historical commits were not individually checked out and their full test suites were not rerun. Findings below refer to reviewed HEAD; a missing requirement can predate these commits and still invalidate a claim of WP completion.

| WP | Status against full Appendix C acceptance | Follow-up findings |
| --- | --- | --- |
| WP0 | Required new contracts/migration missing; old provenance is reusable infrastructure | B1, N3-N5 |
| WP1 | Required target eligibility and fitted preprocessing missing | B2-B3 |
| WP2 | Core self-history fix implemented; predictive persistence incomplete | B6 |
| WP3 | Partial; inductive state improvements and safer role packaging implemented | B7-B8 |
| WP4 | Scaffold only; complete evaluator missing and scorer defects present | B4-B5, N2 |
| WP5 | Partial numerical implementation; state/ownership/input/selection defects | B9-B10, N1 |
| WP6 | Different endpoint objective; solver and prediction contracts incomplete | B11-B12, N1 |
| WP7 | Partial switching component; EM, boundary and frozen-filter contracts incomplete | B12-B13 |
| WP8 | Partial helpers/metadata; diagnostics and ablation integration incomplete | B14 |
| WP9 | Scaffold only; certified generators and actual benchmark/inference runner missing | B15-B16 |
| WP10 | Mandatory relabeling implemented; training ownership/control gates incomplete | B17 |
| WP11 | Required strict profiles, experiment manifests and executable primary handoff missing | B18, N1-N6 |

## Backend findings and acceptance criteria

### B1. P0: WP0 contracts, endpoint lineage and legacy rejection are not implemented

**Evidence:** `src/effectome/data_module/schema.py:14` still declares schema `1.0`; its `ArtifactProvenance` at `:48` has recording-level IDs, not the required fitted-snapshot, endpoint support/availability, raw interval-union and inner/outer ownership contracts. `src/effectome/utils/io.py:34` directly unpickles without migration or scientific eligibility validation. No WP commit changes these files. `src/effectome/data_module/loaders.py:580` falls back to ROI indices only for unresolved names; otherwise `:582` assigns canonical biological indices as ROI IDs, so duplicated resolved names remain duplicated IDs.

**Impact:** callers cannot establish that an endpoint was available at issuance, a fitted transform belongs to the current fold, or equal-sized neural arrays share a registered channel mapping. Legacy artifacts bypass constructor-only schema checks when unpickled. The endpoint source being pinned is useful provenance but does not verify annotation support or independence from the measured neural signal.

**Repair:** implement the WP0 contracts in the existing schema, using unique recording-local ROI IDs separate from canonical names/confidence. Add explicit half-open raw supports, issue/availability times, endpoint kind/vocabulary/annotation-fit provenance, fitted channel/scale snapshots, outer/inner manifests and permitted calibration. Add load-time migration to `legacy_unverified` or validated current schema. Unknown lineage must fail confirmatory validation, not silently mean measured behavior.

**Acceptance:** duplicate canonical names retain distinct ROI IDs; subsets preserve aligned identity arrays; invalid support/ownership is rejected; old pickles load only descriptively unless genuinely migratable. Altering held-out annotations cannot alter training endpoint vocabularies or learned labels. Do not invent unverified biological provenance to satisfy a schema.

### B2. P0: positive lag does not ensure a future target; categorical targets remain modal

**Evidence:** `src/effectome/linking/decoding.py:131` checks source/gaps but never `future.target_start >= origin.context_stop`, anchor validity, or target availability. It accepts `anchors=None`. The separate attribution implementation at `src/effectome/attribution/scoring.py:85` is weaker still, omitting sparse bad-frame checks. `src/effectome/data_module/windowing.py:438` forces every integer endpoint to mode, ignoring `behavior_summary="last"`.

**Reproduction:** four valid anchors with history 30, target 15 and stride 7 return origins for both `q=1` and `q=2`. WP1 requires rejection until `q=3`. This admits targets partially inside already observed history, not strictly future outcomes.

**Repair:** centralize eligibility and use exclusive issue time, actual interval support, all intervening validity/gap flags, target masks and endpoint annotation support. Reject missing anchors in the strict lane. Add an independent categorical `last_valid` rule with selected raw sample and frozen missing/ambiguity policy. Keep mode explicitly legacy.

**Acceptance:** Appendix C's `H=15,s=7,q=1/2/3` cases; equality at `H=s`; invalid intermediate anchors; irregular clocks; mode-versus-last fixture; target annotation extending beyond availability; shared behavior across decoding, attribution and sensitivity consumers.

### B3. P0: primary preprocessing remains window-local, without a fitted training snapshot

**Evidence:** `conf/windowing/primary_physical.yaml:15` keeps `standardize_per_window: true`; `src/effectome/data_module/preprocess.py:165` exposes the old whole-recording function, with no allowed-interval fit/causal state interface. `DependencySupport` in `src/effectome/linking/decoding.py:158` remains three widths, not explicit unions/prefix dependencies. The manifold helper at `notebooks/_shared.py:549` checks equality with `"global"`, unlike consumers that reject all global variants.

**Impact:** windows can express different units and variance ratios as graph changes. Frozen training scale, imputation/filter state and channel ownership are not auditable. Purging a nominal last window does not remove dependence carried through excluded samples by a recursive transform. A global noncausal filter can escape the equality-only check.

**Repair:** implement a training-fitted preprocessor with validated channel map, scale/center, filter and imputation state, fit intervals, reset/burn-in and recorded coefficient-unit transforms. New primary profile uses training-fixed scaling; window scaling remains a named sensitivity. All consumers use one dependency/availability validator.

**Acceptance:** held-out perturbations preserve training state; coefficient and intercept conversions preserve raw predictions; missing frames preserve native time; finite future filters shift issue time or are rejected; prefix state touching a protected block is ineligible. Do not remove permitted past observations from a declared chronological-online protocol.

### B4. P0: WP4 is a split/scoring scaffold, not the required evaluator

**Evidence:** `src/effectome/workflows/evaluation.py:157` supports only `purged_blocked`; `:223` takes already-created windows and defaults to no fitters. `pipeline/08_evaluate.py:28` calls it without fitters, so the CLI issues zero model predictions. No concrete raw-data M0/M1 fitters, nested neural/decoder selection, chronological protocol or leave-animal-out runner were added. Candidate grids in model YAML are not consumed by this runner. `states_enabled`/`groups_enabled` are manifest fields rather than executed ablation controls.

**Additional split bug:** `build_split_manifest` maps the test sets produced by grouped `purged_blocked_splits`, then at `:184` iterates newly generated contiguous index blocks. With multiple unequal or interleaved recording/animal groups, valid grouped test sets need not match those blocks and become spurious unavailable folds. A bounded example with recordings containing 6 and 10 anchors produces valid `(train,test)` sizes `(10,6)` and `(6,10)` from the grouped splitter, but the manifest replaces them with two unavailable `(0,8)` folds. The reason also mislabels arbitrary caught `ValueError`s as insufficient purged data.

**Repair:** construct and persist actual outer/inner raw-source splits before fitting. Implement real probability M0/M1 pipelines with activity history, available behavior/input history, lag-resolved dynamics and training-only class handling. Add neural-loss selection and properly nested decoder-loss selection (or a separately frozen development choice). Preserve attempted fits/origins and specific failure reasons. Derive retained unavailable folds from the actual protocol's candidate groups, not an unrelated partition.

**Acceptance:** CLI from a tiny raw manifest yields nonempty M0/M1 records; all stages are inspected with raw-ID spy assertions; test signal/label changes cannot affect training choices; animal sessions never split across ownership; unequal group-size/interleaved group fixtures work; one-class/rare-class folds have explicit fixed-vocabulary probabilities; optional representations can truly be absent. A zero-fitter run must identify itself as a manifest inspection, never a completed evaluation.

### B5. P1: paired scoring overwrites recordings and silently excludes method failures

**Evidence:** `src/effectome/workflows/evaluation.py:339` keys records only by `(origin_index, animal)`, without recording, fold, target support or endpoint-definition identity. Later records overwrite earlier ones. Only `status="issued"` enters the comparison; unpaired failures disappear without a fixed scoring policy or failure counts in the paired result.

**Reproduction:** two recordings of one animal both have origin 0. M0/M1 probabilities for observed class 0 are `(0.8,0.4)` in recording 1 and `(0.4,0.8)` in recording 2. Correct result: two matched predictions and mean gain zero. Actual result: one match and gain `0.6931471805599453`, dependent on record order.

**Repair:** define a complete stable origin key with dataset/animal/recording/episode, fold, issue and target support, and endpoint definition. Reject duplicates and mismatched observed outcomes/class schemas; do not silently overwrite. Freeze an eligibility/failure/fallback policy and report matched/unmatched/failed counts per model and animal. Use one validated probability-floor/renormalization policy in both probability scorer implementations.

**Acceptance:** the two-recording example, repeated folds, reordered inputs, duplicate detection, mismatched target labels, missing identities and asymmetric model failures. Equal-animal weighting must remain after correct within-animal aggregation.

### B6. P1: WP2 retains self-history numerically but discards the predictive snapshot

**Fulfilled:** `src/effectome/core/causalised_gc.py:309` correctly unions selected cross-neuron support with every target self-lag, stores full coefficients/intercepts/residual variances, and exports a separate cross-only graph. Preserve this repair and its regression tests.

**Gap:** `src/effectome/connectivity/causalised_gc.py:47` returns only the exported graph/lag tensor and summary diagnostics; its local fitted estimator is discarded. The full self-dynamics, intercept, valid response IDs and scale needed to reproduce neural forecasts are not persisted or exposed through WP5's predictor interface. The adapter file named by WP2 was not changed. Summary support counts or a mean residual variance cannot reconstruct the model.

**Repair/acceptance:** persist immutable fitted cGC snapshots with full lag dynamics and support, scale/channel/fit ownership and per-target residual diagnostics. Add forecast capability only when that snapshot is available; do not synthesize it from graph diagonals set to zero. Round-trip serialization must reproduce direct core predictions, including a self-only process and nonzero intercept. Add the correlated-input/omitted-parent/sign and selected-support semantics cases specified in C.4.

### B7. P1: graph-state fitting is improved, but identity, one-state selection and lag centroids remain incomplete

**Fulfilled:** `e17f4d9` uses signed lag/off-diagonal features, training-only standardization/PCA/KMeans and training-only collapsed interpretation centroids, with descriptive centroids separated. A fixed `n_states=1` works.

**Evidence/gaps:** `src/effectome/dynamics/graph_states.py:321` validates node/lag counts, not node identity/order or physical lag schema. Equal-shaped reordered channels are accepted. At `:588`, `select_k` gives one state a NaN silhouette and replaces it whenever a multistate candidate has a finite silhouette; the option does not implement predictive one-state selection. At `:621`, interpretation centroids use `series.matrices`, not lag-resolved tensors. Outer selection remains absent.

**Repair/acceptance:** persist and compare registered node IDs/order, physical lags and representation units. Preserve signed lag-resolved interpretation centroids. Use the evaluator's common-response selection including one state; keep silhouette as a descriptive metric. Tests must permute identities without changing shapes, alter physical lag spacing at the same lag count, preserve opposite-sign lag centroids, and permit a constant model to win predictive selection.

### B8. P1: HMM and temporal roles still lack outer-owned, resumable inference

**Fulfilled:** HMM filtered labels/entropy are explicit; temporal role packaging now uses a fixed seed-ordered reference and label-invariant coassignment, avoiding future-selected references.

**Evidence/gaps:** `src/effectome/dynamics/probabilistic.py:500` scores the entire complement of `fit_indices`, including purge/calibration exclusions. Generic `labels`/`uncertainty` remain smoothed aliases at `:122`. No carried forward filter state or explicit test-index API exists. `src/effectome/community/temporal.py:229` learns scales from the first layer of whichever series is passed; `:283` initializes centers anew, without a reusable outer-training model. At `:313` the update is `rho/counts[state]`, not the required fixed exponential update. `rho=0.25` therefore decays with occupancy instead of retaining the specified meaning.

**Repair/acceptance:** add explicit scoring indices and stateful filtering, a fitted role model with training scales/schema/centers/reference and post-assignment exponential adaptation, plus per-origin state snapshots. Name the decreasing-step algorithm separately if preserved. Test excluded origins never score; frozen fits under future-tail perturbations; five-run prefix invariance; empty groups; hand-computed rho 0/0.25; real-boundary resets; interrupted versus continuous filtering. Label filtered timing as a timing property, not proof of independent graph emissions or biological regimes.

### B9. P1: sparse-VAR observation state compresses time across missing samples

**Evidence:** `src/effectome/connectivity/sparse_var.py:115` discards the `available_inputs` argument; `:126` concatenates `observed[valid]` to history, ignoring gaps in sample indices, chronology and state/model compatibility. The origin is the last supplied sample index, not the exclusive next-sample origin required by C.3. Future inputs are likewise silently discarded in `forecast`.

**Reproduction:** a lag-2 model observing samples `[1, NaN, 3]` at indices `[0,1,2]` stores history `[1,3]` and forecasts without a failure. It treats samples separated by a missing frame as consecutive lags. The otherwise useful `build_lag_design` gap guard is bypassed by the forecast-state path.

**Repair/acceptance:** retain clock/masks or explicitly invalidate/reset lag history across gaps; validate monotone sample IDs and model/recording state identity. Use exclusive origins and persist raw history support. Implement measured inputs with the required family penalty, or reject nonempty inputs until supported instead of silently ignoring them. Test masked/NaN rows, index jumps, out-of-order updates, segment switches, chunk equivalence, wrong-model states and nonzero known input effects.

### B10. P1: constant sparse VAR silently changes into a windowed method

**Evidence:** `src/effectome/connectivity/sparse_var.py:216` reconstructs one raw series from overlapping windows. If shared samples differ, it returns `None`. At `:292`, constant mode then silently fits every window independently. Default preprocessing standardizes each window, so overlapping samples normally differ and trigger this fallback. Conversely, successful reconstruction fits all supplied windows and repeats the result at early anchors, which is retrospective if the series includes the evaluation episode.

**Repair:** static fit must consume permitted raw training rows with explicit sample/source identities and one frozen scale, then apply the frozen model. Reject incompatible input or name a separate windowed adaptation; do not substitute algorithms under `sparse_var`. Keep full-series fitting descriptive only. Convergence failures must not become ordinary graph artifacts or warm starts.

**Acceptance:** overlapping standardized windows fail the constant-fit contract rather than change adaptation; raw overlapping training supports are deduplicated; multiple recordings sharing sample indices do not collide; test tails cannot change a frozen static model; constant versus windowed mode is visible in persisted method identity. Add KKT/proximal-residual checks and nonconvergence tests; current parameter-change stopping in `var_utils.py:174` is not the requested optimality residual.

### B11. P1: TVVAR implements a different objective and can falsely report convergence

**Evidence:** `src/effectome/connectivity/tvvar.py:119` fits one current coefficient to a full window and penalizes distance to a fixed previous endpoint. `:316` repeats this sequentially. C.8 instead requires optimizing all coefficient knots in each allowed buffer, assigning response rows to knots, with global valid-row and knot/link normalization. Those knot assignments and joint buffer optimization are absent. This endpoint regularizer can be kept as a separately named method but cannot be called the specified TVVAR objective.

**Concrete numerical defect:** at `:177` the coefficient solve eliminates an unpenalized intercept by centering, then at `:183` shrinks that intercept toward its previous value. The slope is not recomputed for the penalized intercept. The iterative gradient also uses centered residuals, ignoring this coupling. For smooth ridge-only fitting, `:191` declares convergence with residual zero without checking the reported objective gradient. Standardized fitting additionally reports a raw-unit penalty objective inconsistent with its optimized standardized coefficients.

**Reproduction:** one predictor `X=[1,2,3]`, response `Y=2X`, previous slope 0, previous intercept 10, ridge 0.1, slope temporal penalty 0.2, intercept temporal penalty 1.0. Result: `converged=True`, `proximal_residual=0`, but the derivative of the reported objective with respect to slope is `17.517241379310345`, not zero.

**Further solver concern:** composing group shrinkage about a nonzero previous coefficient and off-diagonal soft thresholding (`:202`) is not, without a derivation, the joint proximal operator of the two translated penalties. A fixed point of that composition is not a sufficient KKT certificate.

**Repair/acceptance:** implement the declared buffer objective with shared inputs, actual knots, normalized components, correct joint intercept/coefficient optimization and a justified smooth or ADMM/primal-dual solver. Store true stationarity/primal/dual residuals and failure status. Compare tiny smooth/fused solutions to an independently solved convex reference, including nonzero means/intercepts, nonzero L1 and previous nonzero coefficients. Verify one knot, zero temporal penalty with correct row weights, large penalty, real gaps, frozen future-tail invariance, instability and resume. Do not change the mathematical specification silently to match the current code.

### B12. P1: TVVAR and switching VAR advertise forecasts they cannot issue

**Evidence:** `src/effectome/connectivity/tvvar.py:288` and `switching_var.py:240` advertise `neural_forecast=True` and `stateful_online=True`, but do not implement fitted models or state/forecast APIs. They inherit `ConnectivityEstimator.initial_state/observe/forecast`, which raise `UnsupportedForecastError` (`base.py:114`).

**Reproduction:** all three `tvvar_smooth`, `tvvar_fused`, `switching_var` registry objects advertise neural forecasts, then fail immediately on `initial_state()`.

**Repair/acceptance:** implement serializable fitted snapshots, initial/filter states, frozen-parameter observation updates, one-step and specified multistep forecasts and graph snapshots. Capability declarations must reflect usable behavior. Every advertised forecast-capable method must pass one shared fit/observe/forecast/serialize/resume contract test and common-mask neural scoring. Missing normalized densities remain explicitly unsupported. Test-time known-input and update policies must be explicit.

### B13. P1: switching VAR does not satisfy its EM objective or sequence-boundary contract

**Evidence:** `src/effectome/connectivity/switching_var.py:34` penalizes each state's regression with occupancy times lambda, without deriving it from a common variance-weighted penalized likelihood. `:195` records unpenalized likelihood only and accepts absolute likelihood change, including decreases, as convergence. It does not enforce generalized-EM nondecrease of a declared penalized objective. `:177` handles low occupancy by fitting uniform weights, then still emits results. `:228` sets convergence true for `max_iter<=1`; a one-iteration synthetic fit reproduced this status.

`fit_switching_var` never uses `LagDesign.segment_ids` or response-index gaps; `_forward_backward` at `:74` processes all retained rows as one episode. Valid design rows separated by missingness/recording boundaries therefore contribute invented HMM transitions. The returned posteriors/likelihood are computed before the final transition/initial-probability update, so the snapshot parameters and inference outputs need not match. `estimate_sequence` at `:288` refits parameters separately in every window; it is not the required training fit with frozen test parameters and carried filtering. KMeans `n_init=5` is not five complete switching-VAR restarts.

**Repair:** choose one explicit penalized observed-data objective and derive weighted coefficient/variance updates accordingly; monitor and reject nonmonotone/unconverged candidates under a frozen policy. Split recursions at valid episode boundaries, learn initial distributions from all segment starts, rerun final inference against the saved parameters, implement genuine training-only restarts and occupancy/rank checks, and keep test parameters frozen. The existing `alpha @ P` helper is useful but must be used by the actual pre-observation forecast API.

**Acceptance:** brute-force tiny likelihood/filter enumeration with multiple segments; invalid-row resets; final saved parameters reproduce returned likelihood/posteriors; decreasing objective and one-iteration budget cannot claim convergence; occupancy rejection is enforced; matched-objective C=1 equivalence; permutation-invariant forecasts; deterministic full restarts; uninterrupted versus chunked filtering. A Gaussian single-state coefficient match alone does not verify the switching likelihood objective.

### B14. P1: transition helpers are partial and mislabel self-transitions as exits

**Fulfilled:** `c01bf5a` adds frozen train fit/test scoring, per-transition scores and explicit edge-censored dwell runs. Actual backend/group-transformation metadata is improved.

**Defect:** `src/effectome/dynamics/transitions.py:194` sets `observed_exits = counts.sum(axis=1)`, which includes staying in the same state. For `[0,0,0,0,0]`, the result is four observed exits despite zero changes of state. Retain a separately named row-trial count if needed; report off-diagonal exits for recurrence/exit evidence.

**Required missing diagnostics:** `transition_diagnostics` at `:253` is a small dictionary of counts/occupancy/dwells, not the required fixed-vocabulary persistence/escape and second-order comparison, recurrence, homogeneity, residual dependence or Chapman-Kolmogorov diagnostics with uncertainty. The new APIs are not integrated into the evaluator; notebooks continue using `fit_transitions`. No outer-owned representation replacement/augmentation ablation or raw-resampling uncertainty path exists.

**Repair/acceptance:** distinguish row trials, actual exits, prior-driven rows and censored dwells; add declared cadence and eligible-pair ownership. Implement the requested baseline comparisons and uncertainty, with no reclustering for two-step checks. Tests: all-self sequence, never-visited state, second-order toy process, fixed-vocabulary two-step example and no purged/gap transition pairs. HSMM/block-model implementation is explicitly conditional, not a prerequisite unless that extension is selected.

### B15. P0 for calibration claims: WP9 raw-process validation is not implemented

**Evidence:** `src/effectome/experiments/validation.py:22` stores suite totals and deterministic seeds only; `conf/validation/revised.yaml` stores counts, not executable scenario cells. `src/effectome/utils/synthetic.py:116` still stabilizes each regime separately; `:168` independently draws regime graphs; `:232` applies instantaneous solves after stabilization. No common-metric switching certificate, effective reduced-form certificate, required burn-in, five temporal hypotheses, observed-projection truth or conditional-M0 endpoint generator exists. WP9's only generator changes are the one-state checks.

**Impact:** individual-regime spectral stability does not satisfy the common-certificate requirement supplied in the report. Calibration cannot claim the prescribed finite benchmark or distinguish latent graph truth from fluorescence finite-lag projection. Constants `500/270/30/120/600` are not executable paired experiments.

**Repair:** implement C.11's generator/observation separation, shared-base edits, common Lyapunov certificate with actual multi-lag and instantaneous reduced-form validation, certified burn-in and finite-rollout checks. Add constant/drift/nonrecurring/recurrent geometric/nongeometric scenarios, measurement/variance nulls, explicit truth IDs, projection convergence checks and endpoints generated from declared M0 plus fresh noise. Expand the manifest into actual deterministic paired cells and a runner using the same evaluator as real data. Persist every attempt, mask/config/hash, runtime/memory method, failure/timeout and resource budget. Keep development and independent evaluation streams separate.

**Acceptance:** the provided individually stable but switching-unstable matrix pair is rejected; p>1 companion shift structure and instantaneous transformations are checked; exact cell counts derive from enumerated scenarios; one-state and native clock/masks work; smoke jobs execute raw preprocessing through scoring with paired methods and complete failure logs. Implement the runner now, but do not launch the 800/920/600-recording grids under the user's no-run instruction.

### B16. P1: probability/multiplicity helpers are not wired into complete inference

**Fulfilled:** `src/effectome/linking/stats.py:155` implements BY q-values with finite-value checks; log loss, Brier score, plus-one p-values and a resolution guard exist.

**Gap:** these helpers do not establish a declared complete secondary family, unavailable-test policy, finite unique-control accounting, selection-aware lag/model null, raw-block refitting or animal-level inferential procedure. Existing endpoint analyses still compute their own p-values and use older score paths. `bootstrap_mean_ci` on dependent windows cannot supply the specified independent-animal inference merely by being available.

**Repair/acceptance:** make the evaluator/validation manifest own complete families, attempted hypotheses, failure statuses, alpha resolution and control enumeration/sampling policy. Implement raw contiguous-block resampling with joint masks/inputs/labels, reset at artificial joins and refit each claimed stage. Report individual paired animal effects and explicit limits of overlapping-training LOAO pilots. Test known BY families with unavailable members, unattainable resolution, finite controls, paired resampling, no across-join rows/transitions and equal animal weighting. Freeze calibration tolerance/practical gain before held-out evaluation, not from its outcomes.

### B17. P1: WP10 corrected the claim language but not the validity gates

**Fulfilled:** `2cce0e6` uses `predictor_feature_sensitivity`, sets biological identification false, returns unavailable effects for invalid surrogate runs, preserves a compatibility wrapper and removes counterfactual candidate promotion. Keep these boundaries.

**Evidence/gaps:** `src/effectome/intervention/surrogate.py:218` uses precomputed full-run features/categories before CV; `:250` validates only mean incremental R-squared above `min_skill`; `:261` refits on all rows. It does not check absolute held-out skill, uncertainty, distribution shift or nested feature ownership. `:400` matches controls using full-series strengths and repeatedly samples a small pool. For one selected node the pool is at most four alternatives, yet `:542` uses the number of repeated draws in the p-value denominator without unique-pool/resolution reporting. No controls becomes numeric `p=1` rather than an explicitly unavailable test. `notebooks/_shared.py:1242` falls back to the top-scoring node even when every node was screened out. `src/effectome/attribution/scoring.py:70` still treats categorical current behavior as a scalar code in its baseline.

**Repair:** freeze candidate selection, categorical schema, matching variables, unique eligible controls and multiplicity family within training; pass a frozen fitted feature/surrogate bundle to held-out sensitivity scoring. Add absolute and incremental performance, uncertainty and shift checks. Distinguish Monte Carlo sampling of a declared finite null from claiming extra independent controls; persist pool size and exact attainable significance, enumerate where appropriate, and flag unavailable controls explicitly. An arbitrary exploratory node edit must be labeled as such rather than silently substituting for a qualified candidate.

**Acceptance:** outer-test perturbations preserve selections/matching/features; relabeling category codes leaves predictions unchanged; weak absolute skill cannot pass on relative improvement alone; a tiny unique control pool triggers the registered resolution policy; no candidate/no controls returns explicit statuses. Perfect sham/monotonicity remains model sensitivity only. A generative biological intervention extension is optional and must remain separate.

### B18. P1: WP11 profiles, strict configuration and reproducible handoff remain absent

**Evidence:** `conf/evaluation/default.yaml:1` is labeled a skeleton and has no development/confirmatory mode, endpoint schema, practical-gain requirement, calibration gate, model grids/budgets, missingness/failure policy or schema migration setting. `EvaluationConfig` has no validation of those contracts. There are no named `legacy_descriptive` and revised primary profiles. Method `extra` dictionaries read known values with `.get`, allowing misspelled/unsupported keys to be ignored. The runnable future command described in Appendix C is not supported by the current entry point.

**Repair/acceptance:** finish a validated frozen schema mirrored consistently in YAML, CLI, notebooks and artifact manifests; reject unknown method keys. Resolve mandatory scientific fields before reading held-out outcomes and fail confirmation when lineage, panel mapping, calibration, practical effect or budget is unresolved. Persist source/lock/config/data/split/endpoint identities and a machine-readable claim-to-test map. Missing optional backends must be explicit unsupported results, never a substitute algorithm. A clean small synthetic rerun from a raw manifest must reconstruct scores without global precomputed learned artifacts.

Structural-break VAR, published dLDS, latent/recurrent models, actual HSMMs, optional block models and generative interventions are conditional extensions in Appendix C. Their absence should be reported as unsupported/deferred, not filled with invented substitutes and not conflated with the mandatory gaps above.

## Notebook and checkpoint findings

### N1. P1: all new sparse-VAR/TVVAR notebook lanes fail at the default chunk size

**Evidence:** `notebooks/_build_notebooks.py:377` emits `CHUNK_SIZE = 16` for every method; `:861`, `:867`, and `:873` include `sparse_var`, `tvvar_smooth`, and `tvvar_fused`. `src/effectome/workflows/connectivity.py:100` rejects more than one chunk for estimators without independent-window capability except `rolling_window`. These three methods declare sequence capability and different mode names. `notebooks/_shared.py:408` passes the chunk size unchanged.

**Trigger/impact:** each configured real recording has more than 16 windows. Each of the nine new dataset/method connectivity notebooks fails before fitting. A bounded probe with `n_windows=17` reproduced the same `ValueError` for all three configurations. Existing helper tests use small windows/older methods and miss this path.

**Required repair:** choose execution from declared capabilities. Preserve the independent-window guard. Run a sequence fit as one explicitly checkpointed stage until stateful chunk/replay exists; implement state snapshots and parent-state hashes before claiming partial resume for sequence models. A single-stage fallback must say that interruption restarts the stage. Do not fix this by lying about independent-window capability. Route corrected primary evaluation through its fold owner rather than fitting static models on the entire test recording.

**Acceptance:** helper integration tests with more than 16 windows for every generated method; compare supported chunk sizes 1/3/16 to uninterrupted execution; interrupt and restart without changing issued results. Test actual notebook config extraction plus helper invocation, without executing biological notebooks. Include missing `windowed_sparse_var` and `switching_var` lanes in the final capability-driven analysis interface or document their explicit availability through the evaluator.

### N2. P0: notebook linking bypasses the revised evaluator and still certifies stitched features

**Evidence:** `notebooks/_shared.py:742` unconditionally requires graph states, communities and a manifold. `:769` onward builds features from precomputed full-run artifacts, and `:1024` calls the old `decode_behavior`. `:1085` onward computes a community increment with current behavior, manifold/activity summaries and collapsed connectivity. It does not implement the primary probability-level M0/M1 comparison. `:687` sets `prospective_eligible` from `cross_fit` alone. The equivalent CLI path does the same in `pipeline/04_manifold.py:224`. No generated notebook calls the new evaluation runner.

**Impact:** users can run the prepared notebooks and obtain legacy descriptive/blocked-CV results while believing they are executing the revised analysis. A global stitched out-of-fold chart does not prove that every downstream decoder-training feature excludes that decoder's test data. C. elegans' retrospective BunDLe lane correctly disables some prospective outputs, but that does not provide the requested primary evaluation. States/groups cannot actually be omitted from these notebooks.

**Required repair:** retain these stages under explicit `legacy_descriptive` naming; build one real outer-owned runner and call it from both CLI and new primary notebooks. Fit preprocessing, graph selection, charts, optional representations and decoder inside the relevant outer/inner ownership. Default primary M0/M1 to no states/groups/manifold dependency. Remove blanket eligibility from `cross_fit=True`; validate concrete fit/dependency manifests. Update the README's method list and execution order to reflect actual supported lanes.

**Acceptance:** same tiny raw inputs/config yield semantically identical CLI and notebook-runner predictions; no optional artifacts exist in the M0/M1 test; held-out signal/label changes cannot change any training-fitted component; future-tail perturbations cannot change earlier issued predictions. Keep BunDLe correspondence explicitly retrospective.

### N3. P1: cached artifacts survive implementation changes

**Evidence:** `src/effectome/workflows/resume.py:75` hashes only config and declared dependency files. The public `execute` path at `:168` uses that signature without a semantic implementation/schema version. WP0 explicitly requires code/lock/schema/split/snapshot identity. Those additions are absent.

**Reproduction:** execute a stage with `config={}` returning `"old implementation"`; execute again with the same config and a changed computation returning `"corrected implementation"`. The result is still `"old implementation"`, with `reused=True`.

**Impact:** switching between the user's WP commits can silently reuse pre-fix outputs under `FORCE=False`, including artifacts from the old signed-refit behavior. Data hashing alone cannot detect changed scientific semantics.

**Required repair:** version scientific artifacts and stage implementations; include code/lock/config/endpoint/split/training-snapshot identities and incoming causal state where relevant. Fit caches should depend on permitted training data only; prediction caches additionally depend on transformed test data. Reject legacy artifacts for confirmation through the loader, not just a dataclass constructor. Preserve readable retrospective copies.

**Acceptance:** same inputs reuse; changes in code/schema/split/scale/endpoint/parent state invalidate the relevant stage; changing only held-out values does not invalidate or refit training-only artifacts; legacy pickles cannot bypass validation.

### N4. P1: downstream helpers load stale artifacts from failed stages

**Evidence:** `notebooks/_shared.py:159` (`require_stage`) checks only whether `artifact.pkl` exists. In `src/effectome/workflows/resume.py:227`, a failed forced/changed-config recomputation writes a failed status but leaves the earlier completed artifact in place.

**Reproduction:** complete a stage, recompute with a new config and a deliberately raised exception, then call `require_stage`. It returns the old artifact despite the stage now being failed. A downstream stage can therefore continue with obsolete inputs while the header reports failure.

**Required repair:** prerequisite loading must verify completed status, matching artifact identity/integrity, expected schema and required upstream lineage. Keep previous artifacts available under explicit historical identities rather than accidentally treating them as the current stage. Handle live-writer state consistently; existence is not a completion contract.

**Acceptance:** completed then failed, interrupted, missing-status, mismatched-artifact and concurrent-writer cases must not serve a current prerequisite; valid completed state must load. No automatic deletion of user outputs.

### N5. P2: copying checkpoints to another machine changes their cache identity

**Evidence:** `src/effectome/workflows/resume.py:67` includes absolute dependency paths in hashed signatures; `notebooks/_shared.py:274` also embeds `source_path` in input-stage config. A bounded probe confirmed that two identical files at different paths produce different signatures.

**Impact:** the README's faster-machine checklist suggests carrying checkpoints, but changing the checkout root invalidates dependent stages and can trigger expensive recomputation. Artifacts also contain absolute model paths that need a relocation policy.

**Required repair:** use logical dataset/source identifiers plus content hashes for semantic identity, and retain absolute paths as non-identity provenance. Resolve artifact/model paths relative to the run manifest or explicitly remap on import. Preserve semantic invalidation from N3.

**Acceptance:** copy a small completed run to another root and verify safe reuse, usable model references and unchanged semantic predictions; changed content still invalidates.

### N6. P2: expanded notebook lint check fails

`ruff check src pipeline tests notebooks` reports six issues: an overlong generated preprocessing header and five import-position errors in the preserved c-GC sweep notebook. The narrower source/helper lint command passes. Fix the generator for generated code and the separately maintained sweep source; do not suppress all notebook linting to hide this distinction.

## Implementation sequencing

Do not treat the presence of an API or YAML candidate grid as completion of the corresponding scientific work package. Implement and test the actual consuming workflow. Preserve the user's historical commits; create separate repair commits with Lore intent/rationale/verification trailers, one coherent repair or WP completion per commit.

1. Complete WP0/WP1 contracts, loader validation, target support and fitted preprocessing; repair N3/N4 cache correctness. These are prerequisites for valid ownership.
2. Complete the raw-recording outer evaluator, probability decoders and fixed failure policy. Integrate neural model selection and optional-stage omission.
3. Repair dynamics numerical/forecast contracts and add real immutable fitted snapshots, filtering state and explicit failure handling.
4. Complete WP9 generator certificates, truth definitions and raw-process benchmark runner before interpreting state/transition evidence.
5. Finish optional-state/group diagnostics and training-owned feature sensitivity without broadening intervention claims.
6. Route primary CLI/notebooks through the same runner; repair N1/N2/N5, regenerate notebooks, and test small synthetic helper paths. Keep progress/resume logs at the actual unit of work.
7. Complete strict WP11 profiles/config validation and manifests. Leave scientific thresholds and endpoint lineage visibly unresolved until verified; confirmation must reject unresolved required fields.

The user's standing constraint remains: **prepare notebooks but do not execute analysis notebooks or large biological/calibration jobs without a new request.** Tiny synthetic tests and static/setup validation are sufficient for this repair handoff's engineering checks; they cannot substitute for the later scientific benchmark.

## Verification and limits

- Existing unit/synthetic suite: 178 passed, 1 skipped (179 collected), exit status 0, with the existing joblib physical-core-detection warning.
- `ruff check src pipeline tests notebooks/_shared.py notebooks/_build_notebooks.py`: passed.
- `mypy src notebooks`: passed for 62 source files, with notes about unchecked untyped BunDLe bodies.
- `ruff check src pipeline tests notebooks`: failed with the six notebook-only issues recorded above.
- `git diff --check d58be17..HEAD`: passed.
- Bounded synthetic probes confirmed N1, N3, N4, N5, B2, B4's grouped-fold mismatch, B5, B9, B11, B12, B13's one-iteration convergence status and B14 without notebook analysis execution.
- Parallel reviewers were attempted but hit an execution usage limit before delivering findings. The final findings were inspected and verified by the main reviewer; this is not represented as completed independent peer review.

No publication grid, biological rerun, independent cohort assessment or new empirical calibration was performed. A skipped optional-backend test and passing synthetic tests do not establish BunDLe installation or faster-machine readiness.

### Minimal numerical reproduction

Run this from the nested repository with `.venv/bin/python`. It uses synthetic arrays only, does not execute notebooks, and does not write analysis artifacts. These are diagnostic examples for the implementing agent to turn into regression tests, not calibration experiments.

```python
import numpy as np
from effectome.connectivity import ConnectivityConfig
from effectome.connectivity.sparse_var import fit_sparse_var_model
from effectome.connectivity.tvvar import fit_tvvar_endpoint
from effectome.connectivity.var_utils import LagDesign
from effectome.workflows.evaluation import PredictionRecord, paired_log_loss_delta

# B11: a reported zero residual does not mean the declared objective is stationary.
x = np.array([[1.0], [2.0], [3.0]])
y = 2 * x
fit = fit_tvvar_endpoint(
    LagDesign(x, y, np.arange(3), 1, 1),
    previous_coefficients=np.zeros((1, 1)),
    previous_intercept=np.array([10.0]),
    lambda2=0.1,
    lambda_temporal=0.2,
    lambda_intercept=1.0,
)
gradient = -2 * x.T @ (y - fit.intercept - x @ fit.coefficients) / 3
gradient += 2 * 0.3 * fit.coefficients
print(fit.converged, fit.proximal_residual, gradient.item())
# Reviewed HEAD: True, 0.0, 17.517241379310345

# B9: missing rows must not become consecutive lag history.
model = fit_sparse_var_model(
    np.random.default_rng(1).normal(size=(30, 1)),
    cfg=ConnectivityConfig(name="sparse_var", max_lag=2),
)
state = model.observe(
    model.initial_state(),
    np.array([[1.0], [np.nan], [3.0]]),
    sample_indices=np.array([0, 1, 2]),
)
print(state.history.ravel(), model.forecast(state, horizon_samples=1).failure_reason)
# Reviewed HEAD: [1., 3.], None

# B5: distinct recordings of the same animal must not overwrite each other.
records = [
    PredictionRecord(
        origin_index=0, fold_id="f", model_name=method, status="issued",
        endpoint_name="e", class_order=(0, 1), probabilities=(p, 1-p),
        observed=0, animal_id="a", recording_id=recording,
    )
    for recording, probabilities in [("r1", (0.8, 0.4)), ("r2", (0.4, 0.8))]
    for method, p in zip(("M0", "M1"), probabilities)
]
print(paired_log_loss_delta(
    records, baseline_model="M0", candidate_model="M1", endpoint_name="e"
))
# Required: two matches, equal-animal gain 0.
# Reviewed HEAD: one match, gain 0.6931471805599453.
```

Changed file for this review: `IMPLEMENTATION_REVIEW.md` only. No production simplification was performed. The handoff simplifies the repair order by making the no-state/no-group M0/M1 evaluator the primary completion gate and keeping optional extensions separate. Remaining risks include unresolved endpoint annotation support, channel identity/panel decisions, untested numerical edge cases, missing end-to-end ownership, and scientific calibration still to be performed.
