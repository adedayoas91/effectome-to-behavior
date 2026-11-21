# Fresh-seed mechanism replication and null calibration

Decision date: 17 November 2026. This protocol is fixed before inspecting any of
the new scientific runs. It is an internal frozen evaluation, not external
preregistration. It extends the favorable mechanism control, preserves the older
failed latent-bistable study, and includes no graph-similarity implementation.

## Question and reason for the change

The completed notebooks show exact native graph recovery in 24 bounded sanity
fits, but all four methods fail the 560-run latent driver benchmark. The revised
mechanism reaches full driver ranking and low intervention-effect error at a
1,024-sample discovery history on one existing development seed. That result does
not establish replication or specificity. The next experiment therefore adds
fresh process seeds and separates threshold selection from positive and null
evaluation. Graph-feature omission and structural intervention remain different
estimands [Williamson et al.](https://arxiv.org/abs/2004.03683); [Pearl](https://proceedings.mlr.press/v6/pearl10a.html); this study evaluates the latter through the already
implemented model-matched readout.

Use the existing observed six-neuron additive-tanh control, 24 independent trials
per seed, trial length 1,152, discovery burn-in 128, twelve outer training trials,
nine inner discovery trials, three penalty/noise-validation trials, and twelve
held-out evaluation trials. Self-history is retained under outgoing disconnection.
The observed endpoint, fitted feature family, ridge grid, 256 paired Monte Carlo
rollouts, eight candidate interventions and existing 0.10 gold-effect threshold
are retained. No simulator coefficients or innovation scales enter native lanes.
Only the explicitly labeled known-support lane receives true adjacency. This
remains a favorable, fully observed control [Peters et al.](https://jmlr.org/papers/v15/peters14a.html); [Hoerl and Kennard](https://doi.org/10.1080/00401706.1970.10488634), not validation of
hidden-state or learned-manifold recovery.

## Pools and comparison ownership

The `replication` profile declares three disjoint, previously unused pools:

| Pool | Seeds | Role |
|---|---|---|
| Calibration global null | 10000–10018 (19) | Select primary per-method cutoffs only |
| Positive test | 20000–20019 (20) | Untouched driver recovery and paired history contrast |
| Null test | 30000–30059 (60) | Untouched whole-seed false-nomination audit |

Primary discovery history is **1,024** samples. Both support controls and all four
native methods are evaluated in every cell. A secondary **256**-sample discovery
history uses the same twenty positive-test recordings, full fitting inputs,
origins, oracle probabilities and rollout streams. Only discovery context changes.
The secondary panel is a paired sensitivity diagnostic: it receives no separately
calibrated cutoff or null-specificity claim. It cannot select the primary history,
which is fixed here from earlier development evidence.

This gives 119 cells, 714 method/control entries, 476 native lane calls and 4,284
within-trial native window fits. Four spawned processes reuse one conventional
runtime per process; internal numerical-library thread counts are limited to one.
The default notebook and CLI output is
`outputs/driver-mechanism-replication/replication/mechanism-replication-01/`.

A separate `pilot` profile uses calibration seeds 1000–1018, positive seeds
1100–1103 and null-test seeds 1200–1205. Its role is engineering verification;
it cannot satisfy the full positive/null count gates. It does not add observations
to the replication pools. Historical seeds 0–2, 20–22, 42, 300–319 and 400–459 are
rejected, as are overlaps or duplicate seed IDs between the new pools.

## Cutoff selection, before any test jobs

For each method, summarize one estimated probability effect per candidate and
supported direction within each calibration-null seed. Define its score as the
maximum over all eight candidate units and both directions. Candidate coverage
must agree between calibration seeds. Both directions need positive origins and
finite scores. A missing/unavailable seed or unsupported direction blocks that
method's calibration; it is never discarded to obtain a favorable cutoff.

Let `n` be the number of planned calibration maxima, `alpha=0.05`, and
`r=ceil((n+1)*(1-alpha))`. Use the r-th ordered maximum as the empirical cutoff;
reject a design with `r>n`. With nineteen calibration seeds this is their maximum.
The **operational** cutoff is the larger of that value and the existing **0.10
minimum probability-effect cutoff**. Nomination requires strict `score > cutoff`;
ties are conservative. The fixed floor preserves the prior minimum meaningful
effect size and can only make nomination more conservative. It is not tuned on
the new positive or null tests. Oracle effects never select the cutoff.

The rank argument gives marginal new-global-null protection under exchangeability
of the complete scoring procedure, including estimator randomization: the chance
that a new null maximum exceeds the r-th calibration order statistic is at most
`(n+1-r)/(n+1) <= alpha`. Strict ties and the fixed floor are conservative. This is
the finite-sample rank principle used in conformal methods [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511), applied here to a whole-seed null maximum rather than a predictive set.
It is **not** a 95% confidence guarantee about the realized cutoff's conditional
false-event probability. Nineteen scores can suffice for marginal 5% calibration
while leaving substantial uncertainty about a particular realized threshold.
The untouched sixty-seed null evaluation is essential.

Here global null means no candidate outgoing intervention affects the selected
endpoint; driver/bystander dependence elsewhere in the graph is retained.
Protection concerns each method under this complete endpoint-null generator. It is
neither simultaneous coverage for four methods nor strong false-discovery control
when some candidates truly influence the endpoint. Positive-test nomination
precision and recall must therefore be evaluated separately.

The runner writes the full source/configuration fingerprint before calibration.
It freezes `calibration.json` before submitting **any** positive, short-history or
null-test job. Resuming verifies that identity and the fixed cutoff. A protocol or
code change requires a new namespace **and fresh test seeds**, not a rerun until
the same test seeds look favorable. The first completed graphs, scores, p-values,
models, rollouts and diagnostics are retained; unavailable lanes remain visible.

## Fixed primary acceptance

Each native method is assessed separately, at the primary history, using:

| Requirement | Frozen gate |
|---|---|
| Calibration | Every planned calibration seed/lane and both directions available |
| Positive evaluation | All 20 planned seeds, both eligible directions |
| Null evaluation | All 60 separate planned null seeds and supported directions |
| Mean directional top-k recall | At least 0.75 |
| Mean directional driver nomination recall | At least 0.75 |
| Mean directional nomination precision | At least 0.75 |
| Mean per-origin intervention-effect MAE | At most 0.05 probability units |
| One-sided exact 95% upper bound on any false nomination | At most 0.05 |

Directions are averaged within process seed before summarizing seeds. Top-k uses
the gold neuron count and the existing tie-aware rule. Nomination recall concerns
intervention-positive neurons; nomination precision concerns all nominated
candidate units/groups. A direction with no nominations has precision zero in
the positive test, so an empty rule cannot pass by vacuity. Missing/ineligible
seed metrics suppress complete-set means rather than silently changing denominators.
Precision, recall and MAE gates make the operational claim more explicit than
ranking alone; they are new prospective gates for this revised control and do not
retroactively relax or replace the old benchmark's criteria.

Known support must pass the same primary readout/null gates. Empty support must
produce identically zero outgoing-effect estimates and no nominations in every
primary positive/null cell. A native method is supported only if its own gates
and both controls pass. `supported_native_methods` lists them; the all-four
`bounded_control_replication_passed` flag requires all four native methods. These
flags concern this favorable control only. Broad `scientific_validation_passed`
and `biological_necessity_validated` remain false.

For the null, one seed contributes one event: any false nominee across candidates
and supported directions. Report the one-sided Clopper–Pearson 95% upper bound
`BetaQuantile(0.95; k+1,n-k)` (one if `k=n`) only on the complete planned test set
[Clopper and Pearson](https://doi.org/10.1093/biomet/26.4.404). It is per method, not neuron-level FDR or simultaneous
method coverage. Zero of sixty gives an upper bound about 0.04870; observed
nonzero failure can prevent the gate from passing. No confidence bound is inferred
from treating directions, windows or Monte Carlo draws as independent trials.

Report raw seed values, means, sample SDs and 2,000-resample descriptive seed-mean
bootstrap intervals [Efron](https://doi.org/10.1214/aos/1176344552). The four native paired 1,024-minus-256 MAE sign tests
form one prespecified Holm family [Holm](https://www.jstor.org/stable/4615733). Missing method contrasts remain in the
family as p=1. Exact-zero differences are ties; sign tests concern tendency rather
than effect magnitude. These secondary p-values cannot override primary gates.

## Randomness, missingness and interpretation

The pinned conventional CPU backend accepts a public `random_state` argument,
but does not forward it to CPU permutation draws in its Numba/executor path.
There is no deterministic public CPU option at the pinned revision. This study
does not monkeypatch or replace the discovery algorithm. Conventional estimator
randomization is part of the fixed procedure. Exchangeability of those randomized
scores is an assumption; controlled or demonstrably independent RNG streams are
not claimed. First completed evidence is retained, including across restarts,
and unfavorable results are not recomputed for selection.

If the revised method fails, document which gates fail before deciding whether
graph similarity addresses the failure. Clustering cannot reconstruct interaction
sign or strength absent from unsigned support, and it is not a remedy for an
uncalibrated nomination rule. This protocol does not authorize mixing graph
comparison results into the fresh-seed replication. Separate graph-similarity
work may follow the resulting methodological decision.

## Running and reviewing

From the application repository:

```sh
# Inventory only; creates no fit outputs.
.venv/bin/python -m synthetic.mechanism_replication --profile replication
# Full frozen study with visible streaming progress.
.venv/bin/python -u -m synthetic.mechanism_replication --profile replication --workers 4 --run
# Read the latest atomic report without fitting.
.venv/bin/python -m synthetic.mechanism_replication --profile replication --status
```

Notebook `synthetic/archive/07_mechanism_replication.ipynb` is ready to run with fitting
enabled and the full profile. Notebooks 00–06 and their saved results are left
unchanged. The final analysis contains primary gate tables, seed-level CSVs,
paired effects and source/protocol fingerprints. The previous result collation
remains the pre-replication record; new results must be labeled separately.

Verified citations and claim limits are in
[mechanism_replication_references.bib](mechanism_replication_references.bib) and
[mechanism_replication_citation_audit.md](mechanism_replication_citation_audit.md).
The original estimand rationale is in
[driver_methodology_revision.md](driver_methodology_revision.md).
