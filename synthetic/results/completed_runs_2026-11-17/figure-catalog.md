# Figure catalog

Each PNG has a matching PDF. Statistical units and references are in the [appendix](stats-appendix.md); [audit JSONs](audits/) preserve numerical lineage. Method order is c-GC, c-GC*, fastC-GC, fastC-GC*, followed or preceded by control lanes where shown.

## 01 — Framework prediction gains

Files: [PNG](figures/01-framework-gains.png), [PDF](figures/01-framework-gains.pdf).

**Purpose.** Show prediction gains over the common activity/history baseline and the full three-seed distribution.

**Source.** The 48 smoke records and 16 saved summaries in [framework_review.json](audits/framework_review.json). Four scenario panels show three seed points, mean and saved percentile 95% interval per method; zero marks equal log loss. Retain negative and near-zero gains.

**Caption.** “Graph augmentation gains in held-out transition log loss (baseline minus augmented, nats; positive favors augmentation). Points show independent raw-process seeds 0–2; summaries show means and existing 2,000-resample percentile 95% intervals. All 48/48 method runs are available. All 16 two-sided sign-test contrasts have Holm-adjusted p=1. Intervals at n=3 are exploratory.”

**Observation.** Every mean is negative in both switching scenarios. Stationary c-GC* and fastC-GC means are 0.000755 and 0.006930 nats; fastC-GC* fixed/stationary gains are roundoff-scale. One-to-one c-GC gain is −0.372170 ± 0.407253 SD nats.

**Decision and limits.** Retain heterogeneous and often adverse changes as development evidence. Three seeds support no winner or equivalence claim. Correspondence and predictive gain are different outcomes; manuscript completion, intervention recovery and biological claims remain unvalidated.

## 02 — Frozen driver gates

Files: [PNG](figures/02-driver-gates.png), [PDF](figures/02-driver-gates.pdf).

**Purpose.** Show the failed frozen recovery and false-nomination gates.

**Source.** Acceptance summaries in [driver_review.json](audits/driver_review.json). Three panels show single-driver top-k, bystander top-k and redundant-group win means with stored percentile 95% intervals, n=20, and gate 0.75. The fourth shows null fractions and one-sided exact 95% upper bounds, n=60, against maximum 0.05. Distinguish bounds from bootstrap intervals.

**Caption.** “Frozen benchmark recovery and null gates. Positive panels average two directional outcomes within each independent seed before summarizing 20 seeds; bars show stored 2,000-resample percentile 95% intervals. The null panel shows the fraction of 60 seeds with any false candidate/group in either direction and its per-method one-sided Clopper–Pearson 95% upper bound. Dashed lines mark 0.75 recovery and 0.05 null-bound gates. All 560/560 runs and 1,120/1,120 direction rows are available and eligible.”

**Observation.** Single-driver means are 0.325, 0.150, 0.200 and 0.025; bystander means are 0.050, 0.025, 0.200 and 0.075; redundant-group wins are 0.200, 0.325, 0.150 and 0.225. Every value is below 0.75. Null upper bounds span 0.446466–0.906693, all exceeding 0.05.

**Decision and limits.** No method passes. Additional mean-gain and group-advantage gates also fail. Gold-sized top-k differs from fixed-cutoff nomination. Hidden confounding is diagnostic and omitted here. Observational alignment does not establish intervention invariance; null bounds provide neither simultaneous method coverage nor neuron-level FDR.

## 03 — Paired mechanism history diagnostic

Files: [PNG](figures/03-mechanism-history.png), [PDF](figures/03-mechanism-history.pdf).

**Purpose.** Show discovery-history sensitivity on one fully observed mechanism.

**Source.** Paired-history records in [mechanism_review.json](audits/mechanism_review.json). Group bars for histories 256 and 1024 within known support, empty support and four native lanes. Show driver top-k and effect MAE without error bars: both histories reuse seed 20.

**Caption.** “Paired discovery-history diagnostic, one raw-process seed 20. Training/evaluation input identities and oracle outputs are shared between histories. Paired bars show the same method at histories 256 and 1024; panels show mean directional top-k recall and effect MAE over eight interventions × 12 held-out origins. No process-level uncertainty or p-value is estimable from this seed.”

**Observation.** Each native lane changes from top-k 1/6 to 1 and effect MAE 0.140055 to 0.001831. Known support stays at top-k 1 and MAE 0.001831; empty support stays at 1/6 and 0.140055. Thus the longer-history native result matches the known-support control on this input.

**Decision and limits.** Independent-seed replication is required. Only known support receives simulator adjacency; native lanes discover support and fit parameters from training data. This observed-coordinate additive-tanh model does not replace the latent bistable/PCA benchmark. Null top-k is unavailable; 0/1 null false nominations has upper bound 0.95. Calibration and scientific/biological validation remain false; archived reruns add no replication.
