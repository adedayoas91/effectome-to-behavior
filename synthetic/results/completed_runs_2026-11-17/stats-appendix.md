# Statistical specification and limits

This collation reads completed `faster-machine-03` artifacts; it fits no models. Counts and definitions are traceable through the [framework](audits/framework_review.json), [driver](audits/driver_review.json), and [mechanism](audits/mechanism_review.json) audits. Availability means execution succeeded, not scientific acceptance.

## Replication and complete denominators

| Stage | Completed / planned | Independent unit and denominator |
|---|---:|---|
| Designed controls, notebooks 00/03 | Four scenarios | One raw-process seed 42; 384 held-out samples per scenario. Twenty circular shifts per scenario are same-recording surrogates. |
| Native discovery sanity | 24/24 method fits | Three raw-process seeds per chain/null condition; four paired GC variants form one discovery family. |
| Shared predictive comparison | 4/4 methods | One seed 42; 164 training and 60 held-out origins, including 30 exits. |
| Framework smoke | 12/12 cells; 48/48 method runs | Three seeds per each of four specifications; four paired methods; 16 planned contrasts. |
| Frozen latent driver benchmark | 140/140 cells; 560/560 method runs | Twenty seeds per each of four non-null scenarios; 60 null seeds. All 1,120/1,120 direction rows are available and eligible. |
| Mechanism smoke and two history reports | 2/2 cells and 12/12 method runs per report | One seed 20 per driver/null scenario; six lanes. All three reports reuse that seed. |

Origins, neurons, edges, directions, trials, Monte Carlo rollouts, method variants and reruns do not increase seed replication. Directions are averaged within a driver seed. Framework region labels are local to each seed: numeric transition codes must not be pooled across seeds. Its switching scenarios score structural recovery on 33/60 pure-history evaluation origins; mixed histories remain in the prediction denominator. Fixed/stationary controls use 32/32 and 60/60 respectively.

## Framework summaries and inference

Gain is baseline minus graph-augmented held-out log loss, in nats; negative values remain in every summary. Mean, sample SD, median, individual seed values and saved 95% percentile intervals are retained. Intervals resample independent seeds 2,000 times, analysis seed 42; they do not resample overlapping origins. Bootstrap attribution follows [Efron (1979)](https://doi.org/10.1214/aos/1176344552). With only three seeds, empirical support is sparse and interval coverage is unstable; more resamples add no independent observations.

The existing two-sided exact sign test tests equal probabilities of positive and negative gain after excluding **exact zero** ties. It tests sign tendency, not mean magnitude, and uses no Gaussian assumption or normality pretest. Raw p-values are 0.25 or 1; all 16 [Holm-adjusted](https://www.jstor.org/stable/4615733) p-values are 1. No superiority or equivalence claim follows. Existing positive fractions count nonzero roundoff gains near 1e-14; these values are preserved but have no substantive predictive interpretation. All requested smoke fits completed; the distinct 7,680-cell manuscript matrix did not.

## Frozen driver acceptance

The protocol is frozen development work, not external preregistration. Mean and sample SD use independent seed means; stored descriptive percentile intervals use 2,000 seed resamples. Required eligibility is at least 20 origins, five events and alignment 0.80 per direction. Gates require availability ≥0.90; mean predictive gain ≥0; single-driver and bystander top-k recall ≥0.75; or redundant-group win fraction ≥0.75 with positive group advantage. Gold effects require paired simulator probability reduction ≥0.10. Gold-sized top-k ranking is distinct from operational nomination at omission score >0.01 nats. Hidden confounding is diagnostic, not a required pass.

Null failure is **any false candidate/group in either direction of one seed**. For k failures among n=60 seeds, the one-sided exact 95% upper bound is Beta quantile(0.95; k+1, n−k), or 1 when k=n. [Clopper–Pearson (1934)](https://doi.org/10.1093/biomet/26.4.404) supplies binomial-tail inversion; coverage is per method, not simultaneous or neuron-level FDR. Counts are 50, 20, 42 and 24/60 for c-GC, c-GC*, fast and fast*: bounds 0.906693, 0.446466, 0.796294 and 0.514351, all above the 0.05 gate. All methods fail overall acceptance. Top-k is unavailable when no gold neurons exist; those rows are not zero-recovery observations.

## Mechanism and historical evidence

History 256 versus 1024 pairs the same seed, training inputs, evaluation initial states and oracle outputs. Each scenario has 12 held-out origins; effect MAE averages eight interventions × 12 origins. Top-k averages eligible directions, using expected recall for exact ties; an all-zero ranking gives 1/6. Null top-k remains unavailable. With n=1 there is no between-seed SD, process-level CI or p-value. Zero false nominations in one null seed has one-sided upper bound 0.95, not established specificity. Same-seed historical reruns and archived seed-20–22 controls are not new confirmation or extra current replicates. No stage establishes biological necessity.

Verified reference keys and retrieval limits are in [analysis-references.bib](analysis-references.bib) and [citation-audit.md](citation-audit.md).
