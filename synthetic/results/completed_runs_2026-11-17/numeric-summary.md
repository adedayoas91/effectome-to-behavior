# Numeric summary: completed synthetic notebooks, 17 November 2026

Run label `faster-machine-03`. Values below are descriptive unless the test/interval is explicitly named. CSV tables retain full floating-point precision. Method entries, directions, trials and overlapping windows are not independent seed replicates.

## Completion

| Stage | Available/planned | Raw seeds per scenario | Scope |
| --- | --- | --- | --- |
| 01 native sanity | 24/24 | 3 | chain/null technical control |
| 02 shared transition example | 4/4 | 1 | development example |
| 04 framework smoke | 48/48 | 3 | four scenarios; publication matrix not executed |
| 05 driver recovery | 560/560 | 20 positive/diagnostic; 60 null | frozen criteria; all methods fail |
| 06 mechanism smoke | 12/12 | 1 | development seed 20 reused; includes four oracle/control entries |
| 06 mechanism history-256 | 12/12 | 1 | development seed 20 reused; includes four oracle/control entries |
| 06 mechanism history-1024 | 12/12 | 1 | development seed 20 reused; includes four oracle/control entries |

## Shared transition example (one seed)

Gain = activity-only baseline log loss minus graph-augmented log loss, in nats; larger is better.

| Method | Baseline | Augmented | Gain | Occupied held-out graph states |
| --- | --- | --- | --- | --- |
| conventional_cgc | 1.393110 | 1.482440 | -0.089330 | 2 |
| conventional_cgc_star | 1.393110 | 1.463098 | -0.069988 | 2 |
| fast_cgc | 1.393110 | 1.381873 | +0.011236 | 1 |
| fast_cgc_star | 1.393110 | 1.362782 | +0.030328 | 1 |

## Framework smoke: all 16 contrasts

Independent n=3 raw process seeds per row. SD is the sample SD. Intervals are the existing 2,000-resample percentile bootstrap of seed means: coverage is unstable at n=3. All Holm-adjusted p=1.000; no corrected evidence for improved prediction.

| Scenario | Method | Mean gain ± SD (nats) | Descriptive 95% bootstrap CI | Holm p |
| --- | --- | --- | --- | --- |
| matched_marginals | conventional_cgc | -0.165287 ± 0.142596 | [-0.251147, -0.000683] | 1.000 |
| matched_marginals | conventional_cgc_star | -0.058948 ± 0.039813 | [-0.104881, -0.034337] | 1.000 |
| matched_marginals | fast_cgc | -0.024108 ± 0.038167 | [-0.059777, +0.016143] | 1.000 |
| matched_marginals | fast_cgc_star | -0.021977 ± 0.019299 | [-0.036162, +0.000000] | 1.000 |
| stationary | conventional_cgc | -0.027961 ± 0.042584 | [-0.073176, +0.011382] | 1.000 |
| stationary | conventional_cgc_star | +0.000755 ± 0.045046 | [-0.051224, +0.028394] | 1.000 |
| stationary | fast_cgc | +0.006930 ± 0.012003 | [-0.000000, +0.020790] | 1.000 |
| stationary | fast_cgc_star | +0.000000 ± 0.000000 | [-0.000000, +0.000000] | 1.000 |
| fixed_circuit | conventional_cgc | -0.003189 ± 0.002491 | [-0.004706, -0.000314] | 1.000 |
| fixed_circuit | conventional_cgc_star | -0.000406 ± 0.000702 | [-0.001217, +0.000000] | 1.000 |
| fixed_circuit | fast_cgc | -0.001878 ± 0.002454 | [-0.004706, -0.000314] | 1.000 |
| fixed_circuit | fast_cgc_star | +0.000000 ± 0.000000 | [+0.000000, +0.000000] | 1.000 |
| one_to_one | conventional_cgc | -0.372170 ± 0.407253 | [-0.798940, +0.012257] | 1.000 |
| one_to_one | conventional_cgc_star | -0.038429 ± 0.089100 | [-0.137601, +0.034872] | 1.000 |
| one_to_one | fast_cgc | -0.073192 ± 0.117622 | [-0.208869, +0.000000] | 1.000 |
| one_to_one | fast_cgc_star | -0.005182 ± 0.008976 | [-0.015547, +0.000000] | 1.000 |

Rounded zeros include floating-point residuals; exact original values and sign-test decisions are retained in CSV.

## Frozen driver-recovery criteria

Positive scenarios have n=20 independent process seeds, averaging the two directions within each seed. Minimum recovery/group-win gate=0.75; null upper-bound gate≤0.05. Hidden confounding is diagnostic.

| Method | Single top-k recall | Bystander top-k recall | Redundant group win | Overall |
| --- | --- | --- | --- | --- |
| conventional_cgc | 0.325 | 0.050 | 0.200 | fail |
| conventional_cgc_star | 0.150 | 0.025 | 0.325 | fail |
| fast_cgc | 0.200 | 0.200 | 0.150 | fail |
| fast_cgc_star | 0.025 | 0.075 | 0.225 | fail |

Individual top-k is undefined for the redundant pair under the frozen intervention threshold; the group-win metric is its prespecified criterion. All required positive scenarios also have negative mean predictive gain. Full means, SDs and descriptive intervals are in `tables/driver-summary.csv`.

| Method | Null seeds with any false nominee | Rate | One-sided 95% exact upper bound |
| --- | --- | --- | --- |
| conventional_cgc | 50/60 | 0.833333 | 0.906693 |
| conventional_cgc_star | 20/60 | 0.333333 | 0.446466 |
| fast_cgc | 42/60 | 0.700000 | 0.796294 |
| fast_cgc_star | 24/60 | 0.400000 | 0.514351 |

## Model-matched mechanism development

One existing independent raw seed (20) per scenario, reused in all panels; six lanes include known- and empty-support controls. Top-k recall is tie-aware and averaged over two directions, not two independent replicates. Effect MAE measures error in oracle probability reductions; factual Brier scores predicted probabilities against observed binary transitions. Lower is better for both.

| Panel | Method | Driver top-k | Effect MAE | Factual Brier | Support recall |
| --- | --- | --- | --- | --- | --- |
| smoke | known_support | 1.000000 | 0.003011068 | 0.075078328 | 1.000 |
| smoke | empty_support | 0.166667 | 0.146972656 | 0.795562744 | 0.000 |
| smoke | conventional_cgc | 0.166667 | 0.146972656 | 0.795562744 | 0.500 |
| smoke | conventional_cgc_star | 0.500000 | 0.087076823 | 0.406508128 | 0.750 |
| smoke | fast_cgc | 0.166667 | 0.146972656 | 0.795562744 | 0.500 |
| smoke | fast_cgc_star | 0.166667 | 0.146972656 | 0.795562744 | 0.500 |
| history-256 | known_support | 1.000000 | 0.001831055 | 0.087547302 | 1.000 |
| history-256 | empty_support | 0.166667 | 0.140055339 | 0.717281342 | 0.000 |
| history-256 | conventional_cgc | 0.166667 | 0.140055339 | 0.717281342 | 0.500 |
| history-256 | conventional_cgc_star | 0.166667 | 0.140055339 | 0.717281342 | 0.500 |
| history-256 | fast_cgc | 0.166667 | 0.140055339 | 0.717281342 | 0.500 |
| history-256 | fast_cgc_star | 0.166667 | 0.140055339 | 0.717281342 | 0.500 |
| history-1024 | known_support | 1.000000 | 0.001831055 | 0.087547302 | 1.000 |
| history-1024 | empty_support | 0.166667 | 0.140055339 | 0.717281342 | 0.000 |
| history-1024 | conventional_cgc | 1.000000 | 0.001831055 | 0.087547302 | 1.000 |
| history-1024 | conventional_cgc_star | 1.000000 | 0.001831055 | 0.087547302 | 1.000 |
| history-1024 | fast_cgc | 1.000000 | 0.001831055 | 0.087547302 | 1.000 |
| history-1024 | fast_cgc_star | 1.000000 | 0.001831055 | 0.087547302 | 1.000 |

For each native method, paired history 1,024 vs 256 gives Δtop-k=+0.833333, Δeffect MAE=−0.138224284 (98.6926% reduction) and ΔBrier=−0.629734039. The paired panels use the same longer recordings, origins and rollout noise; smoke uses a different recording length and is not a controlled history contrast. All null lanes have zero false nominees in one seed; their one-sided 95% exact upper bound is 0.95. These data cannot establish a 5% false-nomination guarantee.

See [stats-appendix.md](stats-appendix.md) for tests and uncertainty limits and [analysis-report.md](analysis-report.md) for interpretation.
