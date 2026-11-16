# First mechanism-readout development results

Date: 15 November 2026. These are development diagnostics, not independent
confirmation or a replacement for the completed failed bistable benchmark.
The methodological reasons, verified references and assumptions are in
[the decision record](driver_methodology_revision.md) and
[citation audit](driver_methodology_citation_audit.md).

## Readout recovery with known support

With supplied generating support, the learned nonlinear mechanism recovered
the intervention-positive driver ranking in both directions for each of three
development seeds (20, 21, 22). These recordings have eight observed nodes and
128 trials, split into 64 training and 64 held-out trials. Mean per-origin
probability-effect absolute error, averaged across units and then seeds, was
0.001352. No false nominee occurred in the three null seeds. The exact one-sided
95% upper bound is nevertheless **0.632**, not a demonstrated 5% guarantee.

The empty-support control produced exactly zero outgoing-disconnection effects;
its tie-aware top-k recall was 0.125 (one of eight neurons in expectation).
The learned lane received known adjacency but did not receive generating
coefficients or innovation scales. This is an oracle-support, model-matched
operating case for parameter learning and rollout attribution.

## Why the initial native-support smoke failed

The reviewed 256-sample smoke fit used nine inner-training histories per native
method. Endpoint-edge detections before the fixed 0.5 aggregation were:

| Method | Source 0 to endpoint | Source 1 to endpoint |
|---|---:|---:|
| Conventional c-GC | 3/9 | 3/9 |
| Conventional c-GC* | 1/9 | 4/9 |
| fastC-GC | 0/9 | 0/9 |
| fastC-GC* | 0/9 | 0/9 |

Conventional detections were removed by the aggregation rule; the fast methods
rejected the edges in the marginal screen before aggregation. Saved lag-one
marginal/conditional evidence reproduces the native graph decisions under the
declared alpha=0.01 and beta=0.005. There is no indicated source/target extraction
error. With no directed path to the endpoint, modeled outgoing effects are zero,
which explains the equality to the empty-support lane. This diagnoses this
control; it does not identify the cause of every failure in the older benchmark.

## Paired history-length experiment

To test resolution/power without relaxing thresholds, we compared 256- and
1,024-sample training histories on **identical full recording arrays**. Both
cases use seed 20, six nodes, 24 independent trials of 1,152 samples, nine
inner-training discovery trials after 128 burn-in samples, three inner-validation
trials, and 12 held-out trials. Mechanism fitting uses the same 12 complete outer
training trials in both cases. The observed endpoint, zero threshold, eight-step
forecast horizon, trailing four-sample target, paired random streams, regularization
grid, alpha, beta and 0.5 support quorum are unchanged. At 10 Hz, the new control's
forecast horizon is 0.8 seconds; this is not the older 3.2-second/PCA target.

| Lane | Top-k recall, history 256 | Top-k recall, history 1,024 | Effect MAE, history 1,024 |
|---|---:|---:|---:|
| Known support | 1.000 | 1.000 | 0.001831 |
| Empty support | 0.167 | 0.167 | 0.140055 |
| Conventional c-GC | 0.167 | 1.000 | 0.001831 |
| Conventional c-GC* | 0.167 | 1.000 | 0.001831 |
| fastC-GC | 0.167 | 1.000 | 0.001831 |
| fastC-GC* | 0.167 | 1.000 | 0.001831 |

The 0.167 values are tie-aware chance expectations, not positive driver findings.
All four imported methods recovered the generating support in the longer-history
driver control. There were no false nominees in the single null seed for either
history setting, but its exact upper bound is **0.95**. Both settings and all
negative outcomes are retained. The two history settings are paired observations
of the same seed, not two independent replications; the related methods are not
independent causal confirmations.

Longer histories are compatible with this control's fixed mechanism. They are
not automatically suitable for rapidly switching biological circuits. Further
development should vary history and persistence jointly, quantify graph recovery,
then add function-class mismatch, redundant drivers and observation stresses.
Choose and freeze the new protocol before using fresh confirmation seeds.

## Reproduction and audit trail

The versioned evidence is in
[`synthetic/results/mechanism_development_2026-11-15/`](../synthetic/results/mechanism_development_2026-11-15/).
It retains reviewed smoke results, the three-seed known-support study, both
history settings, original native support/evidence, source/locks, per-origin
probabilities, negative results and numerical summaries. Its provenance verifies
the implementation fingerprint and file checksums. The original failed study
remains in `synthetic/results/driver_confirmation_v2/` with its original hash.

```bash
.venv/bin/python pipeline/15_mechanism_validation.py --profile smoke
.venv/bin/python pipeline/15_mechanism_validation.py --profile development --oracle-only --run \
  --output-dir outputs/driver-mechanism-validation/development-oracle
.venv/bin/python -m synthetic.mechanism_history_diagnostic
.venv/bin/python -m synthetic.mechanism_history_diagnostic --run
```

Inventory commands do not run fits. Existing directories reject a different
protocol/source identity; use a new output directory after algorithm changes.
Native conventional permutation draws are not fully seeded upstream, so graph
reruns may differ; the evidence archives preserve the actual realization.

An initial pre-review prototype is retained locally under
`outputs/driver-mechanism-validation/smoke-initial-pre-review/`, with its source
archive and a note about the corrected rollout-seed/reporting issues. The
versioned results above use the reviewed implementation. Source code, tests and
all reference records remain separate from simulation replication units.
