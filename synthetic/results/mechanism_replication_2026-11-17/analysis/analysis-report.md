# Fresh-seed mechanism replication

Stage: complete. Completed cells: 119/119. Available lanes: 714/714.

Calibration uses separate global-null seed maxima across all candidates and both directions. The strict-exceedance cutoff is the maximum of the finite-sample order statistic and the existing 0.10 minimum effect size. Calibration is frozen before test jobs are submitted. Protection is marginal under exchangeability, per method under the global null; it is not strong non-null FDR control or simultaneous method coverage.

| Method | Cutoff | Driver top-k | Nomination recall | Nomination precision | Effect MAE | Null false seeds | Null one-sided 95% upper | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| known_support | 0.100000 | NA | NA | NA | 0.002087 | 0/60 | 0.048703 | not passed |
| empty_support | 0.100000 | NA | NA | NA | 0.131191 | 0/60 | 0.048703 | not passed |
| conventional_cgc | 0.100000 | NA | NA | NA | 0.002087 | 0/60 | 0.048703 | not passed |
| conventional_cgc_star | 0.100000 | NA | NA | NA | 0.002087 | 0/60 | 0.048703 | not passed |
| fast_cgc | 0.100000 | NA | NA | NA | 0.002087 | 0/60 | 0.048703 | not passed |
| fast_cgc_star | 0.100000 | NA | NA | NA | 0.002087 | 0/60 | 0.048703 | not passed |

Supported native methods (including control requirements): [].
All-four bounded-control replication passed: False.

Acceptance requires complete calibration, at least 20 positive and 60 separate null seeds, mean top-k/nomination recall/nomination precision >=0.75, mean effect MAE <=0.05, and a one-sided 95% null-event upper bound <=0.05. Empty positive nominations have precision zero; missing or ineligible seed metrics suppress complete-set aggregates. Pilot runs cannot satisfy the full seed-count gates.

## Paired history comparison

| Method | Paired seeds | Mean MAE change (1024 - 256) | Mean top-k change | Holm MAE sign-test p |
|---|---:|---:|---:|---:|
| known_support | 20 | 0.000000 | NA | NA |
| empty_support | 20 | 0.000000 | NA | NA |
| conventional_cgc | 20 | -0.084267 | NA | 0.000031 |
| conventional_cgc_star | 20 | -0.083144 | NA | 0.000031 |
| fast_cgc | 20 | -0.125372 | NA | 0.000008 |
| fast_cgc_star | 20 | -0.125372 | NA | 0.000008 |

History 256 uses the same full training/evaluation inputs as history 1024 for each driver seed. It is a paired sensitivity diagnostic; it has no separately calibrated nomination or null claim. Negative MAE changes favor the longer history. The four native paired sign tests form one fixed Holm family; seed-bootstrap intervals in JSON are descriptive.

## Limits

Conventional CPU permutation randomness is uncontrolled by the public upstream seed API. The fixed procedure retains first completed native evidence and never selects favorable reruns. Exchangeability includes estimator randomization; byte-identical rediscovery is not promised.

This is a favorable fully observed, model-matched control with a predeclared observed endpoint. It does not overturn the old latent-bistable benchmark, validate learned manifold alignment, or establish biological necessity. No graph-similarity extension is included. Protocol and citations: docs/mechanism_replication_protocol.md and its reference/audit files.
