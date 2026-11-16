# Mechanism-readout development diagnostic

Execution complete: True. Available method/recording runs: 12/12 attempted.

This is a model-matched, fully observed positive/null control. Known support is an oracle diagnostic; estimated support comes from the imported CSL packages. It does not replace the completed failed latent-bistable driver study or establish biological necessity.

| Lane | Available/planned | Driver effect MAE | Driver top-k recall | Null false-nomination seeds | Null 95% upper bound |
|---|---:|---:|---:|---:|---:|
| known_support | 6/6 | 0.0014 | 1.0000 (3/3) | 0/3 | 0.6316 |
| empty_support | 6/6 | 0.1133 | 0.1250 (3/3) | 0/3 | 0.6316 |

Effect MAE averages per-origin absolute errors in transition-probability units. Top-k recall uses the number of intervention-positive neurons and averages ties fairly. Each direction is first summarized within its raw-process seed. The null bound concerns any false nominee across units and directions per seed/method, with no simultaneous coverage across methods.

Ranking entries show eligible/planned independent-seed counts. Aggregate top-k recall is suppressed if any planned seed lacks both eligible directions.

The probability-effect nomination cutoff is a provisional development setting, not the old omission-nat cutoff or an independently calibrated significance rule. Few null seeds cannot establish a 5% false-nomination guarantee. Native CPU permutation draws are not fully reproducible upstream; saved graphs/evidence are retained.

Protocol, mechanisms, noise estimates, paired per-origin probabilities and unavailable reasons are in mechanism-validation.json and cells/. Rationale and verified references: docs/driver_methodology_revision.md and docs/driver_methodology_citation_audit.md.
