# Evidence map for the methodological development section

Suggested placement: `\input{sections/methodological_development}` after
`\input{sections/validation}` and before limitations in `main.tex`.

The section is a development decision record. The first five trials are
completed; the sixth now contains a completed exploratory scalar-score and
PCA-dimension diagnostic followed by proposed broader encoder studies.
Repeating the five inspected notebook-08 seeds does not increase the independent
sample size. No old acceptance gate is changed.

Paths below are relative to the inner application repository
`effectome-to-behavior/`.

| Trial | Evidence and numerical checks | Interpretation boundary |
|---|---|---|
| 1. Native recovery and incremental prediction | `synthetic/results/completed_runs_2026-11-17/tables/native-sanity.csv`; `tables/framework-contrasts.csv`; 24/24 technical recovery entries; framework 3 seeds × 4 scenarios × 4 methods = 48, all 16 Holm p-values 1 | Technical controls do not validate the behavioral endpoint; a three-seed pilot does not demonstrate absence of useful predictive information |
| 2. Conditional feature omission | Same bundle's driver tables/audit and `manuscript-260926/sections/driver_methodology_revision.tex`; 560 completed method/recording runs, all four frozen decisions fail | Predictive omission and structural disconnection differ; mismatch is not identified as the sole failure cause |
| 3. Explicit mechanism and paired history | Existing `sections/mechanism_development_results.tex`; development seed 20, all native 256-history ranking fails and 1024-history ranking succeeds; long-history MAE 0.001831, empty support 0.140055 | One-seed, fully observed, model-matched operating case; no arbitrary biological history-length recommendation |
| 4. Fresh calibrated replication | `synthetic/results/mechanism_replication_2026-11-17/README.md`, `analysis/replication-results.tex`, `replication-report.json`, `analysis/eligibility-diagnostics.json`; 119 cells, 714 lanes, 4284 native window fits; MAE 0.002087 ± 0.000805; 0/60 false-event seeds, exact upper 0.048703 per method; eligibility 19/20; source-1 joint effect 0.230469 at seed 20012 | Frozen complete acceptance remains false; predeclared joint candidate is not inferred group discovery; null coverage is per method |
| 5. Learned endpoint | `outputs/learned-manifold-validation/development/endpoint-bridge-03/manifold-validation.json`, `analysis/seed-metrics.csv`, and its independent verification; 10 cases, 70 lanes, 360 native fits, 40/40 supports equal known; three-way MAEs match the manuscript table | Five development seeds, no new calibration or confirmation; full-neural success is not a successful compressed manifold |
| 6. Behavior-prioritized capacity diagnostic | `synthetic/results/behavior_prioritized_probe_2026-11-18/`, including `probe-summary.json`, `probe-protocol.json` and `seed-metrics.csv`; same five inspected development seeds, 10 cases, 70 saved dynamics parameter sets; no new native/mechanism fits; zero parent-label disagreements on 276480 natural neural observations and 1935360 paired-rollout binary events; scalar-score driver/null behavioral MAE 0.001969/0.000073; disclosed PCA k=1..5 driver MAE 0.146908/0.138818/0.138875/0.138818/0.018994 | Algebraic re-expression of the same binary linear readout, not an independent learning improvement or new evidence for intrinsic dimension, reduced sensing, manifold geometry or latent dynamical closure; all full neural inputs and dynamics remain necessary; nonlinear studies remain proposed |

Citation keys already present in the manuscript bibliography:
`adedayo2025cgc`, `adedayo2026fastcgc`, `williamson2023importance`,
`pearl2010causalinference`, `hoerl1970ridge`, `cawley2010overfitting`,
`clopper1934confidence`, `schneider2023cebra`.

The integration additionally uses verified keys `jolliffe2016pca`,
`sani2021psid`, `le2018supervisedautoencoders` and `kumar2023bundleV5` now present
in the manuscript bibliography. The v5-specific BunDLe key avoids mixing
earlier-version title/author metadata with the current preprint. No
bibliography metadata was invented here.

Additional Trial-6 checks:

- In the driver cases, the cumulative four-component PCA explained-variance
  ratios in the saved endpoint artifacts average 0.9044860174101181. The
  statement concerns standardized **neural training variance**, not behavioral
  variance explained or interventional sufficiency.
- Scalar-score equality is with the successful parent full-neural decoder,
  rather than perfect equality with the independently defined behavioral truth.
  The nonzero parent behavioral errors are retained.
- Future targets average the affine decision score and apply the same `>= 0`
  convention. Averaging sigmoid probabilities would be a different endpoint.
- A distributed reference containing the neural bystander requires a newly
  verified null; the present last-coordinate-null generator is not silently
  reused as a behavioral null for an arbitrary new readout.
