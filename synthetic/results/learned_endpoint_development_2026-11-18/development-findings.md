# The mechanism survives; the first compressed endpoint does not

Development evidence, 18 November 2026. The preceding completed replication,
reports and the original [justified plan](learned_manifold_validation_plan.md)
were committed as `fac7ea4` before implementing this experiment. Graph similarity
remains separate. These findings do not change the completed study's gates.

## What was actually run

The initial linear bridge completed **10/10 cases**, comprising driver and
endpoint-null scenarios on **five process seeds, 40000–40004**. Every case ran
the true simulator, fitted known/empty support and all four native support
methods: **70/70 dynamics lanes available, 360 native window fits**. Each neural
fit and its paired futures were reused across seven endpoints. Scaler/PCA and
regions used inner-training trials; behavioral decoders selected regularization
on inner-validation trials and refit only the decoder on outer training.

This is an engineering/development study. Methods, endpoints, trials, origins and
candidate units do not increase the five-seed replication count. No new
calibration or confirmation pool was used. The independent behavioral reference
remains the generator's specified observed endpoint, so this is the initial
scalar-behavior bridge, not yet the distributed/nonlinear main experiment.

The [frozen evidence bundle](../synthetic/results/learned_endpoint_development_2026-11-18/README.md)
contains the executed source, native evidence, frozen endpoint artifacts,
per-rollout events, numerical tables and figures. Original output was
`outputs/learned-manifold-validation/development/endpoint-bridge-01/`.

## Result and interpretation

Mean absolute probability-effect errors below average absolute errors across
the same origins/candidates within each seed, then average the five seeds.
The native-method rows are identical for all four methods in this control; all
**40/40 native consensus supports equal the known support**.

| Scenario / endpoint | True simulator semantic MAE | Native within-chart MAE | Native behavioral MAE |
|---|---:|---:|---:|
| Driver / declared coordinate reference | 0.000000 | 0.001912 | 0.001912 |
| Driver / PCA geometric regions | 0.130737 | 0.002018 | 0.131144 |
| Driver / PCA plus behavioral decoder | 0.138428 | 0.002181 | 0.138818 |
| Driver / full-neural behavioral decoder | 0.000553 | 0.002067 | 0.001969 |
| Null / PCA geometric regions | 0.019238 | 0.002043 | 0.019295 |
| Null / PCA plus behavioral decoder | 0.002832 | 0.001278 | 0.003149 |
| Null / full-neural behavioral decoder | 0.000049 | 0.000122 | 0.000073 |

The central diagnostic is clear: the fitted mechanism agrees closely with the
simulator through each frozen PCA chart, yet the chart measures a different
interventional outcome from the intended behavior. Known support has the same
failure, while full-neural behavioral decoding gives low error. This points to
the compression/learning objective or endpoint construction, rather than graph
discovery, as the immediate bottleneck in this fully observed control.

Behavioral supervision applied only **after** an unsupervised two-dimensional
PCA projection did not repair that projection. Its natural future-window
classification accuracy was about **90.0%**, but factual-origin accuracy was
**46.7%**, and its behavioral effect MAE remained **0.1388**. Natural agreement
over long stationary trial segments cannot replace source and interventional
semantic checks. The full-neural decoder reached **100%** factual-origin
accuracy and about **99.93%** natural target accuracy in the driver scenario.

The negative controls behave as intended. Constant and source-proxy endpoints
have zero within-chart effect error but miss true driver effects; the constant
chart also fails occupancy. The nuisance chart shows nonzero neural effects in
the endpoint-null scenario while behavioral effects remain zero. Thus low chart
error or a genuine neural intervention response alone cannot establish a
behavioral mechanism result. No acceptance threshold was increased to hide
these failures.

## What this changes in the methodology

Keep the neural-space mechanism and the three-way error decomposition. They now
localize a representation failure that a same-chart comparison would have
misclassified as success. The next substantive change should prioritize
behavior during **encoder learning**, with a distributed behavior reference
fixed before fitting, as specified in the plan. Merely replacing graph grouping
or adding a decoder to discarded PCA coordinates does not address the observed
failure.

Behavior-informed representation objectives are motivated by
[CEBRA](https://www.nature.com/articles/s41586-023-06031-6),
[behavior-prioritized identification](https://www.nature.com/articles/s41593-020-00733-0),
and [BunDLe-Net v5](https://www.biorxiv.org/content/10.1101/2023.08.08.551978v5).
Their assumptions and supported intervention classes remain distinct; see the
[verified citation audit](learned_manifold_validation_citation_audit.md).
These references motivate the planned objective, not a claim that an unrun
nonlinear method has already solved this benchmark.

## Verification and version boundaries

An independent NumPy reconstruction checked **160 artifact hashes, 490
endpoint–method–seed rows and 47,040 origin/candidate comparisons**, including
per-origin errors and the empirical event-disagreement triangle bounds. There
were **zero bound violations** at tolerance `1e-12`. This is a check on coupled
finite-sample calculations, not a population confidence statement.

The full implementation suite passed **732 tests with one skip** before the
reviewed guard additions. Review found no training leakage or fresh-run bound
error, but identified missing rejection guards for reused IDs with changed
initial states, changed resume data and inconsistent rollout seeds. Those guards
were then added, including a full evaluation-array fingerprint because cached
natural diagnostics use held-out future observations. A report status flag was
also corrected: the original archived report inherited the inventory's
`fitting_enabled=False`; its actual fitted models, native artifacts and execution
counts remain authoritative. The archived source/report are preserved intact.

The current notebook uses a new label, `endpoint-bridge-03`, after these source
changes. The full five-seed findings above continue to refer to the first
completed evidence, without choosing favorable native reruns. Smoke executions
of the guard revision are engineering checks on the inspected development seed,
not additional independent evidence. Consult the bundle verification record for
the final checks and their exact scopes.
