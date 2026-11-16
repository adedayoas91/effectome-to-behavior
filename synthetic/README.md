# Four-CSL simulation validation

Current simulation results use only conventional c-GC, conventional c-GC*,
fastC-GC and fastC-GC*. Discovery calls the pinned upstream packages; no fitted
VAR readout participates. VAR-estimator results have been removed from the
manuscript and synthetic presentation. Real-data notebooks and VAR backends
remain available. Raw historical output directories are retained for audit and
are not loaded or exported by the current notebooks.

## Environment and run order

From the application repository on the faster machine, restore both locked
runtimes (no raw dataset is required for these simulations):

```bash
git switch dev
git pull --ff-only origin dev
uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks
uv sync --project tools/conventional-cgc --python 3.13.5 --locked
```

Open `synthetic/` in your notebook editor and select `.venv/bin/python` as the
Python 3.14 kernel. Each notebook discovers the repository from its own working
directory and checks installed main-runtime dependencies. Conventional methods
execute in the separately pinned Python 3.13 CPU environment; see
[its installation instructions](../tools/conventional-cgc/README.md). The kernel
extra supplies ipykernel; use your existing notebook editor/frontend.

The filename prefixes now match the recommended order. These are independently
runnable entrypoints, so you can go directly to **06** to evaluate the latest
methodology revision without rerunning the completed old confirmation.

1. [00_ground_truth.ipynb](00_ground_truth.ipynb): generating-process controls.
2. [01_four_csl_sanity.ipynb](01_four_csl_sanity.ipynb): native graph recovery.
3. [02_shared_transition_comparison.ipynb](02_shared_transition_comparison.ipynb):
   one common training-owned manifold/event comparison.
4. [03_correspondence_and_controls.ipynb](03_correspondence_and_controls.ipynb):
   graph/manifold correspondence and descriptive controls.
5. [04_framework_validation.ipynb](04_framework_validation.ipynb): repeated
   framework sensitivity matrix and four-method visual exports.
6. [05_driver_recovery.ipynb](05_driver_recovery.ipynb): intervention-defined
   recovery study, acceptance decisions, seed uncertainty and null calibration.
7. [06_mechanism_driver_validation.ipynb](06_mechanism_driver_validation.ipynb):
   new development-only known-support/estimated-support mechanism diagnostic.
8. [07_graph_similarity.ipynb](07_graph_similarity.ipynb): training-owned grouping
   of planted graphs by neuron identity and recurring topology, with fixed-density
   rewiring controls and an explicit WL collision.

Notebooks inspect saved evidence by default. Execution switches explicitly
start fits; a partial run cannot satisfy full-study acceptance.

### Evaluate the latest changes on the faster machine

Open **06_mechanism_driver_validation.ipynb**, inspect the declared settings,
and use its top parameter cell:

| Purpose | Parameters |
| --- | --- |
| Inspect versioned results, including both history lengths | Leave both run switches `False` |
| Rerun smoke with all four native methods and both support controls | `PROFILE="smoke"`, `ORACLE_ONLY=False`, `RUN=True` |
| Rerun the three-seed known/empty-support readout diagnostic | `PROFILE="development"`, `ORACLE_ONLY=True`, `RUN=True` |
| Rerun the paired 256/1024-history comparison with all four methods | `RUN=False`, `RUN_HISTORY_COMPARISON=True`, `METHODS=None` |

Keep `METHODS=None` for complete native panels. New results are written under
`outputs/driver-mechanism-validation/reruns/<RUN_LABEL>/`; the notebook displays
streamed lane progress and saves `run.log`, per-lane `cells/` caches, native
support evidence, JSON reports and CSV/Markdown analysis. Reuse the label to
resume unchanged code/settings; choose a new label after a code/protocol change.
The paired history comparison is independent of the regular `PROFILE` and
`ORACLE_ONLY` switches. Both native runs can be expensive; the oracle-only run
checks the revised readout before native discovery.

The previously measured result is a **limited operating case**: at 256 samples
the native methods missed endpoint support; at 1024 samples all four recovered
it and ranked drivers correctly on one paired seed. The new three-seed oracle
control checks readout recovery. Neither result overturns the completed old
560-run failure, and the two studies use different data-generating processes
and endpoints. Notebook **05** displays the frozen failure on a clean checkout;
notebook **06** labels versioned historical evidence separately from new reruns.

## Completed native graph sanity

[The frozen three-seed chain/null study](results/four_csl_sanity_2026-11-15/analysis/analysis-report.md)
has24/24 available fits. All four methods recovered exactly0→1 and1→2 in every
chain recording and selected no null links. The isolated neuron was excluded.
Two vector figures, seed tables, native evidence and provenance accompany it.

This is bounded graph sanity, not transition-driver validation. The four
variants share one discovery family. Empty-graph Jaccard is undefined, and
three null seeds cannot establish false-positive calibration. Conventional CPU
permutation randomness is not fully seeded upstream; preserve actual outputs.

## Scheduled-origin driver study

**Completed:** all 560 runs produced usable predictions, and all four methods
failed the frozen recovery criteria. The versioned evidence is in
[`results/driver_confirmation_v2/`](results/driver_confirmation_v2/).
Preserve that conclusion when evaluating the separate revision below. Reproduce
the old study using its frozen source archive; the current implementation has
additional modules and therefore a different implementation fingerprint.

The revised study uses one fixed origin per independent trial, selected before
observing its realized outcome. A256-sample history ends at sample303 and its
strictly future8-sample target ends at335: a3.2-second physical horizon at10Hz.
Each trial has512 samples, cue onset288 and cue offset480. Eight observed neurons
supply four population readouts. A training-fitted one-dimensional PCA chart and
two-region codebook are frozen for held-out trials; truth is used only in the
semantic audit and simulator interventions, not to fit the chart or predictors.

Development seeds10/11 supported the revised timing and chart. Confirmation
uses fresh positive seeds300–319 and null seeds400–459; inspected seed100 is
excluded. The five conditions are single driver, redundant pair, correlated
bystander, no-driver null, and hidden-confounder diagnostic. The complete panel
has140 recordings,560 method/recording runs and71,680 native window fits.

```bash
.venv/bin/python pipeline/13_driver_validation.py --profile confirmatory
```

The parallel runner shares import/JIT startup within each process while creating
fresh upstream estimators for every history. Method/seed caches are atomic and
resumable. The merge verifies identical shared-reference/event/baseline hashes
across methods. The frozen protocol and numerical artifacts are stored under:

```text
outputs/driver-validation/csl_sanity/scheduled-v2/confirmatory/
```

Acceptance thresholds remain unchanged:20 positive seeds and60 null seeds,
20 origins and5 events per direction, semantic agreement≥.8, recovery≥.75,
adequate availability and per-method false-nomination bounds. Actual simulator
outgoing disconnection supplies truth; predictor-feature omission supplies the
candidate score. These are different quantities. Hidden confounding is a
failure-mode diagnostic, not a causal-identification certificate.

Figures show intervention truth, predictive omissions, and recovery/null bounds.
Exact tables retain failed fits, ineligible directions, negative gains and
incomplete seed sets. Intervals use independent seeds; neurons, methods and
multiple origins do not become replicates. Numerical completion does not imply
scientific acceptance. See [the protocol revision](../docs/driver_confirmation_revision.md).

## Mechanism-readout development revision

The [cited decision record](../docs/driver_methodology_revision.md) explains the
change from graph-feature omission to modeled outgoing-disconnection effects.
The existing omission score and completed failed study retain their identities.
The new control is fully observed, uses a declared nonlinear update family, and
has a predeclared directly observed endpoint rather than a learned manifold.

```bash
.venv/bin/python pipeline/15_mechanism_validation.py --profile smoke
.venv/bin/python pipeline/15_mechanism_validation.py --profile smoke --run
.venv/bin/python -m synthetic.mechanism_history_diagnostic
.venv/bin/python -m synthetic.mechanism_history_diagnostic --run
```

The second command includes known-support and empty-support controls plus all
four imported CSL support lanes. `--oracle-only` omits native discovery;
`--method fast_cgc` explicitly restricts estimated-support lanes. Use distinct
`--output-dir` values for different panels or implementation versions. Results,
per-lane caches, native evidence and descriptive reports are written below
`outputs/driver-mechanism-validation/`. Default notebook cells inspect saved
results; the `RUN` switch starts computation.

Regularization and noise estimation use inner whole-trial validation. Discovery
uses only the preceding inner-training trials; the final mechanisms refit all
outer training trials with the selected support. The same held-out initial
states, endpoint and future horizon are used in every lane. Probability-effect
cutoffs are provisional development settings, not the old omission-nat gates.
Fresh confirmation, max-score calibration, redundant-group and observation
stresses remain separate future work. A favorable result here is a limited
operating case, not an overturning of the failed partially observed benchmark.

The [first measured development findings](../docs/mechanism_development_findings.md)
retain both the 256-sample failures and the 1,024-sample operating case. The
paired diagnostic uses identical recordings and endpoints with unchanged
selection cutoffs; its all-four-native comparison still has only one raw-process
seed. Versioned source, native evidence and numerical reports are in
[`results/mechanism_development_2026-11-15/`](results/mechanism_development_2026-11-15/).

## Circuit/manifold controls

The controls retain their known generating recurrences and covariance identity.
Different signed circuits can share the same snapshot law, and different
initial histories can produce different trajectories under one fixed circuit.
These are generating-process constructions, not fitted VAR-estimator results.
Opposite rotations share binary support; the current unsigned effectomes cannot
distinguish sign-only changes. Support-changing controls are required before a
discovery claim about that ambiguity. Never interpret geometric regions as
established biological regimes or predictive importance as biological necessity.

The four-method framework visual exporter requires the original saved detailed
comparison. It never reruns stochastic methods to recreate a favorable panel.
Missing worked-case evidence is explicitly unavailable; historical VAR panels
cannot be silently exported through this workflow.


## Graph similarity grouping diagnostic

Notebook **07_graph_similarity.ipynb** and `pipeline/16_graph_similarity.py`
compare directed, lag-aware edge overlap with a direction/lag-aware WL kernel.
The first retains neuron identity; the second recognises recurring circuit
architecture under neuron relabelling. Medoids use training graphs only, and
held-out identity-archetype and topology-class scores remain separate. Six nodes
and two edges in every main-panel graph control density; one-edge replacements
retain that count. Exact-isomorphism, lag-change, empty-graph and WL-collision
probes expose limits of each comparison.

```bash
.venv/bin/python pipeline/16_graph_similarity.py --profile smoke
.venv/bin/python pipeline/16_graph_similarity.py --profile smoke --run
```

The default inventories 36 quantizer fits and no native CSL fits. The notebook
also defaults to inventory/saved-evidence mode; set `RUN=True` explicitly and use
a new `RUN_LABEL` after a protocol/code change. Reports, seed rows, unavailable
partitions and PDF/PNG figures go under
`outputs/graph-similarity/<PROFILE>/<RUN_LABEL>/` in the notebook. Local saved
configuration must exactly match the requested settings before inspection;
versioned historical evidence is labelled separately. The fixed
K=2/3/4 contrasts do not select a metric or K on held-out labels. A requested K
larger than the training metric can distinguish remains unavailable.

These repeats are graph-sampling seeds, not neural-process or biological
replicates. This is a structural implementation diagnostic; it does not rerun
native discovery or validate behavioral/driver recovery. Read the
[cited methodology](../docs/graph_similarity_methodology.md) for assumptions and
how these graph geometries complement the separate intervention validation.
