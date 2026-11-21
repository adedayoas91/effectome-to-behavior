# Four-CSL simulation validation

The [end-to-end effectome-to-behavior benchmark](../docs/effectome_rollout_pipeline.md)
fits the readout and both mechanisms, chooses forecasts from completed errors,
and scores paired intervention effects. Run `python -m synthetic.behavior_rollout_pipeline
--profile confirmation --run` from the application repository.

The [working mechanism-rollout pipeline](../docs/effectome_mechanism_rollout.md)
uses trailing effectomes to simulate neural trajectories, with a frozen
behavioral readout and paired outgoing-disconnection effects. The standalone
`python -m synthetic.effectome_rollout` runner validates this path with pooled,
fixed-support local and simulator-oracle references. Its outputs are under
`outputs/effectome-rollout/`.

The [post-switch timing study](../docs/effectome_rollout_timing.md) compares the
pooled and fixed-support local fits at 11 delays with two history lengths.
`python -m synthetic.behavior_rollout_timing --profile confirmation --run`
executes the independent timing panel.

The [19 November checkpoint](results/behavior_rollout_2026-11-19/README.md)
contains the completed comparisons, seed tables, and the compiled manuscript.

The development comparison is
[`10_behavior_transition_experiment.ipynb`](10_behavior_transition_experiment.ipynb).
It compares forecasts of future continuous behavior using recent histories,
static graphs, trailing dynamic graphs and oracle mechanisms. It tests driver
pathway switches alongside constant-coupling and nuisance-only controls. Run
All fits five development seeds with fast c-GC and both history lengths under
`outputs/behavior-transition/development/recurrent-coupling-01/`. It also saves
paired intervention effects and training-owned candidate nominations. The
[cited transition plan](../docs/behavior_supervised_transition_plan.md) explains
the timing, comparisons and confirmatory decision.

```sh
.venv/bin/python -u -m synthetic.behavior_transition --run
.venv/bin/python -m synthetic.behavior_transition --status
.venv/bin/python -m synthetic.preflight_tail_transition
```

The fixed-coupling behavioral representation experiment is
[`09_behavior_supervised_validation.ipynb`](09_behavior_supervised_validation.ipynb).
It generates two continuous behavioral coordinates from four observed neural
variables and compares PCA, PLS, full-neural readouts and a jointly trained
BunDLe-inspired encoder. Run All starts the five-seed development panel and
writes progress and detailed results under
`outputs/behavior-supervised/development/distributed-behavior-01/`.
Notebook 09 runs independently. The
[study plan and cited rationale](../docs/behavior_supervised_experiment_plan.md)
explain the generator, objectives and decision rules.

```sh
uv sync --project tools/behavior-encoder --python 3.12 --locked
.venv/bin/python -u -m synthetic.behavior_supervised --run --workers 4
.venv/bin/python -m synthetic.behavior_supervised --status
```

The report, `behavior-supervised-validation.json`, tracks completed cases.
`analysis/seed-metrics.csv` contains the effect and decoding metrics.
The audit command reconstructs effect errors and verifies saved artifact hashes.

```sh
.venv/bin/python -m synthetic.audit_behavior_supervised \
  outputs/behavior-supervised/development/distributed-behavior-01 --plot
```

A complete same-procedure development run enables the separate 99-case
confirmation profile:

```sh
.venv/bin/python -u -m synthetic.behavior_supervised --profile confirmation --run --workers 4 \
  --development-report outputs/behavior-supervised/development/distributed-behavior-01/behavior-supervised-validation.json
```

The earlier scalar-behavior experiment is
[`08_learned_manifold_validation.ipynb`](08_learned_manifold_validation.ipynb).
Run All fits five development seeds (40000–40004), both driver/null scenarios,
all four native methods and oracle/known/empty controls. Seven frozen endpoint
arms share each neural fit and its futures. This is the initial **linear PCA
bridge**. Read the
[plan, justifications and references](../docs/learned_manifold_validation_plan.md).
Go directly to 08 without rerunning earlier notebooks.

```sh
.venv/bin/python -m synthetic.manifold_validation
.venv/bin/python -u -m synthetic.manifold_validation --run --workers 4
.venv/bin/python -m synthetic.manifold_validation --status
```

Results are under
`outputs/learned-manifold-validation/development/endpoint-bridge-03/`.
The runner saves `manifold-validation.json`, per-cell endpoint artifacts,
per-rollout events, native evidence and `analysis/seed-metrics.csv` with a
descriptive report. `--profile smoke` runs one development seed;
`--oracle-only` runs the three mechanism controls. No new calibration or
confirmation pool is enabled, and graph comparison remains separate.

The first five-seed development run is complete and archived under
[`results/learned_endpoint_development_2026-11-18/`](results/learned_endpoint_development_2026-11-18/).
Its executed source is preserved. The `03` default starts a fresh directory after
adding stricter initial-state and resume-data checks; read the archived `01`
report directly to inspect the original evidence.

A separate [behavior-prioritized compression investigation](results/behavior_prioritized_probe_2026-11-18/README.md)
replays the cached mechanisms with a scalar factorization of the trained
full-neural classifier and reports every PCA dimension from one to five.
The scalar code preserves the classifier's low effect error exactly; it is an
equivalent supervised binary readout, not an independently trained nonlinear
manifold. This probe is documented in the [method comparison](../docs/behavior_prioritized_alternatives.md)
and the manuscript's numbered development history. It is separate from notebook
08, uses the same inspected seeds, and performs no native or mechanism refitting.

The preceding experiment is
[`07_mechanism_replication.ipynb`](07_mechanism_replication.ipynb): a full fresh-seed
replication with 19 separate calibration-null seeds, 20 driver-test seeds and
60 null-test seeds. Run All is enabled. The primary history is 1,024 samples;
the same positive-test seeds also run at 256 samples for a paired diagnostic.
The cited design, prospective acceptance gates and calibration limits are in
[`mechanism_replication_protocol.md`](../docs/mechanism_replication_protocol.md).
This remains separate from graph similarity.

That fresh-seed study is now complete. Its
[results and discussion PDF](results/mechanism_replication_2026-11-17/analysis/replication-results.pdf)
and [frozen evidence bundle](results/mechanism_replication_2026-11-17/README.md)
report accurate effects across 20 fresh seeds, no false nominations in 60 null
seeds, and the retained 19/20 individual-ranking eligibility limit. The older
`result_collation.pdf` below remains the pre-replication record for notebooks 00–06.

From the application repository, inventory, run, or inspect progress with:

```sh
.venv/bin/python -m synthetic.mechanism_replication --profile replication
.venv/bin/python -u -m synthetic.mechanism_replication --profile replication --workers 4 --run
.venv/bin/python -m synthetic.mechanism_replication --profile replication --status
```

New results are under
`outputs/driver-mechanism-replication/replication/mechanism-replication-01/`.
The runner writes `study-protocol.json`, locks `calibration.json` before test
jobs, and updates `replication-report.json` after each completed cell. Notebook
runs also stream `run.log`. Keep the code, profile and label unchanged to resume
first-completed evidence; never rerun a result to select a favorable estimate.
The threshold is the larger of the calibration order statistic and the existing
0.10 effect-size floor. Nomination calibration is per method under the global
null; its conditional null rate still needs the separate 60-seed exact-bound audit.

The [compiled synthetic results and discussion](result_collation.pdf) are in
[`result_collation.tex`](result_collation.tex), with a
[verified bibliography](result_collation.bib) and
[citation audit](result_collation-citations.md). Rebuild from this directory with
`tectonic --keep-logs result_collation.tex`.

The completed `faster-machine-03` notebook runs are collated in
[`results/completed_runs_2026-11-17/`](results/completed_runs_2026-11-17/).
Start with its [analysis report](results/completed_runs_2026-11-17/analysis-report.md)
and [numeric summary](results/completed_runs_2026-11-17/numeric-summary.md).
The bundle preserves executed notebooks, source reports, statistical limits and
comparison figures. It contains no graph-similarity results or new fitted runs.

Current simulation results use only conventional c-GC, conventional c-GC*,
fastC-GC and fastC-GC*. Discovery calls the pinned upstream packages; no fitted
VAR readout participates. VAR-estimator results have been removed from the
manuscript and synthetic presentation. Real-data notebooks and VAR backends
remain available. Raw historical output directories are retained for audit and
are not loaded or exported by the current notebooks.

## Environment and run order

From the application repository on the faster machine, restore the locked
runtimes (no raw dataset is required for these simulations):

```bash
git switch dev
git pull --ff-only origin dev
uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks
uv sync --project tools/conventional-cgc --python 3.13.5 --locked
uv sync --project tools/behavior-encoder --python 3.12 --locked
```

Open `synthetic/` in your notebook editor and select `.venv/bin/python` as the
Python 3.14 kernel. Each notebook discovers the repository from its own working
directory and checks installed main-runtime dependencies. Conventional methods
execute in the separately pinned Python 3.13 CPU environment; see
[its installation instructions](../tools/conventional-cgc/README.md). The kernel
extra supplies ipykernel; use your existing notebook editor/frontend.

The filename prefixes now match the recommended order. These are independently
runnable entrypoints. Go directly to **10** for the dynamic-effectome forecasts.

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
8. [07_mechanism_replication.ipynb](07_mechanism_replication.ipynb): completed
   fresh-seed mechanism replication and independent null calibration/testing.
9. [08_learned_manifold_validation.ipynb](08_learned_manifold_validation.ipynb):
   training-frozen learned-endpoint bridge and independent behavioral semantics.
10. [09_behavior_supervised_validation.ipynb](09_behavior_supervised_validation.ipynb):
    supervised compression, continuous behavioral effects and an oracle-checked null.
11. [10_behavior_transition_experiment.ipynb](10_behavior_transition_experiment.ipynb)
    compares prospective behavioral forecasts and context-specific perturbations.

**Run All now starts fitting by default**, as requested for the faster-machine
rerun. Notebook 01 has sanity enabled, and notebooks 02 and 04–10 have their
fitting switches enabled. Notebook 06 also runs the paired 256/1024-history
comparison. Notebooks 00 and 03 compute their generating-process controls
directly. Notebook 05 retains its full 140-recording/560-method-run profile with
four workers; rerunning its existing seeds is reproduction, not fresh independent
confirmation. A partial run cannot satisfy full-study acceptance.

The conventional CPU runtime must be installed before native fitting starts.
Missing runtimes now stop the notebook with the installation command rather than
producing a partially available panel. The top cells print the selected workload,
enabled switches and output directory.

### Evaluate the latest changes on the faster machine

Open **06_mechanism_driver_validation.ipynb**, inspect the declared settings,
and use its top parameter cell:

| Purpose | Parameters |
| --- | --- |
| Run smoke and both history lengths with all four native methods | Keep the shipped `RUN=True`, `RUN_HISTORY_COMPARISON=True`, `METHODS=None` |
| Inspect versioned results without fitting | Explicitly set both run switches `False` |
| Run only smoke with all four native methods and both support controls | `PROFILE="smoke"`, `ORACLE_ONLY=False`, `RUN=True`, `RUN_HISTORY_COMPARISON=False` |
| Run only the three-seed known/empty-support readout diagnostic | `PROFILE="development"`, `ORACLE_ONLY=True`, `RUN=True`, `RUN_HISTORY_COMPARISON=False` |
| Rerun the paired 256/1024-history comparison with all four methods | `RUN=False`, `RUN_HISTORY_COMPARISON=True`, `METHODS=None` |

Keep `METHODS=None` for complete native panels. New results are written under
`outputs/driver-mechanism-validation/reruns/<RUN_LABEL>/`; the notebook displays
streamed lane progress and saves `run.log`, per-lane `cells/` caches, native
support evidence, JSON reports and CSV/Markdown analysis. Reuse the label to
resume unchanged code/settings; choose a new label after a code/protocol change.
The paired history comparison is independent of the regular `PROFILE` and
`ORACLE_ONLY` switches. Both native runs can be expensive; the oracle-only run
checks the revised readout before native discovery. The shipped run label is
`faster-machine-03`; reuse it only to resume identical code/settings.

This run label follows the notebook-backend fix. Restart existing kernels to
load the corrected conventional adapter. It gives the isolated worker a
headless plotting backend while preserving notebook displays. Earlier
unavailable conventional rows caused by that import failure contain missing
metrics, rather than a measured failure to recover graph edges. Notebook 01
also normalizes tuple/list configuration fields before checking saved settings.
Null precision/recall and empty-graph Jaccard remain undefined when their
denominators are zero, even for an available and correct null fit.

### Output directories for the enabled notebook run

| Notebook | New output directory relative to the repository |
|---|---|
| 01 | `outputs/csl-sanity/reruns/faster-machine-03/` |
| 02 and its 03 inspection | `outputs/synthetic-transition-examples/matched_marginals/seed-42/four-csl-sanity/faster-machine-03/` |
| 04 | `outputs/framework-validation/csl_sanity/smoke/reruns/faster-machine-03/` |
| 05 | `outputs/driver-validation/csl_sanity/scheduled-v2/confirmatory/reruns/faster-machine-03/` |
| 06 | `outputs/driver-mechanism-validation/reruns/faster-machine-03/` |

Notebook 06 includes both the regular mechanism panel and the `history-comparison/`
subdirectory. Transfer its JSON reports, `cells/` evidence and `run.log` when
reviewing newly fitted results. Pipeline CLIs still need `--run`; notebook
fitting defaults and CLI inventory defaults are separate controls.

Graph comparison and notebook 07 remain on their separate feature branch. They
are not included in this `dev` run. The supplied inspection-only notebook outputs
were preserved in [`results/notebook_inspection_2026-11-16/`](results/notebook_inspection_2026-11-16/)
before preparing these output-free, fitting-enabled copies.

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
`outputs/driver-mechanism-validation/`. Notebook cells inspect saved
results when the run switches are explicitly disabled; fitting is enabled in the
shipped notebook.

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
