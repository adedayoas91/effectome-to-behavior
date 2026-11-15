# Four-CSL simulation validation

Current simulation results use only conventional c-GC, conventional c-GC*,
fastC-GC and fastC-GC*. Discovery calls the pinned upstream packages; no fitted
VAR readout participates. VAR-estimator results have been removed from the
manuscript and synthetic presentation. Real-data notebooks and VAR backends
remain available. Raw historical output directories are retained for audit and
are not loaded or exported by the current notebooks.

## Environment and run order

Use the project Python3.14 notebook kernel. Conventional methods execute in the
separately pinned Python3.13 CPU environment; see
[its installation instructions](../tools/conventional-cgc/README.md).

1. [00_ground_truth.ipynb](00_ground_truth.ipynb): generating-process controls.
2. [05_four_csl_sanity.ipynb](05_four_csl_sanity.ipynb): native graph recovery.
3. [01_shared_transition_comparison.ipynb](01_shared_transition_comparison.ipynb):
   one common training-owned manifold/event comparison.
4. [02_correspondence_and_controls.ipynb](02_correspondence_and_controls.ipynb):
   graph/manifold correspondence and descriptive controls.
5. [03_framework_validation.ipynb](03_framework_validation.ipynb): repeated
   framework sensitivity matrix and four-method visual exports.
6. [04_driver_recovery.ipynb](04_driver_recovery.ipynb): intervention-defined
   recovery study, acceptance decisions, seed uncertainty and null calibration.

Notebooks inspect saved evidence by default. Execution switches explicitly
start fits; a partial run cannot satisfy full-study acceptance.

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
.venv/bin/python pipeline/13_driver_validation.py --profile confirmatory --run --workers 4
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
