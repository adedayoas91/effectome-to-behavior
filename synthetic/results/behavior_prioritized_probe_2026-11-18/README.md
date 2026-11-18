# Behavior-prioritized compression: retrospective development probe

The exact scalar decision score of the previously trained full-neural classifier
preserves its evaluated labels/events and its low behavioral-effect error. This
is a positive capacity control for the binary outcome, not a new independently
successful learning algorithm or proof of nonlinear manifold geometry.

Start with [the diagnostic report](analysis/analysis-report.md),
[the exact table](analysis/numeric-summary.md), and the
[method comparison](../../../docs/behavior_prioritized_alternatives.md).
The manuscript development section records the motivation, evidence and next
choice for each of six trials. Its complete source/PDF snapshot is archived here.

The same five inspected development seeds (40000--40004) provide ten driver/null
cases and 70 cached neural models. No causal discovery or neural-mechanism fit
was repeated. Forty training-only PCA-dimension fitting calls test dimensions
one, three, four and five; the already frozen two-component endpoint is reused.
The scalar code factors existing trained parameters and has no new fitting step.
Zero disagreements were checked on 276,480 natural labels and 1,935,360 rollout
events. Those are replay checks, not additional independent experimental units.

Files:

- `probe-protocol.json` and `probe-summary.json`: procedure, source/input hashes,
  counts, original rollout template and effective per-cell seeds.
- `seed-metrics.csv`: every endpoint/dynamics/seed/scenario result.
- `probe-evidence.zip`: endpoint artifacts, binary events, cell results and logs.
- `parent-manifold-validation.json` and `parent-run-evidence.zip`: frozen inputs,
  saved dynamics parameters, native evidence and original event arrays.
- `implementation-source.zip`: the neural/endpoint implementation used for replay.
- `independent-verification.json`: direct NumPy reconstruction and source checks.
- `analysis/`: figures, full descriptive statistics, exploratory tests and limitations.
- `manuscript-history-source-and-pdf.zip`: compiled manuscript and source snapshot.
- `manifest.json`: file sizes and SHA256 hashes.

From the application repository using its locked environment, verification and
analysis require no native fitting:

```sh
.venv/bin/python synthetic/results/behavior_prioritized_probe_2026-11-18/audit_probe.py
.venv/bin/python synthetic/results/behavior_prioritized_probe_2026-11-18/analyze_probe.py
```

To repeat the fitting/replay probe, copy `probe_behavioral_compression.py` into a
new empty `outputs/behavior-prioritized-investigation/<LABEL>/` directory and run
that copy. It expects the original notebook-08 input tree under
`outputs/learned-manifold-validation/development/endpoint-bridge-03/`; the parent
evidence ZIP preserves that tree for an isolated reconstruction. Use the archived
implementation if the current generator/endpoint source has changed. Frozen
source/input checks reject inconsistent data. Existing inputs and user notebooks
were preserved. CSV readers must retain the literal scenario name `null`, e.g.
`pd.read_csv(..., keep_default_na=False)`.

Remaining scope: continuous/distributed behavior, a matching oracle-checked null,
a genuinely learned nonlinear bottleneck and fresh calibration/confirmation are
planned. Hidden-state robustness, reduced sensing and autonomous latent dynamics
are not established. No graph branch was merged and no dependency was installed.
