# Synthetic rollout validation

Run the three notebooks in order before opening the C. elegans notebooks.
They use the current support ablation with the tuned pooled/local selector
frozen. [Benchmark design and frozen criteria](../docs/effectome_rollout_support.md)

## Faster-machine setup

From the application repository, sync the two locked environments:

```sh
uv sync --python 3.14 --locked --extra notebooks
uv sync --project tools/behavior-encoder --python 3.12 --locked
```

Select the project Python 3.14 `.venv` kernel. The encoder runs in its
separate Python 3.12 environment. The simulator generates its own recordings.

## Run order

| Notebook | Result |
|---|---|
| [00 · Simulator and sanity](00_simulator_and_sanity.ipynb) | Planted mask diagrams, driver and nuisance controls, a fastC-GC training-window check, and runtime verification |
| [01 · Run support benchmark](01_run_support_benchmark.ipynb) | Paired forecasts for learned, dense, shuffled and direct-history arms, plus simulator and fitted disconnections for the generative arms |
| [02 · Validate and visualize](02_validate_and_visualize.ipynb) | Replayed seed scores and decisions, PNG/PDF figures, and a standalone interactive intervention view |

Notebook 01 starts the **confirmation** profile on Run All. It uses 20
fresh seeds across three scenarios. Set `WORKERS` in notebook 01 for the
available cores and memory. For a short run, set `PROFILE = "smoke"` in
notebooks 01 and 02. Use the same `RUN_ID` in both. Re-running notebook 01
under a matching protocol reuses completed cases.

Notebook 00 fits a small fastC-GC training example. Notebook 01 computes
the effectomes used by the benchmark. It learns a consensus directed mask
from training histories, then fits a pooled signed mechanism and refits
its weights on the trailing observations at every forecast origin. The
mask stays fixed across those origins. Each case keeps the native training
evidence in `native-training.json` and the pooled fits and their forecast
outputs in `support-case.json`. Notebook 02 reads and audits the saved
comparisons. The three notebooks provide their own graph fits.

## Outputs and visualization

The default confirmation directory is
`outputs/effectome-rollout-support/confirmation/support-01/`.
Notebook 01 streams progress and saves it to `run.log`. Inspect
`support-report.json` for case counts and `seed-metrics.csv` for the paired
metrics. The smoke profile writes to a separate `smoke/` branch.

After all cases complete, notebook 02 verifies the case identities and
artifact hashes, reconstructs the selected forecasts, and recalculates
the forecast and intervention scores. Its `validation/` directory contains:

- `forecast_accuracy.png` and `.pdf` for full and early driver forecasts
  and both controls.
- `paired_support_gains.png` and `.pdf` for seed-paired gains against
  the frozen practical margins.
- `disconnection_calibration.png` and `.pdf` for signed fitted versus
  simulator effects on active driver-switch pairs. The scatter uses mean
  shuffle vectors. The seed scores average errors across shuffle draws.
- `interventions.html`, an offline browser view of each saved neuron or
  group at a matched origin. Select the scenario, seed, fitted arm, trial
  and origin to compare both behavioral coordinates with simulator truth.
- `validation.json` for the verified decisions and figure paths.

The figures and browser view are available for smoke and confirmation.
Smoke uses one process seed and has no inferential decision. A completed
confirmation reports the paired tests even when their margins are unmet.

The 11 previous notebooks and their development guide are under
[archive/](archive/README.md). Their original case outputs and evidence
bundles use their historical paths. The three active notebooks can be
regenerated with `.venv/bin/python -m synthetic._build_support_notebooks`.
