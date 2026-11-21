# End-to-end effectome-to-behavior benchmark

One run generates the circuits, trains and freezes the behavioral readout,
estimates the directed support from training histories, fits the pooled and
rolling local mechanisms on that support, and generates paired behavior
forecasts. It then selects a forecast at each origin and estimates the
behavioral effects of outgoing disconnections under the selected mechanism.
The simulator supplies realized behavior for forecast evaluation and paired
intervention truth for effect evaluation.

The earlier online choice used the last 16 completed forecasts and was slow to
revert to pooling after a change. The fixed baseline here retains that rule
and adds a fallback. Two consecutive completed forecasts that favor pooling
suspend local selection. Local selection can resume after four newly completed
forecasts favor the local model on average and the original 16-forecast rule
also favors it. The configuration records both settings before the run. The
selection uses only targets decoded from neural samples available at the
forecast origin. A fresh confirmation panel tests the fixed rule without a
parameter search. [Previous online comparison](effectome_rollout_online.md)

## Evaluation

- Fit the behavior decoder on 10 trials, learn support and fit the pooled
  mechanism on 14 other trials, and forecast on 16 further trials. The local
  fit uses a trailing 256-sample history. Both forecasts use the same observed
  state, support, decoder, and innovation draws at an origin.
- Evaluate a driver switch, a constant circuit, and a nuisance-only switch
  using the four prior exact seed-level criteria. Report full-trajectory
  accuracy against pooling, the first 19.2 seconds against always-refitting,
  and noninferiority to pooling in both controls. The earlier online rule is a
  matched comparison on the same origins. A run can complete successfully
  regardless of its statistical decision.
- For paired intervention evaluation, sample every 32nd forecast origin on
  each held-out trial. Evaluate predefined neuron and group disconnections with
  paired innovations and the same selected mechanism. Score effects on origins
  whose horizon stays inside the current generating regime, and record excluded
  crossings. Forecast scores cover the full origin grid. Effect error is
  descriptive and uses both all candidates and the subset with oracle effect
  magnitude at least 0.05.

## Run

```sh
.venv/bin/python -m synthetic.behavior_rollout_pipeline --profile smoke --run
.venv/bin/python -m synthetic.behavior_rollout_pipeline --profile confirmation --run
.venv/bin/python -m synthetic.behavior_rollout_pipeline --profile confirmation --status
```

Smoke uses seed 88999. Confirmation uses fresh seeds 89000–89019. Outputs are
written to `outputs/effectome-rollout-pipeline/<profile>/fast-fallback-01/`.
The protocol records the fixed parameters and source and runtime identities.
The seed table reports paired metrics, and each case retains the fitted readout,
native support evidence, forecasts, selected origins, and intervention effects.
Tuning is a separate study requiring a new output namespace and fresh
confirmation seeds. Settings for this baseline are recorded in the
[fixed policy configuration](../conf/behavior_rollout_pipeline/fixed.yaml).

## Fixed-policy results

All 60 cases completed, covering 199,680 forecast origins and 6,642 evaluated
intervention origins. In the driver-switch condition, fast fallback reduced
early MAE relative to always refitting by 0.01181, compared with 0.00485 for
the previous online rule on the matched origins. All 20 seeds cleared the early
margin, and both control tests passed. Across the full driver-switch trajectory,
fast fallback achieved MAE 0.20584. The pooled reference achieved 0.21116 and
the previous online rule achieved 0.20263. Thirteen of 20 seeds cleared the
full-trajectory margin, leaving the joint forecast decision below its
acceptance criterion.

Fast fallback selected the local fit at 36% of driver-switch origins, compared
with 64% for the previous rule. Driver-switch MAE on oracle-active effects was
0.07696 for fast fallback, 0.06485 for the previous rule, 0.04714 for always-local,
and 0.09132 for pooling. These numbers provide a baseline for future tuning of
the recovery condition while tracking both forecast and effect accuracy.
[Audited comparison](../outputs/effectome-rollout-pipeline/confirmation/fast-fallback-01/review/analysis-report.md)

The subsequent tuning study evaluates a shorter lookback and quicker recovery
on fresh seeds using the same fitted-mechanism pipeline.
[Tuning protocol and confirmation results](effectome_rollout_tuning.md)
