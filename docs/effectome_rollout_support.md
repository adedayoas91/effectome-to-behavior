# Does learned support improve behavioral rollouts?

The tuning benchmark compared when to refit under the same learned support.
This study freezes the confirmed selector and varies only the support
mask among the generative models. We use the same simulated data,
training partition, readout, lag-one mechanism, ridge penalty, trailing
history, origins, and innovation draws in all three generative arms.
[Confirmed selector](effectome_rollout_tuning.md)

- **Learned support.** Use fastC-GC and the existing training-window consensus
  rule. Fit both the pooled and local mechanisms on this mask.
  [fastC-GC](https://arxiv.org/abs/2610.37883)
- **Dense regularized mechanism.** Allow every directed cross-neuron
  edge and use the same standardized ridge penalty. This tests whether
  screening improves estimation given the same model family.
- **Shuffled support.** Resample each target's allowed parents uniformly
  from its other neurons, preserving the learned count per target and the
  overall edge count. This controls target-level fit dimension. Three
  draws per case in confirmation reduce dependence on a single shuffle.
  Average their errors within each seed. Overlap with the learned mask and
  duplicate shuffled draws remain visible in the case artifacts.
- **Direct neural history.** Train a graph-free forecast on the same
  training windows using neural history and targets decoded by the
  frozen readout. Tune by whole training trial and predict the same
  absolute future behavior target. The direct model provides forecast
  accuracy only. Its forecasts do not define outgoing-disconnection
  effects.

The selector uses completed decoded targets `(lookback=6, margin=0,
consecutive_losses=3, recovery_forecasts=1)` separately for each
generative arm. Forecast error uses realized behavior. Intervention
error uses the same simulator-active pairs and paired rollouts from
unchanged factual origins in each arm. Regime-crossing horizons stay in
the forecast grid and are excluded from effect scoring. This is the
same fully observed simulator and model family as the tuning study.
It retains the existing stable training windows and tests support choice
under that benchmark contract.

## Decision and output

The two primary endpoints are full-trajectory forecast MAE and active
disconnection-effect MAE in the driver-switch condition. For either
endpoint, learned support must improve on both dense and shuffled by
more than 0.005 MAE in a majority of paired process seeds. The forecast
endpoint also requires learned support to stay within 0.005 MAE of both
comparison arms in both controls. These margins retain the tuning study's
forecast tolerance and apply the same absolute behavioral error tolerance
to effects. Each endpoint uses the maximum of its component one-sided
exact sign-test p-values. Holm adjustment covers the two endpoints at
0.05. Early forecasts and the direct comparator are descriptive. The
existing four selector criteria are reported separately for the learned
arm to check that its timing result transfers to this fresh panel.

The confirmation profile uses 20 fresh process seeds (91000–91019).
The smoke profile uses one separate seed, two shuffle draws, and 32
innovations per forecast to exercise all paths. It makes no inferential
decision. The confirmation profile uses the established 128 innovations.

Run from the application repository with its existing fastC-GC and
behavior encoder environments.

```sh
.venv/bin/python -m synthetic.behavior_rollout_support --profile smoke --run
.venv/bin/python -m synthetic.behavior_rollout_support --profile confirmation --run --workers 3
.venv/bin/python -m synthetic.behavior_rollout_support --profile confirmation --status
```

Each profile writes to `outputs/effectome-rollout-support/<profile>/support-01/`.
The protocol freezes the inputs before fitting. Completed cases preserve the
readout, native evidence, masks, forecast choices, shared origin and
intervention references, and individual shuffle comparisons. The seed table
has one row per arm, seed, and scenario, and the report tracks completion
and the confirmation decisions.

The smoke run completed all three scenarios and produced 12 matched
arm-by-scenario rows. Replay reproduced the frozen selector's choices and
the saved forecast and intervention scores. The 20-seed confirmation
remains pending. [Smoke report](../outputs/effectome-rollout-support/smoke/support-01/support-report.json)
