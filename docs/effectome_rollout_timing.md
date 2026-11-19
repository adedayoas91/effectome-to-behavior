# How soon does local refitting help after a coupling change?

The long-history fixed-support fit gave lower average forecast and active-effect
error than support rediscovery in the preceding synthetic comparison. Its
origins followed complete histories in established regimes. This study retains
the same fixed-support mechanism and tests histories that contain both regimes.
[Mechanism-rollout decision](effectome_mechanism_rollout.md)

## Design frozen before evaluation

- Fit the behavior-only readout on 10 trials. Select the directed lag-one mask
  from fast c-GC votes on the 1,024-sample histories of 14 other training trials.
  Reuse their pooled additive fit as the fixed reference.
- Evaluate both switch directions in 48 further trials per seed. At 10 Hz,
  origins are 0, 8, 16, 32, 64, 128, 192, 256, 512, 768, or 1,023 samples
  after the first sample generated under the new mechanism. Use the same origins
  for 256- and 1,024-sample local refits. A local history keeps all observed
  old-regime samples that are still in its trailing window.
- Use a shared readout, ridge penalty, forecast horizon, and paired 128-draw
  innovation bank. The target is the measured mean behavior at steps five
  through eight. The generator's first post-switch sample labels delay zero.
  The regimes and switch times identify evaluation events and oracle truth.
  The local fits receive only the trailing neural windows and a fixed mask.
- Use the same schedules and input innovations for the driver switch, constant
  coupling, and nuisance-only switch. Only the driver switch changes which
  driver reaches behavior. The primary endpoint is the error of predicting
  realized future behavior. A descriptive comparison to the oracle rollout
  mean measures agreement with the simulated conditional behavior expectation.

The independent unit is the process seed. Smoke uses seed 85000. Confirmation
uses 20 fresh seeds from 86000 through 86019. Each seed contributes one paired
forecast MAE gain per condition, history length, and delay. The driver test asks
whether more than half the seeds gain more than 0.005 MAE. Each control test
requires more than half the gains to fall below 0.005. The cell's joint
one-sided exact p-value is the maximum across the three required tests. Ties
count against the relevant condition. Apply Holm's adjustment across all 22
declared history-by-delay cells before identifying each earliest supported grid
point. The earliest sustained point also requires support at every later
measured delay for that history. [Holm, 1979](https://doi.org/10.2307/4615733)

The benchmark measures adaptation after a simulator-known change. It measures
forecasts within a regime after the indexed switch and retains both control
trajectories. The earliest supported grid point depends on the specified history,
effect-size margin and sampling design. Real-time switch detection requires a
separate training-owned rule for choosing between pooling and refitting.

```sh
.venv/bin/python -m synthetic.behavior_rollout_timing --profile smoke --run
.venv/bin/python -m synthetic.behavior_rollout_timing --profile confirmation --run
.venv/bin/python -m synthetic.behavior_rollout_timing --profile confirmation --status
```

Outputs are written to `outputs/effectome-rollout-timing/<profile>/post-switch-01/`.
The runner saves the frozen protocol, seed-level errors and paired forecasts,
including each origin's exact history bounds.

## Independent results

All 60 cases completed. The short history first passed at 19.2 seconds, with
mean driver-switch MAE 0.19559 versus 0.20747 for pooling. The long history
first passed at 76.8 seconds, with mean MAE 0.19873 versus 0.21303. At those
points, respectively, 17 and 20 of the 20 seeds cleared the gain margin and
both control tests passed. Approximately 75% of the fitting lag pairs were
post-switch at each history's first supported delay.

The short-history mean gain remained positive at the last grid point, but its
joint adjusted test failed there. The long history passed at both final grid
points. At the first post-switch sample, the pooled forecast had about 0.05
less driver-switch MAE than either local refit. The result motivates a separate
test of a rule that decides from available neural history when to keep the
pooled mechanism and when to adapt.
[Audited timing results](../outputs/effectome-rollout-timing/confirmation/post-switch-01/review/analysis-report.md)
