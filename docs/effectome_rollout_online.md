# Choosing when to refit from observed neural history

The timing experiment used known synthetic switch times to index forecasts.
It found that pooling beat refitting immediately after a change and that a
256-sample refit first passed its forecast criterion at 19.2 seconds. The new
test evaluates an online choice that can use observed neural history alone.
[Post-switch timing study](effectome_rollout_timing.md)

## Policy and comparisons

1. Fit the readout on ten trials. Estimate fixed support and the pooled
   mechanism using 1,024-sample histories from fourteen other training trials.
   Forecast every 16 samples in sixteen held-out trials. The local mechanism
   refits coefficients on the same support using the last 256 samples.
2. Record both candidate forecasts on matched origins. A target averages steps
   five through eight. Once all of its neural samples have been observed, the
   frozen decoder supplies a behavioral estimate for scoring that forecast.
   A forecast issued now cannot enter the score until its target matures.
   This evaluation follows prediction in time. [Dawid, 1984](https://doi.org/10.2307/2981683)
3. Select the local forecast only if its average reduction in decoded-target
   absolute error over the last completed forecasts exceeds the chosen margin.
   The rule starts with the pooled forecast until its full lookback is available.
   Each trial has an independent history. The policy consumes predicted and
   neural-decoded targets and their times. True behavior and simulator switch
   times enter calibration and evaluation, with the first 192 post-switch
   samples marking the early scoring band. Both models issue forecasts to keep
   their comparison current, even when the pooled forecast is selected.

Five calibration seeds (87000–87004) select a lookback of 4, 8, or 16 completed
forecasts and a score margin of 0, 0.005, 0.01, 0.02, or 0.04. Among rules with
a positive mean early gain over always-local and mean control harm below
0.005, use the largest driver-switch mean gain over always-pooled. Grid order
breaks ties. If the eligible set is empty, use always-pooled. The rule is saved
before forecasting the 20 confirmation seeds (88000–88019).

Confirmation scores the realized future behavioral target. Each seed supplies
one paired gain. Four exact one-sided sign tests must pass at 0.05. In the
driver-switch scenario the rule must beat always-pooled by 0.005 across all
origins and beat always-local by 0.005 in the early scoring band. Each control
requires harm relative to pooling below 0.005. Each test asks whether a
majority of seeds clears its margin. The complete claim requires all four
tests. Selection fractions are descriptive.

```sh
.venv/bin/python -m synthetic.behavior_rollout_online --profile smoke --run
.venv/bin/python -m synthetic.behavior_rollout_online --profile confirmation --run
.venv/bin/python -m synthetic.behavior_rollout_online --profile confirmation --status
```

Outputs are under `outputs/effectome-rollout-online/<profile>/forecast-selection-01/`.
The experiment evaluates forecast choice in a matched, fully observed simulator.
Switch-time localization and performance in neural recordings require separate evidence.

## Independent result

Calibration selected a lookback of 16 completed forecasts and a score margin
of zero. The rule lowered full-trajectory driver-switch MAE from 0.21122 for
pooling to 0.20292. All 20 seeds exceeded the improvement margin. Both control
noninferiority tests passed, with mean harm from selecting local forecasts of
about 0.0012. The always-local arm reached 0.19857 across the full driver-switch
trajectory.

In the first 19.2 seconds after a known switch, the rule improved on always
refitting by 0.00544 on average. Twelve of 20 seeds exceeded the 0.005 margin
and the one-sided exact $p$-value was 0.252. The full decision failed at this
early comparison. The rule selected the local model at 64% of driver-switch
origins and 28% of origins in each control, on average. These results support
tracking completed forecasts as a way to improve on pooling in the switching
simulator. Reliably improving early forecasts remains the acceptance hurdle
for using the rule in the working pipeline.
[Audited online results](../outputs/effectome-rollout-online/confirmation/forecast-selection-01/review/analysis-report.md)
