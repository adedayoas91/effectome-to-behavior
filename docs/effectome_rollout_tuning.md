# Tuning online refit selection

The original selector carried a long forecast-loss average across a coupling
change. Rapid fallback fixed the early comparison, but it selected the local
mechanism infrequently even after the circuit had settled. This study tunes
the selector while holding the readout, support discovery, pooled and local
fits, forecast horizon, and intervention definition fixed. Shorter lookbacks
can clear stale evidence sooner. Fewer required recovery forecasts can return
to the local model sooner. The loss streak and margin control how readily this
happens in the two controls. [Matched fast-fallback comparison](effectome_rollout_pipeline.md)

## Protocol

Replay the grid in [tuning.yaml](../conf/behavior_rollout_pipeline/tuning.yaml)
on the two previously inspected 20-seed panels, 88000–88019 and 89000–89019.
Both panels are now development data. The selection input is the same as
before. Each choice receives the error of past pooled and local forecasts
against neural-decoded targets only after those targets mature. Realized
behavior, simulator switch times, and oracle intervention effects enter
the development assessment of candidate settings.

Require simulated active-effect MAE in the driver-switch scenario to be
at most that of the fixed fallback policy on the later development panel.
Among eligible settings, prefer those that pass all four existing forecast
criteria separately in both development panels. From those, choose the
smallest mean full-trajectory driver-switch MAE, with mean early improvement
and then active-effect MAE as tie-breakers. If none passes, maximize the
smallest count of seeds clearing any one criterion in either panel, then
use the same tie-breakers. The grid contains the fixed fallback as a
reference. This ranking uses development data and carries no confirmatory
significance claim.

Freeze one configuration, then evaluate it on process seeds 90000–90019.
Compare against the fixed fallback, the previous rolling selector, and
always-pooled and always-local on the same origins and simulated effects.
Retain the same four exact sign tests, 0.005 MAE margin, 19.2-second early
window, and intersection decision at 0.05. Report active-effect scores
as descriptive outcomes. The confirmation data will be used once for this
fixed configuration. The forecast loss timing follows sequential
prediction evaluation. [Dawid (1984)](https://doi.org/10.2307/2981683)

## Run and review

Run from the package root with the existing fastC-GC and behavior encoder
environments. The two archived development bundles must be available at the
paths in the tuning configuration.

```sh
.venv/bin/python -m synthetic.behavior_rollout_tuning --select
.venv/bin/python -m synthetic.behavior_rollout_tuning --confirm --workers 3
.venv/bin/python -m synthetic.behavior_rollout_tuning --status
```

The run writes to `outputs/effectome-rollout-tuning/tuning-01/`. The
development grid writes `grid-metrics.csv` and `frozen-policy.json`.
Confirmation fits the two candidate mechanisms and the readout once per
case. All selectors replay the same held-out forecasts and the same paired
intervention vectors. The runner stores matched policy rows in
`confirmation-metrics.csv` and the seed-level decisions in
`tuning-report.json`. The completed cases under `confirmation/cases/`
retain the underlying forecast, readout, support, and intervention evidence.

## Confirmation results

The selected policy uses six completed forecasts with zero margin,
triggers fallback after three consecutive local losses, and requires one
new favorable completed forecast to resume local selection if the rolling
mean also favors it. Of the 270 candidates, 168 met the development effect
requirement and 130 also passed all four forecast tests in both historical
panels. The selected candidate passed all four tests in the fresh panel
of 20 seeds. The run comprised 60 cases, 199,440 forecast origins, and
6,624 evaluated intervention origins. [Frozen selection](../outputs/effectome-rollout-tuning/tuning-01/frozen-policy.json)

| Forecast criterion | Mean gain in MAE | Seeds clearing the margin |
|---|---:|---:|
| Driver switch, full trajectory vs. pooling | 0.00903 | 20/20 |
| Driver switch, first 19.2 s vs. always-local | 0.00922 | 18/20 |
| Constant vs. pooling | −0.00146 | 20/20 |
| Nuisance switch vs. pooling | −0.00146 | 20/20 |

The joint exact test gave $p=0.000201$. On the same seeds, fixed fallback
missed the full-trajectory criterion and the previous online selector
missed the early criterion. Driver-switch full-trajectory MAE was 0.20200
with tuning, 0.20548 with fixed fallback, and 0.20228 with the previous
selector. The paired gain over fixed fallback was 0.00348 MAE with a
pointwise 95% seed-bootstrap interval of [0.00315, 0.00383]. The gain over the
previous selector was 0.00027 with an interval of [−0.00006, 0.00059].
[Audited summary](../outputs/effectome-rollout-tuning/tuning-01/review/analysis-report.md)

Fixed fallback retained a larger early gain over always-local (0.01166
versus 0.00922). The tuned rule selected local refits at 59.7% of
driver-switch origins, compared with 36.4% for fixed fallback. The
active-effect MAE was 0.06664 for tuning, 0.07474 for fixed fallback, and
0.06273 for the previous online rule. Always-local achieved the lowest
full-trajectory (0.19800) and active-effect (0.04492) errors, while the
tuned policy met the early and control criteria as well as the
full-trajectory criterion. [Paired gain plot](../outputs/effectome-rollout-tuning/tuning-01/review/paired-driver-gains.pdf)
