# Independent confirmation of mechanism rollout

All 60 cases completed, with 3,960 native fits. The audit verified 180 artifact hashes, rebuilt the future targets and trailing histories from the simulator, and recomputed errors and decisions.
Frozen forecast decision **fail**.

| Scenario | Pooled MAE | Fixed-support local MAE | Dynamic-support local MAE | Oracle MAE | Dynamic vs pooled gain with 95% interval | Seeds meeting margin | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Driver switch | 0.21886 | 0.20175 | 0.20217 | 0.19864 | 0.01670 [0.00848, 0.02569] | 12/20 | 0.251722 |
| Constant | 0.18396 | 0.18615 | 0.18656 | 0.18410 | -0.00260 [-0.00410, -0.00108] | 20/20 | 0.000001 |
| Nuisance switch | 0.18389 | 0.18607 | 0.18635 | 0.18410 | -0.00246 [-0.00397, -0.00095] | 20/20 | 0.000001 |

## Adaptation and support rediscovery

| Scenario | Gain from local adaptation | Further gain from rediscovering support | Pooled active-effect MAE | Fixed-support active-effect MAE | Dynamic-support active-effect MAE |
|---|---:|---:|---:|---:|---:|
| Driver switch | 0.01711 | -0.00041 | 0.09074 | 0.01515 | 0.01837 |
| Constant | -0.00219 | -0.00041 | 0.00342 | 0.01457 | 0.01761 |
| Nuisance switch | -0.00218 | -0.00028 | 0.00358 | 0.01459 | 0.01787 |

Forecasting used established regimes and fully observed neural states. The fixed-support and intervention comparisons are descriptive. The dynamic-support test concerns the paired forecast contrast against a pooled mechanism under the declared controls.
