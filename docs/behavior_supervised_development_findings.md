# Development results for the continuous behavior experiment

The [study plan](behavior_supervised_experiment_plan.md) specifies a fixed
two-dimensional behavioral outcome across four observed neural variables. The
new null cuts all incoming connections into that block and retains an active
nuisance circuit. Notebook 09 implements the comparison on five development
seeds, 40005–40009, using shared factual origins and paired neural trajectories.

| Frozen readout | Driver MAE | Oracle-positive driver MAE | Driver semantic MAE | Null MAE |
|---|---:|---:|---:|---:|
| Declared behavior | 0.001267 | 0.005069 | 0 | 0 |
| PCA, 2D | 0.046005 | 0.096315 | 0.046094 | 0.011091 |
| PCA, 4D | 0.056476 | 0.109979 | 0.056689 | 0.008154 |
| PLS, 2D | 0.046488 | 0.097076 | 0.046576 | 0.008544 |
| Full-neural linear | 0.043771 | 0.118323 | 0.043833 | 0.002378 |
| Behavior-only network, 2D | 0.010807 | 0.029345 | 0.010363 | 0.001360 |
| Joint behavior and dynamics, 2D | 0.010861 | 0.026781 | 0.010583 | 0.001441 |
| Joint behavior and dynamics, 4D | 0.009103 | 0.024142 | 0.008827 | 0.001065 |
| Full-neural nonlinear | 0.007922 | 0.021765 | 0.007496 | 0.000630 |

The table uses fitted known support and averages five seed summaries. MAE
measures errors in continuous behavioral units. Driver MAE includes all twelve
interventions. The oracle-positive column includes interventions with true
seed-level magnitude at least 0.10. Semantic MAE compares the true dynamics
through the learned readout with the same trajectories through the fixed
behavioral reference.

The joint two-dimensional encoder reduced mean driver MAE by 76.4% relative
to two-dimensional PCA and retained low error on the oracle-positive units.
Its mean within-chart MAE was 0.001557, and its held-out origin decoding MAE
was 0.065057. PCA and PLS retained substantial semantic error even with
known neural support. The behavior-only network reached similar overall
accuracy, so the five-seed experiment leaves the incremental value of the
latent transition loss unresolved. Four dimensions narrowed the error gap
with full-neural nonlinear decoding.

All ten development cases completed with 70 available dynamics lanes and
360 native window fits. Joint two-dimensional driver MAE ranged from 0.010861
to 0.014036 across the four native methods. Oracle-positive intervention MAE
ranged from 0.026781 to 0.039541. At the provisional 0.10 cutoff, each method
recovered all eligible singletons and groups on the five driver seeds without
additional nominations, and made zero nominations on the five null seeds.
These counts describe the development sample.

The [saved report](../outputs/behavior-supervised/development/distributed-behavior-01/behavior-supervised-validation.json)
indexes the per-cell evidence. The [effect-error figure](../outputs/behavior-supervised/development/distributed-behavior-01/analysis/development-effect-error.pdf)
shows the paired seed summaries for known support. The
[audit](../outputs/behavior-supervised/development/distributed-behavior-01/analysis/independent-verification.json)
reconstructs the reported errors and verifies the artifact hashes.

This result concerns an independently specified but simulator-defined behavior
in a fully observed, model-matched system. The new errors use different
units and a different generator from notebook 08. Independent confirmation
requires the separate profile and its calibrated decision rules.
