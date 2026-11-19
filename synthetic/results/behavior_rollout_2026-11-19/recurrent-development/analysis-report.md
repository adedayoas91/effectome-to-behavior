# Recurrent-coupling development results

The full panel completed 15 scenario–seed cases across five independent seeds, with two histories per case and 2,880 native fast c-GC windows.
The audit reconstructed forecast errors, verified trailing histories and whole-trial partitions, and checked generating-regime support and paired control outcomes.

| Scenario | History samples | Recent history MAE | Static graph MAE | Dynamic graph MAE | Oracle rollout MAE | Static rollout MAE | Dynamic rollout MAE | Native support Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constant | 256 | 0.25486 | 0.25420 | 0.25648 | 0.19238 | 0.21415 | 0.21817 | 0.18429 |
| constant | 1024 | 0.26848 | 0.27879 | 0.26674 | 0.19238 | 0.19242 | 0.19040 | 0.95909 |
| driver_switch | 256 | 0.25610 | 0.25597 | 0.24041 | 0.17838 | 0.20490 | 0.20623 | 0.18458 |
| driver_switch | 1024 | 0.24405 | 0.24254 | 0.24204 | 0.17838 | 0.20395 | 0.17862 | 0.95436 |
| nuisance_switch | 256 | 0.25856 | 0.25807 | 0.26369 | 0.19238 | 0.21406 | 0.21835 | 0.18263 |
| nuisance_switch | 1024 | 0.25136 | 0.26110 | 0.24681 | 0.19238 | 0.19240 | 0.19029 | 0.95982 |

## Interpretation

The direct graph-augmented regression and forecast rollouts answer different questions. The direct regression tests whether an origin graph-action feature provides incremental prediction under a common forecast family. The mechanistic rollout tests whether locally estimated coupling improves future simulations compared with pooling couplings into one static mechanism.

These are descriptive development comparisons. The confirmation profile is reserved for independent seeds. The forecast origins follow a complete within-regime observation history.
