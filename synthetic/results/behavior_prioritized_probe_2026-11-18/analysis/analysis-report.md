# Behavior-first compression: an exact readout control and a dimension diagnostic

A one-dimensional affine score formed from the already trained full-neural classifier preserved its source labels and mean-score future events exactly: zero disagreement on 276,480 natural observations and 1,935,360 rollout events. Driver effect MAE remains 0.001969; null effect MAE remains 0.000073. This uses zero new native/mechanism fits, and the score itself has no new fitting step.

| Representation | Dimensions | Driver effect MAE, mean ± SD | Null effect MAE |
|---|---:|---:|---:|
| pca_behavior_k1 | 1 | 0.146908 ± 0.010372 | 0.032682 |
| pca_behavior_k2 | 2 | 0.138818 ± 0.008227 | 0.003149 |
| pca_behavior_k3 | 3 | 0.138875 ± 0.008477 | 0.000920 |
| pca_behavior_k4 | 4 | 0.138818 ± 0.008428 | 0.000684 |
| pca_behavior_k5 | 5 | 0.018994 ± 0.011502 | 0.000545 |
| behavior_score_1d | 1 | 0.001969 ± 0.001074 | 0.000073 |
| full_neural_behavior | 6 | 0.001969 ± 0.001074 | 0.000073 |

The score is the existing classifier expressed through its decision coordinate, not an independent new method. These large event counts audit the transformation; n remains five. All neural variables are still observed and simulated. A one-dimensional binary score does not identify one-dimensional behavioral geometry, reduced sensing or latent dynamics.

PCA dimensions one through five are all disclosed. Four PCs retain 90.45% of standardized neural variance while giving behavioral MAE 0.138818; the squared projection of the standardized behavioral coordinate axis into that PCA subspace is only 0.497672. This geometric diagnostic is not behavioral R-squared or causal recovery. Five PCs reduce MAE to 0.018994 but retain five of six neural directions, whereas the task-aligned score preserves the successful parent readout in one coordinate.

The practical route is to learn the measured behavior objective before selecting a code. Use this supervised score as the binary-task baseline, then evaluate explicitly learned bottlenecks on prespecified continuous/distributed behavior. The new behavior needs a new oracle-checked null if it includes neural coordinates influenced by the current nuisance path. Freeze mappings and evaluation rules before fresh calibration/confirmation. The literature review compares PLS, PSID, BunDLe, CEBRA and supervised autoencoders with their assumptions and runtime constraints.

These are retrospective development probes. No test-pool selection, nonlinear-encoder validation, new causal identifiability claim or graph-branch merge occurred.
