# Learned behavioral endpoint development

Completed cells: 2/2. Available neural-dynamics lanes: 14/14.

This is a PCA bridge using development seeds, not a nonlinear manifold or confirmatory result.
Behavioral error compares fitted chart effects with independently defined simulator behavior. Within-chart error compares fitted and simulator effects through the same frozen chart.

| Scenario | Dynamics | Endpoint | Seeds | Behavioral MAE | Within-chart MAE | Semantic MAE |
|---|---|---|---:|---:|---:|---:|
| driver | conventional_cgc | behavior_reference | 1 | 0.001587 | 0.001587 | 0.000000 |
| driver | conventional_cgc | constant | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | conventional_cgc | full_neural_behavior | 1 | 0.001668 | 0.001872 | 0.000692 |
| driver | conventional_cgc | nuisance | 1 | 0.173177 | 0.001709 | 0.173340 |
| driver | conventional_cgc | pca_behavior | 1 | 0.133179 | 0.002726 | 0.132324 |
| driver | conventional_cgc | pca_geometry | 1 | 0.127157 | 0.002604 | 0.126099 |
| driver | conventional_cgc | proxy | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | conventional_cgc_star | behavior_reference | 1 | 0.001587 | 0.001587 | 0.000000 |
| driver | conventional_cgc_star | constant | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | conventional_cgc_star | full_neural_behavior | 1 | 0.001668 | 0.001872 | 0.000692 |
| driver | conventional_cgc_star | nuisance | 1 | 0.173177 | 0.001709 | 0.173340 |
| driver | conventional_cgc_star | pca_behavior | 1 | 0.133179 | 0.002726 | 0.132324 |
| driver | conventional_cgc_star | pca_geometry | 1 | 0.127157 | 0.002604 | 0.126099 |
| driver | conventional_cgc_star | proxy | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | empty_support | behavior_reference | 1 | 0.120361 | 0.120361 | 0.000000 |
| driver | empty_support | constant | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | empty_support | full_neural_behavior | 1 | 0.120361 | 0.120158 | 0.000692 |
| driver | empty_support | nuisance | 1 | 0.120361 | 0.063802 | 0.173340 |
| driver | empty_support | pca_behavior | 1 | 0.120361 | 0.025553 | 0.132324 |
| driver | empty_support | pca_geometry | 1 | 0.120361 | 0.024373 | 0.126099 |
| driver | empty_support | proxy | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | fast_cgc | behavior_reference | 1 | 0.001587 | 0.001587 | 0.000000 |
| driver | fast_cgc | constant | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | fast_cgc | full_neural_behavior | 1 | 0.001668 | 0.001872 | 0.000692 |
| driver | fast_cgc | nuisance | 1 | 0.173177 | 0.001709 | 0.173340 |
| driver | fast_cgc | pca_behavior | 1 | 0.133179 | 0.002726 | 0.132324 |
| driver | fast_cgc | pca_geometry | 1 | 0.127157 | 0.002604 | 0.126099 |
| driver | fast_cgc | proxy | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | fast_cgc_star | behavior_reference | 1 | 0.001587 | 0.001587 | 0.000000 |
| driver | fast_cgc_star | constant | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | fast_cgc_star | full_neural_behavior | 1 | 0.001668 | 0.001872 | 0.000692 |
| driver | fast_cgc_star | nuisance | 1 | 0.173177 | 0.001709 | 0.173340 |
| driver | fast_cgc_star | pca_behavior | 1 | 0.133179 | 0.002726 | 0.132324 |
| driver | fast_cgc_star | pca_geometry | 1 | 0.127157 | 0.002604 | 0.126099 |
| driver | fast_cgc_star | proxy | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | known_support | behavior_reference | 1 | 0.001587 | 0.001587 | 0.000000 |
| driver | known_support | constant | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | known_support | full_neural_behavior | 1 | 0.001668 | 0.001872 | 0.000692 |
| driver | known_support | nuisance | 1 | 0.173177 | 0.001709 | 0.173340 |
| driver | known_support | pca_behavior | 1 | 0.133179 | 0.002726 | 0.132324 |
| driver | known_support | pca_geometry | 1 | 0.127157 | 0.002604 | 0.126099 |
| driver | known_support | proxy | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | oracle | behavior_reference | 1 | 0.000000 | 0.000000 | 0.000000 |
| driver | oracle | constant | 1 | 0.120361 | 0.000000 | 0.120361 |
| driver | oracle | full_neural_behavior | 1 | 0.000692 | 0.000000 | 0.000692 |
| driver | oracle | nuisance | 1 | 0.173340 | 0.000000 | 0.173340 |
| driver | oracle | pca_behavior | 1 | 0.132324 | 0.000000 | 0.132324 |
| driver | oracle | pca_geometry | 1 | 0.126099 | 0.000000 | 0.126099 |
| driver | oracle | proxy | 1 | 0.120361 | 0.000000 | 0.120361 |
| null | conventional_cgc | behavior_reference | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc | constant | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc | full_neural_behavior | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc | nuisance | 1 | 0.063232 | 0.001790 | 0.063802 |
| null | conventional_cgc | pca_behavior | 1 | 0.001790 | 0.001587 | 0.001424 |
| null | conventional_cgc | pca_geometry | 1 | 0.018880 | 0.002116 | 0.018555 |
| null | conventional_cgc | proxy | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc_star | behavior_reference | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc_star | constant | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc_star | full_neural_behavior | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc_star | nuisance | 1 | 0.063232 | 0.001790 | 0.063802 |
| null | conventional_cgc_star | pca_behavior | 1 | 0.001790 | 0.001587 | 0.001424 |
| null | conventional_cgc_star | pca_geometry | 1 | 0.018880 | 0.002116 | 0.018555 |
| null | conventional_cgc_star | proxy | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | empty_support | behavior_reference | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | empty_support | constant | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | empty_support | full_neural_behavior | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | empty_support | nuisance | 1 | 0.000000 | 0.063802 | 0.063802 |
| null | empty_support | pca_behavior | 1 | 0.000000 | 0.001424 | 0.001424 |
| null | empty_support | pca_geometry | 1 | 0.000000 | 0.018555 | 0.018555 |
| null | empty_support | proxy | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc | behavior_reference | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc | constant | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc | full_neural_behavior | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc | nuisance | 1 | 0.063232 | 0.001790 | 0.063802 |
| null | fast_cgc | pca_behavior | 1 | 0.001790 | 0.001587 | 0.001424 |
| null | fast_cgc | pca_geometry | 1 | 0.018880 | 0.002116 | 0.018555 |
| null | fast_cgc | proxy | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc_star | behavior_reference | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc_star | constant | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc_star | full_neural_behavior | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc_star | nuisance | 1 | 0.063232 | 0.001790 | 0.063802 |
| null | fast_cgc_star | pca_behavior | 1 | 0.001790 | 0.001587 | 0.001424 |
| null | fast_cgc_star | pca_geometry | 1 | 0.018880 | 0.002116 | 0.018555 |
| null | fast_cgc_star | proxy | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | known_support | behavior_reference | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | known_support | constant | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | known_support | full_neural_behavior | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | known_support | nuisance | 1 | 0.063232 | 0.001790 | 0.063802 |
| null | known_support | pca_behavior | 1 | 0.001790 | 0.001587 | 0.001424 |
| null | known_support | pca_geometry | 1 | 0.018880 | 0.002116 | 0.018555 |
| null | known_support | proxy | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | oracle | behavior_reference | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | oracle | constant | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | oracle | full_neural_behavior | 1 | 0.000000 | 0.000000 | 0.000000 |
| null | oracle | nuisance | 1 | 0.063802 | 0.000000 | 0.063802 |
| null | oracle | pca_behavior | 1 | 0.001424 | 0.000000 | 0.001424 |
| null | oracle | pca_geometry | 1 | 0.018555 | 0.000000 | 0.018555 |
| null | oracle | proxy | 1 | 0.000000 | 0.000000 | 0.000000 |

No calibration/test pools were used. Negative charts remain in every comparison. A collapsed chart cannot pass behavioral validity because its within-chart MAE is zero.

Method rationale and verified reference scopes: `docs/learned_manifold_validation_plan.md` and its citation audit.
