# Learned behavioral endpoint development

Completed cells: 10/10. Available neural-dynamics lanes: 70/70.

This is a PCA bridge using development seeds, not a nonlinear manifold or confirmatory result.
Behavioral error compares fitted chart effects with independently defined simulator behavior. Within-chart error compares fitted and simulator effects through the same frozen chart.

| Scenario | Dynamics | Endpoint | Seeds | Behavioral MAE | Within-chart MAE | Semantic MAE |
|---|---|---|---:|---:|---:|---:|
| driver | conventional_cgc | behavior_reference | 5 | 0.001912 | 0.001912 | 0.000000 |
| driver | conventional_cgc | constant | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | conventional_cgc | full_neural_behavior | 5 | 0.001969 | 0.002067 | 0.000553 |
| driver | conventional_cgc | nuisance | 5 | 0.189250 | 0.002132 | 0.189054 |
| driver | conventional_cgc | pca_behavior | 5 | 0.138818 | 0.002181 | 0.138428 |
| driver | conventional_cgc | pca_geometry | 5 | 0.131144 | 0.002018 | 0.130737 |
| driver | conventional_cgc | proxy | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | conventional_cgc_star | behavior_reference | 5 | 0.001912 | 0.001912 | 0.000000 |
| driver | conventional_cgc_star | constant | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | conventional_cgc_star | full_neural_behavior | 5 | 0.001969 | 0.002067 | 0.000553 |
| driver | conventional_cgc_star | nuisance | 5 | 0.189250 | 0.002132 | 0.189054 |
| driver | conventional_cgc_star | pca_behavior | 5 | 0.138818 | 0.002181 | 0.138428 |
| driver | conventional_cgc_star | pca_geometry | 5 | 0.131144 | 0.002018 | 0.130737 |
| driver | conventional_cgc_star | proxy | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | empty_support | behavior_reference | 5 | 0.128532 | 0.128532 | 0.000000 |
| driver | empty_support | constant | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | empty_support | full_neural_behavior | 5 | 0.128532 | 0.128467 | 0.000553 |
| driver | empty_support | nuisance | 5 | 0.128532 | 0.070793 | 0.189054 |
| driver | empty_support | pca_behavior | 5 | 0.128532 | 0.024333 | 0.138428 |
| driver | empty_support | pca_geometry | 5 | 0.128532 | 0.021753 | 0.130737 |
| driver | empty_support | proxy | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | fast_cgc | behavior_reference | 5 | 0.001912 | 0.001912 | 0.000000 |
| driver | fast_cgc | constant | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | fast_cgc | full_neural_behavior | 5 | 0.001969 | 0.002067 | 0.000553 |
| driver | fast_cgc | nuisance | 5 | 0.189250 | 0.002132 | 0.189054 |
| driver | fast_cgc | pca_behavior | 5 | 0.138818 | 0.002181 | 0.138428 |
| driver | fast_cgc | pca_geometry | 5 | 0.131144 | 0.002018 | 0.130737 |
| driver | fast_cgc | proxy | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | fast_cgc_star | behavior_reference | 5 | 0.001912 | 0.001912 | 0.000000 |
| driver | fast_cgc_star | constant | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | fast_cgc_star | full_neural_behavior | 5 | 0.001969 | 0.002067 | 0.000553 |
| driver | fast_cgc_star | nuisance | 5 | 0.189250 | 0.002132 | 0.189054 |
| driver | fast_cgc_star | pca_behavior | 5 | 0.138818 | 0.002181 | 0.138428 |
| driver | fast_cgc_star | pca_geometry | 5 | 0.131144 | 0.002018 | 0.130737 |
| driver | fast_cgc_star | proxy | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | known_support | behavior_reference | 5 | 0.001912 | 0.001912 | 0.000000 |
| driver | known_support | constant | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | known_support | full_neural_behavior | 5 | 0.001969 | 0.002067 | 0.000553 |
| driver | known_support | nuisance | 5 | 0.189250 | 0.002132 | 0.189054 |
| driver | known_support | pca_behavior | 5 | 0.138818 | 0.002181 | 0.138428 |
| driver | known_support | pca_geometry | 5 | 0.131144 | 0.002018 | 0.130737 |
| driver | known_support | proxy | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | oracle | behavior_reference | 5 | 0.000000 | 0.000000 | 0.000000 |
| driver | oracle | constant | 5 | 0.128532 | 0.000000 | 0.128532 |
| driver | oracle | full_neural_behavior | 5 | 0.000553 | 0.000000 | 0.000553 |
| driver | oracle | nuisance | 5 | 0.189054 | 0.000000 | 0.189054 |
| driver | oracle | pca_behavior | 5 | 0.138428 | 0.000000 | 0.138428 |
| driver | oracle | pca_geometry | 5 | 0.130737 | 0.000000 | 0.130737 |
| driver | oracle | proxy | 5 | 0.128532 | 0.000000 | 0.128532 |
| null | conventional_cgc | behavior_reference | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc | constant | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc | full_neural_behavior | 5 | 0.000073 | 0.000122 | 0.000049 |
| null | conventional_cgc | nuisance | 5 | 0.070719 | 0.002124 | 0.070793 |
| null | conventional_cgc | pca_behavior | 5 | 0.003149 | 0.001278 | 0.002832 |
| null | conventional_cgc | pca_geometry | 5 | 0.019295 | 0.002043 | 0.019238 |
| null | conventional_cgc | proxy | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc_star | behavior_reference | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc_star | constant | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | conventional_cgc_star | full_neural_behavior | 5 | 0.000073 | 0.000122 | 0.000049 |
| null | conventional_cgc_star | nuisance | 5 | 0.070719 | 0.002124 | 0.070793 |
| null | conventional_cgc_star | pca_behavior | 5 | 0.003149 | 0.001278 | 0.002832 |
| null | conventional_cgc_star | pca_geometry | 5 | 0.019295 | 0.002043 | 0.019238 |
| null | conventional_cgc_star | proxy | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | empty_support | behavior_reference | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | empty_support | constant | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | empty_support | full_neural_behavior | 5 | 0.000000 | 0.000049 | 0.000049 |
| null | empty_support | nuisance | 5 | 0.000000 | 0.070793 | 0.070793 |
| null | empty_support | pca_behavior | 5 | 0.000000 | 0.002832 | 0.002832 |
| null | empty_support | pca_geometry | 5 | 0.000000 | 0.019238 | 0.019238 |
| null | empty_support | proxy | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc | behavior_reference | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc | constant | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc | full_neural_behavior | 5 | 0.000073 | 0.000122 | 0.000049 |
| null | fast_cgc | nuisance | 5 | 0.070719 | 0.002124 | 0.070793 |
| null | fast_cgc | pca_behavior | 5 | 0.003149 | 0.001278 | 0.002832 |
| null | fast_cgc | pca_geometry | 5 | 0.019295 | 0.002043 | 0.019238 |
| null | fast_cgc | proxy | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc_star | behavior_reference | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc_star | constant | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | fast_cgc_star | full_neural_behavior | 5 | 0.000073 | 0.000122 | 0.000049 |
| null | fast_cgc_star | nuisance | 5 | 0.070719 | 0.002124 | 0.070793 |
| null | fast_cgc_star | pca_behavior | 5 | 0.003149 | 0.001278 | 0.002832 |
| null | fast_cgc_star | pca_geometry | 5 | 0.019295 | 0.002043 | 0.019238 |
| null | fast_cgc_star | proxy | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | known_support | behavior_reference | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | known_support | constant | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | known_support | full_neural_behavior | 5 | 0.000073 | 0.000122 | 0.000049 |
| null | known_support | nuisance | 5 | 0.070719 | 0.002124 | 0.070793 |
| null | known_support | pca_behavior | 5 | 0.003149 | 0.001278 | 0.002832 |
| null | known_support | pca_geometry | 5 | 0.019295 | 0.002043 | 0.019238 |
| null | known_support | proxy | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | oracle | behavior_reference | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | oracle | constant | 5 | 0.000000 | 0.000000 | 0.000000 |
| null | oracle | full_neural_behavior | 5 | 0.000049 | 0.000000 | 0.000049 |
| null | oracle | nuisance | 5 | 0.070793 | 0.000000 | 0.070793 |
| null | oracle | pca_behavior | 5 | 0.002832 | 0.000000 | 0.002832 |
| null | oracle | pca_geometry | 5 | 0.019238 | 0.000000 | 0.019238 |
| null | oracle | proxy | 5 | 0.000000 | 0.000000 | 0.000000 |

No calibration/test pools were used. Negative charts remain in every comparison. A collapsed chart cannot pass behavioral validity because its within-chart MAE is zero.

Method rationale and verified reference scopes: `docs/learned_manifold_validation_plan.md` and its citation audit.
