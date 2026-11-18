# Figure catalog

## 01-compression-effect-error

Purpose: compare all PCA dimensions with the task-aligned existing decoder. Horizontal dashed lines show the same full-neural/1D-score mean; blue lines show PCA means and grey lines are the five paired seed trajectories. There are no confidence error bars. Driver and null panels share development seeds. Source: seed-metrics.csv, known-support lane. Notice that four PCs still fail the driver behavior test, while a fifth helps with weak compression. This motivates behavior-prioritized encoding. Caption must state the score is a re-expression, not a newly trained independent algorithm or intrinsic dimension.

## 02-variance-versus-behavior-direction

Purpose: diagnose why a large neural-variance fraction need not preserve this endpoint. Lines are five-seed means and dots are seed values for standardized variance retained and the squared projection of the standardized last-coordinate axis onto each PCA subspace. Source: geometry arrays in probe-evidence.zip. At four PCs variance exceeds 90% while axis alignment remains about 0.50. This supports choosing the target-aware objective rather than a variance threshold. The projection is not behavioral R-squared, intervention-error explained, or a general nonlinear diagnostic. No uncertainty band or population guarantee is claimed.
