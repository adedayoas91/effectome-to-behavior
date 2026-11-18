# Replication figure catalog

All files have PDF and PNG versions. Source: the completed replication-report.json,
its seed_metrics, summary and paired_history records. No new fitting, threshold
selection or acceptance recomputation occurs during export. Missing complete-set
statistics remain NA in the frozen acceptance tables. Hollow points additionally
show the already-recorded available_mean_descriptive on eligible seeds, without
an interval or a complete-set acceptance claim. Actual eligible/planned counts
remain explicit in the JSON and results discussion. Related methods and
directions are not independent seed replicates.

## 01-fresh-seed-history

Purpose: compare the known/empty-support controls and all native methods at paired
256/1024 histories. Filled dots are complete-set seed means, hollow dots are
eligible-subset descriptions; bars are the runner's 2,000-resample
descriptive seed-bootstrap 95% intervals, with the actual complete counts in JSON.
Dashed lines show the frozen 0.75 ranking and 0.05 MAE gates. Notice whether longer
histories approach the known-support control consistently across fresh seeds.
The secondary history has no calibrated nomination/null claim. The two histories
share inputs within seed; their means are not independent groups.

## 02-nomination-and-null

Purpose: separate positive nomination performance from specificity. Positive
points show recall/precision with descriptive seed-bootstrap 95% intervals. Null
points show any-false-nominee fractions in untouched seeds; caps are one-sided
Clopper–Pearson 95% upper bounds, per method. Dashed thresholds are 0.75 and0.05.
Read actual counts and all gate decisions from analysis-report.md. Calibration
is global-null marginal protection and does not imply strong non-null FDR control.

## 03-paired-mae-changes

Purpose: expose every paired raw-seed MAE change rather than a selected example.
Crosses are independent process-seed differences, dots their mean and bars the
runner's descriptive seed-bootstrap 95% interval. Negative changes favor history
1024. The four planned native paired sign tests use one Holm family in JSON.
Neither a significant history contrast nor a favorable mean overrides a failed
primary nomination, null or control gate. This observed model-matched generator
does not establish robustness to hidden state or biological intervention validity.
