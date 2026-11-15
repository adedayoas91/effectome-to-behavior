# Four imported CSL estimators: native graph sanity check

Available fits: 24/24. Independent data seeds per condition: 3.

| Method | Condition | Available/planned | TP mean ± SD | FP mean ± SD | Recall mean | Exact graphs |
|---|---|---:|---:|---:|---:|---:|
| c-GC | chain | 3/3 | 2.00 ± 0.00 | 0.00 ± 0.00 | 1.000 | 3/3 |
| c-GC | null | 3/3 | 0.00 ± 0.00 | 0.00 ± 0.00 | undefined | 3/3 |
| c-GC* | chain | 3/3 | 2.00 ± 0.00 | 0.00 ± 0.00 | 1.000 | 3/3 |
| c-GC* | null | 3/3 | 0.00 ± 0.00 | 0.00 ± 0.00 | undefined | 3/3 |
| fastC-GC | chain | 3/3 | 2.00 ± 0.00 | 0.00 ± 0.00 | 1.000 | 3/3 |
| fastC-GC | null | 3/3 | 0.00 ± 0.00 | 0.00 ± 0.00 | undefined | 3/3 |
| fastC-GC* | chain | 3/3 | 2.00 ± 0.00 | 0.00 ± 0.00 | 1.000 | 3/3 |
| fastC-GC* | null | 3/3 | 0.00 ± 0.00 | 0.00 ± 0.00 | undefined | 3/3 |

A correct chain graph contains 0→1 and 1→2; neuron 3 is disconnected. Negative coupling on 1→2 is evaluated as support because these effectomes are unsigned.

Pairwise Jaccard scores are in `pairwise-agreement.csv`. Two empty graphs have undefined Jaccard and never count as nontrivial agreement. Compare agreement with truth: methods can share the same false edges.

These four methods form one discovery family. This study checks native graph recovery on directly observed linear Gaussian data; it does not establish manifold transitions, transition-driver recovery, calcium robustness or biological necessity. The upstream conventional CPU permutation RNG is not fully controlled by its random_state argument. Saved artifacts preserve the actual realization; reruns can differ.
