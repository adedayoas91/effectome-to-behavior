# Fresh-seed mechanism replication: completed 17 November 2026

Start with the [compiled results/discussion](analysis/replication-results.pdf)
and its [LaTeX source](analysis/replication-results.tex). The frozen decision tables
are in [analysis-report.md](analysis/analysis-report.md); exact seed metrics,
statistics, eligibility diagnostics and three scientific figures are in analysis/.

The full study completed 119 cells and 714/714 method/control lanes, including 4,284
actual native window fits. Pools were 19 separate calibration-null seeds, 20 fresh
positive-test seeds and 60 untouched null-test seeds; the same 20 positive seeds
also received a paired 256-sample-history comparison. Primary history was 1,024.
All four native methods have primary mean effect MAE 0.0020874; longer histories
reduce mean MAE by 97.55–98.36% relative to 256, with Holm paired p≤0.00003052.
Each native method has 0/60 null false-event seeds and a per-method one-sided 95%
exact upper bound 0.0487029, passing the 0.05 gate.

Individual ranking and nomination are perfect on the 19 eligible seeds. Seed 20012,
source 1, has no individually positive neuron at the frozen 0.10 gold threshold;
its joint driver-pair effect is 0.230469, correctly estimated and nominated by all
four native methods. The known-support control has the same individual-ranking
ineligibility. The prespecified requirement for 20 eligible ranking seeds therefore
remains unmet: **the all-four frozen acceptance flag remains False**. Conditional
19-seed summaries do not replace it. No threshold, denominator or gate was changed
after inspection. The empty-support control behaves as expected. This is a
favorable fully observed/model-matched control; the old latent benchmark remains
a separate failed evaluation. No graph-similarity method was added or evaluated.

Reproduce tables/figures and the report source from the application repository:

```sh
.venv/bin/python -m synthetic.write_mechanism_replication_results --output-root outputs/driver-mechanism-replication/replication/mechanism-replication-01
```

The read-only exporter requires a complete report and never fits, recalibrates or
changes acceptance. To recompile this archived PDF, run
`tectonic --keep-logs replication-results.tex` from this bundle's analysis/ folder.
The ready-to-run entrypoint is synthetic/07_mechanism_replication.ipynb; do not
rerun these inspected test seeds to tune settings. A changed scientific protocol
requires fresh test pools.

The protocol and verified reference/claim audit accompany this bundle. Source,
all completed cell caches, first-completed native graphs/p-values, logs and hashes
are preserved in implementation-source.zip, native-evidence-and-cells.zip and the
provenance/manifest files. Conventional CPU RNG remains uncontrolled by the public
upstream seed API; first-completed evidence is retained without favorable reruns.
Verification: 654 regression tests passed, 1 skipped; Ruff and mypy src/ passed;
952 native artifact hashes and independent seed-level statistics were checked.
