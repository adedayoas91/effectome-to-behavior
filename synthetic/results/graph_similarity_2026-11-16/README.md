# Identity and topology grouping: structural development evidence

Executed 16 November 2026, after the methodology/notebook checkpoint `8ce5ec6`
was pushed to `dev`. That branch remains the stable faster-machine simulation
checkpoint. The graph extension is isolated on `feat/effectome-graph-similarity`.

This is a planted-graph structural diagnostic. It does not refit native CSL,
simulate biological behaviour or test causal driver recovery. The three seeds
are graph-sampling repeats, not neural-process or biological replicates.

- `graph-similarity-protocol.json`: settings and evaluated implementation hash.
- `graph-similarity.json`: all 36 requested fits, including the three unavailable
  clean WL K=4 settings, held-out assignments, input fingerprints and probes.
- `analysis/`: every seed row, distance and characteristic tables, descriptive
  report, and PDF/PNG figures. No held-out score selected K or a winning method.
- `implementation-source.zip`: evaluated source, configurations, locks, tests,
  notebook orchestration and cited rationale. It reproduces the graph diagnostic;
  run the full repository regression suite from the Git checkout, which also
  contains older scientific fixtures and archived studies.
- `execution.txt`, `verification-tests.txt` and `verification.json`: actual run
  and verification evidence; `provenance.json` checksums this bundle.

On the clean controls, identity Jaccard at K=4 recovers the four neuron-specific
archetypes, while directed WL at K=3 recovers three topology classes. Both have
ARI=1 for their different targets. Rewiring degrades both; the complete fixed-K
table retains negative and unavailable outcomes. Zero WL distance between a
six-cycle and two three-cycles coexists with false exact isomorphism, exposing
the kernel's limitation.

The user-supplied e-means paper is assessed as a planned optimizer baseline in
`docs/graph_similarity_methodology.md`; e-means is not implemented or evaluated
by these results. Verified references and their limits are in the adjacent
methodology citation audit and bibliography in `docs/`.

From the application repository with its locked Python 3.14 environment:

```bash
.venv/bin/python pipeline/16_graph_similarity.py --profile smoke
.venv/bin/python pipeline/16_graph_similarity.py --profile smoke --run \
  --output-dir outputs/graph-similarity/new-run
```

Notebook `synthetic/07_graph_similarity.ipynb` inspects this bundle by default;
set `RUN=True` for fresh fits. Existing result directories reject a different
source/protocol identity. The older failed driver study and the separate
mechanism-readout development study retain their original results and scopes.
