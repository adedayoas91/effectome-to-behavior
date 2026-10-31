# Notebook and installed-backend audit — 31 October 2026

## Implementation plan

1. Audit every notebook's active code, labels, environment metadata and shared generator; preserve executed outputs and existing paths.
2. Install the locked Tigramite, igraph/Leiden, scikit-learn and kmedoids backends in the project venv; use published packages without cloning.
3. Retain scikit-learn KMeans/SpectralClustering/PCA. Replace the local k-medoids loop with the package's alternating solver, preserving seeded starting medoids and the labels/medoids interface. Existing two-cluster recovery coverage protects this behavior; add backend integration checks.
4. Keep the project-specific temporal role and HMM procedures distinct from general clustering/community algorithms. Do not substitute static Leiden for temporal role inference.
5. Repair JPCMCI+'s omitted node classification and make Leiden's package requirement explicit instead of silently substituting another algorithm.
6. Verify installed-package execution, all notebook cells, lint, typing and the test suite; document any analyses that were not executed.

## Completed changes

All **172 notebooks (540 code cells)** were audited and updated with Python 3.14
kernel metadata, an environment setup note, and a package-version check. Visible
method labels now identify fastC-GC/fastC-GC*. Configuration keys and directory
paths are unchanged. The generator produces the same setup for future notebooks.
All five existing output blocks, three execution counts, and cell IDs were
preserved. Stored output can predate this migration; it was not regenerated.
No active notebook source clones an upstream repository or injects an upstream
checkout into `sys.path`. Local project discovery remains for nested notebooks.

Installed packages and imports were verified in `.venv/lib/python3.14/site-packages`:

| Use | Installed package | Version |
| --- | --- | --- |
| Fast connectivity | fastcgc | 0.1.0, immutable archive pin |
| PCMCI+ and JPCMCI+ | tigramite | 5.2.10.1 |
| K-means, spectral clustering, PCA | scikit-learn | 1.9.0 |
| Distance-based k-medoids | kmedoids | 0.5.5 |
| Graph representation | igraph / python-igraph | 1.0.0 |
| Leiden community detection | leidenalg | 0.12.0 |

No repository clones were needed. The published kmedoids wheel supports
CPython 3.14 on this platform. The lock records all selected package versions
and archive/wheel hashes. Install the same profile with:

```bash
uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks
```

### Connectivity

The existing `pcmci` key calls Tigramite's official `run_pcmciplus`; `jpcmci`
calls its official `run_jpcmciplus`. These are adapters, not reimplementations.
The previously broken JPCMCI+ default now supplies all-system node
classifications. Context/dummy variables require explicit validated labels;
without them this is a pooled system-variable analysis, not a claim to model
unobserved context. A `conf/connectivity/jpcmci.yaml` preset is now available.
The registry and notebook adapters share the collapse routine, restrict directed
exports to positive lags, and identify weights as MCI dependence statistics.
Unresolved contemporaneous adjacencies are not exported as directed effects.

### Community detection and clustering

Leiden now requires the actual installed packages and no longer silently
substitutes NetworkX greedy modularity. With `symmetrize=False`, its directed
graph retains edges from both triangles. The RB partition rejects negative or
nonfinite weights explicitly; default absolute-weight preprocessing remains.

K-means, spectral clustering, PCA and silhouette scores already used
scikit-learn. The local k-medoids optimization loop has been removed in favor
of `kmedoids.alternating`, keeping seeded initial medoids, the alternating
objective, and the `(labels, medoid_indices)` interface. Package tie handling
can differ from the previous loop, so prior partitions are not asserted to be
bit-identical.

The temporal signed-role model remains project-specific and uses scikit-learn
initialization. Replacing it with static Leiden would change the scientific
question. Optional graph-distance/GW routines and the boundary-aware diagonal
HMM retain their existing implementations. The optional `markov_stability`
entry remains clearly labeled as a NetworkX random-walk/modularity proxy; it
is not presented as the maintained PyGenStability algorithm. No new analysis
lanes were created solely to install packages.

### Verification and scope

Final full suite: **318 passed, 1 skipped** in 144.91 seconds. The two warnings
concern joblib CPU-count discovery and the intentional single-recording invalid
JPCMCI-classification test; neither is a failed check.

All 540 notebook code cells parse. The notebook setup/generator tests pass,
and real installed-package tests recover a known lagged dependence with both
PCMCI+ and pooled JPCMCI+, preserve directed Leiden edges, reject a missing
Leiden package without substituting another algorithm, and recover seeded
k-medoids clusters. An independent read-only API review found no additional
concrete regression. Ruff and mypy pass; installed-package dependency checks
and a locked offline environment sync pass.

The notebook analyses were not executed on biological datasets. Compilation and
small synthetic/package smoke tests establish integration, not empirical
results. Previous outputs and datasets remain intact.

## Changed files

- All 172 `.ipynb` files and `notebooks/_build_notebooks.py`.
- `pyproject.toml`, `uv.lock`, package/notebook READMEs, this audit, and the earlier migration record's install command.
- `src/effectome/connectivity/pcmci.py`, `src/effectome/experiments/adapters.py`, and the new `conf/connectivity/jpcmci.yaml`.
- `src/effectome/community/static.py`, `conf/community/leiden.yaml`, and `src/effectome/dynamics/metrics.py`.
- New `tests/test_installed_backends.py` and `tests/test_notebook_backend_migration.py`.

Official API references: [Tigramite](https://jakobrunge.github.io/tigramite/),
[Leiden](https://leidenalg.readthedocs.io/en/stable/reference.html),
[scikit-learn clustering](https://scikit-learn.org/stable/modules/clustering.html),
and [kmedoids](https://python-kmedoids.readthedocs.io/en/latest/).
