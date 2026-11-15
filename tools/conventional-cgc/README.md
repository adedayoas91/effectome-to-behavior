# Conventional c-GC runtime

This environment installs the author's published c-GC and c-GC* implementation.
The estimator source is imported from the installed package; this directory
contains no estimator implementation.

- Source: [GC-extension](https://github.com/adedayoas91/GC-extension), the current
  destination of the author's historical `causalized-GC` URL.
- Revision: [`d9442b86d7b95d822b047efc29790442693ddff0`](https://github.com/adedayoas91/GC-extension/commit/d9442b86d7b95d822b047efc29790442693ddff0),
  fetched from the public remote before installation.
- Distribution/import: `causalized-gc==0.1.0` / `causalized_gc.GcStar`.
- Runtime: Python 3.13; verified with CPython 3.13.5 on Intel macOS.
- Core numerical dependencies: upstream pins NumPy 2.2.0 and Numba 0.61.2.

The project's Python 3.14 environment cannot load those upstream dependency
versions. The isolated environment allows the main project to call the original
CPU implementation through its subprocess bridge. CUDA, PyTorch, and optional
CDT metrics are unnecessary for these two estimators.

From the package root, restore the exact dependency lock with:

```bash
UV_CACHE_DIR=/tmp/effectome-cgc-uv-cache uv sync \
  --project tools/conventional-cgc --python 3.13.5 --locked
```

The executable is `tools/conventional-cgc/.venv/bin/python`. The environment is
ignored by Git; `pyproject.toml`, `.python-version`, and `uv.lock` specify how to
recreate it. The lock pins the estimator Git revision and all transitive packages.
The locked environment was also checked successfully with `uv sync --locked
--offline`. For subprocesses in restricted environments, set `MPLCONFIGDIR` and
`XDG_CACHE_HOME` to writable cache directories; the upstream CPU import loads
Matplotlib even when no figures are requested.

Direct use within this runtime:

```python
import numpy as np
from causalized_gc import GcStar

# data has shape (neurons, time), with more than 30 usable time samples.
estimator = GcStar(
    n_perm=1000,
    n_pasts=1,
    n_lags=1,
    temporal=True,
    method="cgc",       # "cgc_star" selects c-GC*
    backend="cpu",
)
estimator.fit(np.asarray(data, dtype=np.float64), verbose=0)
graph = estimator.get_connectivity_matrix(
    simulation=True, alpha=0.01, beta=0.001,
)
```

The native graph has source rows and target columns. `simulation=True` merges
only lagged blocks; `simulation=False` also includes the contemporaneous block.
For `n_pasts=1`, `corr_`, `pVal_corr_`, `inv_corr_`, and `pVal_inv_corr_` each have
shape `(2 * neurons, neurons)`: the first row block is lag zero, the second is
lag one. `n_lags` must not exceed `n_pasts`.

Upstream selects edges using both unconditional and conditional permutation
tests and weights selected edges by absolute contemporaneous correlation. Its
returned graph is nonnegative; these weights are not signed regression
coefficients. Preserve the native evidence and distinguish statistical edge
selection from coefficient recovery when interpreting results.

The public constructor accepts `random_state`, but this pinned revision forwards
it only to the GPU implementation. CPU permutation tests use Numba random draws
in two executor threads, so seeding NumPy or passing `random_state` does not
guarantee repeatable CPU permutation p-values. Record this limitation in run
metadata and retain run artifacts for comparison. The upstream permutation code
requires more than 30 samples after lag construction because it draws circular
shifts starting at 30.

Installation verification fits both methods through their public API on a seeded
three-neuron, 256-sample linear process. Finite native graph and evidence shapes
verify runtime operation; a single fit does not establish framework validity.
The smoke process used seed `20261115`, 256 burn-in samples, transition matrix
`[[0.55, 0, 0], [0.75, 0.35, 0], [0, 0.6, 0.3]]` in target-by-source orientation,
independent standard-normal innovations, `n_perm=200`, `n_pasts=n_lags=1`, and
`alpha=beta=0.05`. Both methods recovered the two off-diagonal true edges,
`0 → 1` and `1 → 2`, with no other off-diagonal edges in that run. Both native
graphs retained diagonal self-edges; the calling project must state its diagonal
policy. This runtime check is separate from the replicated validation study.
