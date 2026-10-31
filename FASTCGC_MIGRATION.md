# fastC-GC backend migration — 31 October 2026

## Installed environment

The project `.venv` now uses CPython **3.14.7** and installed **fastcgc 0.1.0**.
The package is installed from the immutable archive at upstream commit
`985de54bf3d2474a5e692ad1d236c74f8452e197`, without a repository clone or upstream
editable checkout. The archive SHA-256 is
`bff6a73540067dfe6d74023d5c12ff688c5ffe7d5c90220cc4da8fd12d82eb8a`.
Both the URL and hash are recorded in `uv.lock`.

Upstream has no PyPI distribution, version tags, or GitHub releases at the time
of migration. The user approved an immutable source pin and Python 3.14
migration in place of a published stable release. Upstream's Python requirement
is `>=3.14,<3.15`; the project now declares the same range. The old environment
was preserved at `/tmp/effectome-venv-py312-20261031` as a temporary rollback copy.

Recreate the installed development/kernel environment with:

```bash
uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks
```

Restart notebook kernels and select the project `.venv` after migration.

## Backend behavior

Existing `cgc` and `cgc_star` config keys and notebook directories are retained.
Their default support backend is now installed `FastGcStar`, with upstream
`method="cgc"` and `method="fcgc"`, respectively. The adapter uses the linear
profile, full pair evaluation, exact complete-shift FFT tests, and a separate
correlation-scale support ridge of `1e-6`.

Positive-lag support is exported to the existing signed ridge-VAR refit. Target
self-history remains in every prediction model, even when cross-edge support is
empty. Dependence scores are not substituted for signed VAR coefficients.
Production YAML uses alpha 0.05 and beta 0.005. Positive thresholds below the
complete-shift resolution are rejected, rather than silently reporting an
unattainable test. Low-level config calls without a beta use alpha in fast mode;
explicit legacy modes preserve their original default thresholds.

The upstream estimator rejects nonfinite recordings. For masked inputs, the
adapter constructs its pinned upstream context from retained, aligned lag-design
columns, retaining the original response indices and avoiding new lagged pairs
across gaps. Circular shifts then operate on that retained-column axis; their
null calibration must account for the mask and does not automatically preserve
physical-time autocorrelation. Diagnostics record the backend, version, source,
conditioning expansion, shift axis, support ridge, and p-value resolution.

`support_test="analytic"` and `support_test="circular_shift"` retain the explicit
legacy comparators. Archive/version and source/lock fingerprints distinguish
new runs from old checkpoints. This migration changes the estimator and can
change graphs; full biological evaluation and calibration were not performed.

## Python compatibility

Stable Hydra 1.3.8 still passes a lazy help object that Python 3.14's argparse
rejects. A small `effectome.utils.hydra` shim converts that help object to a
string through a Hydra-local parser reference. The standard-library parser is
unchanged. The nine Hydra entry points import this wrapper. A metadata type
check in the intervention code was also made explicit for the updated mypy.

## Citations

The manuscript cites four distinct methodological contributions:

- [Original c-GC/c-GC*](https://arxiv.org/abs/2501.02672): RCCP decision rule and conditioning sets.
- [fastC-GC/fastC-GC*](https://arxiv.org/abs/2610.37883): the standardized ridge-residual estimator and FFT implementation.
- [BSS preprocessing study](https://arxiv.org/abs/2608.00655): trace preservation, behavioral/connectivity distortions, and matched preprocessing controls.
- [Conditioning-depth diagnostics](https://arxiv.org/abs/2606.01214): varying conditioning depth while holding reported lags fixed. Its published c-GC comparison is distinguished from the proposed application to the fast backend.

The last two references already had bibliography entries; they are now cited
in Related Work and Methods. An unused duplicate entry for the depth paper was
removed. All 34 cited keys resolve in the rebuilt manuscript.

## Verification at the initial backend migration

- Full suite after the CLI repair: **306 passed, 1 skipped**. One joblib warning reports a fallback from physical to logical CPU count.
- Following the independent review's legacy-default fix, the fast-backend and connectivity tests pass again: **29 passed**, including the additional legacy-default regression. The new fast-backend test file contains 12 passing cases.
- Ruff passes over source, pipeline, tests, and notebook helpers.
- Mypy passes over **65** source/entrypoint files.
- `compileall` succeeds for source, pipeline, and notebook helpers.
- `uv pip check` reports compatible installed packages; a locked offline sync succeeds.
- Tectonic/BibTeX builds the manuscript with no undefined references/citations, duplicate labels, overfull boxes, or TeX errors. Existing minor typography/encoding warnings remain.

## Changed files

Package dependency/environment declarations: `pyproject.toml`, `uv.lock`.
Backend and integration: `src/effectome/core/causalised_gc.py`,
`src/effectome/connectivity/{base,causalised_gc}.py`,
`src/effectome/experiments/adapters.py`, the three c-GC/time-varying connectivity
YAML files, and the legacy sweep notebook's explicit backend setting.
Compatibility: `src/effectome/utils/hydra.py`, the nine Hydra pipeline imports,
and `src/effectome/intervention/surrogate.py`.
Regression coverage: `tests/test_fast_cgc.py`.
Documentation: package and notebook READMEs and this migration record.

Manuscript changes cover the abstract, introduction, related work, methods,
bibliography, citation audit, revision/editorial notes, README, and rebuilt PDF
and logs in `../manuscript-260926/`. No datasets or existing experimental outputs
were changed.

The subsequent all-notebook and maintained-package audit is recorded in [BACKEND_PACKAGE_AUDIT.md](BACKEND_PACKAGE_AUDIT.md).
