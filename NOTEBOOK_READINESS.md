# Notebook run readiness

All 173 notebooks pass bootstrap and configuration-cell execution from their
own directories in the locked Python 3.14 environment. Each check used fresh
globals and temporary checkpoint directories. All 543 code cells parse, and
the obsolete stored error traceback has been removed. Scientific outputs
retained in notebooks may predate the backend migration and must be rerun.

## Changes needed for execution

- All manifold notebooks explicitly default to cross-fitted classical PCA.
  Faithful BunDLe reproduction remains optional, with its original model and
  preprocessing. It requires the `bundle` extra on a supported PyTorch platform;
  no published wheel supports Python 3.14 on Intel macOS.
- Prospective state notebooks checkpoint insufficient-data results and their
  reasons. They clear stale state/transition fits instead of substituting a
  descriptive clustering. Linking reports the unavailable state result.
- Training-owned feature construction now defaults to one neuronal role,
  matching the reviewed configuration.
- Explicit UMAP requests fail clearly if its dependency is missing; they do not
  silently produce PCA results.

## Supplied data preflight

Actual loaders, preprocessing and temporal split construction were checked
without fitting biological connectivity models.

| Recording | Graph windows | Prospective state evaluation |
| --- | ---: | --- |
| C. elegans Worm0 | 186 | Feasible; inner split has 8 training / 19 validation pairs |
| V2a 220119_F2_run11 | 94 | Unavailable: no complete future graph-history pairs |
| V2a 220127_F4_run2 | 95 | Unavailable: purging leaves insufficient training pairs |

The owned-feature chronological split and five-fold PCA split are feasible for
all three recordings. Keep the raw-history purge; an unavailable result is not
evidence for or against discrete states. The primary endpoint comparison and
owned feature sensitivity do not require the graph-state analysis.

## Scope of verification

The full suite passes **378 tests, with 1 optional PyTorch skip**. Regression
tests exercise synthetic analysis stages, installed backends,
checkpoint reuse, and insufficient-data handling. Ruff, mypy, compilation,
locked environment synchronization and installed-package compatibility checks
pass. Full biological experiments across all notebooks were not rerun during
this readiness check. Follow [exec_order.md](exec_order.md) for prerequisites
and [notebooks/README.md](notebooks/README.md) for checkpoint namespaces.
