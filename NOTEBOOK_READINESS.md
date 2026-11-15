# Notebook run readiness

The 173 pre-existing notebooks passed bootstrap and configuration-cell execution
from their own directories during the backend migration. The current collection
has 176 recording/diagnostic notebooks after adding one canonical shared-transition
entrypoint per recording. All code cells parse in the locked Python 3.14
environment. This revision preserves all previous code, local metadata and stored
outputs, including user-generated errors. Stored outputs may predate the current
analysis and must be rerun before interpretation.

Run the five notebooks under [synthetic/](synthetic/README.md) first. They execute
their default code cells in regression checks; the repeated-study notebook
defaults to inventorying its matrix. Full validation execution is explicit.
Then follow raw preprocessing with `09_cross_method/shared_transitions.ipynb`.
All 173 earlier notebooks contain scope notices; independent method/manifold
analyses remain complementary to the canonical shared comparison.

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

The complete regression suite passes **519 tests with one optional skip** after
the final provenance/ownership refinements. The verification record is in
[the delivery audit](docs/framework_validation_audit.md). Regression tests
exercise synthetic analysis stages, installed backends,
checkpoint reuse, and insufficient-data handling. Ruff, mypy, compilation,
locked environment synchronization and installed-package compatibility checks
pass. Full biological experiments across all notebooks were not rerun during
this readiness check. Follow [exec_order.md](exec_order.md) for prerequisites
and [notebooks/README.md](notebooks/README.md) for checkpoint namespaces.
