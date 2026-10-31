# Experiment execution order

Run commands from this repository root on `dev`. Use the project Python 3.14
`.venv`; select that kernel in Jupyter. Keep the same `RUN_ID`, recording, and
method when following a dependency chain.

```bash
uv sync --python 3.14 --locked --extra dev --extra csl --extra community --extra notebooks
```

1. **Check the installation on synthetic data.** Run `pipeline/09_validate.py
   --profile smoke --run --output-dir outputs/validation/smoke`, then
   `pipeline/10_state_calibration.py --profile smoke --output-dir
   outputs/state-calibration/smoke`, both with `.venv/bin/python`.
   Alternatively, run [validation/10_state_calibration.ipynb](notebooks/validation/10_state_calibration.ipynb).
   Calibration has no biological notebook prerequisites.
2. **Prepare each recording.** Run `<recording>/preprocessing/00_preprocess.ipynb`.
   Recording roots are `notebooks/c_elegans/`,
   `notebooks/v2a_rsns/220119_F2_run11/`, and
   `notebooks/v2a_rsns/220127_F4_run2/`.
3. **Run the primary comparison.** Run that recording's
   `evaluation/08_primary_m0_m1.ipynb`. It needs preprocessing only; it constructs
   its own training-owned graph features. States, communities, and manifolds are
   optional and are not prerequisites.
4. **Run optional method analyses** in the order below. Method folders retain
   names such as `c-GC` and `c-GC-star`, whose backends are now fastC-GC/*.

| Notebook under the same recording | Required prior notebooks |
| --- | --- |
| `effectomes/<method>/01_connectivity.ipynb` | `preprocessing/00_preprocess.ipynb` |
| `clustering/<method>/02_cluster_markov.ipynb` | That method's `01_connectivity` |
| `probabilistic_states/<method>/03_hmm.ipynb` | That method's `01_connectivity`; descriptive HMM only |
| `communities/<method>/04_temporal_community.ipynb` | That method's `01_connectivity`; estimates neuronal roles |
| `manifolds/<method>/05_manifold.ipynb` | That method's `01_connectivity` and shared preprocessing artifacts |
| `linking/<method>/06_linking.ipynb` | That method's `02`, `04`, and `05`; HMM `03` is not required |
| `counterfactuals/<method>/07_attribution_counterfactual.ipynb` | Preprocessing only; builds its own purged training-owned features |

Use only notebooks present for the selected method; partial correlation has no
directional `07`. All `05` notebooks default to cross-fitted PCA. Exact C. elegans
BunDLe reproduction is optional and requires PyTorch on a supported platform
(`bundle` extra); Python 3.14 on Intel macOS cannot install its published wheels.

`02` records an explicit unavailable result if raw-history purging leaves too
few samples; `06` then reports that reason. The default V2a recordings have
this limitation. Keep the purge and run the independent analyses as listed.

After `01`, notebooks `02`–`05` may run independently. Different recordings and
method namespaces may run in parallel; never run two writers for the same stage.
`07` reports feature sensitivity, not biological interventions. Structural Leiden
community detection remains available separately through `pipeline/03_community.py
community=leiden` after CLI connectivity. CLI artifacts (`paths.artifacts`) and
notebook checkpoints use different layouts; do not mix them. Role clustering and structural communities
answer different questions.

5. **Compare selection objectives separately.** For the endpoint-directed
   sensitivity, set `evaluation.graph_selection_objective: endpoint_prediction`
   **inside a copied raw-data manifest**; the default is `neural_dynamics`.
   Test with `pipeline/08_evaluate.py
   paths.evaluation_manifest=conf/evaluation-manifests/synthetic-endpoint-sensitivity.yaml
   paths.artifacts=outputs/endpoint-sensitivity` using `.venv/bin/python`.
   Primary raw-manifest settings are not replaced by unrelated Hydra overrides.
6. **Before scientific claims**, freeze settings and run the larger calibration
   profiles on independent draws, including realistic neuron counts and
   history/dwell ratios. Then evaluate independent animals with verified
   endpoints. A smoke run or successful clustering does not validate regimes.

Use `FORCE=False` to resume. Updated code/configuration invalidates stale
checkpoints. Existing notebook outputs were preserved: restart and rerun cells
in order rather than reading stored outputs as results from the new backend.
