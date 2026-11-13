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
   Alternatively, run [10_validation/state_calibration.ipynb](notebooks/10_validation/state_calibration.ipynb).
   Calibration has no biological notebook prerequisites.
2. **Prepare each recording.** Run `<recording>/00_preprocessing/preprocess.ipynb`.
   Recording roots are `notebooks/c_elegans/`,
   `notebooks/v2a_rsns/220119_F2_run11/`, and
   `notebooks/v2a_rsns/220127_F4_run2/`.
3. **Run the primary comparison.** Run that recording's
   `08_evaluation/primary_m0_m1.ipynb`. It needs preprocessing only; it constructs
   its own training-owned graph features. States, communities, and manifolds are
   optional and are not prerequisites.
4. **Run optional method analyses** in the order below. Each numbered stage
   directory contains method-named notebooks for every retained estimator.

| Notebook under the same recording | Required prior notebooks |
| --- | --- |
| `01_effectomes/<method>.ipynb` | `00_preprocessing/preprocess.ipynb` |
| `02_clustering/<method>.ipynb` | That method's `01_effectomes` notebook |
| `03_probabilistic_states/<method>.ipynb` | That method's `01_effectomes` notebook; descriptive HMM only |
| `04_communities/<method>.ipynb` | That method's `01_effectomes` notebook; estimates neuronal roles |
| `05_manifolds/<method>.ipynb` | That method's `01_effectomes` notebook and shared preprocessing artifacts |
| `06_linking/<method>.ipynb` | That method's `02`, `04`, and `05`; HMM `03` is not required |
| `07_counterfactuals/<method>.ipynb` | Preprocessing only; c-GC/c-GC* owned-feature sensitivity |

Use only notebooks present for the selected method; partial correlation has no
directional `07` notebook. All `05` notebooks default to cross-fitted PCA. Exact C. elegans
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
