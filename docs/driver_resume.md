# Resuming the frozen driver study

Completed method/recording checkpoints are reused when their configuration and
implementation signatures match. The checkpoint JSON is committed by atomic
rename after native evidence has been written. A partially written `.tmp` file
is not a completed checkpoint. An interrupted recording/method restarts its
128 origin-window fits; completed recordings and recorded failures are retained.

The resume launcher uses the existing independent shard worker, without the
original bitwise fitted-baseline merger. It validates the frozen code/config,
checkpoint signatures and native archive checksums before starting any fits.
Default execution only inventories completed/remaining work. It must not launch
another writer while the old workers remain active.

From the application root:

```bash
.venv/bin/python -m synthetic.resume_driver_study
# Only after the earlier workers have stopped:
.venv/bin/python -m synthetic.resume_driver_study --run --workers 4
```

If the reporting monitor also stopped, restart it in another terminal:

```bash
.venv/bin/python -m synthetic.finalize_driver_study --watch
```

Keep the frozen source, configuration and the existing output directory. Do not
delete `methods/`, `verified-collection/`, native NPZ archives or the frozen
protocol. If computation already finished, only finalization needs restarting;
its reviewed source-fix verification can resume after interruption.
