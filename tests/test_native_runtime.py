"""Runtime reuse must retain transport isolation and close its child process."""

import json
import os
import sys
from pathlib import Path

import pytest

from effectome.connectivity._native_gc_runtime import (
    conventional_gc_runtime,
    current_conventional_runtime,
)


def test_reuses_process_without_reusing_job_state_and_closes_on_error(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text('''
import json, os, sys
from pathlib import Path
for line in sys.stdin:
    request = json.loads(line)
    if request.get("shutdown"):
        break
    Path(request["output"]).write_text(Path(request["input"]).read_text())
    Path(request["metadata"]).write_text(json.dumps({"pid": os.getpid()}))
    print(json.dumps({"ok": True}), flush=True)
''')
    process = None
    with pytest.raises(RuntimeError, match="scope"):
        with conventional_gc_runtime():
            session = current_conventional_runtime()
            assert session is not None
            pids = []
            for index in range(2):
                source, target, metadata = (tmp_path / f"{name}{index}.json"
                                             for name in ("in", "out", "meta"))
                source.write_text(str(index))
                session.run(Path(sys.executable), worker, source, target, metadata, os.environ)
                assert target.read_text() == str(index)
                pids.append(json.loads(metadata.read_text())["pid"])
            assert len(set(pids)) == 1
            process = next(iter(session.workers.values())).process
            raise RuntimeError("scope")
    assert process is not None and process.poll() == 0
    assert current_conventional_runtime() is None
