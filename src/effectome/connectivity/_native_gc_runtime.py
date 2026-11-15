"""Explicit, process-local reuse of the conventional package's import/JIT runtime."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


class _RuntimeProcess:
    def __init__(self, runtime: Path, worker: Path, environment: Mapping[str, str]) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="effectome-conventional-runtime-")
        self.stderr = tempfile.TemporaryFile(mode="w+b")
        root = Path(self.directory.name)
        self.process = subprocess.Popen(
            [str(runtime), str(worker), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, bufsize=1,
            env={**environment, "MPLCONFIGDIR": str(root / "matplotlib"),
                 "XDG_CACHE_HOME": str(root / "cache")},
        )

    def run(self, input_path: Path, output_path: Path, metadata_path: Path) -> None:
        process = self.process
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("conventional GC runtime has no open transport")
        request = {"input": str(input_path), "output": str(output_path), "metadata": str(metadata_path)}
        try:
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response = process.stdout.readline()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("persistent official conventional GC transport failed") from error
        if not response:
            self.stderr.seek(0)
            detail = self.stderr.read().decode(errors="replace")[-4000:]
            raise RuntimeError(f"official conventional GC runtime exited without a response:\n{detail}")
        try:
            result = json.loads(response)
        except json.JSONDecodeError as error:
            raise RuntimeError("official conventional GC runtime returned an invalid response") from error
        if not result.get("ok"):
            raise RuntimeError(f"official conventional GC runtime failed; no estimator fallback: {result}")

    def close(self) -> None:
        process = self.process
        try:
            if process.poll() is None:
                if process.stdin is not None:
                    try:
                        process.stdin.write('{"shutdown":true}\n')
                        process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            self.stderr.close()
            self.directory.cleanup()


class _ConventionalRuntime:
    """One worker per executable/script pair; each job creates fresh estimators."""

    def __init__(self) -> None:
        self.owner_pid = os.getpid()
        self.workers: dict[tuple[str, str], _RuntimeProcess] = {}

    def run(
        self, runtime: Path, worker: Path, input_path: Path, output_path: Path,
        metadata_path: Path, environment: Mapping[str, str],
    ) -> None:
        if self.owner_pid != os.getpid():
            raise RuntimeError(
                "open conventional GC runtimes cannot be inherited across fork; open in each worker"
            )
        key = (str(runtime.resolve()), str(worker.resolve()))
        if key not in self.workers:
            self.workers[key] = _RuntimeProcess(runtime, worker, environment)
        self.workers[key].run(input_path, output_path, metadata_path)

    def close(self) -> None:
        for worker in self.workers.values():
            worker.close()
        self.workers.clear()


_CURRENT_RUNTIME: ContextVar[_ConventionalRuntime | None] = ContextVar(
    "conventional_gc_runtime", default=None
)


def current_conventional_runtime() -> _ConventionalRuntime | None:
    return _CURRENT_RUNTIME.get()


@contextmanager
def conventional_gc_runtime() -> Iterator[None]:
    """Reuse only imports/JIT within an explicit study scope, closing all children.

    Open this context separately inside each process-pool worker. Native CPU
    permutation evidence remains stochastic; reuse does not repair upstream RNG
    seeding. Environment settings are frozen when each runtime is first started.
    """
    if _CURRENT_RUNTIME.get() is not None:
        yield
        return
    runtime = _ConventionalRuntime()
    token = _CURRENT_RUNTIME.set(runtime)
    try:
        yield
    finally:
        _CURRENT_RUNTIME.reset(token)
        runtime.close()
