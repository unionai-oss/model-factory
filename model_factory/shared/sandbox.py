"""Sandboxed execution of model-generated code against hidden unit tests.

This is the POC analog of poolside's code execution environment: solution
code and its pytest suite are written to a throwaway directory and run in a
resource-limited subprocess. Isolation level (POC-accepted risk, see
docs/SPEC.md §8): separate process, rlimits on CPU/memory/files, isolated
Python (-I), wall-clock timeout, temp cwd. NOT a security boundary against
adversarial code — medium scope moves this to dedicated
container/microVM-isolated executors.

No flyte imports here: unit-testable pure Python.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

DEFAULT_TIMEOUT_S = 10.0
MEMORY_LIMIT_BYTES = 1_000_000_000  # 1 GB address space
CPU_SECONDS = 10


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of one solution-vs-tests execution."""

    passed: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool

    @property
    def summary(self) -> str:
        if self.timed_out:
            return "timeout"
        return "passed" if self.passed else f"failed (rc={self.returncode})"


def _rlimits() -> None:  # pragma: no cover - runs in the child process
    import resource

    # Best-effort per limit: some limits can't be lowered on macOS (dev
    # machines); on Linux task containers they all apply.
    for limit, value in (
        (resource.RLIMIT_CPU, CPU_SECONDS),
        (resource.RLIMIT_AS, MEMORY_LIMIT_BYTES),
        (resource.RLIMIT_NOFILE, 256),
    ):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass
    try:
        os.setsid()  # own process group so cleanup kills grandchildren
    except OSError:
        pass


def run_solution_against_tests(
    solution_code: str,
    test_code: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecutionResult:
    """Write ``solution.py`` + ``test_solution.py`` and run pytest on them.

    Test suites follow the KodCode convention: ``from solution import <fn>``.
    """
    workdir = tempfile.mkdtemp(prefix="mf-sandbox-")
    try:
        with open(os.path.join(workdir, "solution.py"), "w") as f:
            f.write(solution_code)
        with open(os.path.join(workdir, "test_solution.py"), "w") as f:
            f.write(test_code)

        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": workdir,
            "TMPDIR": workdir,
            # no proxy/token env vars leak into generated code
        }
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",  # isolated: ignores PYTHON* env vars and user site
                    "-m",
                    "pytest",
                    "-q",
                    "-x",
                    "-p",
                    "no:cacheprovider",
                    "test_solution.py",
                ],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_rlimits if sys.platform != "win32" else None,
            )
        except subprocess.TimeoutExpired as e:
            return ExecutionResult(
                passed=False,
                returncode=-1,
                stdout=(e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr="timeout",
                timed_out=True,
            )
        return ExecutionResult(
            passed=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=proc.stdout[-4000:],
            stderr=proc.stderr[-4000:],
            timed_out=False,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
