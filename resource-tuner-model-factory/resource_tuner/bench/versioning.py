"""Cache versions that follow the code a task's output actually depends on.

`cache="auto"` hashes only the task function's own body. A station whose
output is decided by *imported* code -- the workload generator, the
measurement harness, an arm's model -- then cache-hits across changes to
that code. It happened: rsb-v1-corpus-2 reused rsb-v1-corpus's
`generate_bench_runs` output after the workload thread caps changed, so 774
v1 runs declare `threads=4` (measured on 3-CPU pods).

These versions hash the source of the modules each stage depends on, so a
cache entry is invalidated exactly when that code changes.
"""

from __future__ import annotations

import hashlib
import pathlib

import flyte

_HERE = pathlib.Path(__file__).parent


def _hash(*names: str) -> str:
    h = hashlib.sha256()
    for n in names:
        h.update((_HERE / f"{n}.py").read_bytes())
    return h.hexdigest()[:16]


#: What a corpus is made of: the code, how it is measured, how rows are labelled.
CORPUS_VERSION = _hash("workloads", "measure", "dataset", "grid")
#: What an arm's predictions and its scorecard depend on.
ARMS_VERSION = _hash("arms", "llm_arms", "gpu_stations", "features", "prompting", "grid", "scoring", "dataset")


def corpus_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"corpus-{CORPUS_VERSION}")


def arms_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"arms-{ARMS_VERSION}")
