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

_PKG = pathlib.Path(__file__).parent.parent  # resource_tuner/


def _hash(*names: str) -> str:
    """Names are modules: "workloads" (this package) or "bench_cpu/workloads"."""
    h = hashlib.sha256()
    for n in names:
        h.update((_PKG / (n if "/" in n else f"bench/{n}")).with_suffix(".py").read_bytes())
    return h.hexdigest()[:16]


#: What the shared LLM arms depend on, for every bench (train_llm_arm serves all three).
_SHARED_LLM = ("llm_arms", "gpu_stations", "spec", "mem_spec", "registry", "common",
               "bench_cpu/spec", "bench_cpu/features", "bench_cpu/scoring", "bench_cpu/grid",
               "bench_gpu/spec", "bench_gpu/features", "bench_gpu/scoring", "bench_gpu/grid")

#: What a corpus is made of: the code, how it is measured, how rows are labelled.
CORPUS_VERSION = _hash("workloads", "measure", "dataset", "grid")
#: What an arm's predictions and its scorecard depend on.
ARMS_VERSION = _hash("arms", "features", "prompting", "grid", "scoring", "dataset", *_SHARED_LLM)

# The CPU and GPU benches' versions are computed when their own stations ask
# for them, not at import: the code bundler ships only modules a deploy
# imports, and a RightSizeCPU deploy does not ship bench_gpu/workloads.py
# (rsc-pilot-1 died on exactly that FileNotFoundError).
_CPU_CORPUS = ("workloads", "measure", "bench_cpu/workloads", "bench_cpu/measure", "bench_cpu/dataset", "bench_cpu/grid")
_CPU_ARMS = ("bench_cpu/arms", "bench_cpu/dataset", "bench_cpu/stations", "arms", *_SHARED_LLM)
_GPU_CORPUS = ("workloads", "measure", "bench_cpu/measure", "bench_gpu/workloads", "bench_gpu/measure",
               "bench_gpu/dataset", "bench_gpu/grid")
_GPU_ARMS = ("bench_gpu/arms", "bench_gpu/dataset", "bench_gpu/stations", "arms", *_SHARED_LLM)


def corpus_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"corpus-{CORPUS_VERSION}")


def arms_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"arms-{ARMS_VERSION}")


def cpu_corpus_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"cpu-corpus-{_hash(*_CPU_CORPUS)}")


def cpu_arms_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"cpu-arms-{_hash(*_CPU_ARMS)}")


def gpu_corpus_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"gpu-corpus-{_hash(*_GPU_CORPUS)}")


def gpu_arms_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"gpu-arms-{_hash(*_GPU_ARMS)}")
