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
_SHARED_LLM = ("llm_arms", "gpu_stations", "spec", "mem_spec", "registry", "common", "hybrid",
               "bench_cpu/spec", "bench_cpu/features", "bench_cpu/scoring", "bench_cpu/grid",
               "bench_gpu/spec", "bench_gpu/features", "bench_gpu/scoring", "bench_gpu/grid")

#: What a corpus is made of: the code, how it is measured, how rows are labelled.
_CORPUS = ("workloads", "measure", "dataset", "grid")
#: What an arm's predictions and its scorecard depend on.
_ARMS = ("arms", "features", "prompting", "grid", "scoring", "dataset", *_SHARED_LLM)

# Every version is computed when a station asks for it, never at import: the
# code bundler ships only modules a deploy imports, and a RightSizeCPU deploy
# does not ship bench_gpu/workloads.py (rsc-pilot-1 died on exactly that
# FileNotFoundError), nor the combined bench's bench/measure.py (rsa-full-1).
_CPU_CORPUS = ("workloads", "measure", "bench_cpu/workloads", "bench_cpu/measure", "bench_cpu/dataset", "bench_cpu/grid")
_CPU_ARMS = ("bench_cpu/arms", "bench_cpu/dataset", "bench_cpu/stations", "arms", *_SHARED_LLM)
_GPU_CORPUS = ("workloads", "measure", "bench_cpu/measure", "bench_gpu/workloads", "bench_gpu/measure",
               "bench_gpu/dataset", "bench_gpu/grid")
_GPU_ARMS = ("bench_gpu/arms", "bench_gpu/dataset", "bench_gpu/stations", "arms", *_SHARED_LLM)


def corpus_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"corpus-{_hash(*_CORPUS)}")


def arms_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"arms-{_hash(*_ARMS)}")


def cpu_corpus_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"cpu-corpus-{_hash(*_CPU_CORPUS)}")


def cpu_arms_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"cpu-arms-{_hash(*_CPU_ARMS)}")


def gpu_corpus_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"gpu-corpus-{_hash(*_GPU_CORPUS)}")


def gpu_arms_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"gpu-arms-{_hash(*_GPU_ARMS)}")


#: The combined bench: one corpus joined from three, and its arms (classical,
#: hosted, supervised decision model, RL), all scored by one cost model.
_ALL_CORPUS = ("bench_all/dataset", "bench_all/grid", "dataset", "grid")
_ALL_ARMS = ("bench_all/grid", "bench_all/dataset", "bench_all/scoring", "bench_all/features", "bench_all/arms",
             "bench_all/prompting", "bench_all/policy", "bench_all/stations", "arms", "features", "scoring",
             "bench_cpu/features", "bench_cpu/scoring", "bench_gpu/features", "bench_gpu/scoring", "llm_arms")


def all_corpus_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"all-corpus-{_hash(*_ALL_CORPUS)}")


def all_arms_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"all-arms-{_hash(*_ALL_ARMS)}")
