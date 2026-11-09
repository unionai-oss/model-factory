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

import ast
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
_ALL_CORPUS = ("bench_all/dataset", "bench_all/grid", "bench_all/k8s", "dataset", "grid")
_ALL_ARMS = ("bench_all/grid", "bench_all/dataset", "bench_all/scoring", "bench_all/features", "bench_all/arms",
             "bench_all/prompting", "bench_all/policy", "bench_all/stations", "bench_all/k8s", "arms", "features",
             "scoring", "hybrid", "bench_all/gating", "bench_all/counterfactual", "shared/llm_client",
             "bench_cpu/features", "bench_cpu/scoring", "bench_gpu/features", "bench_gpu/scoring", "llm_arms")


#: The trained arms (fit_all_arm, train_all_policy, rl_all_policy) key on what
#: decides their predictions only: not the post-hoc stations (pool, gate,
#: scoring slices, analyses), which share ``bench_all/stations.py`` with them.
#: rsk-k8s-1-gate retrained every LLM arm because a gate's code entered the
#: one key all of them shared (round 22).
_ALL_TRAIN = ("bench_all/grid", "bench_all/dataset", "bench_all/scoring", "bench_all/features", "bench_all/arms",
              "bench_all/prompting", "bench_all/policy", "bench_all/k8s", "arms", "features", "scoring", "hybrid",
              "bench_cpu/features", "bench_cpu/scoring", "bench_gpu/features", "bench_gpu/scoring", "llm_arms",
              "shared/llm_client")
#: The top-level names of ``bench_all/stations.py`` a trained arm's output depends on.
_ALL_TRAIN_DEFS = ("_EVAL", "CPU_ARMS", "HOSTED_ARMS", "HOSTED_KNOBS", "GPU_ARMS", "FLYTE_ONLY_SUFFIX",
                   "PLATFORM_SUFFIXES", "TREE_PLATFORM_ARMS", "seed_of", "platform_suffix", "STACK_ARMS", "ALL_ARMS",
                   "HINTS_FILE", "HOSTED_RUNS_PER_SPLIT", "TRAIN_PER_SOURCE", "EPOCHS", "MAX_LEN", "paired_subsample",
                   "_pred_table", "_write_arm", "answer_schema", "zeroshot_predict", "_decode", "_argmax",
                   "_policy_predictions", "knobs", "_save", "train_all_policy", "all_hints", "hinted_all_corpus",
                   "fit_all_arm", "_read_arm", "rl_all_policy")


def _hash_defs(module: str, names: tuple[str, ...]) -> str:
    """The source of `names` (top-level functions, classes, assignments) in `module`, hashed."""
    src = (_PKG / f"{module}.py").read_text()
    seg = {}
    for node in ast.parse(src).body:
        targets = ([node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   else [t.id for t in getattr(node, "targets", [getattr(node, "target", None)]) if isinstance(t, ast.Name)])
        for t in targets:
            seg[t] = ast.get_source_segment(src, node, padded=True) or ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):  # decorators carry the env and cache
                seg[t] = "".join(ast.get_source_segment(src, d) or "" for d in node.decorator_list) + seg[t]
    missing = [n for n in names if n not in seg]
    if missing:
        raise KeyError(f"_hash_defs: {module} has no {missing}")
    return hashlib.sha256("\0".join(seg[n] for n in names).encode()).hexdigest()[:16]


def all_train_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override",
                       version_override=f"all-train-{_hash(*_ALL_TRAIN)}-{_hash_defs('bench_all/stations', _ALL_TRAIN_DEFS)}")


def all_corpus_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"all-corpus-{_hash(*_ALL_CORPUS)}")


def all_arms_cache() -> flyte.Cache:
    return flyte.Cache(behavior="override", version_override=f"all-arms-{_hash(*_ALL_ARMS)}")
