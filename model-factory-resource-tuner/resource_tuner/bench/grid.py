"""The request grid every arm chooses from, and what a request costs.

A request is a cell: (memory step, CPU step). The memory grid is the
"half power of two" ladder people actually write in Kubernetes manifests
(512Mi, 768Mi, 1Gi, 1.5Gi, 2Gi, 3Gi, ...): about 1.4x between steps, so
the grid alone forces at most ~41% over-provisioning and ~20% on average,
against ~100% / ~41% for a pure power-of-two ladder. The coarse grid was
the single biggest waste term in round 16; this one halves it.

Pure logic: no flyte import, safe inside trainers and scorers alike.
"""

from __future__ import annotations

import math

from ..pricing import CPU_DOLLARS_PER_CORE_HR, MEM_DOLLARS_PER_GIB_HR

MEMORY_GRID_MIB: tuple[int, ...] = (
    256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
    8192, 12288, 16384, 24576, 32768, 49152, 65536,
)
CPU_GRID: tuple[float, ...] = (0.5, 1, 2, 3, 4, 6, 8)

#: What the Flyte task process itself holds before user code allocates
#: anything (interpreter + flyte runtime + its imports), MiB. Measured by
#: the `measure_runtime_overhead` probe; the label adds it to the
#: workload's own peak because the pod limit has to cover both.
FLYTE_RUNTIME_OVERHEAD_MIB = 180.0

#: A pod whose limit sits exactly at the working set still dies: allocator
#: slack, page tables, GC timing. The label demands this much headroom.
LABEL_HEADROOM = 1.05


def format_memory(mib: int) -> str:
    if mib % 1024 == 0:
        return f"{mib // 1024}Gi"
    if mib > 1024 and (mib * 2) % 1024 == 0:
        return f"{mib / 1024:g}Gi"
    return f"{mib}Mi"


def parse_memory(text: str) -> int:
    """'1.5Gi' / '768Mi' / '2G' / 3072 -> MiB."""
    if isinstance(text, (int, float)):
        return int(text)
    s = str(text).strip()
    for suffix, mult in _MEMORY_UNITS:
        if s.endswith(suffix):
            return int(round(float(s[: -len(suffix)]) * mult))
    return int(round(float(s)))


# Longest suffix first, so "Gi" is not read as "G" followed by junk.
_MEMORY_UNITS = (
    ("Gi", 1024.0),
    ("Mi", 1.0),
    ("G", 1e9 / 2**20),
    ("M", 1e6 / 2**20),
)


def mem_index_covering(mib: float) -> int:
    """Index of the smallest memory step >= mib (clamped to the top)."""
    for i, step in enumerate(MEMORY_GRID_MIB):
        if mib <= step:
            return i
    return len(MEMORY_GRID_MIB) - 1


def mem_index_at_or_below(mib: float) -> int:
    for i in range(len(MEMORY_GRID_MIB) - 1, -1, -1):
        if MEMORY_GRID_MIB[i] <= mib:
            return i
    return 0


def cpu_index_covering(cores: float) -> int:
    for i, step in enumerate(CPU_GRID):
        if cores <= step + 1e-9:
            return i
    return len(CPU_GRID) - 1


def required_memory_mib(workload_peak_mib: float) -> float:
    """What the pod limit must be for this workload to survive."""
    return (workload_peak_mib + FLYTE_RUNTIME_OVERHEAD_MIB) * LABEL_HEADROOM


def label_memory_index(workload_peak_mib: float) -> int:
    """The oracle's memory cell: cheapest step that covers the requirement."""
    return mem_index_covering(required_memory_mib(workload_peak_mib))


def label_cpu_index(avg_cores: float) -> int:
    """The oracle's CPU cell. CPU is compressible, so 'enough' means the
    measured sustained parallelism, rounded up -- asking for less only
    stretches the run (see scoring), it never kills it."""
    return cpu_index_covering(max(avg_cores, 0.5) * 0.95)


# ── pricing ─────────────────────────────────────────────────────────────
# Two models, both reported; "packed" is the headline.
#
# packed  -- a node is bought whole, and a pod costs its DOMINANT share of
#            one: max(cpu / node_cpu, mem / node_mem) x node $/hr. This is
#            what bin-packing does to a request -- 16Gi with 1 CPU on a
#            4-core/16Gi node strands the other three cores. m6i.xlarge
#            (4 vCPU, 16 GiB, $0.192/hr on-demand, us-east-1), the
#            general-purpose shape most CPU pools are built from.
# fargate -- per-resource linear rates (pricing.py): what a serverless
#            container platform charges. Memory is nearly free here
#            (a core costs as much as 9 GiB), so it understates memory waste.
NODE_CPU, NODE_MEM_GIB, NODE_DOLLARS_PER_HR = 4.0, 16.0, 0.192
PRICING_MODELS = ("packed", "fargate")


def dollars_per_hr(mem_mib: float, cpu: float, pricing: str = "packed") -> float:
    if pricing == "packed":
        return NODE_DOLLARS_PER_HR * max(cpu / NODE_CPU, (mem_mib / 1024) / NODE_MEM_GIB)
    return cpu * CPU_DOLLARS_PER_CORE_HR + (mem_mib / 1024) * MEM_DOLLARS_PER_GIB_HR


def grid_headroom(workload_peak_mib: float) -> float:
    """Over-provisioning the grid alone forces on a perfect predictor."""
    need = required_memory_mib(workload_peak_mib)
    return MEMORY_GRID_MIB[mem_index_covering(need)] / need - 1.0


def log2_mib(mib: float) -> float:
    return math.log2(max(mib, 1.0))
