"""Dollar pricing for resource requests — the business metric.

"How much money am I saving the customer?" is the judgment criterion for
every reward shape, so this module is the single source of $/hr truth used
by training rewards (cost-weighted aggregation), eval (the gate), and the
tune-service value ledger (the dashboard).

Unit rates: AWS Fargate's public per-resource prices (us-east-1, 2026) —
the one AWS price list that decomposes into $/vCPU-hr and $/GiB-hr instead
of bundling them per instance. GPU $/hr is derived from THIS tenant's node
groups (demo.hosted "Node groups" panel, 2026-09-04): take each GPU
instance's on-demand price, subtract the Fargate value of its CPU+memory,
and split the residual across its GPUs.

Per the experiment brief, only the tenant's schedulable GPU pools are
priced/proposable: T4 (g4dn.metal), L4 (g6.2xlarge), L40S (g6e.2xlarge).
"""

from __future__ import annotations

from dataclasses import dataclass

from .policy.actions import GPU_TYPES, GPU_VRAM_MIB, parse_memory_to_mib

# Fargate unit rates, us-east-1 on-demand.
CPU_DOLLARS_PER_CORE_HR = 0.04048
MEM_DOLLARS_PER_GIB_HR = 0.004445

# GPU $/hr residuals, derived from the tenant's node groups:
#   g4dn.metal  $7.824/hr − (96 vCPU + 384GiB) Fargate value, ÷8 T4  → 0.279
#   g6.2xlarge  $0.978/hr − (8 vCPU + 32GiB)   Fargate value, ÷1 L4  → 0.512
#   g6e.2xlarge $2.242/hr − (8 vCPU + 64GiB)   Fargate value, ÷1 L40S → 1.634
GPU_DOLLARS_PER_HR: dict[str, float] = {
    "T4": 0.279,
    "L4": 0.512,
    "L40S": 1.634,
}

# GPU_TYPES / GPU_VRAM_MIB (the proposable grid) live in policy.actions —
# they're action-space facts; this module prices them.


@dataclass(frozen=True)
class NodeGroup:
    """Allocatable resources of a single node (tenant console, 2026-09-04)."""

    instance: str
    cpu_cores: float
    memory_mib: int
    gpu_type: str | None
    gpu_count: int


NODE_GROUPS: tuple[NodeGroup, ...] = (
    NodeGroup("t3a.xlarge", 3.67, 13_700, None, 0),
    NodeGroup("c5.4xlarge", 15.64, 26_900, None, 0),
    NodeGroup("g4dn.metal", 95.44, 354_800, "T4", 8),
    NodeGroup("g6.2xlarge", 7.66, 28_800, "L4", 1),
    NodeGroup("g6e.2xlarge", 7.66, 59_100, "L40S", 1),
)


def dollars_per_hr(
    cpu_cores: float,
    memory_mib: float,
    gpu_type: str | None = None,
    gpu_count: int = 0,
) -> float:
    """$/hr a request costs while it holds its resources."""
    rate = cpu_cores * CPU_DOLLARS_PER_CORE_HR
    rate += (memory_mib / 1024) * MEM_DOLLARS_PER_GIB_HR
    if gpu_count and gpu_type:
        rate += gpu_count * GPU_DOLLARS_PER_HR.get(gpu_type, max(GPU_DOLLARS_PER_HR.values()))
    return rate


def kwargs_dollars_per_hr(kwargs: dict | None) -> float | None:
    """flyte.Resources-style kwargs dict → $/hr; None if unpriceable.

    Accepts gpu as an int count (typed T4, the tenant's cheapest pool) or a
    "TYPE:N" string.
    """
    if not kwargs or "cpu" not in kwargs or "memory" not in kwargs:
        return None
    try:
        cpu_raw = kwargs["cpu"]
        if isinstance(cpu_raw, str):
            s = cpu_raw.strip().lower()
            cpu = float(s[:-1]) / 1000 if s.endswith("m") else float(s)
        else:
            cpu = float(cpu_raw)
        mem = parse_memory_to_mib(kwargs["memory"])
    except (ValueError, TypeError):
        return None
    gpu_type, gpu_count = None, 0
    raw_gpu = kwargs.get("gpu")
    if raw_gpu:
        if isinstance(raw_gpu, str) and ":" in raw_gpu:
            gpu_type, _, n = raw_gpu.partition(":")
            try:
                gpu_count = int(n)
            except ValueError:
                gpu_count = 1
        elif isinstance(raw_gpu, str) and raw_gpu in GPU_VRAM_MIB:
            gpu_type, gpu_count = raw_gpu, 1
        else:
            try:
                gpu_type, gpu_count = "T4", int(raw_gpu)
            except (ValueError, TypeError):
                return None
    return dollars_per_hr(cpu, mem, gpu_type, gpu_count)


def cheapest_gpu_for(vram_mib: float, margin: float = 0.1) -> str | None:
    """Smallest-priced tenant GPU whose VRAM covers `vram_mib` (+margin);
    None if nothing fits (the task needs multi-GPU — out of scope)."""
    need = vram_mib * (1 + margin)
    fitting = [t for t in GPU_TYPES if GPU_VRAM_MIB[t] >= need]
    return min(fitting, key=GPU_DOLLARS_PER_HR.__getitem__) if fitting else None
