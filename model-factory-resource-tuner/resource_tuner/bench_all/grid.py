"""The joint request: memory, CPU, and a GPU (or none).

The three axes are the three benches' own grids, so a cell means the same
thing here as on its bench:

    memory  RightSizeMem's half-power-of-two ladder (256Mi ... 64Gi)
    cpu     RightSizeCPU's core steps (0.25 ... 8)
    gpu     "none", or RightSizeGPU's GPU memory cells (1GiB ... 44GiB),
            each of which lands on the cheapest device that holds it

Pure logic: no flyte, no torch.
"""

from __future__ import annotations

from ..bench.grid import MEMORY_GRID_MIB, format_memory, mem_index_covering
from ..bench_cpu.grid import CPU_GRID, format_cores, index_covering
from ..bench_gpu.grid import DEVICES, GPU_MEM_GRID_GIB, device_for, format_gib, mem_index_covering as gpu_cell_covering

N_MEM = len(MEMORY_GRID_MIB)
N_CPU = len(CPU_GRID)
#: gpu index 0 is "no GPU"; index k >= 1 is GPU memory cell k - 1.
N_GPU = 1 + len(GPU_MEM_GRID_GIB)
NO_GPU = 0

MEM_OPTIONS: tuple[str, ...] = tuple(format_memory(m) for m in MEMORY_GRID_MIB)
CPU_OPTIONS: tuple[str, ...] = tuple(format_cores(c) for c in CPU_GRID)
GPU_OPTIONS: tuple[str, ...] = ("none",) + tuple(format_gib(g) for g in GPU_MEM_GRID_GIB)


def mem_cell(mib: float) -> int:
    return mem_index_covering(mib)


def cpu_cell(cores: float) -> int:
    return index_covering(cores)


def gpu_cell(peak_gib: float | None) -> int:
    """The gpu index covering a GPU memory peak; None (a CPU task) -> no GPU."""
    return NO_GPU if peak_gib is None else 1 + gpu_cell_covering(peak_gib)


def gpu_gib(g: int) -> float:
    return GPU_MEM_GRID_GIB[g - 1]


def device_of(g: int) -> int | None:
    """Index into DEVICES a gpu choice lands on, or None for no GPU."""
    return None if g == NO_GPU else device_for(gpu_gib(g))


__all__ = ["CPU_GRID", "DEVICES", "GPU_MEM_GRID_GIB", "MEMORY_GRID_MIB", "N_CPU", "N_GPU", "N_MEM", "NO_GPU",
           "MEM_OPTIONS", "CPU_OPTIONS", "GPU_OPTIONS", "mem_cell", "cpu_cell", "gpu_cell", "gpu_gib", "device_of"]
