"""GPU memory cells, the machine ladder, utilization buckets, and the labels.

The request is GPU memory; it lands on the cheapest machine shape whose GPU
memory holds it. Since big-2 the ladder is every GPU shape demo.hosted
provisions, single- and multi-GPU (round 21, from the cluster's node pools):

    T4      g4dn.xlarge    1 x 15 GiB      L40S    g6e.2xlarge   1 x 44 GiB
    L4      g6.2xlarge     1 x 22 GiB      4xL4    g6.12xlarge   4 x 22 GiB
    A10G    g5.2xlarge     1 x 22 GiB      8xT4    g4dn.metal    8 x 15 GiB
                                           4xL40S  g6e.12xlarge  4 x 44 GiB

A multi-GPU machine only helps a task whose code spreads its model over
every visible GPU (``sharded``: layers or embedding tables placed by
``torch.cuda.device_count()``); any other task uses one of its GPUs and
pays for all of them. Whether a run *fits* a shape it was not measured on
is modelled (``fits``): one device must hold the whole peak; a sharded run
on n devices holds its share, (peak - context) x imbalance / n + context,
with the imbalance measured where the run was measured on several GPUs.

Pure logic: no flyte, no torch.
"""

from __future__ import annotations

from typing import NamedTuple

#: GPU memory a task asks for, GiB: dense where tasks are, with a cell at each
#: shape's capacity so "fits a T4" / "fits a 4xL4" is a cell boundary.
GPU_MEM_GRID_GIB: tuple[float, ...] = (1, 1.5, 2, 3, 4, 6, 8, 10, 12, 15, 22, 32, 44, 64, 88, 120, 176)


class Device(NamedTuple):
    name: str
    n_gpus: int
    per_gpu_gib: float  # usable memory per GPU (NVML total, rounded down)
    price: float  # $/hour, the whole node, on-demand us-east-1
    instance: str

    @property
    def capacity_gib(self) -> float:
        return self.n_gpus * self.per_gpu_gib


DEVICES: tuple[Device, ...] = (
    Device("T4", 1, 15.0, 0.526, "g4dn.xlarge"),
    Device("L4", 1, 22.0, 0.978, "g6.2xlarge"),
    Device("A10G", 1, 22.0, 1.212, "g5.2xlarge"),
    Device("L40S", 1, 44.0, 2.242, "g6e.2xlarge"),
    Device("4xL4", 4, 22.0, 4.602, "g6.12xlarge"),
    Device("8xT4", 8, 15.0, 7.824, "g4dn.metal"),
    Device("4xL40S", 4, 44.0, 10.493, "g6e.12xlarge"),
)

#: CUDA context + allocator reserve every used GPU carries, GiB.
CONTEXT_GIB = 0.5
#: A sharded run measured on one GPU: how unevenly its layers would split
#: (embeddings on the first GPU, the LM head on the last).
DEFAULT_IMBALANCE = 1.25

#: Mean GPU utilization, in 10-point buckets.
UTIL_BUCKETS: tuple[str, ...] = tuple(f"{lo}-{lo + 10}%" for lo in range(0, 100, 10))
#: "Accurate" for a utilization prediction: within 10 points.
UTIL_MARGIN = 10.0
#: "Accurate" for a memory prediction: within 20% of the peak (or 0.5 GiB).
MEM_MARGIN_REL = 0.2
MEM_MARGIN_ABS = 0.5

#: Union's dcgm-exporter samples every 30 s (its default; unionai/cloud sets no -c).
DCGM_INTERVAL_S = 30


def format_gib(g: float) -> str:
    return f"{g:g}GiB"


def mem_index_covering(gib: float) -> int:
    for i, step in enumerate(GPU_MEM_GRID_GIB):
        if gib <= step + 1e-9:
            return i
    return len(GPU_MEM_GRID_GIB) - 1


def device_for(gib: float) -> int:
    """Index into DEVICES of the cheapest shape whose GPU memory covers `gib`
    (the largest shape if none does)."""
    fits = [i for i, d in enumerate(DEVICES) if d.capacity_gib + 1e-9 >= gib]
    if not fits:
        return len(DEVICES) - 1
    return min(fits, key=lambda i: DEVICES[i].price)


def next_device_up(i: int) -> int | None:
    """After a CUDA OOM on shape i: the cheapest shape with more GPU memory."""
    bigger = [j for j, d in enumerate(DEVICES) if d.capacity_gib > DEVICES[i].capacity_gib]
    return min(bigger, key=lambda j: DEVICES[j].price) if bigger else None


def per_gpu_need(peak_gib: float, n_gpus: int, sharded: bool, imbalance: float = DEFAULT_IMBALANCE) -> float:
    """GiB the busiest GPU holds when a run with this peak runs on `n_gpus` GPUs."""
    if n_gpus == 1 or not sharded:
        return peak_gib
    return max(peak_gib - CONTEXT_GIB, 0.0) * max(imbalance, 1.0) / n_gpus + CONTEXT_GIB


def fits(dev: int, peak_gib: float, sharded: bool = False, imbalance: float = DEFAULT_IMBALANCE) -> bool:
    d = DEVICES[dev]
    return per_gpu_need(peak_gib, d.n_gpus, sharded, imbalance) <= d.per_gpu_gib + 1e-9


def oracle_device(peak_gib: float, sharded: bool = False, imbalance: float = DEFAULT_IMBALANCE) -> int | None:
    """The cheapest shape the run fits, or None if none does."""
    ok = [i for i in range(len(DEVICES)) if fits(i, peak_gib, sharded, imbalance)]
    return min(ok, key=lambda i: DEVICES[i].price) if ok else None


def oracle_cell(peak_gib: float, sharded: bool = False, imbalance: float = DEFAULT_IMBALANCE) -> int:
    """The smallest memory cell that lands on the oracle's shape (a sharded run
    whose share does not fit 4xL4 has to ask past 88 GiB to reach 8xT4)."""
    dev = oracle_device(peak_gib, sharded, imbalance)
    lo = mem_index_covering(peak_gib)
    if dev is None:
        return lo
    for i in range(lo, len(GPU_MEM_GRID_GIB)):
        if device_for(GPU_MEM_GRID_GIB[i]) == dev:
            return i
    return lo


def label_mem_index(peak_gib: float) -> int:
    return mem_index_covering(peak_gib)


def util_index(util_pct: float) -> int:
    return min(9, max(0, int(util_pct // 10)))


def util_mid(idx: int) -> float:
    return idx * 10 + 5.0


def mem_within_margin(pred_gib: float, peak_gib: float) -> bool:
    return abs(pred_gib - peak_gib) <= max(MEM_MARGIN_ABS, MEM_MARGIN_REL * peak_gib)
