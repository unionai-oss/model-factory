"""GPU memory cells, the device ladder, utilization buckets, and the labels.

Pure logic: no flyte, no torch.
"""

from __future__ import annotations

#: GPU memory a task asks for, GiB. Dense where tasks are, and with cells at
#: each device's usable memory so "fits a T4" is a cell boundary.
GPU_MEM_GRID_GIB: tuple[float, ...] = (1, 1.5, 2, 3, 4, 6, 8, 10, 12, 15, 22, 44)


#: name, usable GiB (what NVML reports as total, rounded down), $/GPU-hour
#: on-demand us-east-1 for the single-GPU shape demo.hosted provisions.
DEVICES: tuple[tuple[str, float, float, str], ...] = (
    ("T4", 15.0, 0.526, "g4dn.xlarge"),
    ("L4", 22.0, 0.978, "g6.2xlarge"),
    ("A10G", 22.0, 1.212, "g5.2xlarge"),
    ("L40S", 44.0, 2.242, "g6e.2xlarge"),
)

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
    """Index into DEVICES of the cheapest device whose memory covers `gib`
    (the largest device if none does)."""
    fits = [i for i, d in enumerate(DEVICES) if d[1] + 1e-9 >= gib]
    if not fits:
        return len(DEVICES) - 1
    return min(fits, key=lambda i: DEVICES[i][2])


def next_device_up(i: int) -> int | None:
    """After a CUDA OOM on device i: the cheapest device with more memory."""
    bigger = [j for j, d in enumerate(DEVICES) if d[1] > DEVICES[i][1]]
    return min(bigger, key=lambda j: DEVICES[j][2]) if bigger else None


def label_mem_index(peak_gib: float) -> int:
    return mem_index_covering(peak_gib)


def util_index(util_pct: float) -> int:
    return min(9, max(0, int(util_pct // 10)))


def util_mid(idx: int) -> float:
    return idx * 10 + 5.0


def mem_within_margin(pred_gib: float, peak_gib: float) -> bool:
    return abs(pred_gib - peak_gib) <= max(MEM_MARGIN_ABS, MEM_MARGIN_REL * peak_gib)
