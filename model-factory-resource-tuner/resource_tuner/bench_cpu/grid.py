"""The CPU decision grid, the platform's measurement window, and the label.

Pure logic: no flyte import.
"""

from __future__ import annotations

#: Candidate core counts, the way requests are written (250m steps up to 2,
#: then half and whole cores). The widest gap is 25% (4 -> 5), so the
#: oracle's cell is always within the bench's accuracy margin of the peak.
CPU_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)

#: Union's task monitoring scrapes cAdvisor every 15 s and charts irate()
#: over consecutive scrapes: CPU is seen in 15-second windows.
PLATFORM_WINDOW_S = 15
#: The measurement pod's CPU limit. Workloads size their own pools to at
#: most 8 threads; the extra core is the harness and the Flyte runtime.
MEASURE_POD_CPU = 9

#: "Accurate" for a peak prediction: within 25% of the measured peak, or a
#: quarter core for small peaks (a 0.3-core task is not 'wrong' at 0.5).
MARGIN_REL = 0.25
MARGIN_ABS = 0.25


def format_cores(c: float) -> str:
    return f"{c:g}"


def index_covering(cores: float) -> int:
    """Smallest cell >= cores (clamped to the top)."""
    for i, step in enumerate(CPU_GRID):
        if cores <= step + 1e-9:
            return i
    return len(CPU_GRID) - 1


def label_peak_index(peak_cores: float) -> int:
    """The oracle's peak cell: the smallest request that covers the peak window."""
    return index_covering(peak_cores)


def label_mean_index(mean_cores: float) -> int:
    return index_covering(mean_cores)


def within_margin(predicted: float, actual: float) -> bool:
    return abs(predicted - actual) <= max(MARGIN_ABS, MARGIN_REL * actual)


def peak_window(series: list[float], window_s: int = PLATFORM_WINDOW_S) -> float:
    """Busiest `window_s`-second mean of a per-second core series.

    Sliding, not phase-aligned: the platform's scrapes land at an arbitrary
    phase, and the label is the worst any phase could see."""
    if not series:
        return 0.0
    w = min(window_s, len(series))
    s = sum(series[:w])
    best = s
    for i in range(w, len(series)):
        s += series[i] - series[i - w]
        best = max(best, s)
    return best / w
