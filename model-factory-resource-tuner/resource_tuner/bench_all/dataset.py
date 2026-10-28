"""One corpus from three: RightSizeMem, RightSizeCPU and RightSizeGPU rows,
labelled on every axis each one measured.

    source  memory label            cpu label                    gpu label
    mem     ru_maxrss (+ runtime)   mean cores (no 1 s series)   none
    cpu     ru_maxrss (+ runtime)   busiest 15 s window          none
    gpu     -- (the instance's)     -- (the instance's)          peak device memory

A GPU task's host memory and cores come with the GPU instance it lands on,
so its memory and CPU requests are not scored; a CPU task's right GPU
answer is "none". Splits are each bench's own (by task; held-out families
stay ``ood``), and run ids are prefixed by source so they never collide.

History entries are rewritten to one schema (``peak_mem_mib``,
``peak_cores``, ``mean_cores``, ``peak_gpu_mib``, ``mean_util``, whichever
were measured) so one prompt and one feature set read all three.

Pure logic: no flyte.
"""

from __future__ import annotations

import json

from ..bench import dataset as mem_dataset
from ..bench.grid import required_memory_mib
from ..bench_gpu import grid as gpu_grid
from . import grid

SPLITS = mem_dataset.SPLITS
SOURCES = ("mem", "cpu", "gpu")

#: Columns every unified row carries (the scorers' and arms' inputs).
KEEP = ("run_id", "task_id", "family", "driver", "run_idx", "split", "task_code", "inputs_json",
        "input_bytes_json", "inputs_desc", "wall_s")


def _history(entries: list[dict], source: str) -> list[dict]:
    out = []
    for h in entries:
        e = {"inputs_desc": h.get("inputs_desc", ""), "input_bytes": h.get("input_bytes", {}),
             "wall_s": h.get("wall_s", 0.0)}
        if source == "mem":
            e.update(peak_mem_mib=h.get("peak_rss_mib", 0.0), mean_cores=h.get("avg_cores", 0.0))
        elif source == "cpu":
            e.update(peak_cores=h.get("peak_cores", 0.0), mean_cores=h.get("mean_cores", 0.0))
            if "peak_rss_mib" in h:
                e["peak_mem_mib"] = h["peak_rss_mib"]
        else:
            e.update(peak_gpu_mib=h.get("peak_gpu_mib", 0.0), mean_util=h.get("mean_util", 0.0))
        out.append(e)
    return out


def unify_row(r: dict, source: str) -> dict:
    out = {k: r.get(k) for k in KEEP}
    out["run_id"] = f"{source}:{r['run_id']}"
    out["task_id"] = f"{source}:{r['task_id']}"
    out["source"] = source
    out["driver"] = r.get("driver") or r.get("memory_driver")  # RightSizeMem names it memory_driver
    out["history_json"] = json.dumps(_history(json.loads(r.get("history_json") or "[]"), source))
    out["n_history"] = int(r.get("n_history") or 0)
    out["ok"] = bool(r.get("ok")) and not r.get("censored") and not r.get("fills_device")
    if source in ("mem", "cpu"):
        need = float(r["required_mib"]) if source == "mem" else required_memory_mib(float(r["peak_rss_mib"]))
        cores = float(r["avg_cores"]) if source == "mem" else float(r["peak_cores"])
        out.update(need_mib=need, t_peak_frac=float(r.get("t_peak_frac") or 0.5),
                   need_cores=cores, mean_cores=float(r.get("avg_cores") if source == "mem" else r["mean_cores"]),
                   cpu_series_json=(r.get("cpu_series_json") or "[]") if source == "cpu" else "[]",
                   peak_gpu_gib=float("nan"), sharded=False, gpu_imbalance=1.0)
        out["label_mem"] = grid.mem_cell(need)
        out["label_cpu"] = grid.cpu_cell(cores)
        out["label_gpu"] = grid.NO_GPU
    else:
        peak = float(r["peak_gpu_gib"])
        sharded = bool(r.get("sharded"))
        imb = float(r.get("gpu_imbalance") or gpu_grid.DEFAULT_IMBALANCE)
        out.update(need_mib=float("nan"), t_peak_frac=float(r.get("t_peak_frac") or 0.1),
                   need_cores=float(r.get("mean_cpu_cores") or 0.0), mean_cores=float(r.get("mean_cpu_cores") or 0.0),
                   cpu_series_json="[]", peak_gpu_gib=peak, sharded=sharded, gpu_imbalance=imb)
        out["label_mem"] = -1
        out["label_cpu"] = -1
        # the cheapest shape the run fits (a sharded run's share may need 8xT4, not 4xL4)
        out["label_gpu"] = 1 + gpu_grid.oracle_cell(peak, sharded, imb)
    # unscoreable: a CPU-bench row whose RSS was not recorded, or a need above the top cell
    if source == "cpu" and not float(r.get("peak_rss_mib") or 0.0):
        out["ok"] = False
    if source != "gpu" and out["need_mib"] > grid.MEMORY_GRID_MIB[-1]:
        out["ok"] = False
    return out


def unify(mem_rows: list[dict], cpu_rows: list[dict], gpu_rows: list[dict]) -> list[dict]:
    return ([unify_row(r, "mem") for r in mem_rows] + [unify_row(r, "cpu") for r in cpu_rows]
            + [unify_row(r, "gpu") for r in gpu_rows])


def usable(row: dict) -> bool:
    return bool(row.get("ok"))


def by_split(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for r in rows:
        if usable(r):
            out[r["split"]].append(r)
    return out


def balanced(rows: list[dict], per_source: int, seed: int = 0) -> list[dict]:
    """At most `per_source` rows of each source, deterministic: the LLM arms'
    training budget, so the largest corpus (RightSizeMem) does not drown the
    other two."""
    from random import Random

    out = []
    for s in SOURCES:
        mine = sorted((r for r in rows if r["source"] == s), key=lambda r: r["run_id"])
        Random(seed).shuffle(mine)
        out += mine[:per_source] if per_source else mine
    Random(seed + 1).shuffle(out)
    return out
