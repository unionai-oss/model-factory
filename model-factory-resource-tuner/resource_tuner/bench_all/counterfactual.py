"""Does an approach read the manifest? A controlled test (round 22).

A run and its K8s twin differ in more than the staging: the twin also drops
the Flyte runtime, so a model that has learned only "a Job needs 180 MiB less
than a Flyte task" scores well on run-vs-twin pairs without reading
``medium: Memory`` at all (on k8s-1, 2,400 of the 2,726 test pairs whose label
moved, moved down). Here the platform is held fixed: each sampled run is
rendered as a Job under all four stagings -- same function, inputs and task,
only the manifest, the fetch code and the history's units changing with the
staging -- and every approach sizes all four. A request that reads the
manifest moves when, and the way, the label moves:

    contrast         what changes                         the label
    medium           emptyDir {} -> {medium: Memory}      + staged bytes (often a cell or more)
    fetcher          aws-cli init container -> boto3       + 48 MiB (rarely a cell)

Pure logic: no flyte, no torch.
"""

from __future__ import annotations

import re
import zlib
from random import Random

import numpy as np

from . import dataset, k8s, scoring

#: The second test here: the task's name. ``render_task`` names the
#: TaskEnvironment after the family ("pandas-join"), and a Job manifest repeats
#: that name; a real task has a name too, but the bench's names are its family
#: labels. ``neutral`` renames both to a hash of the task, and the trained
#: models size the same runs again (no retraining): how much of the score is the name?
NEUTRAL = "-neutral"

#: (from, to): the medium contrast at each fetcher, the fetcher contrast at each medium.
CONTRASTS = {
    "medium": (("init-disk", "init-tmpfs"), ("boto3-disk", "boto3-tmpfs")),
    "fetcher": (("init-disk", "boto3-disk"), ("init-tmpfs", "boto3-tmpfs")),
}
SEP = "#"


def rendition_id(flyte_run_id: str, staging: str) -> str:
    return f"{flyte_run_id}{k8s.SUFFIX}{SEP}{staging}"


def sample_runs(rows: list[dict], n_per_split: int, seed: int = 5) -> list[dict]:
    """Usable CPU-node Flyte runs of test and ood, `n_per_split` each, one per
    task at most, stratified by family."""
    out = []
    for split in ("test", "ood"):
        by_fam: dict[str, dict[str, dict]] = {}
        for r in sorted(rows, key=lambda r: r["run_id"]):
            if (r["split"] == split and dataset.platform_of(r) == "flyte" and r["source"] != "gpu"
                    and dataset.usable(r)):
                by_fam.setdefault(r["family"], {}).setdefault(r["task_id"], r)
        rng = Random(seed)
        fams = sorted(by_fam)
        per = max(1, n_per_split // max(1, len(fams)))
        for f in fams:
            tasks = sorted(by_fam[f].values(), key=lambda r: r["run_id"])
            out += rng.sample(tasks, min(per, len(tasks)))
    return out


def renditions(rows: list[dict], picked: list[dict]) -> list[dict]:
    """Each picked run as a Job under every staging; its history is its task's
    earlier runs rendered under the same staging."""
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        if dataset.platform_of(r) == "flyte":
            by_task.setdefault(r["task_id"], []).append(r)
    out = []
    for p in picked:
        for st in k8s.STAGINGS:
            t = next(x for x in k8s.task_twins(by_task[p["task_id"]], st) if x["twin_of"] == p["run_id"])
            out.append({**t, "run_id": rendition_id(p["run_id"], st), "task_id": f"{t['task_id']}{SEP}{st}"})
    return [r for r in out if dataset.usable(r)]


def report(rends: list[dict], picks: dict[str, dict[str, dict]]) -> dict:
    """variant -> per contrast: label-moved share, direction right on moved
    pairs, request kept on unmoved pairs, exact; and regret per staging."""
    by_run: dict[str, dict[str, dict]] = {}
    for r in rends:
        base, st = r["run_id"].rsplit(SEP, 1)
        by_run.setdefault(base, {})[st] = r
    out = {}
    for v, p in picks.items():
        res: dict = {"split": {}}
        for split in ("test", "ood"):
            runs = {b: s for b, s in by_run.items() if next(iter(s.values()))["split"] == split}
            rec: dict = {}
            for name, pairs in CONTRASTS.items():
                moved, kept, exact, n_moved, n = [], [], [], 0, 0
                for s in runs.values():
                    for a, b in pairs:
                        if a not in s or b not in s or s[a]["run_id"] not in p or s[b]["run_id"] not in p:
                            continue
                        dl = int(s[b]["label_mem"]) - int(s[a]["label_mem"])
                        dp = int(p[s[b]["run_id"]]["mem_idx"]) - int(p[s[a]["run_id"]]["mem_idx"])
                        n += 1
                        if dl:
                            n_moved += 1
                            moved.append(np.sign(dp) == np.sign(dl))
                            exact.append(dp == dl)
                        else:
                            kept.append(dp == 0)
                rec[name] = {"pairs": n, "label_moved": n_moved / max(n, 1),
                             "moved_direction": float(np.mean(moved)) if moved else float("nan"),
                             "moved_exact": float(np.mean(exact)) if exact else float("nan"),
                             "unmoved_kept": float(np.mean(kept)) if kept else float("nan")}
            for st in k8s.STAGINGS:
                rs = [s[st] for s in runs.values() if st in s and s[st]["run_id"] in p]
                ratios = [scoring.ratio(r, *(int(p[r["run_id"]][k]) for k in ("mem_idx", "cpu_idx", "gpu_idx")))
                          for r in rs]
                rights = [scoring.mem_right(r, int(p[r["run_id"]]["mem_idx"])) for r in rs]
                rec[f"staging={st}"] = {"n": len(rs), "regret": float(np.mean(ratios)) - 1 if rs else float("nan"),
                                        "mem_right": float(np.mean(rights)) if rs else float("nan")}
            res["split"][split] = rec
        out[v] = res
    return out


def neutral(row: dict) -> dict:
    """The row with its task's name (the family's, in render_task) replaced by a hash."""
    name = k8s.job_name(row["task_code"]) if "TaskEnvironment(" in row["task_code"] else None
    if name is None:  # a twin's job.py: the name lives in the manifest
        m = re.search(r"generateName: (\S+)-\n", row.get("manifest") or "")
        name = m.group(1) if m else None
    if not name:
        return {**row, "run_id": row["run_id"] + NEUTRAL}
    new = f"task-{zlib.crc32(row['task_id'].split('@')[0].encode()) & 0xFFFFFF:06x}"
    sub = lambda t: re.sub(rf"\b{re.escape(name)}\b", new, t or "")  # noqa: E731
    return {**row, "run_id": row["run_id"] + NEUTRAL, "task_code": sub(row["task_code"]),
            "manifest": sub(row.get("manifest"))}


def name_report(rows: list[dict], picks: dict[str, dict[str, dict]]) -> dict:
    """variant -> split -> regret with the real name, with a neutral one, and their paired difference."""
    out = {}
    for v, p in picks.items():
        res = {}
        for split in ("test", "ood"):
            d, a, b = [], [], []
            for r in rows:
                n = r["run_id"] + NEUTRAL
                if r["split"] != split or r["run_id"] not in p or n not in p:
                    continue
                ra = scoring.ratio(r, *(int(p[r["run_id"]][k]) for k in ("mem_idx", "cpu_idx", "gpu_idx")))
                rb = scoring.ratio(r, *(int(p[n][k]) for k in ("mem_idx", "cpu_idx", "gpu_idx")))
                a.append(ra), b.append(rb), d.append(rb - ra)
            if d:
                res[split] = {"n": len(d), "named": float(np.mean(a)) - 1, "neutral": float(np.mean(b)) - 1,
                              "diff": float(np.mean(d))}
        out[v] = res
    return out
