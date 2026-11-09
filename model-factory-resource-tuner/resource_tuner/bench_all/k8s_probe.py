"""In-cluster probes behind the Kubernetes twin's accounting (``k8s``, round 22).

Two questions, each answered by measurement on demo.hosted pods:

1. **What does staging cost?** (``staging_probe``) For inputs of 64 MiB to
   4 GiB uploaded to the cluster's object store: the resident memory boto3
   adds to a job (import, client, ``download_file``; peak during the
   transfer and what stays after it), the transfer's throughput and cores,
   and how much of a file written to a memory-backed volume the container's
   memory cgroup is charged (``memory.stat`` shmem). These set the
   constants in ``k8s``.
2. **Does the twin's label hold for real workloads?** (``validate``) A
   stratified sample of RightSizeMem and RightSizeCPU runs, each run twice
   in one pod: as the Flyte bench measured it (the harness, the control),
   and as its K8s twin -- the rendered ``job.py`` staging its inputs from
   the object store, with the init-container stagings done by a separate
   process, as an init container is. The twin's memory is read where the
   kernel charges it: the job's own ``ru_maxrss`` plus the tmpfs pages
   (cgroup ``shmem``) while it runs. Agreement with ``k8s.twin``'s label,
   by ladder cell, is the result.

The memory-backed volume is the pod's ``/dev/shm`` (``Resources(shm=...)``
mounts an ``emptyDir {medium: Memory}`` there), the same kind of volume the
twins' manifests declare.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import flyte
import flyte.io
import numpy as np

from ..bench import workloads as mem_w
from ..bench.common import read_rows
from ..bench.grid import FLYTE_RUNTIME_OVERHEAD_MIB, LABEL_HEADROOM, mem_index_covering
from ..bench.llm_arms import TrainConfig
from ..bench.measure import _base_env, _maxrss_mib, prepare_inputs
from ..bench_cpu import workloads as cpu_w
from ..bench_gpu import workloads as gpu_w
from ..shared.reporting import Reporter
from . import grid, k8s, policy
from .envs import allbench_train_env, k8s_probe_env

TMPFS = "/dev/shm"
CG = "/sys/fs/cgroup"


# ── cgroup reads ────────────────────────────────────────────────────────


def cgroup_version() -> int:
    return 2 if os.path.exists(f"{CG}/memory.stat") and os.path.exists(f"{CG}/cgroup.controllers") else 1


def mem_stat() -> dict[str, int]:
    """This container's memory cgroup counters (bytes): anon, shmem, file, current."""
    path = f"{CG}/memory.stat" if cgroup_version() == 2 else f"{CG}/memory/memory.stat"
    out: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            k, v = line.split()
            out[k] = int(v)
    cur = f"{CG}/memory.current" if cgroup_version() == 2 else f"{CG}/memory/memory.usage_in_bytes"
    with open(cur) as f:
        out["current"] = int(f.read().strip())
    if "anon" not in out and "rss" in out:  # cgroup v1 names
        out["anon"] = out["rss"]
    if "file" not in out and "cache" in out:
        out["file"] = out["cache"]
    return out


class Sampler:
    """Samples the cgroup's anon + shmem every 20 ms while a child runs."""

    def __init__(self):
        self.peak = {"anon": 0, "shmem": 0, "anon_shmem": 0}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            s = mem_stat()
            a, sh = s.get("anon", 0), s.get("shmem", 0)
            self.peak["anon"] = max(self.peak["anon"], a)
            self.peak["shmem"] = max(self.peak["shmem"], sh)
            self.peak["anon_shmem"] = max(self.peak["anon_shmem"], a + sh)
            time.sleep(0.02)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)


#: Prepended to every measured child: at exit it writes its own VmHWM. A
#: child's ``ru_maxrss`` from ``wait4`` is not its own peak -- at ``exec`` the
#: kernel folds the *parent's* high-water mark into the child's
#: (``setmax_mm_hiwater_rss`` on the old mm), so a child of a 400 MiB parent
#: reports >= 400 MiB however little it used (rt-k8sprobe-1: 228-490 MiB
#: ru_maxrss for children whose own VmHWM was 74-80 MiB).
_SELF_HWM = r"""
import atexit as _ae, os as _os
def _rsk_dump():
    p = _os.environ.get("RSK_HWM_FILE")
    if p:
        with open(p, "w") as f:
            f.write(" ".join(l.split()[1] for l in open("/proc/self/status") if l.startswith("VmHWM:")))
_ae.register(_rsk_dump)
"""


def _run_child(argv: list[str], env: dict | None = None) -> dict:
    """Run `python argv...` and measure it three ways: its own VmHWM (at exit),
    ``ru_maxrss`` from ``wait4`` (inherits the parent's peak, kept to measure
    that), and the container cgroup's anon + shmem peak over the run."""
    hwm_file = tempfile.mktemp(prefix="rsk-hwm-")
    env = {**(env or _base_env(1)), "RSK_HWM_FILE": hwm_file}
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        base = mem_stat()
        with Sampler() as smp:
            t0 = time.monotonic()
            proc = subprocess.Popen([sys.executable, *argv], stdout=out, stderr=err, env=env)
            _, status, ru = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)  # reaped by wait4
            wall = time.monotonic() - t0
        out.seek(0), err.seek(0)
        own = 0.0
        if os.path.exists(hwm_file):
            with open(hwm_file) as f:
                own = float(f.read().strip() or 0) / 1024
            os.remove(hwm_file)
        return {"rc": proc.returncode, "maxrss_mib": _maxrss_mib(ru), "own_hwm_mib": own,
                "cg_anon_mib": (smp.peak["anon"] - base.get("anon", 0)) / 2**20,
                "cg_peak_mib": (smp.peak["anon_shmem"] - base.get("anon", 0) - base.get("shmem", 0)) / 2**20,
                "cg_shmem_peak_mib": smp.peak["shmem"] / 2**20, "cpu_s": ru.ru_utime + ru.ru_stime,
                "wall_s": wall, "stdout": out.read()[-4000:], "stderr": err.read()[-2000:]}


def _self_hwm_mib() -> float:
    with open("/proc/self/status") as f:
        return next(float(ln.split()[1]) / 1024 for ln in f if ln.startswith("VmHWM:"))


_DOWNLOAD = r"""
import json, os, sys, time, gc
def vm(k):
    for line in open("/proc/self/status"):
        if line.startswith(k):
            return int(line.split()[1]) / 1024
t0 = time.monotonic()
base = vm("VmRSS:")
import boto3
s3 = boto3.client("s3")
t_import = time.monotonic() - t0
client = vm("VmRSS:")
bucket, key = sys.argv[1].removeprefix("s3://").split("/", 1)
t1 = time.monotonic()
s3.download_file(bucket, key, sys.argv[2])
t_dl = time.monotonic() - t1
gc.collect()
print(json.dumps({"base_rss": base, "client_rss": client, "after_rss": vm("VmRSS:"), "hwm": vm("VmHWM:"),
                  "t_import": t_import, "t_download": t_dl, "boto3": boto3.__version__}))
"""


def _write_random(path: str, mib: int) -> None:
    rng = np.random.default_rng(mib)
    with open(path, "wb") as f:
        f.writelines(rng.integers(0, 256, 16 * 2**20, dtype=np.uint8).tobytes() for _ in range(mib // 16))


@k8s_probe_env.task(report=True)
async def staging_probe(sizes_mib: list[int] | None = None, repeats: int = 2) -> str:
    """boto3's footprint, transfer throughput and cores, and tmpfs charging (question 1)."""
    sizes_mib = sizes_mib or [64, 512, 2048, 4096]
    out = {"cgroup": cgroup_version(), "cpu_count": os.cpu_count(), "runs": []}
    out["python_pass"] = _run_child(["-c", "pass"])["maxrss_mib"]
    out["python_boto3_client"] = _run_child(["-c", "import boto3; boto3.client('s3')"])["maxrss_mib"]
    work = tempfile.mkdtemp(prefix="k8sprobe-")
    for mib in sizes_mib:
        src = os.path.join(work, f"blob-{mib}.bin")
        _write_random(src, mib)
        uri = (await flyte.io.File.from_local(src)).path
        os.remove(src)
        for rep in range(repeats):
            for where in ("disk", "tmpfs"):
                dest = os.path.join(TMPFS if where == "tmpfs" else work, f"dl-{mib}-{rep}.bin")
                before = mem_stat()
                r = _run_child(["-c", _SELF_HWM + _DOWNLOAD, uri, dest])
                after = mem_stat()
                info = json.loads(r["stdout"].strip().splitlines()[-1]) if r["rc"] == 0 else {"error": r["stderr"]}
                size = os.path.getsize(dest) if os.path.exists(dest) else 0
                if os.path.exists(dest):
                    os.remove(dest)
                freed = mem_stat()
                out["runs"].append({
                    "mib": mib, "rep": rep, "where": where, "uri": uri, **info,
                    "child_maxrss_mib": r["maxrss_mib"], "child_own_hwm_mib": r["own_hwm_mib"],
                    "child_cg_anon_mib": r["cg_anon_mib"], "child_cpu_s": r["cpu_s"], "child_wall_s": r["wall_s"],
                    "cores_during_download": r["cpu_s"] / max(info.get("t_download", r["wall_s"]), 1e-6),
                    "MBps": size / 1e6 / max(info.get("t_download", 1e-6), 1e-6),
                    "shmem_delta_mib": (after["shmem"] - before["shmem"]) / 2**20,
                    "shmem_freed_mib": (after["shmem"] - freed["shmem"]) / 2**20,
                    "current_delta_mib": (after["current"] - before["current"]) / 2**20,
                    "peak_anon_shmem_delta_mib": r["cg_peak_mib"],
                    "file_bytes": size,
                })
                print(json.dumps(out["runs"][-1]), flush=True)
    rep = Reporter("K8s twin: staging probe", f"cgroup v{out['cgroup']}, {out['cpu_count']} CPUs")
    rep.kv({"python -c pass (MiB)": f"{out['python_pass']:.1f}",
            "import boto3 + client (MiB)": f"{out['python_boto3_client']:.1f}"})
    rep.table(["MiB", "where", "after RSS", "HWM", "MB/s", "cores", "shmem delta", "peak anon+shmem delta"],
              [[str(x["mib"]), x["where"], f"{x.get('after_rss', 0):.0f}", f"{x.get('hwm', 0):.0f}", f"{x['MBps']:.0f}",
                f"{x['cores_during_download']:.2f}", f"{x['shmem_delta_mib']:.0f}", f"{x['peak_anon_shmem_delta_mib']:.0f}"]
               for x in out["runs"]])
    await rep.flush()
    return json.dumps(out)


# ── question 2: real workloads, Flyte control vs K8s twin ───────────────

_INIT = r"""
import sys, boto3
s3 = boto3.client("s3")
for uri, path in zip(sys.argv[1::2], sys.argv[2::2]):
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    s3.download_file(bucket, key, path)
"""

#: The Flyte bench's measured child, as ``bench.measure._CHILD`` runs it.
_CONTROL = _SELF_HWM + r"""
import json, sys
spec = json.loads(sys.argv[1])
ns = {}
exec(compile(open(spec["harness"]).read(), "workload", "exec"), ns)
ns["main"](*spec["paths"])
"""

#: The twin runs as ``python job.py args``, the hook prepended to the file itself
#: (rt-k8sval-2 ran it through runpy, which broke process pools: a worker cannot
#: unpickle a function defined in a runpy module).


def _job_source(module: str, stage: str) -> str:
    """The rendered job.py, its /data volume pointed at this pod's staging dir,
    and the upload skipped (the twin's bucket is fictional)."""
    src = module.replace('"/data', f'"{stage}')
    return src.replace("    s3.upload_file(", "    pass  # s3.upload_file(")


def validate_one(row: dict, staging: str, uris: dict[str, str], paths: list[str], sizes: dict[str, int],
                 work: Path) -> dict:
    """Control (the harness on local files, as the Flyte bench measured it), then
    the twin: its ``job.py`` staging the same files from the object store."""
    inputs = json.loads(row["inputs_json"])
    harness = work / "workload.py"
    harness.write_text(row["harness_code"])
    parent_hwm = _self_hwm_mib()
    control = _run_child(["-c", _CONTROL, json.dumps({"harness": str(harness), "paths": paths})])
    for p in paths:
        os.remove(p)
    stage = TMPFS if k8s.is_tmpfs(staging) else str(work / "data")
    os.makedirs(stage, exist_ok=True)
    job = work / "job.py"
    job.write_text(_SELF_HWM + _job_source(k8s.module_code(row["task_code"], inputs, staging), stage))
    staged = [os.path.join(stage, k8s._file_name(s)) for s in inputs]
    env = {**_base_env(1), "OUTPUT_URI": "s3://unused/out/"}
    shmem0 = mem_stat().get("shmem", 0)
    if not k8s.is_boto3(staging):  # the init container: another process, done before the job starts
        init = _run_child(["-c", _INIT, *[x for s, p in zip(inputs, staged) for x in (uris[s["name"]], p)]])
        if init["rc"] != 0:
            return {"run_id": row["run_id"], "staging": staging, "ok": False, "error": "init: " + init["stderr"][-300:]}
        args = staged
    else:
        env.update({k8s._env_name(s): uris[s["name"]] for s in inputs})
        args = []
    pre = mem_stat().get("shmem", 0) - shmem0  # what the init container left on the tmpfs
    job_run = _run_child([str(job), *args], env=env)
    for p in staged:
        if os.path.exists(p):
            os.remove(p)
    tmpfs_mib = max(job_run["cg_shmem_peak_mib"] - shmem0 / 2**20, 0.0) if k8s.is_tmpfs(staging) else 0.0
    # two instruments, each compared like for like:
    #   rss: own VmHWM (what ru_maxrss would be without the parent's) + the tmpfs pages
    #   cgroup: the container's anon (+ shmem) delta, all processes, as the kernel charges it
    pred_rss = k8s.container_peak_mib(control["own_hwm_mib"], sizes, staging)
    meas_rss = job_run["own_hwm_mib"] + tmpfs_mib
    pred_cg = (control["cg_anon_mib"] + (k8s.BOTO3_CG_MIB if k8s.is_boto3(staging) else 0.0)
               + (k8s.staged_bytes(sizes) / 2**20 if k8s.is_tmpfs(staging) else 0.0))
    meas_cg = job_run["cg_peak_mib"] + pre / 2**20
    return {
        "run_id": row["run_id"], "family": row["family"], "staging": staging,
        "ok": job_run["rc"] == 0 and control["rc"] == 0,
        "error": (job_run["stderr"] if job_run["rc"] else control["stderr"] if control["rc"] else "")[-300:],
        "input_mib": sum(sizes.values()) / 2**20, "parent_hwm_mib": parent_hwm,
        "corpus_peak_mib": float(row["peak_rss_mib"]),
        "control_own_hwm_mib": control["own_hwm_mib"], "control_maxrss_mib": control["maxrss_mib"],
        "control_cg_mib": control["cg_anon_mib"],
        "job_own_hwm_mib": job_run["own_hwm_mib"], "job_maxrss_mib": job_run["maxrss_mib"],
        "job_cg_mib": meas_cg, "tmpfs_mib": tmpfs_mib,
        "pred_rss_mib": pred_rss, "meas_rss_mib": meas_rss, "pred_cg_mib": pred_cg, "meas_cg_mib": meas_cg,
        "cell_pred": mem_index_covering(pred_rss * LABEL_HEADROOM),
        "cell_meas": mem_index_covering(meas_rss * LABEL_HEADROOM),
        "cell_corpus": mem_index_covering(k8s.twin_need_mib(
            float(row.get("required_mib") or _flyte_need(float(row["peak_rss_mib"]))), sizes, staging)),
        "job_wall_s": job_run["wall_s"], "job_cpu_s": job_run["cpu_s"],
        "control_wall_s": control["wall_s"], "control_cpu_s": control["cpu_s"],
    }


def _flyte_need(peak_rss_mib: float) -> float:
    """The corpus's Flyte need as the round-22 sweeps labelled it (k8s-1: the 180 MiB constant)."""
    return (peak_rss_mib + FLYTE_RUNTIME_OVERHEAD_MIB) * LABEL_HEADROOM


@k8s_probe_env.task(retries=1)
async def validate_shard(rows_json: str) -> str:
    """Each (row, staging) of the shard: upload its inputs, then control and twin (``validate_one``)."""
    out = []
    for item in json.loads(rows_json):
        row, staging = item["row"], item["staging"]
        with tempfile.TemporaryDirectory(prefix="k8sval-") as d:
            work = Path(d)
            inputs = json.loads(row["inputs_json"])
            try:
                paths, sizes = prepare_inputs(inputs, work, int(row["seed"]))
                uris = {s["name"]: (await flyte.io.File.from_local(p)).path for s, p in zip(inputs, paths)}
                res = await asyncio.to_thread(validate_one, row, staging, uris, paths, sizes, work)
            except Exception as e:  # noqa: BLE001 -- one bad run must not sink the shard
                res = {"run_id": row["run_id"], "staging": staging, "ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
        print(json.dumps(res), flush=True)
        out.append(res)
    return json.dumps(out)


def sample_rows(rows: list[dict], n: int, max_need_mib: float, max_input_mib: float, seed: int = 0) -> list[dict]:
    """Usable runs, family-stratified, small enough for one probe pod (tmpfs included)."""
    ok = [r for r in rows if r.get("ok") and not r.get("censored") and float(r.get("peak_rss_mib") or 0) > 0
          and float(r["peak_rss_mib"]) + 200 < max_need_mib
          and sum(json.loads(r.get("input_bytes_json") or "{}").values()) / 2**20 < max_input_mib]
    by_fam: dict[str, list[dict]] = {}
    for r in sorted(ok, key=lambda r: r["run_id"]):
        by_fam.setdefault(r["family"], []).append(r)
    rng = random.Random(seed)
    per = max(1, n // max(1, len(by_fam)))
    out = []
    for fam in sorted(by_fam):
        out += rng.sample(by_fam[fam], min(per, len(by_fam[fam])))
    return out


_KEEP = ("run_id", "family", "task_code", "harness_code", "inputs_json", "input_bytes_json", "seed", "peak_rss_mib",
         "required_mib")


@k8s_probe_env.task(report=True)
async def validate(mem: flyte.io.File, cpu: flyte.io.File, n_per_source: int = 96, shard_size: int = 8) -> str:
    """Question 2 over a stratified sample of both CPU-side corpora; every staging equally often."""
    picked = []
    for name, f in (("mem", mem), ("cpu", cpu)):
        rows = await read_rows(f)
        picked += [{**{k: r.get(k) for k in _KEEP}, "run_id": f"{name}:{r['run_id']}"}
                   for r in sample_rows(rows, n_per_source, max_need_mib=8192, max_input_mib=3072)]
    items = [{"row": r, "staging": k8s.STAGINGS[i % len(k8s.STAGINGS)]} for i, r in enumerate(picked)]
    shards = [items[i: i + shard_size] for i in range(0, len(items), shard_size)]
    results = []
    for chunk in await asyncio.gather(*(validate_shard(json.dumps(s, default=str)) for s in shards)):
        results += json.loads(chunk)
    good = [x for x in results if x.get("ok")]
    rep = Reporter("K8s twin: validation", f"{len(good)} of {len(results)} runs measured as control + twin")
    for st in k8s.STAGINGS:
        sub = [x for x in good if x["staging"] == st]
        if not sub:
            continue
        rss = np.array([x["meas_rss_mib"] / max(x["pred_rss_mib"], 1e-6) for x in sub])
        cg = np.array([x["meas_cg_mib"] / max(x["pred_cg_mib"], 1e-6) for x in sub])
        rep.kv({st: f"n={len(sub)}; measured/predicted, own RSS: median {np.median(rss):.3f} "
                    f"(p10 {np.percentile(rss, 10):.3f}, p90 {np.percentile(rss, 90):.3f}); cgroup: median "
                    f"{np.median(cg):.3f} (p10 {np.percentile(cg, 10):.3f}, p90 {np.percentile(cg, 90):.3f}); "
                    f"same cell {np.mean([x['cell_meas'] == x['cell_pred'] for x in sub]):.1%}, within one "
                    f"{np.mean([abs(x['cell_meas'] - x['cell_pred']) <= 1 for x in sub]):.1%}"})
    infl = np.array([x["corpus_peak_mib"] / max(x["control_own_hwm_mib"], 1e-6) for x in good])
    if len(infl):
        rep.kv({"corpus peak / re-measured own peak": f"median {np.median(infl):.3f} (p10 {np.percentile(infl, 10):.3f}, "
                                                      f"p90 {np.percentile(infl, 90):.3f})"})
    await rep.flush()
    return json.dumps(results)


# ── trainer throughput on the L4 at the K8s context length ──────────────


def synthetic_rows(n: int, seed: int = 0) -> list[dict]:
    """Prompt-shaped rows (Flyte and twin) from the workload generators, with
    random labels: what a training step costs, not what it learns."""
    rng = random.Random(seed)
    rows: list[dict] = []
    gens = (("mem", mem_w), ("cpu", cpu_w), ("gpu", gpu_w))
    while len(rows) < n:
        src, mod = gens[len(rows) // 2 % 3]
        t = mod.make_task(rng.choice(sorted(mod.FAMILIES)), rng.randint(0, 2**31), n_runs=2)
        ins = t.run_inputs(1)
        sizes = {x["name"]: rng.randint(10**6, 2 * 10**9) for x in ins}
        hist = [{"inputs_desc": mem_w.describe_inputs(t.run_inputs(0), sizes), "input_bytes": sizes, "wall_s": 30.0,
                 **({"peak_gpu_mib": 9000.0, "mean_util": 80.0} if src == "gpu" else
                    {"peak_mem_mib": 900.0, "peak_cores": 1.2, "mean_cores": 1.0})}]
        r = {"run_id": f"{src}:{t.task_id}-r1", "task_id": f"{src}:{t.task_id}", "source": src, "run_idx": 1,
             "split": "train", "task_code": mem_w.render_task(t), "inputs_json": json.dumps(ins),
             "input_bytes_json": json.dumps(sizes), "inputs_desc": mem_w.describe_inputs(ins, sizes),
             "history_json": json.dumps(hist), "wall_s": 30.0, "need_mib": 1500.0, "need_cores": 1.0,
             "mean_cores": 1.0, "t_peak_frac": 0.5, "cpu_series_json": "[]", "platform": "flyte", "manifest": "",
             "label_mem": rng.randrange(grid.N_MEM) if src != "gpu" else -1,
             "label_cpu": rng.randrange(grid.N_CPU) if src != "gpu" else -1,
             "label_gpu": rng.randrange(1, grid.N_GPU) if src == "gpu" else 0}
        tw = k8s.twin(r)
        tw.update(label_mem=r["label_mem"], label_cpu=r["label_cpu"], label_gpu=r["label_gpu"])
        rows += [r, tw]
    return rows[:n]


@allbench_train_env.task(report=True)
async def train_timing(n_rows: int = 480, max_len: int = 3072) -> str:
    """Seconds per training sample and per scored row for the decision model on this GPU."""
    import torch

    rows = synthetic_rows(n_rows)
    cfg = TrainConfig(model_name="Qwen/Qwen3-1.7B", epochs=1, head="pointer", bench="all", max_len=max_len)
    t0 = time.time()
    tok, model, head, hist = policy.train_supervised(rows, cfg)
    train_s = time.time() - t0
    t1 = time.time()
    policy.logits(tok, model, head, rows, cfg)
    score_s = time.time() - t1
    out = {"gpu": torch.cuda.get_device_name(0), "n": len(rows), "max_len": max_len, "train_s": train_s,
           "s_per_train_sample": train_s / len(rows), "s_per_scored_row": score_s / len(rows),
           "max_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "loss": hist}
    rep = Reporter("Decision-model throughput", out["gpu"])
    rep.kv({k: str(v)[:120] for k, v in out.items()})
    await rep.flush()
    return json.dumps(out)


# ── question 3 (after review): the twin in real pods ────────────────────

from . import pod_probe  # noqa: E402


def job_template(image: str, row: dict, staging: str, uris: dict[str, str], limit_mib: int,
                 init_limit_mib: int = 256) -> flyte.PodTemplate:
    """The twin's pod as its manifest declares it (``k8s.manifest``), plus Flyte's
    waiting primary: the init container (256Mi, the manifest's) for the init
    stagings, the emptyDir with or without ``medium: Memory``, and the job in a
    container of its own whose limit is ``limit_mib``."""
    from kubernetes.client import (V1Container, V1EmptyDirVolumeSource, V1EnvVar, V1PodSpec, V1ResourceRequirements,
                                   V1Volume, V1VolumeMount)

    inputs = json.loads(row["inputs_json"])
    mount = [V1VolumeMount(name="data", mount_path="/data")]
    src = k8s.module_code(row["task_code"], inputs, staging).replace("    s3.upload_file(", "    pass  # s3.upload_file(")
    env = [V1EnvVar(name=v, value="1") for v in k8s.THREAD_ENV] + [
        V1EnvVar(name="OUTPUT_URI", value="s3://unused/out/"), V1EnvVar(name="JOB_SRC", value=src),
        V1EnvVar(name="WRAP_SRC", value=pod_probe.WRAP)]
    files = [f"/data/{k8s._file_name(s)}" for s in inputs]
    init = []
    if k8s.is_boto3(staging):
        env += [V1EnvVar(name=k8s._env_name(s), value=uris[s["name"]]) for s in inputs]
        args: list[str] = []
    else:
        args = files
        init = [V1Container(name="stage-inputs", image=image, volume_mounts=mount,
                            command=["python", "-c", pod_probe.INIT, *[x for s, f in zip(inputs, files)
                                                                       for x in (uris[s["name"]], f)]],
                            resources=V1ResourceRequirements(requests={"cpu": "500m", "memory": "256Mi"},
                                                             limits={"cpu": "500m", "memory": f"{init_limit_mib}Mi"}))]
    job = V1Container(name="main", image=image, volume_mounts=mount, env=env,
                      command=["python", "-c", pod_probe.BOOT, *args],
                      resources=V1ResourceRequirements(requests={"cpu": "2", "memory": f"{limit_mib}Mi"},
                                                       limits={"cpu": "2", "memory": f"{limit_mib}Mi"}))
    vol = V1Volume(name="data", empty_dir=V1EmptyDirVolumeSource(medium="Memory") if k8s.is_tmpfs(staging)
                   else V1EmptyDirVolumeSource())
    spec = V1PodSpec(containers=[V1Container(name="primary", volume_mounts=mount), job], init_containers=init or None,
                     volumes=[vol])
    return flyte.PodTemplate(pod_spec=spec, primary_container_name="primary")


@k8s_probe_env.task(retries=1)
async def upload_inputs(rows_json: str) -> str:
    """Write each row's inputs (``bench.measure.prepare_inputs``) and upload them: run_id -> {uris, sizes}."""
    out = {}
    for row in json.loads(rows_json):
        with tempfile.TemporaryDirectory(prefix="k8spod-") as d:
            inputs = json.loads(row["inputs_json"])
            paths, sizes = prepare_inputs(inputs, Path(d), int(row["seed"]))
            uris = {s["name"]: (await flyte.io.File.from_local(p)).path for s, p in zip(inputs, paths)}
        out[row["run_id"]] = {"uris": uris, "sizes": sizes}
    return json.dumps(out)


def _twin_label_mib(row: dict, sizes: dict[str, int], staging: str) -> float:
    return k8s.twin_need_mib(float(row.get("required_mib") or _flyte_need(float(row["peak_rss_mib"]))), sizes, staging)


#: The waiting primary's limit: the pod's memory limit is the sum of its containers', so
#: whatever of this the primary does not use is slack a pod-level charge can hide in.
PRIMARY_MIB = 256


async def _one_pod_run(image: str, row: dict, staging: str, up: dict, limit_mib: int, kind: str,
                       init_limit_mib: int = 256) -> dict:
    tmpl = job_template(image, row, staging, up["uris"], limit_mib, init_limit_mib)
    try:
        t = pod_probe.job_pod.override(pod_template=tmpl,
                                       resources=flyte.Resources(cpu=0.5, memory=f"{PRIMARY_MIB}Mi"))
        res = json.loads(await t(row["run_id"]))
    except Exception as e:  # noqa: BLE001 -- an init container OOM fails the pod: that is a result
        res = {"run_id": row["run_id"], "status": "pod-failed", "error": f"{type(e).__name__}: {e}"[-600:]}
    return {**res, "kind": kind, "staging": staging, "limit_mib": limit_mib, "init_limit_mib": init_limit_mib}


async def _flyte_run(row: dict, up: dict) -> dict:
    need = _flyte_need(float(row["peak_rss_mib"]))
    mem = int(min(24576, max(2048, 2 * need + 1024)))
    try:
        t = pod_probe.flyte_task_measure.override(resources=flyte.Resources(cpu=2, memory=f"{mem}Mi"))
        return json.loads(await t(json.dumps({k: row[k] for k in ("run_id", "task_code")}), json.dumps(up["uris"])))
    except Exception as e:  # noqa: BLE001
        return {"run_id": row["run_id"], "ok": False, "error": f"{type(e).__name__}: {e}"[-600:]}


@pod_probe.k8s_pod_driver_env.task(report=True)
async def validate_pods(mem: flyte.io.File, cpu: flyte.io.File, image: str, n_per_source: int = 48,
                        n_init_tmpfs_large: int = 8, upload_shard: int = 6) -> str:
    """Question 3: each sampled run as a real Flyte task pod, as its twin's pod
    under a generous limit, at the twin's label cell, and one cell below.
    Stagings rotate; ``n_init_tmpfs_large`` extra runs with inputs over 256 MiB
    are staged init-tmpfs, which is where the manifest's 256Mi init container
    and the kernel's charge-to-the-writer rule meet."""
    picked = []
    for name, f in (("mem", mem), ("cpu", cpu)):
        rows = await read_rows(f)
        picked += [{**{k: r.get(k) for k in _KEEP}, "run_id": f"{name}:{r['run_id']}"}
                   for r in sample_rows(rows, n_per_source, max_need_mib=12288, max_input_mib=3072, seed=3)]
        big = [r for r in rows if r.get("ok") and not r.get("censored") and float(r.get("peak_rss_mib") or 0) > 0
               and 300 < sum(json.loads(r.get("input_bytes_json") or "{}").values()) / 2**20 < 3072
               and float(r["peak_rss_mib"]) < 8192]
        random.Random(9).shuffle(big)
        picked += [{**{k: r.get(k) for k in _KEEP}, "run_id": f"{name}:{r['run_id']}", "_force": "init-tmpfs"}
                   for r in big[: n_init_tmpfs_large // 2]]
    items = []
    for i, r in enumerate(picked):
        items.append((r, r.pop("_force", None) or k8s.STAGINGS[i % len(k8s.STAGINGS)]))
    shards = [[r for r, _ in items[i: i + upload_shard]] for i in range(0, len(items), upload_shard)]
    uploads: dict = {}
    for chunk in await asyncio.gather(*(upload_inputs(json.dumps(s, default=str)) for s in shards)):
        uploads.update(json.loads(chunk))
    jobs = []
    for r, st in items:
        up = uploads[r["run_id"]]
        sizes = {k: int(v) for k, v in up["sizes"].items()}
        label = grid.MEMORY_GRID_MIB[grid.mem_cell(_twin_label_mib(r, sizes, st))]
        below = grid.MEMORY_GRID_MIB[max(grid.mem_cell(_twin_label_mib(r, sizes, st)) - 1, 0)]
        staged = k8s.staged_bytes(sizes) / 2**20 if k8s.is_tmpfs(st) else 0.0
        generous = int(min(20480, 2 * float(r["peak_rss_mib"]) + staged + 1024))
        # init-tmpfs: k8s-1's manifest gave the init container 256Mi, which the staged bytes
        # (charged to the container that writes them) OOM-kill (rt-k8spod-pilot-2); give it
        # room for them, and also try the job at its label without the staged bytes: pages
        # the init container leaves are reparented to the pod, whose limit is the sum of the
        # containers' -- do they still count?
        init_lim = int(256 + staged) + 1 if k8s.is_tmpfs(st) and not k8s.is_boto3(st) else 256
        jobs += [_flyte_run(r, up), _one_pod_run(image, r, st, up, generous, "measure", init_lim),
                 _one_pod_run(image, r, st, up, label, "label", init_lim),
                 _one_pod_run(image, r, st, up, below, "below", init_lim)]
        if init_lim > 256:
            bare = grid.MEMORY_GRID_MIB[grid.mem_cell(_twin_label_mib(r, sizes, st) - staged * LABEL_HEADROOM)]
            jobs += [_one_pod_run(image, r, st, up, bare, "label-without-staged", init_lim),
                     _one_pod_run(image, r, st, up, label, "label-init256", 256)]
    results = await asyncio.gather(*jobs)
    meta = {r["run_id"]: {"staging": st, "family": r["family"], "peak_rss_mib": r["peak_rss_mib"],
                          "required_mib": r.get("required_mib"), "sizes": uploads[r["run_id"]]["sizes"],
                          "twin_label_mib": _twin_label_mib(r, {k: int(v) for k, v in uploads[r["run_id"]]["sizes"].items()}, st)}
            for r, st in items}
    rep = Reporter("K8s twin: real pods", f"{len(items)} runs x (Flyte task, twin pod: measure, @label, @label-1)")
    by = {}
    for x in results:
        by.setdefault(x.get("kind", "flyte"), []).append(x)
    rep.kv({k: f"{len(v)} pods; " + ", ".join(f"{s}={sum(1 for y in v if y.get('status', 'ok' if y.get('ok') else 'failed') == s)}"
                                               for s in sorted({y.get('status', 'ok' if y.get('ok') else 'failed') for y in v}))
            for k, v in by.items()})
    await rep.flush()
    return json.dumps({"meta": meta, "results": results})
