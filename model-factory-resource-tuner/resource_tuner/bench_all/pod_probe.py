"""The twin's labels, checked in real pods (round 22, after review).

rt-k8sval-2 ran the control, the "init container" and ``job.py`` as sibling
processes in one Flyte pod: one memory cgroup, ``/dev/shm`` for the volume.
That checks the arithmetic of the accounting, not Kubernetes. Here every
sampled run is measured in pods of its own:

- **flyte** (``flyte_task_measure``): the run as the Flyte task it is -- the
  rendered task module, its ``File`` inputs downloaded by Flyte from the
  object store -- in a fresh task pod. The container cgroup's anon + shmem is
  sampled from task start, so the Flyte runtime's own footprint (the 180 MiB
  the Flyte labels add) is measured, not assumed.
- **job** (``job_pod``, measure): the twin's pod spec as the manifest
  declares it -- an init container with the manifest's 256Mi limit for the
  init stagings, the emptyDir with or without ``medium: Memory``, the job in
  its own container -- under a generous limit. The job container samples its
  own cgroup, so pages are counted where the kernel charges them (a tmpfs
  page goes to the container that wrote it).
- **job @ label**: the same pod with the job container's limit at the twin's
  label cell, and one cell below: does the label run, and does the cell below
  fail?

Flyte runs a task as the primary container of a pod it builds; here the
primary only waits (in its own container and cgroup) for the job container's
result on the shared volume. This module imports nothing heavy at the top, so
``flyte_task_measure``'s process starts as a bare Flyte task does.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time

import flyte
import flyte.io

from ..bench.envs import bench_image
from ..config import cluster_env_vars
from .envs import k8s_probe_env

CG = "/sys/fs/cgroup"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

k8s_pod_image = bench_image.with_pip_packages("boto3>=1.35,<2", "kubernetes>=30,<37")
#: The task pods. A job pod's primary only waits; its job container's limit is set per call.
k8s_pod_env = flyte.TaskEnvironment(
    name="rt-k8spod",
    image=k8s_pod_image,
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    interruptible=True,
    env_vars=cluster_env_vars(),
)


def mem_stat() -> dict[str, int]:
    out = {}
    with open(f"{CG}/memory.stat") as f:
        for ln in f:
            k, v = ln.split()
            out[k] = int(v)
    return out


def _status_kib(key: str) -> float:
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith(key + ":"):
                return float(ln.split()[1])
    return 0.0


class Sampler:
    """Peak anon, shmem, anon + shmem and file of this container's cgroup, every 50 ms."""

    def __init__(self):
        self.peak = {"anon": 0, "shmem": 0, "anon_shmem": 0, "file": 0}
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            s = mem_stat()
            a, sh = s.get("anon", 0), s.get("shmem", 0)
            for k, v in (("anon", a), ("shmem", sh), ("anon_shmem", a + sh), ("file", s.get("file", 0))):
                self.peak[k] = max(self.peak[k], v)
            time.sleep(0.05)

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)


def _mib(b: float) -> float:
    return b / 2**20


#: The sweep's driver (``k8s_probe.validate_pods``): reads the corpora and builds the pod templates.
k8s_pod_driver_env = flyte.TaskEnvironment(
    name="rt-k8spod-driver",
    image=k8s_pod_image,
    resources=flyte.Resources(cpu=2, memory="12Gi"),
    env_vars=cluster_env_vars(),
    depends_on=[k8s_pod_env, k8s_probe_env],
)


# ── the Flyte task, measured as a Flyte task ────────────────────────────


@k8s_pod_env.task(retries=3)
async def flyte_task_measure(row_json: str, uris_json: str, memory_mib: int = 16384) -> str:
    """Run the rendered Flyte task module's function in this task pod, on its
    inputs in the object store, and read the container's cgroup throughout.
    ``base`` is the runtime alone: the task process before the workload's
    module is imported."""
    row, uris = json.loads(row_json), json.loads(uris_json)
    base = mem_stat()
    base_rss = _status_kib("VmRSS") / 1024
    for v in THREADS:  # the measurement's thread floor (bench.measure._base_env)
        os.environ[v] = "1"
    out = {"run_id": row["run_id"], "base_anon_mib": _mib(base.get("anon", 0)), "base_rss_mib": base_rss}
    t0 = time.monotonic()
    try:
        with Sampler() as smp:
            # a real module, as Flyte loads one: a process pool's workers unpickle its functions by name
            import importlib
            import sys
            import tempfile

            d = tempfile.mkdtemp(prefix="rsk-task-")
            with open(os.path.join(d, "rsk_task_module.py"), "w") as f:
                f.write(row["task_code"])
            sys.path.insert(0, d)
            ns = vars(importlib.import_module("rsk_task_module"))
            fn = next(v for k, v in ns.items() if k != "env" and hasattr(v, "func") and hasattr(v, "forward"))
            args = {name: flyte.io.File.from_existing_remote(u) for name, u in uris.items()}
            await fn.func(**args)
        out.update(ok=True, peak_anon_shmem_mib=_mib(smp.peak["anon_shmem"]), peak_anon_mib=_mib(smp.peak["anon"]),
                   peak_file_mib=_mib(smp.peak["file"]), own_hwm_mib=_status_kib("VmHWM") / 1024)
    except Exception as e:  # noqa: BLE001 -- recorded, not raised: one bad run must not fail the sweep
        out.update(ok=False, error=f"{type(e).__name__}: {e}"[:400])
    out["wall_s"] = time.monotonic() - t0
    return json.dumps(out)


# ── the twin's pod: init container, volume, job container ───────────────

#: The job container's entrypoint: write job.py and the wrapper from the env,
#: then exec the wrapper, which runs the job as its child and samples this
#: container's own cgroup (the wrapper's ~10 MiB is read and reported).
BOOT = ("import os, sys\n"
        "open('/tmp/job.py', 'w').write(os.environ['JOB_SRC'])\n"
        "open('/tmp/wrap.py', 'w').write(os.environ['WRAP_SRC'])\n"
        "os.execvp(sys.executable, [sys.executable, '/tmp/wrap.py', '/tmp/job.py', *sys.argv[1:]])\n")

WRAP = r'''
import json, os, subprocess, sys, threading, time
CG = "/sys/fs/cgroup"
def stat():
    d = {}
    for ln in open(f"{CG}/memory.stat"):
        k, v = ln.split(); d[k] = int(v)
    return d
def kv(path):
    d = {}
    for ln in open(path):
        k, v = ln.split(); d[k] = int(v)
    return d
os.makedirs("/data/out", exist_ok=True)
res_path, beat = "/data/out/_probe.json", "/data/out/_beat"  # job.py writes /data/out/result.json itself
base = stat()
peak = {"anon": 0, "shmem": 0, "anon_shmem": 0, "file": 0}
done = threading.Event()
def sample():
    while not done.is_set():
        s = stat(); a, sh = s.get("anon", 0), s.get("shmem", 0)
        for k, v in (("anon", a), ("shmem", sh), ("anon_shmem", a + sh), ("file", s.get("file", 0))):
            peak[k] = max(peak[k], v)
        open(beat, "w").write(str(time.time()))
        time.sleep(0.05)
th = threading.Thread(target=sample, daemon=True); th.start()
own = [l for l in open("/proc/self/status") if l.startswith("VmRSS:")]
t0 = time.time()
p = subprocess.Popen([sys.executable, *sys.argv[1:]], stderr=subprocess.PIPE, text=True)
_, err = p.communicate()
done.set(); th.join(2)
res = {"rc": p.returncode, "wall_s": time.time() - t0, "stderr": (err or "")[-600:],
       "base": {k: base.get(k, 0) for k in ("anon", "shmem", "file")}, "peak": peak,
       "wrapper_rss_kib": int(own[0].split()[1]) if own else 0,
       "memory_peak": int(open(f"{CG}/memory.peak").read()) if os.path.exists(f"{CG}/memory.peak") else -1,
       "memory_max": open(f"{CG}/memory.max").read().strip(), "events": kv(f"{CG}/memory.events"),
       "cpu": kv(f"{CG}/cpu.stat")}
json.dump(res, open(res_path + ".tmp", "w")); os.rename(res_path + ".tmp", res_path)
'''

#: The init container: the manifest's ``aws s3 cp``, as boto3 (the probe image
#: has no aws-cli), then its own cgroup's view of what it left behind.
INIT = r'''
import json, os, sys, boto3
s3 = boto3.client("s3")
for uri, path in zip(sys.argv[1::2], sys.argv[2::2]):
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    s3.download_file(bucket, key, path)
st = {}
for ln in open("/sys/fs/cgroup/memory.stat"):
    k, v = ln.split(); st[k] = int(v)
pk = "/sys/fs/cgroup/memory.peak"
json.dump({"shmem": st.get("shmem", 0), "anon": st.get("anon", 0), "file": st.get("file", 0),
           "memory_peak": int(open(pk).read()) if os.path.exists(pk) else -1,
           "sizes": {p: os.path.getsize(p) for p in sys.argv[2::2]}}, open("/data/init.json", "w"))
'''


@k8s_pod_env.task(retries=3, timeout=flyte.Timeout(max_runtime=3600))
async def job_pod(run_id: str, deadline_s: int = 1800) -> str:
    """The primary container of a twin's pod (``job_template``): wait for the job
    container's result on the shared volume. A job container killed by the
    kernel writes nothing; its heartbeat going stale says so."""
    t0, last_beat = time.time(), time.time()
    res_path, beat = "/data/out/_probe.json", "/data/out/_beat"
    out: dict = {"run_id": run_id}
    while True:
        if os.path.exists(res_path):
            with open(res_path) as f:
                out["job"] = json.load(f)
            out["status"] = "done"
            break
        if os.path.exists(beat):
            last_beat = max(last_beat, os.path.getmtime(beat))
            if time.time() - last_beat > 45:
                out["status"] = "killed"  # the job container (wrapper and job) was OOM-killed or died
                break
        elif time.time() - t0 > 600:
            out["status"] = "never-started"
            break
        if time.time() - t0 > deadline_s:
            out["status"] = "timeout"
            break
        await asyncio.sleep(1.0)
    if os.path.exists("/data/init.json"):
        with open("/data/init.json") as f:
            out["init"] = json.load(f)
    out["waited_s"] = time.time() - t0
    return json.dumps(out)
