"""The Kubernetes twin of every measured run: the same workload as a plain
Python module in a ``batch/v1`` Job, labelled from its Flyte measurement.

A Flyte task is a pod: the workload code that ran under Flyte runs the same
way as ``python job.py`` in a container. What differs is everything around
the code -- how the inputs get into the pod, where they are staged, what
else lives in the container -- and that is exactly what moves the request.
So the twin is not measured again (the user code is identical); its labels
are the Flyte measurement plus an explicit accounting of the staging,
calibrated by in-cluster probes (``k8s_probe``, round 22):

    staging       who fetches the inputs          where they land
    init-disk     an aws-cli init container       emptyDir on node disk
    init-tmpfs    an aws-cli init container       emptyDir, medium: Memory
    boto3-disk    the job itself, boto3           emptyDir on node disk
    boto3-tmpfs   the job itself, boto3           emptyDir, medium: Memory

The two ``-disk`` / ``-tmpfs`` pairs render the same ``job.py``; one line of
the manifest (``medium: Memory``) is the difference, and it charges every
staged byte to the pod's memory cgroup (a tmpfs page counts as memory; the
kernel cannot reclaim it the way it reclaims page cache). The arms have to
read the manifest to see it. Accounting per twin (``twin_need_mib``):

- memory: the workload's own peak (its Flyte need without the Flyte runtime
  and the label headroom), plus boto3's resident footprint for the boto3
  stagings (48 MiB, measured; the transfer itself never peaks above it),
  plus the staged inputs and the result file for the tmpfs ones, then the
  bench's 5% headroom. No Flyte runtime: ``python job.py`` does not
  load it. Page cache from reading a disk-backed file is reclaimable and is
  not charged (``ru_maxrss`` never counted it either).
- CPU: an init container's download runs before the job, under its own
  request; a boto3 download runs inside the job, at ``DL_CORES`` for
  bytes / ``DL_BYTES_PER_S`` seconds, so it is prepended to the job's
  per-second core series (or folded into its mean) and can become the
  busiest 15 s window of a light job with a large input.
- GPU: unchanged; staging does not touch the device.

Twins keep their run's split (by task), so a test task's twin is a test
task. Their history is the twin's own: earlier runs' container peaks, as a
metrics stack (cAdvisor / Prometheus) would report them, not Flyte's.

Pure logic: no flyte, no boto3.
"""

from __future__ import annotations

import json
import math
import re
import zlib

from ..bench.grid import LABEL_HEADROOM
from ..bench_cpu.grid import peak_window
from . import grid

PLATFORMS = ("flyte", "k8s")
#: The Flyte runtime's footprint in a task's container, MiB: what a Flyte
#: task's memory need adds to its workload's own peak. Measured in real task
#: pods (rt-k8spod-1, 100 runs): 55 MiB of anon at task start (p10-p90
#: 54.9-55.0), and a container peak a median 70 MiB (p10 53, p90 242) above
#: the same run's Job workload -- running the workload inside the runtime's
#: process costs a little more than the runtime at rest. Through k8s-1 the
#: labels added ``bench.grid.FLYTE_RUNTIME_OVERHEAD_MIB`` = 180, a figure no
#: probe in the repo produced (the single-axis benches keep it): 70% of real
#: Flyte containers fell a cell below their label at 180, 33% at 70.
FLYTE_RUNTIME_MIB = 70.0
STAGINGS = ("init-disk", "init-tmpfs", "boto3-disk", "boto3-tmpfs")
#: How often each staging is drawn, per task (a task's runs share a manifest).
STAGING_WEIGHTS = (0.3, 0.2, 0.3, 0.2)
SUFFIX = "@k8s"

# ── calibration: rt-k8sprobe-1 (2026-11-07; boto3 1.43, 8-CPU pod, 64 MiB-4 GiB objects) ──
#: Resident set boto3 adds to a job that imports it, builds a client and
#: downloads its inputs. Alone: VmHWM 74-80 MiB against a bare interpreter's
#: 11 MiB (rt-k8sprobe-1), and the peak *is* the end state (no transient
#: above it). Inside a real workload it shares modules the workload already
#: loaded: +48 MiB median (p10 38, p90 61; rt-k8sval-2, n=24), MiB.
BOTO3_MIB = 48.0
#: The same as the container cgroup charges it (anon only: shared-library
#: pages are file-backed): 57-63 MiB.
BOTO3_CG_MIB = 61.0
#: Transfer: t = bytes / 440 MB/s + 0.4 s per object (0.54 s for 64 MiB,
#: 10.0-10.3 s for 4 GiB), at 1.6-2.2 cores (s3transfer's 10 threads), after
#: ~0.7 s of import and client setup.
DL_BYTES_PER_S = 440e6
DL_LATENCY_S = 0.4
DL_SETUP_S = 0.7
DL_CORES = 1.8
#: The job's result (a JSON summary) is written to the staging volume before
#: it is uploaded; on tmpfs its page is charged too. A file on a memory-backed
#: emptyDir is charged byte for byte (shmem delta = file size, 64 MiB-4 GiB).
RESULT_BYTES = 4096
_PAGE = 4096

#: The init container's own request (aws-cli); its limit adds the staged bytes on tmpfs (``init_limit_mib``).
INIT_BASE_MIB = 256
BUCKET = "batch-data"
INIT_IMAGE = "amazon/aws-cli:2.17.0"
REGISTRY = "registry.internal/batch"
#: What the measurement ran with (``bench.measure._base_env``), so the twin's
#: manifest carries it: BLAS/OpenMP would otherwise size to the node.
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def staging_for(task_id: str) -> str:
    u = (zlib.crc32(f"k8s-staging:{task_id}".encode()) % 10_000) / 10_000
    acc = 0.0
    for s, w in zip(STAGINGS, STAGING_WEIGHTS):
        acc += w
        if u < acc:
            return s
    return STAGINGS[-1]


def is_tmpfs(staging: str) -> bool:
    return staging.endswith("-tmpfs")


def is_boto3(staging: str) -> bool:
    return staging.startswith("boto3-")


def staged_bytes(sizes: dict[str, int]) -> int:
    """Bytes a tmpfs holds for these inputs and the result, page-rounded."""
    return sum(_PAGE * math.ceil(b / _PAGE) for b in sizes.values()) + RESULT_BYTES


def download_s(sizes: dict[str, int]) -> float:
    return DL_SETUP_S + sum(sizes.values()) / DL_BYTES_PER_S + DL_LATENCY_S * len(sizes)


# ── labels ──────────────────────────────────────────────────────────────


def workload_peak_mib(flyte_need_mib: float) -> float:
    """The workload's own peak RSS, from its Flyte need (``flyte_need_mib``)."""
    return flyte_need_mib / LABEL_HEADROOM - FLYTE_RUNTIME_MIB


def flyte_need_mib(peak_mib: float) -> float:
    """A Flyte task's memory need for a workload peak: the runtime on top, then the headroom."""
    return (peak_mib + FLYTE_RUNTIME_MIB) * LABEL_HEADROOM


def container_peak_mib(peak_mib: float, sizes: dict[str, int], staging: str) -> float:
    """The job container's peak memory as the cgroup sees it, MiB."""
    tmpfs = staged_bytes(sizes) / 2**20 if is_tmpfs(staging) else 0.0
    return peak_mib + (BOTO3_MIB if is_boto3(staging) else 0.0) + tmpfs


def need_from_peak(peak_mib: float, platform: str) -> float:
    """The limit a container needs for a memory peak as each platform reports it:
    Flyte's peaks are the workload's (the runtime comes on top); a K8s
    container peak is everything in the container already."""
    if platform == "k8s":
        return peak_mib * LABEL_HEADROOM
    return flyte_need_mib(peak_mib)


def twin_need_mib(flyte_need_mib: float, sizes: dict[str, int], staging: str) -> float:
    return need_from_peak(container_peak_mib(workload_peak_mib(flyte_need_mib), sizes, staging), "k8s")


def _twin_cpu(row: dict, sizes: dict[str, int], staging: str) -> dict:
    """The job's CPU picture with an in-process download in front of it."""
    if not is_boto3(staging):
        return {}
    d = download_s(sizes)
    wall = float(row.get("wall_s") or 0.0)
    out = {"wall_s": wall + d,
           "t_peak_frac": (d + float(row["t_peak_frac"]) * wall) / max(wall + d, 1e-9)}
    series = json.loads(row.get("cpu_series_json") or "[]")
    if series:  # RightSizeCPU rows: per-second cores, label = busiest window
        twin = [DL_CORES] * max(1, round(d)) + series
        out.update(cpu_series_json=json.dumps([round(x, 4) for x in twin]),
                   need_cores=peak_window(twin), mean_cores=sum(twin) / len(twin))
    elif row["source"] == "mem":  # RightSizeMem: the label is the mean
        mean = (float(row["mean_cores"]) * wall + DL_CORES * d) / max(wall + d, 1e-9)
        out.update(need_cores=mean, mean_cores=mean)
    return out


# ── the rendering ───────────────────────────────────────────────────────

_SIG = re.compile(r"^async def (\w+)\((.*)\) -> dict:$")
_DOWNLOAD = re.compile(r"^\s+(\w+)_path = await \w+\.download\(\)$")


def _inputs_of(row: dict) -> list[dict]:
    return json.loads(row["inputs_json"])


def _file_name(spec: dict) -> str:
    return f"{spec['name']}.{spec['fmt']}"


def _env_name(spec: dict) -> str:
    return f"INPUT_{spec['name'].upper()}"


def job_name(task_code: str) -> str:
    m = re.search(r'TaskEnvironment\(name="([^"]+)"', task_code)
    return m.group(1) if m else "batch-job"


def module_code(task_code: str, inputs: list[dict], staging: str) -> str:
    """``bench.workloads.render_task``'s Flyte module, as the plain ``job.py``
    a Kubernetes Job runs: same body, paths instead of ``File``s, and a
    ``__main__`` that stages inputs and writes the result for upload."""
    lines = task_code.rstrip("\n").split("\n")
    head, body, fn = [], [], None
    in_body = False
    for ln in lines:
        if not in_body:
            if ln in ("import flyte", "from flyte.io import File") or ln.startswith("env = flyte.TaskEnvironment("):
                continue
            if ln == "@env.task":
                continue
            m = _SIG.match(ln)
            if m:
                fn = m.group(1)
                args = ", ".join(f"{s['name']}_path: str" for s in inputs)
                body.append(f"def {fn}({args}) -> dict:")
                in_body = True
                continue
            head.append(ln)
        elif not _DOWNLOAD.match(ln):
            body.append(ln)
    if fn is None:
        raise ValueError("module_code: not a render_task module")
    while head and not head[-1].strip():
        head.pop()
    std = ["import json", "import os"] + (["import boto3"] if is_boto3(staging) else ["import sys"])
    have = set(head)
    head += [s for s in std if s not in have]
    if is_boto3(staging):
        args = ", ".join(f'fetch(os.environ["{_env_name(s)}"])' for s in inputs)
        main = (
            "def fetch(uri: str) -> str:\n"
            '    bucket, key = uri.removeprefix("s3://").split("/", 1)\n'
            '    path = os.path.join("/data", os.path.basename(key))\n'
            "    s3.download_file(bucket, key, path)\n"
            "    return path\n\n\n"
            'if __name__ == "__main__":\n'
            '    s3 = boto3.client("s3")\n'
            f"    result = {fn}({args})\n"
            '    os.makedirs("/data/out", exist_ok=True)\n'
            '    with open("/data/out/result.json", "w") as f:\n'
            "        json.dump(result, f)\n"
            '    bucket, key = os.environ["OUTPUT_URI"].removeprefix("s3://").split("/", 1)\n'
            '    s3.upload_file("/data/out/result.json", bucket, key + "result.json")\n'
        )
    else:
        args = ", ".join(f"sys.argv[{i + 1}]" for i in range(len(inputs)))
        main = (
            'if __name__ == "__main__":\n'
            f"    result = {fn}({args})\n"
            '    os.makedirs("/data/out", exist_ok=True)\n'
            '    with open("/data/out/result.json", "w") as f:\n'
            "        json.dump(result, f)\n"
        )
    return "\n".join(head) + "\n\n\n" + "\n".join(body) + "\n\n\n" + main


def _uri(name: str, task_key: str, run_idx: int, leaf: str) -> str:
    return f"s3://{BUCKET}/{name}/{task_key}/run-{run_idx}/{leaf}"


def init_limit_mib(sizes: dict[str, int], staging: str) -> int:
    """The init container's memory limit. A tmpfs page is charged to the container
    that writes it, so an init container staging onto ``medium: Memory`` must hold
    every staged byte: at k8s-1's flat 256Mi it is OOM-killed staging more than
    ~200 MiB (rt-k8spod-pilot-2). The pages outlive it, charged to the pod."""
    if not is_tmpfs(staging) or is_boto3(staging):
        return INIT_BASE_MIB
    return INIT_BASE_MIB + math.ceil(staged_bytes(sizes) / 2**20)


def manifest(task_code: str, inputs: list[dict], staging: str, task_key: str, run_idx: int,
             sizes: dict[str, int] | None = None) -> str:
    """The Job a platform team would write for this module; ``resources`` is
    the request being decided. Flow style keeps it to a few hundred tokens."""
    name = job_name(task_code)
    image = f"{REGISTRY}/{name}:{zlib.crc32(task_code.encode()) & 0xFFFFFFFF:08x}"
    out_uri = _uri(name, task_key, run_idx, "out/")
    threads = [f"{{name: {v}, value: \"1\"}}" for v in THREAD_ENV]
    mount = "volumeMounts: [{name: data, mountPath: /data}]"
    y = ["apiVersion: batch/v1", "kind: Job", "metadata:", f"  generateName: {name}-",
         f"  labels: {{app.kubernetes.io/name: {name}}}", "spec:", "  backoffLimit: 0", "  template:",
         "    spec:", "      restartPolicy: Never"]
    files = [f"/data/{_file_name(s)}" for s in inputs]
    if not is_boto3(staging):
        cps = " && ".join(f"aws s3 cp ${_env_name(s)} {p}" for s, p in zip(inputs, files))
        envs = ", ".join(f"{{name: {_env_name(s)}, value: \"{_uri(name, task_key, run_idx, _file_name(s))}\"}}"
                         for s in inputs)
        y += ["      initContainers:", "        - name: stage-inputs", f"          image: {INIT_IMAGE}",
              f"          command: [sh, -c, \"{cps}\"]", f"          env: [{envs}]",
              f"          resources: {{requests: {{cpu: 500m, memory: {INIT_BASE_MIB}Mi}}, "
              f"limits: {{memory: {init_limit_mib(sizes or {}, staging)}Mi}}}}",
              f"          {mount}"]
        cmd = (f"[sh, -c, \"python /app/job.py {' '.join(files)} && "
               f"aws s3 cp /data/out/result.json $OUTPUT_URI\"]")
        env = threads + [f"{{name: OUTPUT_URI, value: \"{out_uri}\"}}"]
    else:
        cmd = "[python, /app/job.py]"
        env = ([f"{{name: {_env_name(s)}, value: \"{_uri(name, task_key, run_idx, _file_name(s))}\"}}"
                for s in inputs] + threads + [f"{{name: OUTPUT_URI, value: \"{out_uri}\"}}"])
    y += ["      containers:", "        - name: main", f"          image: {image}", f"          command: {cmd}",
          "          env:"] + [f"            - {e}" for e in env]
    y += ["          resources: {}  # to be decided", f"          {mount}", "      volumes:",
          "        - name: data", "          emptyDir: {medium: Memory}" if is_tmpfs(staging) else "          emptyDir: {}"]
    return "\n".join(y) + "\n"


# ── twins ───────────────────────────────────────────────────────────────


def _task_key(task_id: str) -> str:
    return f"{zlib.crc32(task_id.encode()) & 0xFFFFFFFF:08x}"


def _twin_history(entries: list[dict], staging: str, earlier: dict[str, dict]) -> list[dict]:
    """A Flyte row's history, as the twin's metrics would have recorded it."""
    out = []
    for h in entries:
        e = dict(h)
        sizes = h.get("input_bytes") or {}
        if "peak_mem_mib" in h:
            e["peak_mem_mib"] = round(container_peak_mib(float(h["peak_mem_mib"]), sizes, staging), 1)
        if is_boto3(staging):
            d = download_s(sizes)
            e["wall_s"] = round(float(h.get("wall_s", 0.0)) + d, 1)
            twin = earlier.get(h.get("inputs_desc", ""))
            if twin is not None and "peak_cores" in h:  # the earlier run's twin, from its series
                e["peak_cores"], e["mean_cores"] = round(twin["need_cores"], 3), round(twin["mean_cores"], 3)
            elif "mean_cores" in h:
                w = float(h.get("wall_s", 0.0))
                e["mean_cores"] = round((float(h["mean_cores"]) * w + DL_CORES * d) / max(w + d, 1e-9), 3)
                if "peak_cores" in h:
                    e["peak_cores"] = round(max(float(h["peak_cores"]), DL_CORES * min(d, 15) / 15), 3)
        out.append(e)
    return out


def twin(row: dict, earlier: dict[str, dict] | None = None, staging: str | None = None) -> dict:
    """The K8s twin of one unified (Flyte) row. ``staging`` overrides the
    task's own draw (``staging_for``): the counterfactual renditions of one
    run under every staging (``counterfactual``)."""
    staging = staging or staging_for(row["task_id"])
    inputs = _inputs_of(row)
    sizes = json.loads(row.get("input_bytes_json") or "{}")
    out = dict(row)
    out.update(run_id=row["run_id"] + SUFFIX, task_id=row["task_id"] + SUFFIX, platform="k8s", staging=staging,
               twin_of=row["run_id"],
               task_code=module_code(row["task_code"], inputs, staging),
               manifest=manifest(row["task_code"], inputs, staging, _task_key(row["task_id"]), int(row["run_idx"]),
                                 sizes))
    out.update(_twin_cpu(row, sizes, staging))
    if row["source"] != "gpu":
        out["need_mib"] = twin_need_mib(float(row["need_mib"]), sizes, staging)
        out["label_mem"] = grid.mem_cell(out["need_mib"])
        out["label_cpu"] = grid.cpu_cell(float(out["need_cores"]))
        if out["need_mib"] > grid.MEMORY_GRID_MIB[-1]:
            out["ok"] = False
    out["history_json"] = json.dumps(_twin_history(json.loads(row.get("history_json") or "[]"), staging, earlier or {}))
    for k in ("_series", "_oracle"):
        out.pop(k, None)
    return out


def task_twins(runs: list[dict], staging: str | None = None) -> list[dict]:
    """The twins of one task's Flyte runs, in run order. A twin's history reads
    the twins of the task's earlier runs (their CPU series include the download)."""
    earlier: dict[str, dict] = {}
    out = []
    for r in sorted(runs, key=lambda r: int(r["run_idx"])):
        t = twin(r, earlier, staging)
        out.append(t)
        earlier[r.get("inputs_desc", "")] = t
    return out


def with_twins(rows: list[dict]) -> list[dict]:
    """Every Flyte row and its K8s twin."""
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    out = list(rows)
    for runs in by_task.values():
        out += task_twins(runs)
    return out
