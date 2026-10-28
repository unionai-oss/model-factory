"""Which GPU node shapes does demo.hosted actually schedule, and what do they look like?

    uv run flyte --config .flyte/config.yaml run probe_gpu_shapes.py probe_all

One pod per shape (the RightSizeGPU workload image), each reporting what
NVML and the cgroup see: device names, usable memory, host CPUs and memory.
A shape that never schedules shows up as a timed-out child, not a failure
of the whole probe.
"""

from __future__ import annotations

import asyncio
import json
import os

import flyte

from resource_tuner.bench_gpu.envs import gpu_workload_image

#: (env name, gpu request, cpu, memory) -- each sized under its node's allocatable.
SHAPES = (
    ("t4x1", "T4:1", 3, "13Gi"),
    ("t4x8", "T4:8", 88, "320Gi"),
    ("l4x1", "L4:1", 7, "26Gi"),
    ("l4x4", "L4:4", 44, "160Gi"),
    ("a10gx1", "A10G:1", 7, "26Gi"),
    ("l40sx1", "L40s:1", 7, "54Gi"),
    ("l40sx4", "L40s:4", 44, "320Gi"),
)

_envs = {
    name: flyte.TaskEnvironment(name=f"rt-gpuprobe-{name}", image=gpu_workload_image,
                                resources=flyte.Resources(cpu=cpu, memory=mem, gpu=gpu, disk="40Gi"))
    for name, gpu, cpu, mem in SHAPES
}


def _report() -> str:
    import pynvml

    pynvml.nvmlInit()
    devs = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(h)
        devs.append({"name": name if isinstance(name, str) else name.decode(),
                     "total_mib": pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20,
                     "pci": pynvml.nvmlDeviceGetPciInfo(h).busId if hasattr(pynvml.nvmlDeviceGetPciInfo(h), "busId") else ""})
    node = os.environ.get("NODE_NAME", "")
    return json.dumps({"devices": devs, "cpus": os.cpu_count(), "node": node,
                       "visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                       "instance": _instance_type()}, default=str)


def _instance_type() -> str:
    try:
        import urllib.request

        tok = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        t = urllib.request.urlopen(tok, timeout=2).read().decode()
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": t})
        return urllib.request.urlopen(req, timeout=2).read().decode()
    except Exception as e:  # noqa: BLE001
        return f"? ({type(e).__name__})"


@_envs["t4x1"].task(timeout=flyte.Timeout(max_runtime=600, max_queued_time=3600))
def probe_t4x1() -> str:
    return _report()


@_envs["t4x8"].task(timeout=flyte.Timeout(max_runtime=600, max_queued_time=3600))
def probe_t4x8() -> str:
    return _report()


@_envs["l4x1"].task(timeout=flyte.Timeout(max_runtime=600, max_queued_time=3600))
def probe_l4x1() -> str:
    return _report()


@_envs["l4x4"].task(timeout=flyte.Timeout(max_runtime=600, max_queued_time=3600))
def probe_l4x4() -> str:
    return _report()


@_envs["a10gx1"].task(timeout=flyte.Timeout(max_runtime=600, max_queued_time=3600))
def probe_a10gx1() -> str:
    return _report()


@_envs["l40sx1"].task(timeout=flyte.Timeout(max_runtime=600, max_queued_time=3600))
def probe_l40sx1() -> str:
    return _report()


@_envs["l40sx4"].task(timeout=flyte.Timeout(max_runtime=600, max_queued_time=3600))
def probe_l40sx4() -> str:
    return _report()


_probes = {"t4x1": probe_t4x1, "t4x8": probe_t4x8, "l4x1": probe_l4x1, "l4x4": probe_l4x4,
           "a10gx1": probe_a10gx1, "l40sx1": probe_l40sx1, "l40sx4": probe_l40sx4}

driver = flyte.TaskEnvironment(name="rt-gpuprobe", image=gpu_workload_image,
                               resources=flyte.Resources(cpu=1, memory="1Gi"), depends_on=list(_envs.values()))


@driver.task
async def probe_all() -> str:
    names = list(_probes)
    res = await asyncio.gather(*(_probes[n].aio() for n in names), return_exceptions=True)
    out = {n: (json.loads(r) if not isinstance(r, BaseException) else f"{type(r).__name__}: {str(r)[:300]}")
           for n, r in zip(names, res)}
    print(json.dumps(out, indent=2), flush=True)
    return json.dumps(out)
