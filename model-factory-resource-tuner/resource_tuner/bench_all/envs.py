"""Task environments for the combined bench.

Nothing here measures a workload: the corpus is joined from the three
benches' measured corpora. The orchestration / classical-arm env is the
benches' CPU shape (t3a.xlarge has headroom); the decision model trains and
RL-tunes on the benches' T4 env (round 22, the K8s release: on an L4).
"""

from __future__ import annotations

import os

import flyte

from ..bench.envs import bench_gpu_env, bench_gpu_image, bench_image
from ..config import LLM_SERVICE_SECRET, cluster_env_vars

#: The K8s twin's probes (``k8s_probe``): the bench's workload image plus
#: boto3, the client a Kubernetes job stages its inputs with. A
#: memory-backed /dev/shm stands in for the twins' ``emptyDir {medium: Memory}``.
k8s_probe_image = bench_image.with_pip_packages("boto3>=1.35,<2")
k8s_probe_env = flyte.TaskEnvironment(
    name="rt-k8sprobe",
    image=k8s_probe_image,
    resources=flyte.Resources(cpu=8, memory="24Gi", disk="60Gi", shm="12Gi"),
    interruptible=True,
    env_vars=cluster_env_vars(),
)

#: The decision model's trainer since round 22: an L4 (g6.2xlarge, sm_89, bf16)
#: by default, for the twice-as-large K8s corpus at a 3,072-token context.
#: Override per deploy with RSA_GPU=T4:1. The benches' own LLM arms stay on
#: ``bench.envs.bench_gpu_env``.
RSA_GPU = os.environ.get("RSA_GPU", "L4:1")
_TRAIN_POD = {  # under each node's allocatable
    "L4:1": dict(cpu=6, memory="26Gi"),   # g6.2xlarge: 7.9 CPU / 29 GiB
    "T4:1": dict(cpu=3, memory="13Gi"),   # g4dn.xlarge
}[RSA_GPU]
allbench_train_env = flyte.TaskEnvironment(
    name="rt-allbench-train",
    image=bench_gpu_image,
    resources=flyte.Resources(**_TRAIN_POD, gpu=RSA_GPU, disk="80Gi", shm="auto"),
    env_vars={**cluster_env_vars(), "RSA_GPU": RSA_GPU, "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
)

allbench_env = flyte.TaskEnvironment(
    name="rt-allbench",
    image=bench_image,
    resources=flyte.Resources(cpu=2, memory="10Gi"),
    secrets=[flyte.Secret(key=LLM_SERVICE_SECRET, as_env_var="LLM_SERVICE_API_KEY")],
    env_vars=cluster_env_vars(),
    depends_on=[bench_gpu_env, k8s_probe_env, allbench_train_env],
)
