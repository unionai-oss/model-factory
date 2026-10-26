"""Task environments for the combined bench.

Nothing here measures a workload: the corpus is joined from the three
benches' measured corpora. The orchestration / classical-arm env is the
benches' CPU shape (t3a.xlarge has headroom); the decision model trains and
RL-tunes on the benches' T4 env.
"""

from __future__ import annotations

import flyte

from ..bench.envs import bench_gpu_env, bench_image
from ..config import LLM_SERVICE_SECRET, cluster_env_vars

allbench_env = flyte.TaskEnvironment(
    name="rt-allbench",
    image=bench_image,
    resources=flyte.Resources(cpu=2, memory="10Gi"),
    secrets=[flyte.Secret(key=LLM_SERVICE_SECRET, as_env_var="LLM_SERVICE_API_KEY")],
    env_vars=cluster_env_vars(),
    depends_on=[bench_gpu_env],
)
