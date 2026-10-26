"""RightSizeGPU Bench: predict a GPU task's memory and utilization before it runs.

The GPU half of right-sizing. From a task's source (PyTorch training and
inference code), its input metadata and its run history, an arm predicts

- **peak GPU memory** -- the most device memory the task holds, as the
  platform sees it (``DCGM_FI_DEV_FB_USED``: CUDA context, caching
  allocator and all). That decides which GPU the task can run on: a
  request maps to the cheapest device on demo.hosted whose memory covers
  it (T4 16 GB < L4 24 GB < L40S 48 GB; A10G 24 GB costs more than an L4
  and is never the cheapest), and a task that does not fit dies with a
  CUDA OOM and is retried on the next device up;
- **mean GPU utilization** -- the share of time a kernel is running
  (``DCGM_FI_DEV_GPU_UTIL``), which says whether the task can use the GPU
  it is given at all (an input-bound loader at 20% wants a smaller, shared
  or no GPU).

Workload length follows the platform's GPU metrics: Union's dcgm-exporter
keeps its default 30 s collection interval (unionai/cloud dcgm_exporter.tf
sets no ``-c``) and Prometheus scrapes it every 15 s, so a run must last
several 30 s samples to be seen at all. Every workload trains or infers
for a ``TIME_BUDGET_S`` of 2-3 minutes (4-6 DCGM samples), well under the
20-30 minute ceiling that keeps the bench's GPU bill small.

Modules mirror RightSizeMem's and RightSizeCPU's.
"""
