"""RightSizeCPU Bench: predict a task's peak CPU before it runs.

The question is the CPU half of right-sizing: given a task's source, its
input metadata and its run history, how many cores will it use at its
busiest -- as the platform measures it? Union's task monitoring reads CPU
from cAdvisor scraped every 15 s and charts ``irate`` over consecutive
scrapes (unionai/cloud: operator Prometheus ``scrape_interval: 15s``;
usage service ``irate(container_cpu_usage_seconds_total[5m])``), so the
bench's label is the busiest **15-second window** of a measured run.

Peak CPU is set mostly by code, not input size: explicit thread counts
(``n_jobs``, ``SET threads``, ``torch.set_num_threads``,
``POLARS_MAX_THREADS``), what a thread pool actually parallelizes (pure
Python threads hold the GIL; ``zlib``/``hashlib``/numpy release it),
process pools, serial libraries (pandas), and I/O waits. Every workload
runs for 1-2 minutes -- several 15 s windows -- so the peak is a real
sustained window, not a startup spike.

Modules mirror RightSizeMem's (``resource_tuner.bench``): grid, workloads,
measure, dataset, features, scoring, arms, spec, envs, stations, factory.
"""
