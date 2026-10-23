# Round 18 — RightSizeCPU and RightSizeGPU benches (small releases)

**Date:** 2026-10-23 · **Branch:** `round-18-rightsize-cpu-gpu`

## Context

Round 17's RightSize Bench was a memory benchmark; it is renamed
**RightSizeMem Bench** (code, reports and docs; the deployed factory
`rightsize-bench` and its `rt-bench-*` artifacts keep their names, because
they hold round 17's published versions). This round adds the other two
halves of right-sizing, measured the same way (real code executed in pods,
labels from the kernel or the driver, every arm restricted to what a
scheduler sees at submit time: source, input metadata, run history):

- **RightSizeCPU** (`resource_tuner/bench_cpu/`): predict a task's peak CPU
  -- the busiest 15-second window -- and its mean CPU.
- **RightSizeGPU** (`resource_tuner/bench_gpu/`): predict a PyTorch task's
  peak GPU memory (which decides the device) and its mean GPU utilization.

The question for this round: do small releases of both benches run end to
end on demo.hosted, are their labels sound, and does the RightSizeMem
ordering (rules < trees < LLMs, decision head competitive) hold?

### Measurement windows come from Union's own metrics pipeline

Workload length is set by how often the platform measures (read from
`unionai/cloud`, 2026-10-23):

- **CPU:** the operator's Prometheus scrapes cAdvisor every **15 s**
  (`operator/deploy/helm/templates/monitoring/prometheus/configmap.yaml`:
  global `scrape_interval: 15s`; the `kubernetes-cadvisor` job inherits it),
  and task CPU in the console is
  `irate(container_cpu_usage_seconds_total[5m])`
  (`usage/deploy/values.yaml`), i.e. the rate over the last two scrapes. The
  label is therefore the busiest **sliding 15 s window** of a per-second
  series read from the container's cgroup (`cpu.stat`), the counter
  cAdvisor exports. Workloads loop for a 60-90 s `TIME_BUDGET_S`: >= 4
  windows (median 5.0).
- **GPU:** the `gpu-metrics` job scrapes dcgm-exporter every 15 s, but
  dcgm-exporter itself collects every **30 s** -- the chart (4.7.1,
  `infra/terraform/modules/dataplane/k8s_base/common/dcgm_exporter.tf`) sets
  no `-c`, so the upstream default holds. The console's GPU memory and
  utilization are `DCGM_FI_DEV_FB_USED` and `DCGM_FI_DEV_GPU_UTIL`. The bench
  samples the same NVML counters every 0.5 s; workloads run for a 120-180 s
  budget: >= 4 DCGM samples (median 5.2), far under the 20-30 min ceiling.

### Design

- **Shared LLM stack.** `bench/spec.py` + `registry.py` name what differs
  per bench (two decisions and their option lists, prompt, parser, decoder,
  headline). The zero-shot, SFT and Jev-style decision arms
  (`llm_arms.py`) and the T4 trainer (`gpu_stations.train_llm_arm`) read the
  spec, so one implementation serves all three benches; RightSizeMem's spec
  reproduces its round-17 prompt and decoding exactly (its 43 tests pass
  unchanged).
- **RightSizeCPU workloads** (11 families, 2 held out): explicit pool sizes
  (`OPENBLAS_NUM_THREADS`, `POLARS_MAX_THREADS`, `SET threads`, `n_jobs`,
  `torch.set_num_threads`, `threadpool_limits`, `pa.set_cpu_count`), thread
  pools on pure Python (GIL: ~1 core) vs `zlib`/`hashlib`/numpy (GIL
  released), process pools capped by their chunk count, serial pandas,
  I/O-bound fetch loops, and serial-then-parallel phases. One workload per
  9-CPU interruptible pod at a time. BLAS/OpenMP are pinned to one thread
  unless the code sets them, and the prompt says so.
- **RightSizeGPU workloads** (8 families, 2 held out): tabular MLP, CNN,
  GPT-style LM, encoder inference (with/without `inference_mode`, outputs
  kept on device), two-tower recsys (sparse vs dense Adam), an input-bound
  dataloader, blocked similarity GEMM, kNN (`cdist` + `topk`). Pass 1 on a
  T4 (g4dn.xlarge); CUDA OOMs re-measured on an L40S.
- **Device ladder** (from the tenant's GPU pools): a GPU memory prediction
  maps to the cheapest device that holds it -- T4 15 GiB $0.526/h <
  L4 22 GiB $0.978/h < L40S 44 GiB $2.242/h. A10G (22 GiB, $1.212/h) is
  never the cheapest. A run that does not fit dies with a CUDA OOM after the
  part of the run before its peak and is retried one device up.

## Run ledger

Console base: `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/<run>`

| run | what | outcome |
|---|---|---|
| [rsc-pilot-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-pilot-1) | CPU pilot, 1 task/family | **FAILED** at import in 24 s: `versioning.py` hashed `bench_gpu/workloads.py` at module import, and the code bundler only ships modules the deploy imports. Fix: the CPU/GPU cache versions are computed when their own stations ask (commit with this entry) |
| [rsc-pilot-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-pilot-2) | CPU pilot, 22 runs | **succeeded**: all 22 measured from the cgroup; per-family peaks match the code (pure-Python threads 1.01 cores at any pool size, `n_jobs` forests ~5.4, I/O fetcher 0.06, mixed phases peak 3.0 / mean 2.0) |
| [rsg-pilot-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-pilot-1) | GPU pilot, 16 runs on T4 | **succeeded** in 15 min. Two findings changed the code before the release: a CNN's cuDNN workspace search logged failed 3-8 GB allocations, fell back, and peaked at 15.1 of 15.1 GiB -- and succeeded (now flagged `fills_device`, excluded from memory-accuracy columns only); 14 of 16 runs read >90% utilization (loader family reweighted toward per-sample Python augmentation) |
| [rsc-small-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-small-1) | **RightSizeCPU small-1**: 11 fam x 10 tasks x 2 runs = 220 runs, 37 shards; 8 arms | **succeeded** (2 h 15 min, 4.7 pod-hours of measurement). 4 shards were spot-reclaimed twice each ("terminated in response to imminent node shutdown") and succeeded on attempt 3 -- each reclaim discarded up to 5 finished runs |
| [rsg-small-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-small-1) | **RightSizeGPU small-1**: 8 fam x 8 tasks x 2 runs = 128 runs, 26 T4 shards + 1 L40S shard; 8 arms | **succeeded** in 40 min (5.7 GPU-hours). Leaderboard **superseded**: memory was decoded by expected cost, and because ~95% of runs fit the cheapest device, every distributional arm collapsed to one cell (15 GiB) -- gbt-dist, sft-sampled and jev-pointer scored identically |
| [rsg-small-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-small-2) | re-materialize after switching GPU to accuracy decoding | **no-op**: every arm/scorecard partition already existed, so the factory reused them all |
| [rsg-small-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-small-3) | `--rebuild rt-gpubench-arm --downstream` | **ran the old code**: every fit used task version `37ff8d2f...`, the same as rsg-small-1 -- `flyte factory deploy` pinned station versions at 17:24 and the 18:07 station redeploy never reached the factory. Results discarded |
| [rsg-small-4](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-small-4) | factory redeployed, then rebuild arms + downstream | **succeeded**: the GPU leaderboard below |

## Results

Small releases: test n=52 (CPU) / 22 (GPU), unseen families n=40 / 32. A
95% interval on a share near 75% at n=52 is about ±12 points -- read the
orderings, not the decimals.

### RightSizeCPU small-1 (220 runs; train 116)

Headline: peak predicted within ±25% of the measured busiest-15 s window
(or 0.25 core). `cpu regret`: core-seconds reserved + 1 oracle core-second
per second of throttling delay, replayed against each run's own series.

| arm | test within ±25% | test under-predicted | test MAE (cores) | test cpu regret | **unseen within ±25%** | unseen MAE |
|---|---|---|---|---|---|---|
| static (one request) | 34.6% | 30.8% | 1.13 | 44.8% | 32.5% | 1.55 |
| vpa (history max x margin) | 67.3% | 23.1% | 0.62 | 18.3% | 67.5% | 0.88 |
| **code-rule** (largest declared pool) | **80.8%** | 40.4% | 0.74 | 18.0% | **95.0%** | 0.55 |
| gbt (quantile trees) | 65.4% | 34.6% | 0.61 | 30.9% | 35.0% | 1.36 |
| gbt-dist | 73.1% | 26.9% | 0.51 | 22.5% | 47.5% | 1.20 |
| **zero-shot 27B** | 78.8% | 23.1% | **0.43** | **15.4%** | 92.5% | **0.50** |
| SFT 1.7B, greedy | 71.2% | **7.7%** | 0.54 | 17.5% | 67.5% | 1.00 |
| SFT 1.7B, 8 samples | 61.5% | 13.5% | 0.76 | 41.5% | 67.5% | 0.96 |
| decision 1.7B, pointer (argmax) | 63.5% | 7.7% | 0.64 | 24.0% | 67.5% | 0.92 |
| decision 1.7B, pointer | 71.2% | 17.3% | 0.49 | 19.5% | 82.5% | 0.78 |

Test by driver (within ±25%): the code-rule is perfect on `serial`,
`processes` and `phases` and fails the GIL (`gil` 25%: it reads
`max_workers=8` on pure-Python threads as 8 cores) and I/O (0%); the
zero-shot 27B gets the GIL right (88%) but the serial-then-parallel phases
wrong (12%); gbt-dist and the decision head get the GIL right (100% / 75%).

### RightSizeGPU small-1 (126 usable runs; train 62)

Headline: peak GPU memory within ±20% of the measured peak (or 0.5 GiB;
runs that filled the device are not scored on it). The device and its
dollars follow from the prediction.

| arm | test mem within ±20% | test within 1 cell | test MAE (GiB) | test $ regret | test OOM | unseen mem within ±20% | unseen util within ±10 pts |
|---|---|---|---|---|---|---|---|
| static | 9.1% | 54.5% | 2.19 | 16.5% | 9.1% | 23.3% | 87.5% |
| vpa | 40.9% | 77.3% | 1.49 | 8.4% | 4.5% | 43.3% | 93.8% |
| size-rule | **54.5%** | **81.8%** | **1.39** | 8.4% | 4.5% | 36.7% | 93.8% |
| gbt | 36.4% | 68.2% | 1.98 | 16.5% | 9.1% | 30.0% | 87.5% |
| gbt-dist | 50.0% | 63.6% | 1.95 | 16.5% | 9.1% | 20.0% | 87.5% |
| zero-shot 27B | 36.4% | 54.5% | 2.79 | 8.4% | 4.5% | 40.0% | 40.6% |
| SFT 1.7B, greedy | 50.0% | 50.0% | 2.32 | 16.5% | 9.1% | **50.0%** | 87.5% |
| SFT 1.7B, 8 samples | 31.8% | 50.0% | 2.33 | 16.5% | 9.1% | 36.7% | 87.5% |
| decision 1.7B, pointer | 36.4% | 36.4% | 2.52 | 16.5% | 9.1% | 30.0% | 87.5% |

## Findings

1. **Both benches run end to end as Union factories on demo.hosted**, each
   in one materialization: 220 CPU runs in 2 h 15 min (4.7 pod-hours, spot),
   128 GPU runs in 40 min (5.7 T4-hours + one L40S shard). Every run spans
   >= 4 of the platform's own measurement windows.
2. **Peak CPU is a property of the code, not the input.** Two runs of the
   same task at different input sizes agree to 0.7% at the median (p90
   11%). This is why the CPU bench separates arms by *reading*: the
   code-rule (parse the declared pool size) reaches 81% / 95%, while trees
   on input features fall to 35% on unseen families.
3. **The zero-shot 27B is the strongest CPU arm on balance** -- lowest MAE
   (0.43 / 0.50 cores), lowest throttling cost (15.4%), and it alone gets
   both explicit pools and the GIL mostly right. The code-rule's 95% on
   unseen families is a property of those two families (duckdb and
   pyarrow's CSV reader honor their declared pool sizes), not a general
   win: it scores 25% on GIL-bound threads and 0% on I/O.
4. **The 1.7B decision head learns CPU from 116 rows** (71% / 82.5%, ahead
   of gbt and SFT on unseen families) and its expected-margin decoding
   beats its own argmax by 8-15 points -- the round-17 pattern, at a
   fiftieth of the data. SFT greedy has the fewest under-predictions (7.7%).
5. **For CPU, accuracy and cost disagree; accuracy is the target.** Under a
   throttling-only cost a request below the peak is often *cheaper* (a
   throttled second costs less than an idle core), so a cost-optimal
   decoder predicts low on purpose (`test_cost_optimal_decoding_under_predicts`).
   The headline and decoders target "within ±25% of the peak".
6. **GPU memory accuracy is the open problem at this size.** With 62
   training rows no arm exceeds 55% within ±20% on test (size-rule) or 50%
   on unseen families (SFT greedy). The decision head collapsed to a single
   cell (its expected-margin and argmax decodings are identical); it needs
   the bigger release before it says anything.
7. **On this tenant's GPU ladder the device choice is nearly free.** ~95% of
   runs fit a T4, so "anything up to 15 GiB" costs the same and
   cost-decoding erased the memory prediction (rsg-small-1). The bench
   scores memory accuracy and derives the device from it.
8. **NVML/DCGM utilization saturates**: 108 of 126 runs read 90-100%.
   `GPU_UTIL` counts any queued kernel, so a constant "90-100%" is right 88%
   of the time; the input-bound loader and the recsys model are the only
   informative cases. The zero-shot 27B under-predicts utilization badly
   (MAE 13-23 points). `DCGM_FI_PROF_SM_ACTIVE` (already enabled in Union's
   dcgm-exporter config) is the better label; NVML cannot read it.
9. **Conv nets fill the device.** cuDNN's workspace search grows into
   whatever memory exists (7 runs at >= 95% of the T4). A platform reading
   peak FB_USED for such a task sees the device's size, not the task's
   need; the label must be "fits device D", which the bench records.

## Code changes

- `bench/spec.py`, `bench/mem_spec.py`, `bench/registry.py`,
  `bench/common.py`; `llm_arms.py` and `gpu_stations.py` read a spec.
- `bench_cpu/` and `bench_gpu/` packages; `bench_cpu.py`, `bench_gpu.py`,
  `bench_cpu_factory.py`, `bench_gpu_factory.py`.
- `bench/measure.py`: integer `.npy` inputs can declare a value range
  (token ids < vocabulary).
- `bench/versioning.py`: CPU/GPU cache versions computed lazily
  (rsc-pilot-1).
- `bench_gpu`: `fills_device`; memory-accuracy decoding (rsg-small-1).
- Tests: `tests/test_bench_cpu.py`, `tests/test_bench_gpu.py` (incl. pointer
  offsets against the real Qwen3 tokenizer for both new prompts).
- `CLAUDE.md`: `flyte factory deploy` pins station versions; existing
  partitions are not re-run without `--rebuild ... --downstream`.

## Open / next

- **Scale both releases** before reading arm orderings closely: ~40
  tasks/family (CPU ~880 runs, ~20 pod-hours spot; GPU ~640 runs, ~30
  T4-hours), shards of 3 runs so a spot reclaim loses less.
- GPU utilization label: a profiling metric (SM active) via DCGM, or more
  input-bound workloads; measure a subset on L4 to show device dependence.
- Add the Qwen3-4B decision head and the gbt x decision pool (RightSizeMem's
  best policy) once the corpora are large enough to train them.
