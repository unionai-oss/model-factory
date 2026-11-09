# Round 22: RightSize Bench, Flyte and Kubernetes -- every run and its K8s twin

## Context

The combined bench (round 21) sizes Flyte tasks. Flyte runs on Kubernetes, so the
same workload code, with the same inputs, can run as a plain Python module
in a Kubernetes Job. Customers on plain Kubernetes are the larger market.
This round asks three questions.

1. Can the bench cover Kubernetes Jobs without re-measuring? The measured
   user code is the same. What changes is how inputs reach the pod, where
   they are staged, and what else runs in the container. Can that be
   accounted for exactly, and checked by measurement?
2. Does a model trained on both platforms size both? In particular, does it
   read a Job manifest well enough to see a difference of one line,
   `emptyDir: {medium: Memory}`, which charges every staged byte to the
   container's memory?
3. Does a model trained on Flyte data alone transfer to Kubernetes? This is
   the question for a customer whose history is all Flyte.

The bench is the largest the standing constraints allow: every run of the
largest measured releases (RightSizeMem v2, RightSizeCPU big-1, RightSizeGPU
big-2) and each run's K8s twin. The decision model trains on every row its
trainer's 12 h budget allows.

## The Kubernetes twin (`bench_all/k8s.py`)

For each Flyte run (`bench.workloads.render_task`'s module), the twin consists of:

- **`job.py`**: the same function body, with paths instead of `File`s and
  no `flyte` import, plus a `__main__` that stages the inputs, calls the
  function, writes `result.json` to the staging volume and uploads it. The
  disk and tmpfs variants of a staging render the same `job.py`.
- **a `batch/v1` Job manifest**: image, command, the input URIs, the
  thread floor the measurement used (`OMP/OPENBLAS/MKL_NUM_THREADS=1`), the
  staging volume, and `resources: {}  # to be decided`.

Each task draws one of four stagings; a task's runs share it.

| staging | share | who fetches the inputs | where they land | memory label |
|---|---|---|---|---|
| init-disk | 30% | aws-cli init container | emptyDir (node disk) | workload peak |
| init-tmpfs | 20% | aws-cli init container | emptyDir, medium: Memory | + staged bytes |
| boto3-disk | 30% | the job, boto3 | emptyDir (node disk) | + boto3 (48 MiB) |
| boto3-tmpfs | 20% | the job, boto3 | emptyDir, medium: Memory | + boto3 + staged bytes |

The workload peak is the Flyte run's need without the Flyte runtime (180
MiB) or the 5% headroom. `python job.py` does not load the runtime. The
label puts the headroom back. A boto3 download runs inside the job at 1.8
cores, for bytes / 440 MB/s + 0.4 s per object + 0.7 s of setup. That
download is prepended to the job's per-second CPU series (RightSizeCPU
rows), or folded into its mean (RightSizeMem rows), so it can become a
light job's busiest 15 s window. The GPU label is unchanged.

A twin keeps its run's split (by task) and its crossfit fold
(`hybrid.fold_of` strips the `@k8s` suffix). Its history holds the earlier
runs' container peaks, as cAdvisor would report them, not Flyte's. The
prompt shows both the code and the manifest, and it ends the same way on
both platforms, so the pointer head's option offsets are unchanged. The
trees get five new token flags: `kind: Job`, `initContainers`,
`medium: Memory`, `import boto3`, `download_file`. The `rules` arm reads
history in each platform's units.

## Run ledger

| run | what | outcome |
|---|---|---|
| [rt-k8sprobe-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sprobe-1) | staging probe: boto3 footprint, transfer throughput and cores, tmpfs charging, 64 MiB-4 GiB objects, 2 repeats, disk vs a memory-backed `/dev/shm` | succeeded. Results under Findings 1; they set `k8s`'s constants. Also found the `ru_maxrss` inheritance (Finding 2) |
| [rt-l4timing-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-l4timing-1) | decision-model throughput on an L4 at 3,072 tokens | **failed** at import: the trainer env set `RSB_GPU=L4:1`, which `bench/envs.py` reads as the T4 bench's GPU (`KeyError`). Fix: the trainer env carries `RSA_GPU` instead |
| [rt-l4timing-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-l4timing-2) | the same, fixed | succeeded: L4 admitted in minutes, 1.02 s per training sample including the model load (480 rows), 0.245 s per scored row, 10.0 GiB peak. Hence 5,000 rows per source and platform, one epoch |
| [rt-k8sval-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sval-1) | validation: real workloads as the Flyte control and as the K8s twin | **aborted**: its instrument (`ru_maxrss` of each child) reports the parent's peak, not the child's (Finding 2) |
| [rt-k8sval-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sval-2) | the same, with the child's own VmHWM (atexit hook) and the container cgroup's anon + shmem as instruments; 192 runs (96 RightSizeMem, 96 RightSizeCPU), every staging 48 times | succeeded, 184 of 192 returned, 176 measured: Finding 3. The 8 that failed are process-pool workloads, which the probe's runpy wrapper broke (a pool worker cannot unpickle a function defined in a runpy module). The twin itself runs as `python job.py` and runs them (checked locally). The wrapper is now prepended to `job.py` itself. 8 runs (one shard) returned nothing |
| [rsk-k8s-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-1) | **RightSize Bench, Flyte + K8s (k8s-1)**: v2 + CPU big-1 + GPU big-2 and their twins (test n=10,190, unseen n=12,832); static, rules, gbt-dist, zeroshot-27b, jev-pointer, jev-pointer-flyte, RL, pools | succeeded, 21 actions, 12.2 h (both decision models ~8 h on an L4, RL 3.6 h). Findings 4-6 |
| [rt-k8sgate-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sgate-1) + [-score](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sgate-1-score) | `gate_all_arms` (new) on k8s-1's arms: the novelty gate at five val quantiles, and the Flyte -> K8s adapter | succeeded. The gate: Finding 7. The adapter fixed memory and overshot CPU on RightSizeMem rows, whose CPU label is a mean, not a busiest window. Fixed in `gating.adapt_probs` |
| [rt-k8sgate-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sgate-2) + [-score](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sgate-2-score) | the same, with the CPU floor on window-labelled rows only | succeeded. The adapter's CPU and GPU picks from the Flyte rendition were still worse than the model's own on the K8s prompt. Next: memory only |
| [rt-k8sgate-2-signals](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sgate-2-signals) | `gate_signals` (analysis): per evaluated run, candidate novelty signals (kNN distance to training runs, disagreement, entropies) and every arm's cost ratio | succeeded, 28,058 runs, ~1.5 h (900k cost evaluations). kNN separates unseen families perfectly: Finding 8 |
| [rt-k8sgate-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sgate-3) + [-score](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8sgate-3-score) | the adapter on memory only, from the Flyte rendition or the Job's own prompt | succeeded: Finding 7 |
| [rsk-k8s-1-gate](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-1-gate) | the factory with `rt-k8sbench-gate` (kNN / numeric kNN / disagreement gates, adapters) on k8s-1's arms, leaderboard rebuilt (`--rebuild rt-k8sbench-leaderboard`) | succeeded, 23 actions, 12.1 h. **It retrained every LLM arm**, which was not the intent: adding `gating` to the combined bench's cache version (and editing `stations`) changed the cache key of every station that uses it, so every station re-ran. static, rules and gbt-dist reproduced their scorecards exactly. The result is a second, independent training of the suite (replicate 2): Findings 7-9 |

## Findings

### 1. Staging costs are small and exact, except for tmpfs

From rt-k8sprobe-1 (boto3 1.43.126, cgroup v2):

- boto3 alone: a bare interpreter holds 11 MiB. After import and client
  setup it holds 59 MiB; after the download, 74-80 MiB (growing slowly
  with object size). The high-water mark is the end state: the transfer has
  no transient peak above it.
- Throughput: 122 MB/s for 64 MiB, 315 for 512 MiB, 385-404 for 2 GiB,
  417-431 for 4 GiB. That fits bytes / 440 MB/s + 0.4 s per object. The
  transfer uses 1.6-2.8 cores (s3transfer's 10 threads).
- A file on the memory-backed volume is charged to the container's memory
  byte for byte (shmem delta = file size, 64 MiB-4 GiB) and released on
  delete. On node disk the same file appears in `memory.current` as page
  cache, which the kernel reclaims before it OOM-kills. The twin's
  accounting does not charge it, as `ru_maxrss` never did.

### 2. A child's `ru_maxrss` includes its parent's peak

`os.wait4` returns a child's `ru_maxrss`. On Linux this includes the
parent's high-water mark at the moment the child `exec`s: `exec_mmap`
folds the old memory map's hiwater into the child's `signal->maxrss`, and
under fork or vfork the old memory map is the parent's. In the probe, every
child reported 228-490 MiB, the probe process's own peak at the time. Their
own VmHWM was 74-80 MiB. The validation therefore measures a child's own
VmHWM, written by an atexit hook, plus the container cgroup's anon + shmem
peak.

The RightSize benches measure with `ru_maxrss`, so their labels have a
floor at the measuring process's peak. rt-k8sval-2 re-measured 176 corpus
runs. The median ratio of corpus peak to own peak is 1.001, but 27% of runs
are inflated by more than 5% (RightSizeCPU 39%, RightSizeMem 17%). Almost all
of those corpus peaks sit at 157-161 MiB: the measuring parent's peak. Their
own peaks are 12-160 MiB. Four RightSizeCPU runs read 290-414 MiB against own
peaks below 375.

What this does to a label depends on the platform:

- **Flyte labels** (+180 MiB runtime, +5%): the floor moves the ladder cell
  up one step for 10 of 80 RightSizeCPU runs and 3 of 96 RightSizeMem runs
  (one RightSizeMem run moves down).
- **K8s labels** (no runtime): it moves no cell. Floor and own peak both fall
  in the bottom cell (256Mi).

The labels are left as they are. This is a stated limit of every RightSize
release so far: the smallest Flyte workloads' labels are conservative by up
to one cell.

### 3. The twin's labels hold under measurement (rt-k8sval-2)

176 runs (88 RightSizeMem, 88 RightSizeCPU; 44 per staging), each measured
twice in the same pod:

- once as the Flyte bench measured it (the control);
- once as the rendered `job.py`, staging its inputs from the cluster's object
  store. The init stagings ran a separate process, as an init container would.

The accounting predicts the job from the control:

| staging | n | measured / predicted (median, p5-p95) | same ladder cell | within one |
|---|---|---|---|---|
| init-disk | 44 | 1.000 (0.993-1.032) | 95.5% | 100% |
| init-tmpfs | 44 | 1.000 (0.886-1.017) | 97.7% | 100% |
| boto3-disk | 44 | 0.999 (0.964-1.044) | 100% | 100% |
| boto3-tmpfs | 44 | 1.000 (0.914-1.027) | 97.7% | 100% |

- With an init container, the job *is* the workload. Job minus control own
  peak: median +0.02 MiB (p10 -3.6, p90 +4.4).
- tmpfs charges exactly the staged bytes: charged / input size is 1.0000
  (max 1.0014).
- Inside a real workload, boto3 adds 48.0 MiB median (p10 36, p90 60), not
  the standalone 70 MiB: it shares modules the workload already loaded.
  `k8s.BOTO3_MIB = 48`. The cgroup shows 40.5 MiB, because the cgroup counts
  shared pages once.

End to end, compare the twin label the bench actually uses (from the
*corpus* Flyte peak, through the accounting) with the job's measured cell.
It is the same cell for 90.3% of runs and within one cell for 100%. The
misses are conservative: 9.1% are one cell over and 0.6% (one run) one cell
under. By source, the same-cell rate is 85.0% for RightSizeCPU and 94.8% for
RightSizeMem. By staging, it is 95.5% for every staging except boto3-tmpfs
(75.0%). The over-cells come from the Flyte label floor (Finding 2), which
the twin inherits whenever staged bytes or boto3 lift the twin out of the
bottom cell.

### 4. One model sizes both platforms (rsk-k8s-1)

Mean regret against the oracle, by platform. The memory slices are CPU-node
runs; the GPU label is the same on both platforms.

| arm | test: Flyte / K8s | unseen: Flyte / K8s | K8s memory right (test) |
|---|---|---|---|
| static | 91.8% / 89.3% | 113.4% / 115.1% | -- |
| rules | 39.4% / 42.4% | 48.0% / 49.8% | -- |
| gbt-dist | 14.6% / 11.6% | 67.8% / 69.9% | 92.8% |
| zeroshot-27b | 81.1% / 93.3% | 128.0% / 138.1% | -- |
| jev-pointer | 10.8% / 13.4% | 48.3% / 50.3% | 91.0% |
| rl-pointer | 14.2% / 13.4% | 36.5% / **34.5%** | 78.3% |
| rl-pointer-argmax | 7.7% / 8.3% | 42.6% / 43.1% | 89.1% |
| pool-jev-pointer | 6.4% / 7.1% | 57.8% / 59.8% | 95.1% |
| pool-rl-pointer-argmax | **6.0% / 6.6%** | 70.2% / 68.1% | 93.8% |

- **Trained on both platforms, an arm's K8s regret is within 1-3 points of
  its Flyte regret**, on test and on unseen families alike. The decision
  model's K8s memory accuracy is the same across stagings: jev-pointer gets
  90.2% / 90.8% / 91.4% / 92.0% right on init-disk / init-tmpfs /
  boto3-disk / boto3-tmpfs.
- **It reads the one-line difference.** Each run and its twin form a pair.
  When the label moves between them, jev-pointer's memory request moves in
  the same direction in 76.9% of test pairs, and pool-rl-pointer-argmax's in
  81.2%. The rules move theirs in 43.4%, zero-shot 27B in 37.9%.
- **Unseen families remain the bottleneck, not the platform.** The best
  unseen-family regret is rl-pointer's 34.5-36.5%, the same as round 21's
  35.5% on Flyte alone. Tripling the bench's rows with twins changed none of
  this.

### 5. A model trained on Flyte alone does not transfer as is

`jev-pointer-flyte`: the same 1.7B decision model, trained on the same
number of steps, but on Flyte rows only.

| | test: Flyte / K8s regret | K8s memory right | pairs: moves with the label |
|---|---|---|---|
| jev-pointer (both platforms) | 10.8% / 13.4% | 91.0% | 76.9% |
| jev-pointer-flyte | 14.3% / 18.7% | 39.9% (Flyte: 72.7%) | 35.0% |

It sizes a K8s Job about as it would the Flyte task. By staging, its
memory is right on 28.7-54.6% of runs, and its pair-direction score (35%)
is below the rules'. Regret hides part of this,
because over-requesting memory is cheap next to an OOM. Its K8s regret is
still 4.4 points over its Flyte regret, against 2.6 for the model trained
on both.

### 6. The minimax-best arm

The best arm on the worse of test and unseen families is **rl-pointer:
13.8% / 35.5%**. For comparison:

- rl-pointer-argmax: 8.0% / 42.9%;
- pool-rl-pointer: 10.4% / 45.6%;
- jev-pointer: 12.1% / 49.3%.

The pooled arms win in distribution (6.2-6.8%) but lose unseen families
(58-69%). The trees' weight is tuned on val, which holds seen families only,
and so it is too high for families the trees have never seen. As in round 21,
no single arm is best on both.

### 7. The Flyte -> K8s adapter fixes memory accuracy, not regret

`<arm>-adapt` (`gating.adapted`) gives a Flyte-only model's memory
distribution for a Job's Flyte rendition, moved through the twin's
accounting. `-adapt-prompt` instead reads the Job's own prompt as Flyte
units. CPU and GPU come from the model's own reading of the Job.
rt-k8sgate-1/-2 established those choices: the rendition's CPU and GPU
picks were worse.

| replicate | arm | test: all / K8s regret | K8s memory right | pairs: moves with the label |
|---|---|---|---|---|
| 1 | jev-pointer-flyte | 16.5% / 18.7% | 39.9% | 35.0% |
| 1 | jev-pointer-flyte-adapt | 17.0% / 19.7% | 65.1% | 73.7% |
| 1 | pool-jev-pointer-flyte | 6.2% / 7.4% | 77.9% | 48.6% |
| 1 | pool-jev-pointer-flyte-adapt | 9.6% / 14.3% | 84.8% | 81.4% |
| 2 | jev-pointer-flyte | 16.6% / 18.0% | 30.5% | 37.7% |
| 2 | jev-pointer-flyte-adapt | 17.9% / 20.6% | 51.2% | 65.6% |
| 2 | pool-jev-pointer-flyte | 7.3% / 7.1% | 84.4% | 56.4% |
| 2 | pool-jev-pointer-flyte-adapt | 11.2% / 15.0% | 85.1% | 81.7% |

Replicate 1 is rt-k8sgate-3 on rsk-k8s-1's arms; replicate 2 is
rsk-k8s-1-gate.

- The adapter makes a Flyte-only model see the staging. Pair direction goes
  from 35-38% (below the rules) to 66-82%, the level of a model trained on
  both platforms (Finding 4). K8s memory accuracy rises by 20-25 points.
- **It does not lower regret.** On both replicates, every adapted arm's K8s
  regret is 1-8 points above the unadapted arm's. The unadapted model sizes
  a tmpfs Job as if it were the Flyte task, and so under-requests by the
  staged bytes. Most of those misses are small enough that the model's own
  margin covers them. The adapter's corrections are sometimes too large, and
  an over-request that moves a run up a cell costs more than the misses it
  fixes. Regret rewards a cheap, slightly-too-small memory request on the
  run that would not have OOMed; memory accuracy does not.
- **For a customer whose history is all Flyte, the honest answer is to
  retrain on their K8s runs.** The adapter is a fallback, until there is K8s
  history, for when staging-aware memory matters more than mean cost (for
  example, an OOM costs more than this scorer charges).

### 8. A novelty gate gets the best arm on each split

`gate-<signal>-q<q>` (`gating.gate`): pool-rl-pointer-argmax's request on
runs whose novelty signal is at most the signal's val q-quantile, and
rl-pointer's on the rest. The threshold needs no family labels: val holds
seen families only, so q = 0.95 sends 5% of in-distribution runs to
rl-pointer by construction. Three signals were tried:

- knn: mean distance to the 10 nearest training runs in the trees'
  standardized features;
- knn-numeric: the same over input sizes and history only, without code
  tokens;
- js: the trees' and rl-pointer's disagreement.

Replicate 2 (rsk-k8s-1-gate; the factory's `rt-k8sbench-gate` node):

| arm | test | unseen | worse of the two | K8s: test / unseen | unseen runs routed out | test runs routed out |
|---|---|---|---|---|---|---|
| pool-rl-pointer-argmax | **6.6%** | 94.5% | 94.5% | 7.3% / 98.9% | -- | -- |
| rl-pointer | 13.9% | **36.8%** | 36.8% | 13.7% / 35.4% | -- | -- |
| **gate-knn-q95** (headline) | **6.9%** | **36.8%** | **36.8%** | 7.8% / 35.4% | 100% | 5.3% |
| gate-knn-q99 | 6.8% | 36.8% | 36.8% | 7.6% / 35.4% | 100% | 2.1% |
| gate-knn-numeric-q95 | 7.2% | 68.8% | 68.8% | 8.0% / 68.8% | 30.5% | 4.5% |
| gate-js-q50 | 9.8% | 40.8% | 40.8% | 10.5% / 40.3% | 75.6% | 50.7% |
| gate-js-q95 | 7.4% | 84.9% | 84.9% | 8.2% / 91.4% | 20.1% | 5.7% |

On replicate 1 (the offline analysis of rt-k8sgate-2-signals, before the
station existed), the same gate scored 6.7% test and 35.5% unseen. That
replicate's arms gave 6.3% for pool-rl-pointer-argmax on test and 35.5% for
rl-pointer on unseen.

- **The kNN gate matches the best single arm on each split, on both
  replicates.** It loses 0.3-0.4 points on test, for the 5% of runs it
  routes out, and nothing on unseen families. On the worse of the two, it
  ties rl-pointer. Against rl-pointer, it is 7 points better on test.
  Against every other arm, it is 6-58 points better on the worse of the two.
- **Caveat, and it belongs in any claim made from this: the bench is the
  kNN gate's best case.** Its AUROC, val vs unseen, is 0.9999. Six of the
  seven unseen families import a library that no training run imports, so
  a code-token feature constant on train puts them at the far edge of the
  space. Without code tokens (knn-numeric), AUROC is 0.67. That gate routes
  out only 30% of unseen runs, and its unseen regret is 68.8%, near the
  trees'. The disagreement signal is weaker still (AUROC 0.70). In
  production, "imports something the history has never seen" is a real and
  cheap signal. A new workload that reuses familiar libraries in an
  unfamiliar way would not trip it, and this bench does not measure that
  case.
- **Mean regret and dollars disagree on unseen families.** Cost per 1,000
  unseen runs, against an oracle of $515:
  - gbt-dist: $955
  - knn-numeric gate: $990
  - pool-rl-pointer-argmax: $1,042
  - kNN gate and rl-pointer: $1,066

  The scorer averages per-run cost ratios. Dollars are dominated by the
  large jobs, which the trees size better. On test, the kNN gate's $328 is
  17% over the oracle's $280, against $794 for static requests (-59%) and
  $437 for rules (-25%). That in-distribution number is the one to ship.
  The unseen-family dollar gap to the trees is open.

### 9. Two trainings of the same suite

rsk-k8s-1-gate retrained every LLM arm (ledger), so each arm has two
independent trainings on the same data. The deterministic arms (static,
rules, gbt-dist) reproduced exactly. For the LLM arms:

| arm | test: rep 1 / rep 2 | unseen: rep 1 / rep 2 |
|---|---|---|
| jev-pointer | 12.1% / 13.2% | 49.3% / 51.1% |
| jev-pointer-flyte | 16.5% / 16.6% | 54.3% / 43.8% |
| rl-pointer | 13.8% / 13.9% | 35.5% / 36.8% |
| rl-pointer-argmax | 8.0% / 7.3% | 42.9% / 50.9% |
| pool-rl-pointer | 10.4% / 9.4% | 45.6% / 55.6% |
| pool-rl-pointer-argmax | 6.3% / 6.6% | 69.1% / 94.5% |
| zeroshot-27b | 87.2% / 86.7% | 133.1% / 130.0% |

- **Test regret is stable to about 1 point.** Unseen-family regret is not:
  the argmax and pooled variants move by 8-25 points between trainings. The
  Bayes-decoded rl-pointer moves by 1.3.
- The minimax ranking holds on both replicates: rl-pointer, and the kNN
  gate over it, are best on the worse of the two splits. A single
  replicate's unseen number for any other arm should be read as +-10
  points.
- Note (2026-11-08) on Finding 4: "within 1-3 points" holds for every
  decoded arm on replicate 2 (largest gap 3.0). Two argmax read-outs differ
  by more: jev-pointer-argmax on test, +6.9 points on K8s, and
  pool-rl-pointer-argmax on unseen families, +8.7.

- Note (2026-11-08) on Finding 4's pair metric: `scoring.paired`'s
  `mem_shift_direction` is computed over *every* CPU-node pair, not only
  those where the label moves: a pair counts when sign(request change) ==
  sign(label change), so "both unchanged" counts. "When the label moves
  between them" above overstates it.

Write-up: `benchmarks/rightsize-bench-k8s.html` and
`benchmarks/rightsize-bench-k8s-abstract.html` (replicate 2, with
replicate 1 alongside).

## Code changes

- `bench_all/k8s.py` (new): stagings, `module_code`, `manifest`, the label
  accounting (`twin_need_mib`, the CPU series with the download in front),
  `twin`, `with_twins`; constants calibrated by the probes.
- `bench_all/k8s_probe.py` (new): `staging_probe`, `validate` /
  `validate_shard`, and `train_timing`. Children are measured by their own
  VmHWM and by the cgroup, never by `ru_maxrss` alone.
- `bench_all/dataset.py`: `platform` / `staging` / `manifest` / `twin_of`
  columns, `with_k8s`; `balanced` draws per source *and* platform.
- `bench_all/prompting.py`: a platform-neutral system prompt (memory-backed
  emptyDir pages are charged to the container); K8s rows show `job.py` and
  the manifest, and history in container units.
- `bench_all/features.py`: K8s token flags. `bench_all/arms.py`: `rules`
  reads history through `k8s.need_from_peak`.
- `bench_all/scoring.py`: `{split}/platform=`, `{split}/staging=` and
  `{split}/pairs` slices. `paired` asks whether an arm's memory request
  moves the way the need moved between a run and its twin.
- `bench_all/stations.py`: `join_corpora(platforms=)`;
  `train_all_policy(platforms=, max_len=)`; the `jev-pointer-flyte`
  ablation (Flyte rows only, twice the epochs, the same steps); a paired
  hosted subsample; leaderboard tables for platform, staging and pairs.
  The decision model and RL train on `rt-allbench-train` (L4 by default,
  `RSA_GPU`).
- `bench_all/factory.py`: `build_graph` is parametric; `rightsize_k8s_bench`
  (`rt-k8sbench-*`, sources pinned, 5,000 rows per source and platform,
  3,072 tokens, no stacked arm). Entrypoint `bench_k8s_factory.py`; probes
  `bench_k8s_probe.py`.
- `bench/hybrid.py`: `fold_of` keeps a twin in its task's fold.
  `bench/versioning.py`: `bench_all/k8s` and `hybrid` are in the combined
  bench's cache versions.
- `bench_all/gating.py` (new): the novelty signals (`novelty_knn`, with and
  without code tokens; `disagreement`), `gate` (val-quantile threshold),
  `auroc`, and the Flyte -> K8s adapter (`staging_seen`, `adapt_probs`,
  `adapted`; memory only from the rendition or the Job's prompt).
- `bench_all/stations.py`: `gate_all_arms` (every signal at six val
  quantiles, the headline `gate-rl-pointer` = kNN at 0.95, the adapted
  Flyte-only arms) and `gate_signals` (analysis: per-run signals and every
  arm's cost ratio). The leaderboard gains a worse-of-test-and-unseen table
  and takes the gate scorecard.
- `bench_all/factory.py`: `build_graph(gate=)`. `rightsize_k8s_bench` builds
  `rt-k8sbench-gate` and its scorecard.
- `bench/versioning.py`: `bench_all/gating` is in the combined bench's
  cache version. This is what retrained the suite in rsk-k8s-1-gate. Every
  arm station keys its cache on the whole of `_ALL_ARMS`, so a post-hoc
  arm's code busts the trained arms. Open: key each station on its own
  dependencies.
- `tests/test_bench_k8s.py` (new, 16 tests: the adapter, the gate's
  quantile, `gate_all`'s variants).
