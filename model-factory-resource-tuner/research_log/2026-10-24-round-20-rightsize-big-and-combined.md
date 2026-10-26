# Round 20 — RightSizeCPU / RightSizeGPU at RightSizeMem's scale, and one bench for the whole request

**Date:** 2026-10-24 · **Branch:** `round-18-rightsize-cpu-gpu`

## Context

Two asks:

1. **Run RightSizeCPU and RightSizeGPU at RightSizeMem's order of magnitude.**
   RightSizeMem v2 measured 14,400 runs; the full-1 releases (round 19)
   measured 880 and 640. The `big` releases target ~10^4 each: 9,900 CPU
   runs (11 families x 450 tasks x 2) and 10,000 GPU runs (8 x 625 x 2).
2. **One benchmark and one RL training setup across all three.** A
   scheduler makes the whole request at once -- memory, CPU, and a GPU or
   none -- and the axes interact. The combined **RightSize Bench**
   (`resource_tuner/bench_all/`) joins the three measured corpora, scores
   the joint request under one cost model, and adds an RL arm trained
   against that cost model.

## Why the big releases needed new measurement plumbing

rsc-full-1's measurement shards waited, not worked: reading each shard
attempt's phase transitions, **never more than 5 CPU shards were RUNNING at
once** (mean 4.8 over 4.1 h; one shard waited 18 min for 3.4 min of work).
The 9-CPU interruptible pods fit one per spot c5.4xlarge and that node group
sat at "max node group size reached". GPU shards ran 18.5 at a time (max
35). At those rates, 10^4 runs would take ~60 h (CPU) and ~30 h (GPU).

A probe ([rt-pod-probe-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-pod-probe-1))
tried four pod shapes: a 15-CPU spot pod landed on a 16-core node and a
T4:4 pod on a 4-T4 node; 32-CPU pods (spot or on-demand) did not schedule.
The pod's cgroup is read-only, so per-workload sub-cgroups are out.

- **CPU: two workloads per pod, each pinned to its own 7 cores.** A 15-CPU
  pod (`rt-cpubench-measure-pair`) runs two slots; each slot thread sets its
  own affinity, which its children (input prep and the workload) inherit.
  The cgroup counter would sum both, so a shared run reads its own process
  tree, now keeping exited children (`measure.TreeCpu`; summing only the
  live tree made the series stall when a pool shut down). The `big` workload
  profile caps pools at 6 (below the slot) and draws time budgets of 30/45/60 s
  instead of 60/75/90 s. 8-run shards.
- **GPU: half-length budgets** (60/90/120 s instead of 120/150/180 s; a
  training loop reaches its peak in its first steps) and 6-run shards. T4:1
  pods were never the bottleneck.
- **The release name picks the profile** (`common.profile_for_release`):
  `big-*` releases use the `big` profile and the pinned pairs; every row
  carries its `profile`.

The pinned pairs were checked against the cgroup before spending 10^4 runs
on them: the same 44 runs measured both ways
([rsc-pairpilot-pair-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-pairpilot-pair-1) vs
[rsc-pairpilot-single-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-pairpilot-single-1)):
**44/44 paired labels within ±25% of the one-per-pod cgroup label, 43/44 in
the same CPU cell, median ratio 1.000** (range 0.97-1.45).

## Workload fixes (both open items from round 19)

- **`encoder_infer` with gradients on** (no `inference_mode`): every such
  run OOMed a 44 GiB L40S in round 19 (26/26), with or without
  `keep_on_gpu`, and 2/2 again in the big pilot
  ([rsg-bigpilot-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-bigpilot-1)).
  The cause was not the kept outputs: each output tensor holds its forward
  graph, so activations accumulate for the whole run. The outputs are now
  detached; the label is one batch's saved activations.
- **`keep_on_gpu`** flushes to the host every 16/64/256 batches (a code
  knob), so its peak is set by the code, not the time budget.

## RightSizeGPU's `vpa` == `size-rule` (round 19, finding 7)

Explained, not a bug. Peak GPU memory does not scale with input bytes in
this bench (full-1 train: log-log correlation -0.08; quantile slopes -0.06 /
-0.01 / +0.05 at q = 0.5 / 0.75 / 0.9). With the slope clipped at 0 the
size rule's warm path *is* vpa's (largest history peak x margin), and its
cold path, an intercept-only quantile fit, *is* static's quantile. Both
tuned to the same knobs on val.

## The combined bench

| | |
|---|---|
| corpus | `rt-bench-corpus[v2]` + `rt-cpubench-corpus[release]` + `rt-gpubench-corpus[release]`, as factory **sources**, joined by `join_corpora` |
| request | memory (RightSizeMem's 17 cells) x CPU (RightSizeCPU's 16) x GPU (none, or RightSizeGPU's 12 memory cells) |
| labels | mem rows: memory + mean cores; cpu rows: memory (ru_maxrss) + busiest 15 s window; gpu rows: peak device memory (a GPU pod's host memory and cores come with its instance) |
| cost | dollars to SUCCEEDED, one simulation: host OOM -> retry at 2x; CPU below need -> throttled replay of the measured series; CUDA OOM -> next device up; a GPU task with no GPU -> fails at start, retried on a T4; a GPU for a CPU task -> the GPU instance's price for the whole run. Retries and lateness carry the benches' delay penalties |
| headline | mean per-run regret vs the oracle; also first-attempt success, all-axes-right, per-axis accuracy |

Arms: `static`, `rules` (a GPU iff the code mentions CUDA; history x margin;
declared pools), `gbt-dist` (19-quantile trees per axis + a needs-a-GPU
classifier, no-GPU threshold tuned on val), `zeroshot-27b`, `jev-pointer`
(the pointer decision model with three heads: Qwen3-1.7B + LoRA,
cross-entropy on each measured axis, 3,500 training runs per source, 2
epochs), and **`rl-pointer`**: the supervised model, RL-tuned against the
bench's own cost model. RL is GRPO without generation -- per prompt, 16
joint actions are sampled from the three heads in one forward pass, each is
scored by the simulator (reward = -regret ratio, capped at 20), advantages
are group-normalized, and the loss is the policy gradient plus a KL penalty
(beta 0.05) to the supervised policy. A GPU task's memory and CPU log-probs
are left out of its gradient (they do not move its reward).

## Run ledger

| run | what | outcome |
|---|---|---|
| [rt-pod-probe-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-pod-probe-1) | pod-shape probe: 32-CPU spot, 32-CPU on-demand, 15-CPU spot, T4:4 | 15-CPU spot (16-core node) and T4:4 ran; both 32-CPU pods unschedulable. Aborted once read |
| [rsc-pairpilot-pair-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-pairpilot-pair-1) | `big-pilot`, 2 tasks/family (44 runs), two pinned slots per pod | succeeded, 44/44 measured (process tree) |
| [rsc-pairpilot-single-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-pairpilot-single-1) | the same 44 runs, one per pod | succeeded (cgroup); the comparison above |
| [rsg-bigpilot-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-bigpilot-1) | `big-pilot` GPU, 2 tasks/family (32 runs) | succeeded: 30 measured, median wall 93 s; 2 grad-enabled `encoder_infer` runs OOMed the L40S -> detach fix |
| [rsa-full-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsa-full-1) | combined bench on the full-1 corpora (shakedown) | **FAILED** at `join_corpora`: `bench/versioning.py` hashed `bench/measure.py` at import, and the combined bench's bundle does not ship it. Fix: every cache version is computed on demand |
| [rsa-full-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsa-full-2) | combined bench on v2 + full-1 + full-1, all arms + RL | **FAILED** at `fit_all_arm[gbt-dist]`: the arm's meta kept every list in the fitted state, and gbt-dist's state holds lists of fitted trees (`float() argument must be ... not 'HistGradientBoostingRegressor'`). The other 4 arms (incl. jev-pointer training) succeeded. Fix: `stations.knobs` keeps only JSON-serializable state; the arm test now serializes it |
| [rsa-full-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsa-full-3) | rsa-full-2 relaunched after the fix | succeeded (~8 h): 5 arms, RL (300 steps, 79 min on a T4), leaderboard -- results below |
| [rsc-big-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-big-1) | **RightSizeCPU big-1**: 9,900 runs in pinned pairs; 9 arms | succeeded in 23.1 h: 1,238 shards measured in 17.6 h (4-9 pods at a time), 9,900/9,900 runs ok; arms + scoring 5.5 h. Results below |
| [rsg-big-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-big-1) | **RightSizeGPU big-1**: 10,000 runs; 9 arms | **FAILED** at `rt-gpubench-arm[big-1, sft]`: `max_runtime of 8h0m0s exceeded`. Training finished (~980 steps, 5,129 train runs); decoding ~4,800 eval runs greedy + 8 samples (batch 2) did not. Corpus and the other 8 arms succeeded. 42 of 184 L40S re-measure shards TIMED_OUT (their runs are missing from the corpus) |
| [rsg-big-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-big-2) | relaunch with `train_llm_arm`'s max_runtime raised to 14 h in `bench/gpu_stations.py` | **aborted** after 2 min: that file is in every LLM arm's cache version (`_SHARED_LLM`), so all three LLM arms started retraining (~10 T4-hours). Fix: revert it, override the timeout at the GPU bench's call site (`train_llm_arm.override(timeout=...)` in `bench_gpu/stations.py`) |
| [rsg-big-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-big-3) | relaunch with the call-site override | succeeded in 8.4 h: jev-pointer and jev-pointer-4b cache-hit within minutes, sft retrained (4.1 h training + decoding, 8.3 h station), corpus reused from rsg-big-1. Results below |
| [rsa-big-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsa-big-1) | **combined bench big-1**: v2 + CPU big-1 + GPU big-1 corpora, 5 arms + RL | succeeded in 8.4 h: supervised decision model 5.7 h (10,500 train runs), RL 2.7 h (370 steps), zeroshot 69 min. Results below |

## RightSizeCPU big-1 (rsc-big-1)

Splits 5,628 train / 838 val / 1,634 test / 1,800 OOD; zeroshot-27b on a
297 + 300 subsample. Within ±25% of the measured peak (headline):

| arm | test | OOD | test under | test MAE | test CPU regret |
|---|---|---|---|---|---|
| static | 31.0% | 26.7% | 41.7% | 1.31 | 53.1% |
| vpa | 64.6% | 60.6% | 27.1% | 0.71 | 22.1% |
| code-rule | 73.6% | 79.4% | 41.3% | 0.89 | 19.8% |
| gbt | 92.4% | 65.8% | 4.5% | 0.35 | -0.1% |
| gbt-dist | 94.6% | 68.4% | 24.3% | 0.26 | -2.7% |
| zeroshot-27b | 68.4% | 83.7% | 25.6% | 0.59 | 33.7% |
| sft-greedy | 94.9% | 75.2% | 2.2% | 0.40 | 1.4% |
| sft-sampled | 94.1% | 83.1% | 9.1% | 0.29 | -2.1% |
| jev-pointer-argmax | 94.7% | 74.7% | 2.7% | 0.40 | 1.5% |
| **jev-pointer** | **97.9%** | 83.9% | 31.7% | **0.24** | -4.7% |
| jev-pointer-4b-argmax | 94.1% | 80.7% | 1.7% | 0.41 | 1.8% |
| **jev-pointer-4b** | 97.6% | **86.1%** | 32.1% | 0.24 | -4.5% |

1. **The decision models lead on both splits once there is data.** 1.7B
   97.9% test (small-1: 71.2%, full-1: 86.6%); 4B best on OOD at 86.1%.
   They score 96-100% on every driver.
2. **No arm improved on OOD beyond full-1** (4B 88.7% -> 86.1%, 1.7B
   85.0% -> 83.9%, gbt-dist 76.2% -> 68.4%): more data from the 9 training
   families teaches those families.
3. **Trees collapse on pyarrow_csv** (gbt-dist 48% vs 86-90% for the code
   readers); on duckdb_sql they are best (88.8%).
4. **zeroshot-27b trails on test (68.4%) because of phased code (6%)**; it
   was the best arm on small-1.
5. **code-rule's blind spots hold at scale**: io 0%, gil 63%; its small-1
   OOD 95% (40 runs) is 79.4% at 1,800.
6. **Cost-optimal decoding now nearly agrees with accuracy decoding** at
   penalty >= 1 (94-98% within ±25%), unlike small-1; below 1 it still
   throttles >93% of runs on purpose.
7. Corpus: a task's two runs agree to 0.46% median (p90 10%); median peak
   2.08 cores, max 6.33 (inside the 7-core slot); median 3.0 windows, min
   2.0 (30 s budgets); sampled CPU seconds / kernel rusage median 1.000.

8. Arm metadata (rsc-big-1): jev-pointer trained 2.6 h (1,056 steps,
   loss 1.62 -> 0.50, T = 1.25), 154 ms/decision on a T4; 4B 3.8 h (704
   steps, T = 1.14), 345 ms; SFT 2.8 h, greedy 1.5 s / 8 samples 2.7 s,
   chose margin k = 0; zeroshot-27b 2.6 s median, 230 completion tokens,
   **96 of ~900 answers unparseable** (fell back to 1 core), which is part
   of its test gap. Tuned knobs: static q 0.5 (2 cores), vpa margin 0.9,
   code-rule factor 0.8, gbt q 0.5.

Pages: [RightSizeCPU Bench (big-1)](https://claude.ai/artifact/EVom2ZzCJdA8eV4tW9VCAP),
[abstract](https://claude.ai/artifact/9ffYsA79KJNvFPKm78Geh4)
(`benchmarks/rightsize-cpu-bench-big*.html`).

## Combined-bench shakedown: rsa-full-3 (v2 + CPU full-1 + GPU full-1)

The full-1 CPU and GPU corpora are small, so this run trains on 4,325 rows
(3,500 RightSizeMem, the rest CPU and GPU) and tests on 2,551 test /
2,641 OOD runs, of which only 101 test runs are GPU runs. It is a
shakedown of the pipeline and the RL loop; the numbers to quote come from
rsa-big-1.

| arm | test regret | geo | first attempt ok | all axes right | mem / cpu / gpu src regret | OOD regret |
|---|---|---|---|---|---|---|
| static | 37.9% | 21.8% | 88.0% | 5.9% | 29.7 / 121.8 / 112.6% | 45.3% |
| rules | 31.8% | 19.3% | 92.5% | 29.6% | 32.1 / 38.1 / 15.1% | 59.4% |
| gbt-dist | 23.7% | 19.3% | 97.3% | 90.5% | 25.1 / 8.1 / 12.2% | **33.3%** |
| zeroshot-27b (292 runs) | 44.4% | 27.2% | 84.6% | 28.1% | 39.3 / 35.7 / 66.9% | 51.9% |
| jev-pointer (decoded) | 23.6% | 16.2% | 93.3% | 77.3% | 24.9 / 10.3 / 12.2% | 34.9% |
| jev-pointer-argmax | 47.1% | 29.3% | 78.7% | 69.7% | 48.6 / 47.3 / 12.2% | 69.4% |
| **rl-pointer-argmax** | **12.6%** | **6.4%** | 93.8% | 72.5% | **13.1 / 4.3** / 12.2% | 33.5% |
| rl-pointer (decoded) | 31.2% | 25.9% | 97.8% | 54.6% | 33.3 / 9.6 / 12.2% | 40.1% |

Findings (shakedown scale; to be re-tested on big-1):

1. **RL against the cost model roughly halves in-distribution regret.**
   The supervised pointer model's best decode was 23.6% (tied with
   gbt-dist, 23.7%); RL-tuned and read by argmax it is 12.6%, with the gain
   on the memory rows (24.9 -> 13.1%) and the CPU rows (10.3 -> 4.3%). The
   training reward moved from -1.27 to -1.10 (sampled-action cost ratio)
   over 300 steps; KL to the supervised policy rose to ~1.7 nats in the
   first 20 steps and then held there.
2. **After RL, argmax is the decode; the risk-aware decoder hurts.** The
   supervised model needs the cost-aware decoder (argmax 47.1% vs 23.6%
   decoded). The RL policy has already folded the asymmetric cost into its
   probabilities, so decoding on top of it adds a second safety margin
   (31.2%, first-attempt success up to 97.8% but over-provisioned).
   Calibration agrees: the RL policy's memory and CPU temperatures both
   hit the calibrator's upper bound (5.0, vs 1.14 / 1.04 supervised) --
   the RL heads are sharp, not calibrated.
3. **No OOD gain.** On unseen families RL argmax (33.5%) ties gbt-dist
   (33.3%) and jev-pointer (34.9%). Every learned arm does badly on the OOD
   CPU rows (gbt-dist 70%, jev 92%, RL 80%), where `rules` -- history x
   margin, else the declared pool size -- is best (13.4%). The code's
   declared parallelism generalizes; the learned CPU heads do not.
4. **The GPU slice is saturated at this size.** All three learned arms
   score the identical 12.2% on the 101 GPU test runs: GPU cost depends
   only on the device (4 choices), all three pick the same device on every
   run, and they miss the same 6 runs (94.1% right). RL did not move it.
5. **zeroshot-27b** is the worst learned arm (44.4%) and produced
   unparseable answers on 183 of its ~600 prompts (counted as the
   fallback request).

## RightSizeGPU big-1 (rsg-big-3; corpus from rsg-big-1)

Measurement (rsg-big-1): 1,667 six-run T4 shards in 7.6 h, then 184 L40S
re-measure shards for the runs that CUDA-OOMed the T4, in 4.1 h. Of
10,000 runs, 9,566 are usable: 5,021 train / 721 val / 1,413 test /
2,411 OOD (knn_search, recsys_train). Excluded: 219 runs whose peak
filled the device, 165 runs that OOMed the T4 and whose L40S re-measure
shard timed out, and 50 that OOMed even the L40S (> 44 GiB). Median peak
2.4 GiB (p90 11.2, max 42.6); 94.8% of usable runs fit a T4, 2.1% need
an L4, 3.1% an L40S. Median mean utilization 95% (78% of runs >= 90%;
loader_bound median 20%). Median wall 94 s.

| arm | test mem ±20% | within 1 cell | test GPU regret | test OOM | test util ±10 | OOD mem ±20% | OOD regret |
|---|---|---|---|---|---|---|---|
| static | 8.7% | 22.7% | 34.1% | 6.4% | 80.3% | 10.3% | 21.9% |
| vpa | 35.3% | 57.9% | 22.9% | 4.1% | 89.8% | 33.1% | 16.4% |
| size-rule | 37.1% | 60.3% | 22.9% | 4.1% | 89.8% | 34.8% | 16.4% |
| gbt | 60.3% | **94.0%** | **8.4%** | 1.8% | **97.4%** | 25.0% | 16.3% |
| **gbt-dist** | **71.3%** | 89.8% | 34.1% | 6.4% | 97.4% | 25.7% | 21.9% |
| zeroshot-27b (~295) | 37.5% | 53.7% | 26.2% | 5.3% | 56.0% | 37.3% | 25.2% |
| sft-greedy | 58.5% | 86.0% | 13.7% | 2.8% | 95.5% | 44.5% | 15.4% |
| sft-sampled | 60.7% | 83.0% | 34.1% | 6.4% | 95.2% | 47.0% | 21.9% |
| jev-pointer-argmax | 56.9% | 88.2% | 13.1% | **1.7%** | 95.2% | 38.9% | 16.3% |
| jev-pointer | 63.3% | 85.5% | 28.1% | 5.2% | 95.2% | 45.0% | 19.3% |
| jev-pointer-4b-argmax | 57.1% | 84.0% | 15.5% | 2.9% | 94.2% | 42.3% | **15.2%** |
| **jev-pointer-4b** | 64.9% | 81.7% | 33.6% | 6.3% | 94.2% | **50.5%** | 21.9% |

1. **Memory accuracy finally moves with data** (full-1, 104 test runs ->
   1,413): gbt-dist 50.5% -> 71.3%, jev-pointer 42.6% -> 63.3%, 4B
   42.6% -> 64.9%, sft-greedy 40.6% -> 58.5%. The rules and the 27B stay
   at 35-38%.
2. **Unseen families: the code readers lead, trees collapse.** 4B 50.5%,
   sft-sampled 47.0%, jev-pointer 45.0%; gbt-dist falls from 71.3% to
   25.7% (24% on knn_search), below size-rule (34.8%).
3. **The accuracy decoder never leaves the T4.** gbt-dist, sft-sampled and
   static score the identical 6.4% OOM / 34.1% regret: within-±20%
   decoding never picks the wide 22 / 44 GiB cells, so every run above
   15 GiB OOMs once. Argmax reads do pick them (jev argmax 1.7% OOM, 13.1%
   regret; gbt 8.4%). For dollars, read the GPU arms by argmax or by the
   cost decoder (the frontier), not by the accuracy decoder.
4. **Utilization is no longer saturated but still easy**: 78% of runs are
   >= 90% busy; trees and the trained LLMs are 94-97% within 10 points,
   the 27B 56%.
5. Arm metadata: jev-pointer 3.9 h training (5,129 runs), T = 0.99, 180
   ms/decision on a T4; 4B 5.7 h, 391 ms; sft 4.1 h, greedy 1.7 s / 8
   samples 3.1 s per decision, 22.5 generated tokens, margin k = 0, no
   parse failures. Knobs: static q 0.75 (5.9 GiB), size-rule q 0.5
   margin 1.0, vpa margin 1.0, gbt q 0.5.

Pages: [RightSizeGPU Bench (big-1)](https://claude.ai/artifact/J2FcFNpDNFQ6bgWSQqBaoJ),
[abstract](https://claude.ai/artifact/6hkCMHaTb4uVofC27LniLb)
(`benchmarks/rightsize-gpu-bench-big*.html`).

Correction (2026-10-26): the rsg-big-1 ledger row and the combined-bench
section say the runs of the 42 timed-out L40S shards are "missing from
the corpus". They are in it, as 165 censored rows (CUDA OOM on the T4, no
L40S measurement), and are excluded from scoring like the 50 runs that
OOMed the L40S itself.

## Combined bench big-1 (rsa-big-1: v2 + CPU big-1 + GPU big-1)

The GPU big-1 corpus is rsg-big-1's (its corpus step succeeded; the runs
of the 42 timed-out L40S shards are missing from it). Eval: 2,697 val /
5,363 test / 6,534 OOD runs (test: 2,316 mem, 1,634 cpu, 1,413 gpu);
the decision model trains on 3,500 runs per source. zeroshot-27b is
scored on a 300-run subsample per split. Mean per-run regret vs the
oracle (lower is better):

| arm | test regret | geo | first attempt ok | all axes right | mem / cpu / gpu src regret | OOD regret | OOD cpu src |
|---|---|---|---|---|---|---|---|
| static | 85.9% | 66.7% | 68.1% | 1.9% | 51.5 / 113.1 / 110.9% | 94.0% | 121.6% |
| rules | 34.0% | 15.0% | 94.8% | 37.3% | 36.1 / 53.4 / 8.2% | **32.5%** | **16.7%** |
| gbt-dist | 14.1% | 9.8% | 97.0% | **92.1%** | 25.2 / 1.4 / 10.4% | 33.2% | 69.1% |
| zeroshot-27b (300 runs) | 49.1% | 30.0% | 87.0% | 28.3% | 34.7 / 64.5 / 50.2% | 49.5% | 14.7% |
| jev-pointer-argmax | 24.3% | 14.6% | 89.1% | 84.0% | 42.1 / 10.8 / 10.6% | 49.7% | 95.7% |
| jev-pointer (decoded) | 9.7% | 5.2% | 95.5% | 89.3% | **16.3** / -0.3 / 10.4% | 34.1% | 71.4% |
| **rl-pointer-argmax** | **8.0%** | **2.8%** | 94.7% | 87.8% | 17.1 / **-4.6** / **7.6%** | 37.3% | 73.3% |
| rl-pointer (decoded) | 19.2% | 13.9% | 96.5% | 74.8% | 36.3 / 2.6 / 10.4% | 34.0% | 51.7% |

1. **At 2.4x the supervised data the supervised model catches up and RL's
   lead shrinks to 1.7 points** (9.7% -> 8.0% test regret; rsa-full-3:
   23.6% -> 12.6%). The decoded supervised model is now the best arm on
   the memory rows (16.3% vs RL 17.1%); RL's remaining gain is on the CPU
   rows (-0.3% -> -4.6%) and the GPU rows (10.4% -> 7.6%, the first time
   any arm moved the GPU slice). Training reward: sampled-action regret
   ~13% over the first 50 steps, ~6% over the last 50; KL to the
   supervised policy ~2.1 nats at the end (beta 0.05).
2. **On unseen families RL is worse than the model it started from**
   (OOD 37.3% vs 34.1%) and `rules` is the best arm on OOD for the first
   time (32.5%, with gbt-dist 33.2%). The OOD CPU rows carry it: rules
   16.7% (declared pools), every learned arm 52-96%. The decoded RL policy
   does best of the learned arms there (51.7%), its extra safety margin
   paying off where the model is wrong.
3. **The RL heads are still sharp, not calibrated**: memory and CPU
   temperatures again hit the calibrator's cap (5.0, 5.0; GPU 3.1), and
   the cost-aware decode on top of them over-provisions (19.2% test,
   first-attempt 96.5%). Supervised temperatures 1.09 / 1.04 / 0.94.
4. **Negative regret on the CPU rows**: the oracle asks for the cell that
   covers the busiest 15 s, and a request a cell lower throttles briefly
   for less than the core-hours it saves. RL found this (-4.6%), as the
   decoded supervised model nearly did (-0.3%). The oracle is the label,
   not the cost minimum; per-run regrets on the CPU source should be read
   as "at or below the label's cost".
5. **gbt-dist is the most accurate request (92.1% all axes right) at
   14.1% regret**: trees get each axis right more often, but miss by more
   when they miss on the memory rows (25.2% mem-source regret vs 16-17%).
6. zeroshot-27b: 206 of ~900 answers unparseable (fallback 4Gi / 1 core
   / no GPU), 2.4 s median, 46 completion tokens; 49.1% test regret.
   No arm asked for a GPU on a CPU task, and every arm except static and
   zeroshot gave every GPU task a GPU.

Pages: [RightSize Bench (big-1)](https://claude.ai/artifact/AGe3ANw4KKjGfVfFDcFeqM),
[abstract](https://claude.ai/artifact/LJceQj2Na4jwmmdPhyYx1u)
(`benchmarks/rightsize-bench-combined*.html`).

## Code changes

- `bench_cpu/workloads.py`, `bench_gpu/workloads.py`: workload profiles
  (`big`: CPU pools <= 6, budgets 30-60 s; GPU budgets 60-120 s);
  `encoder_infer` detaches grad-enabled outputs and flushes kept outputs.
- `bench_cpu/measure.py`: `TreeCpu` (exited children kept), `slot_cpus`,
  `shared=` (process-tree source), `ru_cpu_s` recorded.
- `bench_cpu/stations.py` / `envs.py`: `measure_cpu_pair_shard` on a 15-CPU
  pod, `pairing=` override; `RELEASE_TASKS["big"] = 450`.
- `bench_gpu/stations.py`: `RELEASE_TASKS["big"] = 625`, 6-run shards for big.
- `bench/common.py`: `profile_for_release`. `bench/versioning.py`: lazy
  versions; the combined bench's caches.
- `bench_all/` (new): grid, dataset, features, prompting, scoring, arms,
  policy (three-head pointer, supervised + GRPO), envs, stations, factory;
  `bench_all.py`, `bench_all_factory.py` entrypoints.
- `bench_gpu/stations.py`: the GPU bench calls `train_llm_arm` with a 14 h
  max_runtime (rsg-big-1's sft arm ran past 8 h). A timeout override does
  not change the cache key; editing a hashed module does (rsg-big-2).
- `bench_all/stations.py`: `knobs` -- an arm's meta keeps only its
  JSON-serializable fitted state (rsa-full-2).
- Tests: `tests/test_bench_all.py` (cost-model ordering, information
  boundary, every arm, a supervised + RL step on a tiny random Qwen3, bundle
  guard), big-profile / tree-CPU / encoder tests in the CPU and GPU suites.
