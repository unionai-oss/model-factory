# Round 21: RightSizeGPU big-2 (the whole machine ladder) and trees x LLM hybrids on every bench

## Context

Two questions from round 20.

1. **RightSizeGPU's workloads never needed more than a T4.** In big-1, 94.8%
   of usable runs fit a T4 (2.1% an L4, 3.1% an L40S), so the device
   choice was nearly trivial and the accuracy decoder never left the T4.
   Does the bench still rank the arms the same way when the workloads
   span every GPU shape demo.hosted provisions, including multi-GPU nodes?
2. **The trees and the LLMs fail differently** (GPU big-1: gbt-dist 71% in
   distribution, 26% on unseen families; the decision models 63-65% and
   45-51%). Does combining them beat both? Every LLM arm (SFT /
   autoregressive, decision head 1.7B and 4B; the combined bench's RL
   policy) is run two ways with the bench's own gbt-dist:
   - **stacked** (`stack-<arm>`): the trees' prediction is written into the
     LLM's prompt; training rows carry out-of-fold predictions (5 folds by
     task), so the LLM learns how far to trust the trees on runs they did
     not fit;
   - **pooled** (`pool-<variant>`): a logarithmic opinion pool of the two
     calibrated distributions, the weight tuned on val.

The benches run with them: RightSizeCPU big-1 (same corpus, every arm
re-fit plus the stacked arms and the pool), RightSizeGPU big-2 (new
workloads, new corpus), and the combined bench big-2 (v2 + CPU big-1 +
GPU big-2, with the stacked decision model and RL from it).

## The machine ladder (from the cluster's GPU node pools, 2026-10-26)

| shape | instance | GPUs | usable GiB | $/h (on-demand) |
|---|---|---|---|---|
| T4 | g4dn.xlarge | 1 x T4 | 15 | 0.526 |
| L4 | g6.2xlarge | 1 x L4 | 22 | 0.978 |
| A10G | g5.2xlarge | 1 x A10G | 22 | 1.212 (never cheapest) |
| L40S | g6e.2xlarge | 1 x L40S | 44 | 2.242 |
| 4xL4 | g6.12xlarge | 4 x L4 | 88 | 4.602 |
| 8xT4 | g4dn.metal | 8 x T4 | 120 | 7.824 |
| 4xL40S | g6e.12xlarge | 4 x L40S | 176 | 10.493 |

A multi-GPU machine only holds more for code that spreads its model over
every visible GPU (`sharded`); other code uses one of its GPUs and pays
for all of them. Fit on a shape a run was not measured on is modelled
(`grid.fits`): one GPU holds the whole peak, or a sharded run's share,
(peak - 0.5 GiB context) x imbalance / n + 0.5, fits each GPU, with the
imbalance measured where the run was measured on several GPUs (1.25
otherwise).

## Run ledger

| run | what | outcome |
|---|---|---|
| [rt-gpuprobe-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-gpuprobe-1) | one pod per GPU shape (T4, 8xT4, L4, 4xL4, A10G, L40S, 4xL40S) | T4, 8xT4, L4, 4xL4 scheduled within 2 min; A10G in 19 min; both L40S shapes still waiting at 25 min (Karpenter claims for g6e.2xlarge / g6e.12xlarge created at 15:28). Hence the measurement ladder below |
| [rsg-big2pilot-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-big2pilot-1) | big-2 pilot: 4 tasks/family (80 runs), every measurement pass | **aborted** in its L40S pass: the 4xL40S shard would have queued up to 4 h, and the probe's 4xL40S pod never got a node in its hour (MaxQueuedTimeExceeded). T4, 4xL4 and 8xT4 passes worked (e.g. sharded recsys_train measured across 8 T4s at 49 and 64 GiB; mlp_tabular four at a time on 4xL4). Fix: 4xL40S shards for single-GPU runs queue 1 h, then fall back to single-L40S shards |
| [rsg-big2pilot-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-big2pilot-2) | the same pilot, relaunched with the fallback | T4 / 4xL4 / 8xT4 shards cache-hit (the stations are not in the corpus cache version); 70% of the 80 runs fit a T4, and the training family that teaches sharding (lm_finetune) never left it (7-12 GiB, matching params x 16 B + activations). Fix: lm_finetune d 2048-5120 (0.6-7B params), larger cnn_train and encoder_infer (by estimate lm_finetune now 25% T4 / 22% L4 / 39% L40S / 14% multi-GPU). The L40S fallback works: the 4xL40S per-GPU shard timed out in its 1 h queue, then 2 single-L40S shards measured the 7 runs that outgrew an L4 (gpt_train 43.0 GiB, blocked_gemm 41.0, knn_search 26.0; 3 knn / gemm runs still OOM at 44 GiB, censored at the top of the single-GPU ladder). Succeeded |
| [rsg-big2-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-big2-1) | **RightSizeGPU big-2**: 10,000 runs, 9 arms + 3 stacked + pool | running. T4 pass: 1,659 of 1,667 shards succeeded; 8 shards (48 runs, 0.5%) failed all 3 attempts with the **pod** OOMKilled (exit 137): a large mlp_tabular run's host RAM outgrew the T4 pod's ~13 GiB and took its 5 shard-mates with it (e.g. action aj813mr9dkoomysu7r2fe5fy7). Those runs drop out, slightly thinning the host-RAM-heavy end. Fix for the next release: cap each measured child's address space so the child dies, not the pod. Then 459 8xT4 shards (about 1,840 sharded runs past a T4) and 118 4xL4 shards queued; the 8xT4 pass is bounded by the g4dn.metal pool's 5 nodes. **Multi-GPU capacity mostly did not come:** all 118 4xL4 shards timed out in the queue (2 h x 3 attempts, "didn't match Pod's node affinity/selector", although 4xL4 scheduled in 2 min in the probe); 229 of 459 8xT4 shards timed out; 4xL40S scheduled 6 of 340 whole-node shards and 0 of 157 per-GPU shards. Single-GPU runs past a T4 fell through to single-L40S shards (469 succeeded), so they were measured on an L40S rather than an L4. Corpus: 8,380 usable, 1,569 censored, 51 failed (48 lost shards). Measured on T4 7,056 / L40S 1,324; 815 runs spread over several GPUs. Arms: 6 LLM trainings (stack-sft 9+ h, the longest), pool, leaderboard; **succeeded** |
| [rsa-big-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsa-big-2) | **combined bench big-2**: v2 + CPU big-1 + GPU big-2; arms static, rules, gbt-dist, zeroshot-27b, jev-pointer, stack-jev-pointer; RL from jev-pointer and from stack-jev-pointer; per-axis pool | succeeded. Test n=5,095, unseen n=6,416. The local watcher died once on an API timeout (the run did not); the watcher now retries |
| [rsc-big-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-big-2) | **RightSizeCPU big-1, round 21**: corpus reused (cache hit); 9 arms re-fit + 3 stacked + pool | succeeded (finished 18:11). Single arms reproduce round 20 to within ±1.5 pt in distribution (test); the hybrids are below |

## Findings

### RightSizeGPU big-2 corpus: the ladder is no longer trivial, but its top end is censored

Cheapest shape a usable run fits (8,380 runs): T4 6,256 (74.7%), L4 548
(6.5%), L40S 1,245 (14.9%), 4xL4 214 (2.6%), 8xT4 74 (0.9%), 4xL40S 3,
none 40. Big-1 was 94.8% T4. Only the three sharded families reach the
multi-GPU shapes (llm_generate up to 127.5 GiB, recsys_train 122.5,
lm_finetune 83.6). The 1,569 censored runs are concentrated where
multi-GPU shards never got a node: lm_finetune 433, llm_generate 358,
encoder_infer 211, recsys_train 210. A censored run's lower bound is
known (it ran out of memory on the largest shape it reached). Even so,
the corpus under-samples exactly the large, sharded runs this release
was built to add. The capacity, not the workloads, was the limit.

### Combined bench big-2 (rsa-big-2): the best test regret yet, and the worst unseen

Mean regret against the oracle over memory + CPU + GPU (lower is
better). Test has 5,095 runs and unseen 6,416. "All right" means all
three axes were picked correctly.

| arm | test regret | test $ regret | test all right | unseen regret | unseen $ regret | unseen 1st try ok |
|---|---|---|---|---|---|---|
| rules | 39.4% | 55.6% | 32.0% | 48.0% | 120.1% | 87.3% |
| gbt-dist | 15.4% | 31.7% | 90.4% | 90.8% | 107.9% | 85.2% |
| jev-pointer | 14.0% | 43.3% | 84.3% | 58.8% | 112.0% | 77.6% |
| rl-pointer-argmax | 9.8% | 39.5% | 82.0% | 47.8% | 97.3% | 77.2% |
| rl-pointer | 20.3% | 55.4% | 67.5% | **35.5%** | 113.8% | 87.7% |
| stack-jev-pointer | 10.1% | 29.0% | 91.0% | 94.6% | 106.9% | 82.3% |
| **stack-rl-pointer-argmax** | **3.2%** | **13.0%** | 92.2% | 102.0% | 123.6% | 86.1% |
| stack-rl-pointer | 16.1% | 25.6% | 78.5% | 76.9% | 99.7% | **93.5%** |
| pool-jev-pointer (w_trees 0.4) | 6.9% | 28.7% | 91.1% | 52.2% | 94.8% | 81.1% |
| pool-rl-pointer (w_trees 0.6) | 13.3% | 32.6% | 86.9% | 53.1% | **87.7%** | 85.2% |

- **In distribution, the trees and the RL policy compound.** RL from the
  stacked decision model, read by argmax, reaches 3.2% test regret, a
  third of the best single arm (rl-pointer-argmax 9.8%; round 20's best
  was 8.0%). It also has 13.0% $ regret and 92.2% of runs with all three
  axes right. The pool of the plain decision model with the trees
  (6.9%) is the best of the hybrids that do not use RL.
- **On unseen families the stacked arms are the worst learned arms**:
  regret is 77-126%, against 35.5% for rl-pointer and 48.0% for rules.
  The GPU axis drives this (stack-rl-pointer-argmax: 208% GPU regret on
  unseen against 42% for rl-pointer-argmax), where the trees fail
  hardest (gbt-dist: 167%). RL from a stacked start does not undo the
  dependence on the hint. It sharpens it, because the hint is right in
  every rollout.
- **The pool is the compromise.** pool-jev-pointer roughly halves the
  decision model's test regret (14.0% to 6.9%) and also cuts its unseen
  regret (58.8% to 52.2%). It is the only hybrid that does better than
  its LLM member on both splits. That is because the combined pool's
  weight was tuned on a val split that includes the memory and CPU
  axes, where the trees hold up better. pool-stack-jev-pointer tuned
  w_trees to 0.0, so it is stack-jev-pointer unchanged.
- Across rounds, unseen regret is higher for every arm than in round 20
  (gbt-dist 33.2% to 90.8%), because GPU big-2's unseen families include
  the multi-GPU llm_generate and recsys_train. Compare arms within a
  round, not across.

### RightSizeGPU big-2 (rsg-big2-1): the same split, sharper

GPU memory predicted within ±20% (n = 1,202 test runs, 2,390 OOD runs
from recsys_train, knn_search and llm_generate); $ regret is the cost
over the oracle's machine, after the climb:

| arm | test | test $ regret | test OOM | OOD | OOD $ regret | OOD OOM |
|---|---|---|---|---|---|---|
| gbt-dist (the anchor) | 71.9% | 48.5% | 13.1% | 9.8% | 93.5% | 6.7% |
| jev-pointer | 58.2% | 65.3% | 14.7% | 33.6% | 155.8% | 24.9% |
| jev-pointer-4b | 55.5% | 50.8% | 7.9% | **42.1%** | 98.5% | 16.5% |
| sft-sampled | 53.1% | 53.2% | 9.2% | 35.8% | 106.4% | 14.8% |
| stack-jev-pointer | 73.4% | 43.8% | 11.6% | 20.5% | 87.5% | 6.0% |
| stack-jev-pointer-argmax | 59.3% | 24.3% | 4.2% | 15.9% | 129.3% | 2.9% |
| pool-jev-pointer (w_trees 0.7) | 73.7% | 47.2% | 13.1% | 27.7% | **84.7%** | 7.0% |
| pool-stack-jev-pointer (w_trees 0.3) | **74.8%** | 46.5% | 12.6% | 16.2% | 90.9% | 6.3% |

- **In distribution, the hybrids mostly beat both of their members.**
  6 of the 9 decoded hybrids do; stack-jev-pointer-4b and pool-sft-sampled
  are 0.1 pt below the trees, and stack-sft-sampled (65.0%) is clearly
  below them. The lowest test $ regret is pool-stack-jev-pointer-argmax
  (23.4%). pool-stack-jev-pointer reaches 74.8%, against 71.9% for the trees and
  58.2% for the decision model alone. Stacking lifts the decision model
  by 15 points (58.2% to 73.4%), more than on CPU, because here the trees
  are the stronger member.
- **On unseen families the trees collapse (9.8%) and drag every hybrid
  down.** The 4B decision model alone is best at 42.1%; the hybrids land
  at 5-28%. This is the CPU finding again, more extreme. The pool weight
  is tuned on val, where the trees are the better member, so it settles
  at 0.7-0.9 for the plain LLMs.
- **The exception is cost and OOM on unseen families.** The hybrids keep
  the trees' habit of asking big. They OOM 3-7% of the time against
  15-25% for the LLMs alone. pool-jev-pointer's $ regret of 84.7% is the
  lowest of any arm on OOD. So a hybrid is less accurate on new families
  but cheaper after the climb, because a first-try OOM costs a full rerun.
- The corpus caveat above applies: the largest sharded runs are mostly
  censored, so the OOD slice under-samples the multi-GPU shapes.

### RightSizeCPU big-1 (rsc-big-2): the hybrids win in distribution and lose on unseen families

Within ±25% of the oracle's peak cores (n = 1,634 test runs, 1,800 OOD
runs, i.e. runs from unseen families):

| arm | test | test under | OOD | OOD under |
|---|---|---|---|---|
| gbt-dist (the anchor) | 94.6% | 24.3% | 68.4% | 34.3% |
| jev-pointer | 97.7% | 32.0% | 85.6% | 34.2% |
| jev-pointer-4b | 97.9% | 28.5% | **86.3%** | 34.7% |
| stack-jev-pointer | **98.5%** | 30.4% | 71.7% | 41.7% |
| stack-jev-pointer-4b | 97.2% | 32.9% | 76.0% | 41.4% |
| stack-sft-greedy | 96.0% | 1.8% | 60.8% | 15.6% |
| pool-jev-pointer (w_trees 0.7) | 97.6% | 29.1% | 73.1% | 39.4% |
| pool-jev-pointer-4b (w_trees 0.5) | 98.3% | 28.3% | 82.2% | 39.6% |
| pool-stack-jev-pointer (w_trees 0.4) | 98.2% | 29.2% | 69.0% | 43.4% |

- **In distribution, the hybrids add up to 0.8 pt** (stack-jev-pointer
  98.5% vs jev-pointer 97.7%; pool-jev-pointer-4b 98.3% vs 97.9%) and
  stack-sft-greedy trims the throttle rate to 1.8%.
- **On unseen families every hybrid is worse than its LLM alone**, by 4
  to 17 pt (stack-jev-pointer 71.7% vs 85.6%; stack-sft-sampled 68.4% vs 84.4%), and lands near the trees
  (68.4%). Stacked models learned to copy the hint: out-of-fold hints are
  right 95% of the time in distribution, so following them is what the
  training loss rewards. The pool weight is tuned on val, which is in
  distribution: trees alone miss 5.0% of val and the LLM 2.0%, so
  w_trees settles at 0.4-0.7, a weight the trees do not earn on new
  families.
- Neither combination tells the LLM *when* the trees are out of their
  depth. Untested next steps: tune the pool weight on held-out families
  (folds by family rather than by task), or give the stacked arm hints
  from trees fit without the run's family, so the hint is as wrong in
  training as it is on new families.

### Summary across the three benches

Feeding the trees into the LLM, by stacking or pooling, is the best
option on every bench for workloads like the training set. CPU: 98.5%
within ±25%. GPU: 74.8% within ±20%. Combined: 3.2% regret. It is the
worst learned option on workloads unlike anything seen in training,
because both methods learn how far to trust the trees from data where
the trees are right. The next round should tune that trust on held-out
families: cross-fit by family rather than by task, and tune the pool
weight on a family-held-out val. Then rerun these three benches.

## Code changes

- `bench_gpu/grid.py`: the machine ladder as `Device(name, n_gpus,
  per_gpu_gib, price, instance)`; cells to 176 GiB; `fits`,
  `per_gpu_need`, `oracle_device`, `oracle_cell` (a sharded run whose
  share does not fit 4xL4 has to ask past 88 GiB to land on 8xT4).
- `bench_gpu/scoring.py`: `simulate` climbs shapes by `grid.fits`; the
  oracle is the cheapest shape a run fits.
- `bench_gpu/measure.py`: NVML on every GPU a run can see; `summarize`
  folds per-GPU peaks into one single-GPU-equivalent peak (less the extra
  CUDA contexts) and an imbalance; `measure_many` runs one workload per
  GPU (`per_gpu`) or one across all (`whole`).
- `bench_gpu/stations.py` / `envs.py`: the measurement climbs
  (`_climb`): single-GPU code T4 -> 4xL4 (one run per L4) -> 4xL40S (one
  run per L40S); sharded code T4 -> 8xT4 (the run across all 8) -> 4xL40S
  (across all 4).
- `bench_gpu/workloads.py`: single-GPU families scaled up; new
  `lm_finetune` (train; layers placed over every GPU, full / last-4 /
  head-only fine-tuning, AdamW / SGD / Adafactor) and `llm_generate`
  (held out; fp16 decoder with a preallocated KV cache, layers over every
  GPU); `recsys_train` (held out) shards its embedding tables by row over
  every GPU. 10 families, `big` = 500 tasks x 2 runs; budgets 45-90 s.
- `bench/hybrid.py` (new): `crossfit`, `hint_text`, `log_pool`,
  `pool_variants`; `bench/spec.py`: `with_hint` puts the hint before the
  options (the pointer head's offsets, counted from the end, hold --
  `tests/test_hybrid.py`).
- `bench/gpu_stations.py`: `train_llm_arm` takes `stack-<arm>`; CPU / GPU
  stations: `STACK_ARMS`, `pool_<bench>_arms`, leaderboards take the pool
  scorecard; factories: `rt-<bench>bench-pool[-scorecard]`.
- `bench_all`: `stack-jev-pointer` (hints in the three-axis prompt; saved
  with the model so its RL phase prompts the same), `rt-allbench-rl-stack`,
  `rt-allbench-pool` (per-axis pool, one weight, no-GPU threshold tuned),
  CPU corpus pinned to big-1 (`CPU_RELEASE`); the GPU label is the
  oracle's cell.
