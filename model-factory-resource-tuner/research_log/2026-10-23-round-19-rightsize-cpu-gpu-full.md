# Round 19 — RightSizeCPU and RightSizeGPU full releases

**Date:** 2026-10-23 to 2026-10-24 · **Branch:** `round-18-rightsize-cpu-gpu`

## Context

Round 18's small releases ran end to end, but their learned arms were
data-starved: RightSizeGPU's decision head fit a constant cell from 60
training rows, and no GPU arm beat 55% within ±20%. This round asks
whether the rankings hold at 4-5x the data, and adds the 4B decision model
(`jev-pointer-4b`, RightSizeMem's best single arm) to both benches.

## Changes before launch

- **Release size comes from the release name** (`bench/common.py`
  `tasks_for_release`): `small-*` keeps 10 (CPU) / 8 (GPU) tasks per
  family, `full-*` is 40. The factories no longer pass a size.
- **3-run measurement shards** (were 6 CPU / 5 GPU): rsc-small-1 had 4
  shards spot-reclaimed twice each, each reclaim discarding up to 5
  finished runs.
- **Shard `max_runtime`** 3600 s -> 1200 s (CPU, ~6 min of work) and
  1800 s (GPU, ~10-12 min), so a stranded shard retries sooner.
- **`jev-pointer-4b`** added to both benches' LLM arms and leaderboards.
- Stations redeployed, then both factories (round 18: factory deploy pins
  station versions).

Not changed: GPU utilization is still NVML `utilization.gpu`. DCGM's
SM-active is the better label but is a DCGM profiling metric; NVML cannot
read it on a T4 (GPM needs Hopper), so it cannot be measured in-pod.

## Run ledger

| run | what | outcome |
|---|---|---|
| [rsc-full-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsc-full-1) | **RightSizeCPU full-1**: 11 fam x 40 tasks x 2 runs = 880 runs, 294 shards of 3; 9 arms | **succeeded** in 4 h 44 min. All 880 runs measured (18.8 h of workload wall time). 9 shards were retried once (spot), none twice. LLM arms trained 21-36 min each on a T4 |
| [rsg-full-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsg-full-1) | **RightSizeGPU full-1**: 8 fam x 40 tasks x 2 runs = 640 runs, 214 T4 shards + 16 L40S shards; 9 arms | **succeeded** in 2 h 15 min (26.6 h of workload wall time). 612 usable; 28 failed even on the L40S (26 encoder_infer, 2 cnn_train); 33 measured on L40S; 17 filled the device. No shard retried |

## Results

Test n=134 (CPU) / 104 (GPU), unseen families n=160 / 160. A 95% interval
on a share near 85% at n=134 is about ±6 points; near 40% at n=104, about
±9.

### RightSizeCPU full-1 (880 runs; train 524)

| arm | test within ±25% | test under | test MAE | test cpu regret | **unseen within ±25%** | unseen MAE |
|---|---|---|---|---|---|---|
| static | 40.3% | 38.1% | 1.19 | 43.2% | 31.2% | 1.46 |
| vpa | 67.9% | 19.4% | 0.65 | 16.7% | 59.4% | 0.95 |
| code-rule | 72.4% | 44.8% | 0.77 | 15.2% | 84.4% | 0.69 |
| gbt | **89.6%** | 7.5% | 0.34 | 0.7% | 65.0% | 0.77 |
| gbt-dist | 88.8% | 24.6% | 0.33 | -0.2% | 76.2% | 0.70 |
| zero-shot 27B | 74.6% | 21.6% | 0.53 | 23.3% | 83.8% | 0.63 |
| SFT 1.7B, greedy | 85.1% | 4.5% | 0.55 | 8.7% | 74.4% | 0.66 |
| SFT 1.7B, 8 samples | 83.6% | 6.7% | 0.44 | 3.2% | 83.8% | 0.57 |
| decision 1.7B, argmax | 81.3% | **1.5%** | 0.66 | 10.6% | 67.5% | 0.88 |
| decision 1.7B, pointer | 86.6% | 9.7% | 0.35 | -0.1% | 85.0% | 0.60 |
| decision 4B, argmax | 86.6% | 5.2% | 0.55 | 7.6% | 75.0% | 0.67 |
| **decision 4B, pointer** | **89.6%** | 20.9% | **0.27** | -1.9% | **88.7%** | **0.55** |

Test by driver (within ±25%; n = threads 76, gil 12, io 14, phases 12,
processes 8, serial 12): the 4B decision head is 91% / 83% / 71% / 100% /
88% / 100%; gbt-dist 86% / 100% / 71% / 100% / 100% / 100%; zero-shot 27B
86% / 75% / 64% / **8%** / 50% / 100%; code-rule 91% / 33% / 0% / 67% / 50%
/ 100%.

### RightSizeGPU full-1 (612 usable runs; train 310)

| arm | test mem within ±20% | test within 1 cell | test MAE (GiB) | test $ regret | test OOM | unseen mem within ±20% | unseen util within ±10 |
|---|---|---|---|---|---|---|---|
| static | 9.9% | 30.8% | 3.83 | 42.6% | 5.8% | 19.0% | 92.5% |
| vpa | 36.6% | 65.4% | 2.70 | 21.3% | 2.9% | 34.8% | 96.3% |
| size-rule | 36.6% | 65.4% | 2.70 | 21.3% | 2.9% | 34.8% | 96.3% |
| gbt | 45.5% | **76.0%** | **1.96** | 24.9% | 3.8% | 29.1% | 78.1% |
| **gbt-dist** | **50.5%** | 74.0% | 2.14 | 42.6% | 5.8% | 32.3% | 78.1% |
| zero-shot 27B | 34.7% | 53.8% | 3.70 | 39.0% | 5.8% | 31.6% | 46.3% |
| SFT 1.7B, greedy | 40.6% | 59.6% | 3.20 | 21.3% | 2.9% | **37.3%** | 96.3% |
| SFT 1.7B, 8 samples | 41.6% | 59.6% | 3.00 | 42.6% | 5.8% | 34.8% | 95.6% |
| decision 1.7B, pointer | 42.6% | 62.5% | 2.78 | 42.6% | 5.8% | 34.2% | 92.5% |
| decision 4B, pointer | 42.6% | 59.6% | 3.15 | 42.6% | 5.8% | 27.8% | 92.5% |

## Findings

1. **At 4x the data, RightSizeCPU's ordering flips: learned models beat
   the zero-shot 27B and the code-rule.** The 4B decision head is best on
   both splits (89.6% / 88.7% within ±25%, MAE 0.27 / 0.55 cores), with
   quantile trees level on test (89.6% / 88.8%) but 12-24 points behind on
   unseen families. The zero-shot 27B fell from 78.8% to 74.6% on test (it
   did not change; the small-release test set was easier) and is now
   behind every fine-tuned arm. Round 18's reading ("27B strongest on
   balance") was a small-sample artifact.
2. **Reading code is what generalizes.** On unseen families the arms that
   read the source (4B 88.7%, 1.7B 85.0%, code-rule 84.4%, 27B 83.8%, SFT
   sampled 83.8%) lead the trees (65.0% / 76.2%). In-distribution the
   trees catch up because 44 hand-picked code tokens cover the training
   families' pool APIs.
3. **The 27B still cannot do phased code** (8% on serial-then-parallel,
   same as round 18's 12%): it predicts the average, not the parallel
   peak. Every fine-tuned arm learned it (75-100%). The code-rule still
   fails the GIL (33%) and I/O (0%).
4. **Accuracy decoding beats argmax again** for both decision heads (+5 /
   +3 points on test, +17.5 / +13.7 on unseen), the RightSizeMem pattern,
   at the cost of more under-predictions (20.9% vs 5.2% for the 4B). The
   argmax heads have the fewest under-predictions of any arm (1.5% for
   1.7B).
5. **Negative CPU regret is expected, not a bug.** gbt-dist, both decision
   heads and the 4B come in 0.1-1.9% *below* the oracle's core-seconds:
   the oracle requests the cell covering the peak window, and a run that
   sits at its peak only part of the time is cheaper slightly throttled
   (round 18, finding 5). This is why accuracy is the headline.
6. **RightSizeGPU memory accuracy stays near 50% at 5x the data.** Best on
   test is gbt-dist (50.5%), best on unseen is SFT greedy (37.3%); the
   decision heads (42.6%) no longer collapse to one cell (their argmax and
   expected decodings now differ) but do not beat trees. The 4B adds
   nothing over the 1.7B (42.6% both; 27.8% vs 34.2% unseen). The GPU
   label is harder than CPU: within a family, peaks span 2.5-18x with code
   choices (batch, AMP, checkpointing, what stays on device) that interact
   multiplicatively.
7. **`vpa` and `size-rule` scored identically on every GPU metric.** Both
   use the task's own history when it exists, and validation tuning may
   have chosen settings where the two coincide. Not yet checked; it does
   not change the ordering, but it should be confirmed before the page
   names them separately.
8. **28 GPU runs are unmeasurable as written.** 26 are `encoder_infer`
   with `keep_on_gpu`: outputs accumulate for the whole time budget, so
   peak memory grows with run length, not with code, and exceeds 44 GiB.
   That is a workload-design flaw (the label depends on the 2-3 min budget),
   not a property of the task; the family should cap its output count.
9. **Utilization is still saturated** (498 of 612 runs at 90-100%); a
   constant scores 87-93% within ±10 points. No arm beats the training
   median meaningfully; the zero-shot 27B is worst (46%).
10. **3-run shards held up.** 9 of 294 CPU shards were retried once
    (spot), none twice; no GPU shard retried.

## Code changes

- `bench/common.py`: `tasks_for_release`; `bench_cpu`/`bench_gpu`
  stations: `RELEASE_TASKS`, 3-run shards, tighter `max_runtime`,
  `jev-pointer-4b` arm and leaderboard rows; factories stop passing a size.
- `tests/test_bench_cpu.py`: release-prefix sizing test (427 tests pass).
- Benchmark pages for the round-18 small releases (`benchmarks/rightsize-cpu-bench*.html`,
  `benchmarks/rightsize-gpu-bench*.html`), published privately; not yet
  updated with these full-release numbers.

## Open / next

- Update the four benchmark pages with full-1 (CPU's headline changed).
- Pool gbt-dist with the 4B decision head on CPU (RightSizeMem's best
  policy); the two err on different drivers (trees: unseen APIs; LLM:
  under-prediction).
- GPU: cap `encoder_infer`'s accumulated outputs; investigate vpa ==
  size-rule; try a utilization label that varies (SM-active via a DCGM
  sidecar, or more input-bound families).
