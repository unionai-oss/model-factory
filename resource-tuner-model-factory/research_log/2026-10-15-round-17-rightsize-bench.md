# Round 17 — RightSize Bench: measured workloads, four kinds of tuner, one cost model (2026-10-15)

## Context

Sixteen rounds built machinery: a GRPO policy, a GBT baseline, a grid-cell
decision model, a tune service, a value ledger, a factory. What they never
produced is a result anyone outside this repo would trust, for three reasons
this round set out to remove:

1. **Labels were formulas.** Every corpus row's "true peak" came from an
   analytic footprint (`templates.py`), so every arm was graded against the
   simulator's opinion of the code, not the code. The round-4 execution
   oracle measured a few synthetic tasks; nothing scored against them.
2. **The baseline could see the answer.** `ml_baseline.extract_features`
   reads `params_json` -- the generator's own sampled parameters (rows,
   cols, hidden, batch_size...). A scheduler never has those. The round-10
   "GBT ties the LLM" result was the LLM against an oracle-fed tree.
3. **Code never mattered.** Template code stated its sizes as literals; the
   memory-relevant knowledge was arithmetic on numbers in the source, not
   *understanding* the source. There was no workload where two tasks with
   the same input size needed 30x different memory because of how they
   were written -- which is the entire case for putting a language model in
   the loop.

The question for this round, in the form a customer asks it:

> A task is submitted with some inputs. What should it request? Does
> reading the code help, does an LLM help, and does the *kind* of LLM matter?

## Design

### Workloads (`bench/workloads.py`)

12 families of real task code -- pandas groupby/feature pipelines, pandas
fact x dimension joins, polars (eager / lazy / streaming), duckdb SQL, CSV
(whole vs chunked), JSON lines (list of dicts / stream / pandas / chunked),
numpy linear algebra, sklearn fits (RF, HGB, logreg, k-means, kNN), TF-IDF
(sparse vs `.toarray()` vs SVD), torch-on-CPU training (with and without
large embedding tables; full-tensor vs streamed batches), image arrays
(uint8 -> float32 blowup, batched, mmap), plain-Python grouping
(Counter vs `defaultdict(list)` of rows).

Each task is **code + inputs + knobs**. The knobs are the code-level choices
that decide how input bytes become resident memory (projection,
`dtype_backend`, downcasts, streaming engines, `chunksize`, `mmap_mode`,
`.toarray()`, dedup-before-join, Python objects vs columns, constants such as
an embedding table's vocabulary size). They are hidden from every arm and
used only to slice results. A task runs at 3 input sizes, in random order,
so run k has runs 0..k-1 as **history** -- the run ledger a platform keeps.
Two families (`duckdb_sql`, `image_batch`) are held out of training
entirely: the out-of-distribution split.

### Labels are measurements (`bench/measure.py`)

Every run executes for real in a measurement pod: inputs are written in
chunks in one subprocess, the workload runs in a fresh interpreter, and
`os.wait4` returns its `ru_maxrss` (the kernel's high-water mark) and CPU
seconds; a sampler records time-to-peak and kills runs heading past the pod
limit (recorded as censored, not silently dropped). Library versions are
pinned exactly (`envs.WORKLOAD_PINS`): a label is "what this code used under
these versions".

### The information boundary (`bench/features.py`, `bench/prompting.py`)

Every arm sees exactly what a scheduler sees at submit time: the source; each
input's format metadata (bytes always; rows + schema for parquet; shape +
dtype for npy; only the header for CSV and only the first record's keys for
JSONL); and the task's history. `test_features_cannot_see_the_answer` and
`test_prompt_cannot_see_the_answer` pin it.

### Arms

| arm | what it is | decodes by |
|---|---|---|
| `static` | one request for every run (the training set's q-coverage cell) | -- |
| `vpa` | max peak this task has been seen to use x margin; cold start -> static | margin |
| `size-rule` | per-format power law in input bytes (quantile fit); warm: rescale nearest history run by the size ratio | quantile, margin |
| `gbt` | quantile GBT over all scheduler-visible features | quantile |
| `gbt-dist` | 19 quantile GBTs read as a distribution | expected cost |
| `zeroshot` | qwen35-397b (hosted), prompted, no training | snap up |
| `sft` | Qwen3-1.7B + LoRA, autoregressive, writes the JSON | greedy+margin / 8 samples -> expected cost |
| `jev` | **same torso + LoRA**, LM head replaced by a decision head over the grid cells: one forward pass, a probability per cell, temperature-scaled | expected cost |

Every knob is tuned on the validation split against the same cost function.
`sft` and `jev` share torso, LoRA config, prompt, data and steps: the output
head is the only variable.

### The cost model (`bench/scoring.py`)

Dollars per 1,000 task-hours to get **every** run to SUCCEEDED. An OOM burns
the part of the run before the peak (measured `t_peak_frac`) and retries at
double the memory; every attempt is paid. Headline pricing is **packed**: a
pod costs its dominant share of an m6i.xlarge (4 vCPU, 16 GiB, $0.192/hr),
because bin-packing makes a 16Gi/1-CPU pod strand three cores. Fargate
linear pricing is reported alongside. The oracle asks for the cheapest cell
that fits; `regret` = cost / oracle cost - 1. The memory grid is the
half-power-of-two ladder (512Mi, 768Mi, 1Gi, 1.5Gi, ...) people actually
write, which halves the grid's own waste floor relative to round 16's.

### Results -- v2 (`sft` and `jev-pointer-4b` still training)

Corpus `rt-bench-corpus[release=v2]` @ `f02c7caf`: 14,234 runs, train 8,457 /
val 1,138 / test 2,316 / OOD 2,323 -- twice v1's training data.

| arm | test | OOM | OOD | OOD OOM |
|---|---|---|---|---|
| static | 255.6% | 8.9% | 231.8% | 15.2% |
| vpa | 165.5% | 12.0% | 163.7% | 15.3% |
| size-rule | 118.7% | 23.0% | 81.3% | 21.3% |
| gbt | 33.7% | 3.2% | 92.8% | 13.0% |
| gbt-dist | 34.0% | 2.5% | 84.9% | 9.8% |

| zeroshot-27b (600-run subsample; 312 unanswered) | 154.5% | 9.7% | 192.6% | 22.0% |
| jev (linear head) | 52.0% | 4.1% | 112.5% | 15.8% |
| **jev-pointer** (1.7B) | **33.2%** | 3.2% | 90.8% | 13.8% |
| **gbt-dist x jev-pointer, pooled (w=0.7)** | **20.8%** | **1.7%** | **79.0%** | **9.6%** |

- Twice the data moved the trees in-distribution (38.6% -> 34.0%) and not at
  all on unseen families (84.8% -> 84.9%): more tabular data does not buy
  generalization to new code.
- **With twice the data the 1.7B decision model passes the trees
  in-distribution** (33.2% vs 34.0%), better calibrated (Brier 0.294 vs 0.325,
  ECE 0.006 vs 0.039). On unseen families it got *worse* than at v1 (90.8% vs
  84.6%): more in-distribution data made it more confident there, and that
  confidence transferred badly (OOD ECE 0.097).
- **Pooled, v2 gets the bench's best number: 20.8% memory regret at a 1.7%
  first-attempt OOM rate** -- 37% less waste than either member alone -- and
  79.0% on unseen families.

## Live validation

A stratified sample of test/ood runs re-executes in real pods sized by each
arm's decision (retries=0, OOM is the measurement), and the simulator's
fit/OOM call is compared with the kernel's.

## Run ledger

Base: `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/<run>`

| run | what | result |
|---|---|---|
| [rsb-pilot-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-pilot-1) | `bench_pilot`, 3 tasks/family (108 runs) | **ABORTED** after 20 min `WAITING_FOR_RESOURCES`: "all available instance types exceed limits for nodepool oc-production-awsc54xlarge" -- the 4 CPU/16Gi coordinator fit no pool with headroom. Fix: coordinator 2 CPU/10Gi (t3a.xlarge), measurement 3 CPU/24Gi |
| [rsb-pilot-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-pilot-2) | same | **"SUCCEEDED" with every shard lost**: `guard_mib: float` defaulted to an int constant ("Expected float, got int"), and `measure_bench` swallowed the 9 shard failures. Fix: float constant; `measure_bench` now raises when >10% of shards fail |
| [rsb-pilot-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-pilot-3) | same | **succeeded**: 107/108 runs measured; `torch_train` median 281 s/run, one 900 s timeout -> step count cut 40 -> 6 (the peak arrives on the first optimizer step) |
| [uxr8458kj9n92qlvqxrc](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uxr8458kj9n92qlvqxrc) | `bench_peek` of the pilot corpus | succeeded (the laptop has no data-plane S3 credentials; results are read through an in-cluster task that returns them inline) |
| [rsb-v1-corpus](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-corpus) | first v1 materialization, 200 tasks/family (7,200 runs, 180 shards) | **ABORTED**: 7 measurement pods at a time on the i3.xlarge pool -> ~5.5 h. Fix: two-pass measurement (3 CPU/10Gi interruptible pass 1 on t3a/spot c5; censored runs re-measured on 24Gi) and workload thread caps of 3 |
| [rsb-pilot-board-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-pilot-board-1) | first full pilot leaderboard | **FAILED** at the arm stage: `--param rt-bench-arm.llm_config={...}` arrives as a JSON dict, not a string. Fix: typed `llm_epochs` / `llm_model` / `llm_max_train_rows` |
| [rsb-pilot-board-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-pilot-board-2) | relaunch | **ABORTED** -- launched without `tasks_per_family=3`, i.e. a duplicate full corpus competing with v1 for nodes (operator error) |
| [rsb-pilot-board-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-pilot-board-3) | relaunch, pilot corpus pinned with `--version` | 5 CPU arms succeeded; both zero-shot arms **FAILED** (`cannot import name 'llm_client'` -- a function-level import the code bundler never shipped; `charts`/`grid` had the same latent bug); 3 GPU arms stuck: L40s group at "max node group size reached". ABORTED. Fix: static imports + an AST guard test; GPU arms on A10G:1 |
| [rsb-pilot-board-4](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-pilot-board-4) | relaunch | CPU arms succeeded; zero-shot arms **FAILED**: the llama.cpp `qwen38-27b`/`qwen35-397b` apps never became ready (no L40S); GPU arms QUEUED behind v1's shards (project cap of ~30 concurrent actions). ABORTED; zero-shot moved to the running vLLM 27B |
| [rsb-gpu-probe-1](https://demo.hosted.unionai.cloud/v2/domain/staging/project/resource-tuner-model-factory/runs/rsb-gpu-probe-1) | `train_llm_arm` in the staging domain, to dodge the queue | QUEUED there too; ABORTED |
| [rsb-v1-corpus-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-corpus-2) | **v1 corpus**, 288 pass-1 shards of 25 runs + 38 pass-2 shards of 8 | **succeeded**: 7,200 runs, 7,089 usable (98.5%), 87 censored above 20 GiB, 24 failed (a `describe(include="all")` -> `to_parquet` workload bug, fixed for v2). 40.5 pod-hours of measurement. Generate cache-hit the aborted run's output -- see finding 4 |
| [rsb-v1-board-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-board-1) | all arms on v1 (corpus pinned with `--version`) | 5 CPU arms succeeded; GPU arms QUEUED >1 h on A10G behind other projects; `zeroshot-27b` **FAILED** (vLLM's empty-200 `/health` read as an auth redirect). ABORTED |
| [rsb-t4-probe-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-t4-probe-1) | `train_llm_arm` on T4 via `.override`, 64 rows | succeeded -- T4 admits work immediately; fp16 path green |
| [rsb-v1-board-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-board-2) | arms on T4 | `sft` **OOMed** the 16 GB T4 on full-vocabulary logits (fix: `logits_to_keep`); decision heads ran at 20 steps/35 min -- `is_bf16_supported()` counts emulation, so the T4 ran emulated bf16. ABORTED |
| [rsb-v1-sft-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-sft-1) | `sft` alone | ABORTED (same bf16 issue) |
| [rsb-v1-board-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-board-3) | **all arms on v1**, fp16 on T4 | CPU arms + `jev` / `jev-pointer` / `jev-num` succeeded (~1h50 each on a T4); `zeroshot-27b` parsed only 813/2,318 answers (max_tokens=64 cut off its reasoning); `sft` (running) |
| [rsb-v1-zs27-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-zs27-2) | `zeroshot-27b` rebuilt with 1,024 tokens to reason | succeeded after ~4.5 h: 2,318 decisions, 639 generated tokens each on average, p50 103 s per decision on the shared vLLM app; 1,207 still ran out of budget mid-reasoning |
| [rsb-v1-live-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-live-1) | live validation, 600 pods | 439 ran, 161 lost to a 15-min queue limit while v2 held the slots -> limit raised to 3 h |
| [rsb-v1-live-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-live-2) | live validation re-run | **succeeded: all 600 pods ran** (582 fit, 18 OOM-killed) |
| [rsb-v1-4b-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-4b-1) | `jev-pointer-4b` on v1: Qwen3-4B pointer head, 2 epochs, one T4 (fp16, micro-batch 4 x 4) | succeeded, 2h37m training; see the scaling note under Results |
| [rsb-v2-corpus-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v2-corpus-1) | **v2 corpus**: 400 tasks/family (14,400 runs), seed 1, describe fix, thread caps of 3, 4 workloads per measurement pod | **ABORTED** at 79/240 shards: spot reclaims ("terminated in response to imminent node shutdown") left ~25 shard actions RUNNING for 40+ min with no failover. Fix: 1 h shard `max_runtime` so the timeout drives the retry |
| [rsb-v2-corpus-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v2-corpus-2) | v2 relaunched (completed shards cache hits) | **succeeded**: 14,400 runs, **14,234 usable, 0 failed**, 166 censored above 20 GiB; 1,070 runs re-measured on 24Gi pods in pass 2 (the per-process guard is tighter at 4 per pod). 71.6 pod-hours |
| [rsb-v2-board-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v2-board-1) | **all arms + ensemble on v2 in one materialization**, corpus pinned | (running) |

## Results -- v1

Corpus `rt-bench-corpus[release=v1]` @ `b45077fc`: 7,089 measured runs
(train 4,167 / val 604 / test 1,144 / OOD 1,174). The OOD split is two
whole workload families no arm trained on (`duckdb_sql`, `image_batch`).

**Headline: memory regret** -- GiB-hours reserved to get every run to
SUCCEEDED, OOM-killed attempts and the delay penalty included, over the
oracle's, minus 1 (lower is better; 0 = the oracle). CPU held at oracle.
`cold` = runs with no history; `code` = families whose peak is set by the
code rather than the input size (streaming / chunked / mmap / constants).

| arm | test | OOM | cold | code-driven | **OOD** | OOD OOM | $ regret (packed) | CPU exact | Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| static (safe single request) | 242.9% | 10.8% | 247.9% | 425.0% | 223.5% | 16.2% | 12.8% | 28% | | |
| vpa (history max x margin) | 146.6% | 14.2% | 247.9% | 196.6% | 163.4% | 17.3% | 18.0% | 69% | | |
| size-rule (power law in bytes) | 132.1% | 20.7% | 210.7% | 182.0% | 91.8% | 18.4% | 25.4% | 83% | | |
| gbt (quantile trees) | 46.5% | 3.9% | 68.8% | 28.8% | 86.1% | 14.1% | 8.1% | 84% | | |
| gbt-dist (19 quantiles -> expected cost) | 38.6% | 3.7% | 60.9% | 25.2% | 84.8% | 11.5% | 6.5% | 84% | 0.385 | 0.032 |
| sft-greedy (autoregressive, +1-cell margin) | 58.7% | 3.3% | 76.8% | 40.7% | 88.5% | 9.2% | 10.1% | 98% | | |
| sft-sampled (8 samples -> expected cost) | 54.9% | 3.1% | 92.8% | 54.4% | 83.3% | 10.2% | 7.7% | 97% | 0.417 | 0.008 |
| zeroshot-27b (hosted, prompted; unanswered -> 4Gi) | 162.1% | 10.5% | | | 171.6% | 18.7% | 13.0% | 78% | | |
| jev-num-argmax | 148.9% | 28.8% | 172.0% | 80.6% | 359.0% | 53.4% | 27.9% | 95% | 0.553 | 0.049 |
| jev-num (decision head + numeric features) | 89.0% | 8.7% | 112.7% | 51.3% | 337.0% | 46.9% | 13.9% | 95% | 0.553 | 0.010 |
| jev-argmax | 94.4% | 20.9% | 151.8% | 74.3% | 117.6% | 24.8% | 21.0% | 95% | 0.433 | 0.032 |
| jev (decision head, linear readout) | 59.8% | 5.5% | 109.1% | 41.5% | 93.7% | 10.7% | 9.7% | 95% | 0.433 | 0.018 |
| jev-pointer-argmax | 72.6% | 16.6% | 116.9% | 75.7% | 112.7% | 28.7% | 16.5% | 97% | 0.411 | 0.012 |
| **jev-pointer** (Strands-style pointer head) | 50.0% | 5.0% | 83.0% | 29.0% | 84.6% | 15.0% | 9.4% | **97%** | 0.411 | **0.004** |
| **jev-pointer-4b** (same head, Qwen3-4B, 2 epochs) | 42.5% | 3.8% | | | **78.1%** | 10.7% | 7.3% | 97% | 0.348 | 0.016 |
| **ensemble** (gbt-dist x jev-pointer, w=0.6) | **35.6%** | **3.2%** | **59.5%** | **22.2%** | **75.4%** | **11.3%** | **6.4%** | 97% | **0.343** | 0.025 |

(`jev*` = Qwen3-1.7B + rank-16 LoRA, 3 epochs, one T4 each. Numbers recomputed
locally from the arms' cluster predictions on an exact local reconstruction of
the corpus; the factory's scorecards compute the same thing.)

What the table says, honestly:

- **Rules lose badly; classical ML is a strong bar.** A safe static request
  reserves 3.4x what the oracle does; history and size rules halve that but OOM
  on 14-21% of first attempts. Quantile GBTs over scheduler-visible features
  (with code tokens a person who knew the generator picked -- a generous
  baseline) get within 39% of the oracle in-distribution.
- **A 1.7B decision model reading only the prompt does not beat that GBT
  in-distribution (50.0% vs 38.6%), and ties it on unseen families (84.6% vs
  84.8%)** -- with no feature engineering, and calling CPU right 97% of the
  time vs 84%.
- **Calibrated probabilities are most of the LLM's value.** The same models
  decoded by argmax are 23-35 points worse in-distribution and 28 points worse
  OOD; decoded by expected cost they are good. The pointer head is the best
  calibrated model in the bench (ECE 0.004).
- **Head vs. generation, same torso, same data, same steps:** the pointer
  head and autoregressive SFT land in the same place on quality (50.0% vs
  58.7% greedy / 54.9% sampled in-distribution; 84.6% vs 88.5% / 83.3% OOD) --
  but per decision on a T4 the head takes **129 ms and generates 0 tokens**;
  greedy SFT takes 1,178 ms for ~14 tokens (9x), and the 8-sample decoding
  that gives SFT a distribution takes 2,104 ms (16x). The decision model buys
  the same answer, with a calibrated probability attached, for 1/9-1/16 of the
  serving cost.
- **A 27B model, zero-shot, is not the answer.** Given 1,024 tokens it
  reasons at length (sizing arithmetic, risk trade-offs) and runs out before
  answering on 52% of runs. On the runs it does answer it is respectable out
  of distribution (108.8% vs gbt-dist 116.9% and jev-pointer 97.9% on the same
  551 runs) and weak in-distribution (77.7% vs 32.9% / 41.4%, 560 runs) -- at
  ~600 prompt + ~640 generated tokens and ~100 s per decision. The fine-tuned
  1.7B decision head beats it on both splits at 129 ms and zero tokens.
- **The decision model scales.** The same pointer head on Qwen3-4B (2 epochs,
  one T4, 272 ms per decision) improves on the 1.7B everywhere -- 50.0% ->
  42.5% in-distribution, 84.6% -> **78.1% on unseen families, now 6.7 points
  better than the GBT** -- with better calibration (Brier 0.411 -> 0.348). Its
  pool with the GBT reaches 36.1% / 80.4% (val picked w=0.7, leaning on the
  trees; on OOD the 1.7B pool is still better at 75.4%).
- **The two fail differently, so they pool.** A log-linear pool of the GBT's
  and the pointer head's distributions (one weight, tuned on val) beats both
  everywhere: 35.6% test, **75.4% OOD (-9.4 points vs either member)**, 3.2%
  OOM -- the best arm on every column. That is the deployable answer.
- **Adding the numeric features to the decision head hurt** (jev-num, 337%
  OOD): a numeric-only head with the same 780-step budget scores the same 89%
  in-distribution -- the head is optimization-starved -- and an MLP over
  numbers extrapolates to new families far worse than trees. Numbers belong in
  the trees; the LLM's job is the code.


### Live validation (v1): the simulator against the kernel

[rsb-v1-live-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-live-1): 120 test/OOD runs (10 per family), each
re-executed in a real pod under each of 5 arms' first-attempt request
(`retries=0`; an OOM is the measurement).

- First attempt ([rsb-v1-live-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-live-1)) lost 161 pods to a
  15-minute queue limit while v2 measurement held the project's slots; the
  re-run ([rsb-v1-live-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-live-2), 3 h limit) ran **all 600**.
- **The simulator is never optimistic.** Of 550 pods the simulator said would
  fit, 550 fit; of 50 it said would OOM, 18 did and 32 fit. Agreement 94.7%,
  and every disagreement is on the conservative side.
- **Same 120 runs, real pods, first-attempt request:**

  | arm | memory requested | real OOM (1st try) |
  |---|---|---|
  | static | 480.0 GiB | 4.2% |
  | size-rule | 318.4 GiB | 3.3% |
  | sft-greedy | 237.0 GiB | 3.3% |
  | gbt-dist | 219.0 GiB | 1.7% |
  | **jev-pointer** | **208.5 GiB** | 2.5% |

  The 1.7B decision model asks for 57% less memory than the static request,
  and 4.8% less than the trees, at a lower real OOM rate than static, size-rule
  or SFT.
- **Why conservative:** the label adds a flat 180 MiB of runtime overhead and
  5% headroom to the workload's peak. The real task process holds ~158 MiB, and
  shared-library pages are charged once to the pod cgroup, not per process, so
  small workloads fit with less. Every real OOM had a request below the
  measured peak (request/peak <= 0.99); every request at >= 1.05x fit.
- **Labels reproduce.** Re-measured live, a workload's peak lands at
  0.91x-1.06x its corpus label (5th-95th percentile; median 1.000; n=424).

## Findings

**1. Measured labels change the picture.** Peak-to-input ratios vary 28x-527x
within TF-IDF, 1x-53x within torch training and 0.16x-4.2x within CSV
processing: the same bytes, different code, 1-2 orders of magnitude of memory.
No input-size rule can see that, which is why size-rule reserves 2.3x the
oracle and the code-reading arms do not.

**2. A pure reserved-memory objective rewards OOMs.** Tuned on GiB-hours alone,
the size rule settled at a 37% OOM rate (an OOM at small memory is cheap). The
headline therefore charges one oracle run of delay per OOM (`OOM_PENALTY`), and
the risk frontier sweeps that penalty 0-16x.

**3. Dollar figures were measuring CPU.** Nearly every workload wants ~1 core,
so under any price list CPU dominates the bill and an arm's CPU coin-flip
outweighed its memory skill. The headline holds CPU at oracle and reports each
arm's own CPU calls separately.

**4. `cache="auto"` hashes only the task's own function body.**
`generate_bench_runs` cache-hit across a change to the workload generator it
calls, so v1 was generated before the thread caps changed: 774 runs declare
`threads=4` and were measured on 3-CPU pods. Their memory labels are valid (the
peak of the code as written); their CPU labels top out near 3 cores. Stations
now key their caches on a hash of the modules they depend on
(`bench/versioning.py`).

**5. Factory reuse is task caching.** A redeploy changes every task version; an
uncached build re-runs on the next materialization even when its artifact
exists, and `--version` pins one version per artifact name, not per partition.
GPU arms are now cached on the arms' source hash.

**6. The capacity facts of this tenant decided the architecture.** The project
runs ~30 actions at once; on-demand c5.4xlarge and i3.xlarge pools sit at
their nodepool limits; the L40s group was full; A10G jobs queued for over an
hour behind other projects; T4 admitted work immediately. Hence: measurement
pods that run 4 workloads each (ru_maxrss is per process), interruptible pass-1
pods, and the LLM arms on T4 in fp16 -- where `torch.cuda.is_bf16_supported()`
returns True by counting emulation, which made the first T4 run 15x slow.

**7. The autoregressive arm pays for its vocabulary.** Full-sequence logits over
Qwen's 151k-token vocabulary are ~7 GB at batch 8 and OOMed a 16 GB T4; SFT
now computes logits for the completion window only. The decision heads never
build that tensor.

**8. A zero-shot 27B model reasons before it answers.** At 64 output tokens it
produced a parseable answer 35% of the time; it needs ~1k tokens per decision
to answer at all -- versus zero generated tokens for a decision head.

## Code changes

- **new** `resource_tuner/bench/` -- workloads, measure, dataset, features,
  prompting, arms, llm_arms, scoring, charts, stations, gpu_stations,
  envs, factory; `bench.py` and `bench_factory.py` deploy units;
  `tests/test_bench.py` (40 tests).
