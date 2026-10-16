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

### Live validation

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
| [rsb-v1-corpus-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-v1-corpus-2) | **v1 corpus**, 288 pass-1 shards of 25 + 38 pass-2 shards | (running) |

## Results

(pending)

## Findings

(pending)

## Code changes

- **new** `resource_tuner/bench/` -- workloads, measure, dataset, features,
  prompting, arms, llm_arms, scoring, charts, stations, gpu_stations,
  envs, factory; `bench.py` and `bench_factory.py` deploy units;
  `tests/test_bench.py` (40 tests).
