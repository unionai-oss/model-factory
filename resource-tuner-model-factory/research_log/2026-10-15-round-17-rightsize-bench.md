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
| [rsb-pilot-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsb-pilot-1) | generate + measure, 3 tasks/family (108 runs) | (pending) |

## Results

(pending)

## Findings

(pending)

## Code changes

- **new** `resource_tuner/bench/` -- workloads, measure, dataset, features,
  prompting, arms, llm_arms, scoring, charts, stations, gpu_stations,
  envs, factory; `bench.py` and `bench_factory.py` deploy units;
  `tests/test_bench.py` (40 tests).
