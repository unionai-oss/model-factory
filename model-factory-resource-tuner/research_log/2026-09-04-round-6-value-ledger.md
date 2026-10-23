# 2026-09-04 — round 6: value ledger + multi-workflow demo

## Context

Round 5 proved impact once, in a controlled A/B. This round makes the
value *continuously measurable*: the tune service now keeps an
append-only ledger of every proposal and every reported outcome, and its
dashboard shows cumulative CPU/mem saved plus a registry of every task
the service has ever tuned. To exercise it, a second demo
(`tuned_workflows_demo.py`) runs three representative workflow shapes
(data engineering, ML training, batch inference) at two input scales
each — six tuned invocations per driver run.

## What was built

- **`resource_tuner/tune_store.py`**: append-only state on
  `flyte.io.Dir` by construction — one immutable JSON shard per record
  (`{kind}-{time_ns}-{uuid8}.json`) under a stable URI, written via
  `flyte.storage.put_stream`, read back incrementally with `Dir.walk()`
  (only unseen shards fetched). Safe for scale-to-zero replicas and
  concurrent writers: nothing is ever rewritten. Aggregations
  (`savings_of`, `savings_series`, `task_registry`, `totals`) are pure
  functions over record lists — unit-tested without storage
  (`tests/test_tune_store.py`).
- **`rt-tune` additions**: proposals persist to the store off the
  request path (`fire_and_forget`); new `POST /v1/outcome` records what
  actually happened under a proposal (fit/oom + measured
  `peak_rss_mib`); new `GET /` **value dashboard** — totals, cumulative
  mem/cpu-saved charts, full task registry table.
- **`@tune.resources` upgrades**: the invocation's actual inputs ride
  along as the input profile (same task, different `rows=` → different
  digest → separately-modeled proposal); outcomes auto-report when the
  wrapped task returns a dict containing `peak_rss_mib`; failures report
  an outcome (OOM detected from the error) then re-raise.
- **`tuned_workflows_demo.py`**: `sessionize_clickstream` (pandas),
  `train_churn_model` (sklearn GBM), `embed_product_catalog` (numpy
  matmul), all with the classic padded prior (4 CPU / 8Gi);
  `workflows_demo_driver` invokes each at small and large scales.

## Run ledger

Base URL: `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/`

| run | what | outcome |
|---|---|---|
| [ud655fkrz2ddgmf24n6p](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ud655fkrz2ddgmf24n6p) | workflows demo, first attempt | all 6 children failed: `'TunedTask' object has no attribute 'native_interface'` — the in-pod runtime resolves the module attribute where the real task lived and expects the full TaskTemplate surface. Fixed with `__getattr__` forwarding on the wrapper (runtime uses `.execute`, never `__call__`, so no re-tuning in the child). |
| [urbtknzlkzp4wbwwfgxs](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/urbtknzlkzp4wbwwfgxs) | workflows demo, wrapper fixed | 5/6 succeeded under tuned 4Gi (peaks 149–766 MiB vs the 8Gi prior); `embed/large` (1.5M×768 catalog ≈ 6GB working set) **OOMKilled under the tuned 4Gi** — recorded in the ledger as an OOM outcome. Priors arrived empty (`parent_env` is a weakref on the v2 template — must be dereferenced), so savings showed 0. |
| [ulk4j9ckds2zfxw96vg7](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ulk4j9ckds2zfxw96vg7) | workflows demo, prior fix | Same profile with real priors recorded: 5/6 fit (peaks 150–767 MiB under 4Gi vs the 8Gi prior), `embed/large` OOMKilled again (cached 4Gi proposal). Dashboard after the run: 36 records, 18 proposals / 3 tasks in the registry, model+cache answer rate 100%, **24 GiB cumulative memory-request savings, 2 cores CPU savings**, fit rate 56% (the ledger honestly counts the 6 wrapper-bug failures from the first run and both real embed OOMs). |

## Findings

- **Wrapping a task template requires full duck-typing.** The decorator
  pattern works at invocation time, but the child pod re-resolves the
  same module attribute and drives it as a `TaskTemplate`
  (`native_interface`, `execute`, `report`). Any wrapper must forward
  the whole surface (`__getattr__` → wrapped task).
- **`parent_env` on v2 task templates is a weakref** — read
  `task.resources` first, and call `parent_env()` when falling back to
  the env's declaration.
- **The uniform-4Gi caveat from round 5 has a real cost**: the model
  proposes ~4Gi regardless of scale hints in the input profile, which
  saves 4Gi/invocation on small workloads but OOMs a genuinely large one
  (`embed/large`). The ledger now makes that visible per task — the
  training signal for input-profile sensitivity is the obvious next
  round.
- **Append-only shards are the right state shape for scale-to-zero
  apps**: replicas die without flushing anything (every record is
  durable the moment it's written), and concurrent writers can't
  conflict (unique names, no shared file).

## Code changes

All on branch `flyte-2.7-metrics` (rides with rounds 3–5):
`resource_tuner/tune_store.py` (new), `tune_service.py` (store +
`/v1/outcome` + value dashboard), `tune.py` (`__getattr__` forwarding,
input-profile, outcome auto-report, weakref-aware prior),
`tuned_workflows_demo.py` (new), `tests/test_tune_store.py` (new).

## Key links

- Value dashboard: <https://rt-tune-resource-tuner-model-factory-development.apps.demo.hosted.unionai.cloud/>
- Store URI: `s3://union-oc-production-demo-raw/rt-tune-store/resource-tuner-model-factory/development`
