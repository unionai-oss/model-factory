# model-factory

Monorepo of model-factory subprojects (`model-factory-basic/`,
`model-factory-decision/`, `model-factory-resource-tuner/`), each
self-contained (own `pyproject.toml`, `uv.lock`, `.flyte/config.yaml`, tests,
docs). Run commands from inside the subproject directory. Cluster:
demo.hosted.unionai.cloud.

## Research log — non-negotiable

Any experimentation (cluster runs, training, evals, synthetic data, dark
trigger runs, factory materializations) **requires updating
`<factory>/research_log/`** in the corresponding factory directory: run ledger
with Union console URLs, findings, code changes, index/standing-results
refresh. Failures are logged too, linked to their fixes. Use the
`run-experiments` skill for the full procedure; log entries ride the same
branch/PR as the experiment.

## Factories

All three projects declare their pipeline as a Union **factory** (preview;
needs `flyte>=2.10.0` + `flyteplugins-union>=0.12.0`). The graph lives in
`<pkg>/factory.py`, with a thin top-level `factory.py` as the deploy
entrypoint (the CLI loads a module by path, which a package-relative import
cannot satisfy).

- **Station tasks return plain `File`/`Dir`.** The factory declares and
  publishes each artifact — name, kind, partition values, parent versions — by
  calling the task with `produces_artifacts=True` inside a
  `flyte.artifacts.produces(...)` block. A task that wraps its own output in
  `flyte.artifacts.new()` is fighting that.
- **An artifact's partition schema is immutable** once its first version is
  published. Declaring new dimensions on a name that already exists
  unpartitioned is rejected at `factory deploy`; publish under a new name.
  This is why model-factory-basic uses `mf-*` and the resource-tuner's
  decision arm uses `rt-*`.
- **A partition value injected into a same-named task parameter is typed.** A
  `factory.Daily` dimension arrives as a `datetime`, not a `str`.
- **`flyte factory deploy` does not build app images.** Deploy the app env
  once first (`flyte deploy app.py <app_env>`); on a cold registry that deploy
  itself fails resolving the artifact its parameter is bound to, which is
  expected — the image is built by then. Bootstrap order is
  stations → app env → factory → materialize target → materialize endpoint.
- **`flyte factory deploy` pins station task versions.** The graph's
  `Task.get(name, auto_version="latest")` resolves once, at factory deploy:
  redeploying stations alone changes nothing a materialization runs, and
  `--rebuild <artifact> --downstream` then re-runs the *old* code
  (rsg-small-3 re-fit every arm on the pre-change version). Redeploy
  stations, then the factory. A task called from inside a station runs from
  that station's code bundle, not the latest deploy of its own env.
- **Existing partitions are not re-run**, even after a code change:
  re-materializing a release whose arm artifacts exist reuses all of them
  (rsg-small-2 launched nothing). `--rebuild <artifact> --downstream`
  rebuilds that artifact and everything built from it; `--plan` does not
  show the effect of `--rebuild`.
- **An app env's spec is compiled into the factory's graph**, so changing the
  app env means redeploying the factory too. A failed serve redeploy leaves
  the previous revision serving, so verify a serve step via the app's
  `conditions`/`spec.inputs`, not by curling the endpoint.
- **`flyte.app.Parameter(mount=...)` pointing outside the app's writable
  working directory crash-loops the container.** Use
  `flyte.app.get_parameter(name)`, which returns the already-downloaded path.
- Task images must pin a `flyte` floor that **tracks the deploying SDK**; an
  image a minor behind cannot read a factory task spec.

## Operational notes

- resource-tuner triggers declare `auto_activate=True`: deploys leave dark
  mode LIVE (and rebind triggers to the new task version). Corollaries:
  a deploy also re-activates a trigger someone manually paused — pause by
  editing `auto_activate`, not just the console toggle; and any factory
  adding a trigger should set `auto_activate=True` unless it has a reason
  not to.
- `factory.on(...)` triggers are declared on the `Factory`, not on a task,
  and they are **LIVE as soon as the factory is deployed** — a declared
  `flyte.Cron` fired on schedule with no manual activation (run
  `u883f0f834236535d`, 2026-10-02 06:00 UTC). One `factory.on(event, a, b)`
  compiles into one trigger per target, so it shows up as several runs.
  A triggered materialization with nothing changed is all cache hits and
  finishes in seconds (9.1s for a 6-station graph), so it is cheap to leave
  armed -- **but only when it re-targets an existing partition**: a cron on a
  `Daily`-partitioned factory builds a *new* day every time it fires, i.e. a
  full rebuild (model-factory-decision round 2 was retraining a GPU suite
  nightly). `factory.on` has no `auto_activate`; the off switch is not
  declaring the trigger. One `factory.on(cron, a, b)` makes one trigger per
  target and each re-runs shared upstream work: target only the endpoint. `factory.on(<source>)` needs a `factory.source` — an artifact
  published outside the factory; everything a factory builds itself is
  reached by materializing it, not by a trigger.
- The code bundler ships only statically-imported modules; import task
  dependencies at module top, not inside task bodies.
- `flyte.remote.Artifact.listall` works on the demo tenant. Partition values
  read back from `spec.timePartition` (one time partition) and
  `spec.partitions.value.<key>.staticValue` (string partitions) — two
  different places.
- A ~10^6-row corpus cannot be read with `pd.read_parquet`; stream it
  (`pyarrow.ParquetFile.iter_batches`) with an explicit column projection.
- Helper calls inside async flyte tasks only fail on the cluster, after
  everything upstream has already run. Pin cross-module call shapes with a
  local test instead (see
  `model-factory-resource-tuner/tests/test_decision_model.py`).

## Caching and reuse (learned across all three factories, 2026-10)

- **Factory reuse IS task caching.** Flyte 2 task caching is off by default;
  an uncached build re-runs on every materialization even if its artifact
  exists (`factory plan` will still say "existing"). A redeploy changes every
  task version. `--version name=<v>` pins one version per artifact name, so it
  cannot pin a partitioned suite.
- **`cache="auto"` hashes only the task's own function body**, not modules it
  imports, and not profile/config *contents* (only their names). A station
  whose output is decided by imported code or a profile cache-hits across
  changes to it. Key the cache on what matters:
  `flyte.Cache(behavior="override", version_override=<hash of those sources>)`
  (resource-tuner `bench/versioning.py`), and treat profile names as immutable.
- A cache-hit GPU task still waits for GPU admission before its cache lookup
  (20+ min on a busy pool).

## Capacity on demo.hosted (2026-10)

- A project runs ~30 actions at once; beyond that they sit QUEUED with no
  cluster events. Throughput lever: more work per pod (e.g. several measured
  subprocesses per pod), not more pods.
- On-demand c5.4xlarge and i3.xlarge pools sit at their nodepool limits;
  `interruptible=True` reaches the spot pools. Spot reclaims can strand an
  action in RUNNING ("terminated in response to imminent node shutdown"):
  set `max_runtime` near the expected duration so the timeout drives the retry.
- GPUs: L40s:1 frequently at "max node group size"; A10G jobs can queue >1 h
  behind other projects; T4:1 (g4dn.xlarge, request <= 3 CPU / 13Gi) admits
  quickly. `torch.cuda.is_bf16_supported()` says True on a T4 (it counts
  emulation) -- check `get_device_capability()[0] >= 8`, else fp16 + GradScaler.
- Read action failure/scheduling reasons from `Action.details()` ->
  `pb2.attempts[-1].clusterEvents`; the laptop has no data-plane S3 credentials,
  so read artifacts through a small in-cluster task that returns them inline.

## LLM services and models

- llm-service apps: task pods use `http://<app>.llm-service-development.svc.cluster.local`;
  public URLs need a real User-Agent and the API key. vLLM apps answer `/health`
  with an empty 200 and validate the `model` field (read it from `/v1/models`).
  A shared 27B can run at ~1 token/s under load.
- vLLM inside a plain debian task image needs `VLLM_USE_FLASHINFER_SAMPLER=0`.
- Generative training over a 151k vocabulary: compute logits for the completion
  window only (`logits_to_keep`); full-sequence logits OOM a 16 GB GPU.
- Temperature-scale **logits**, not probabilities: fp32 softmax saturates to 1.0.
