# Research notes: Flyte v2 / Union building blocks (verified against SDK 2.6.0)

Compiled 2026-08-12 from Union docs, SDK examples, and direct inspection of the
installed `flyte==2.6.0` package. Where docs and SDK disagree, the SDK wins
(noted inline).

## 1. TaskEnvironment

```python
import flyte

env = flyte.TaskEnvironment(
    name="gpu_env",
    resources=flyte.Resources(cpu=6, memory="4Gi", gpu="L4:1"),
    image=flyte.Image.from_debian_base().with_pip_packages("torch", "wandb"),
    secrets=flyte.Secret(key="NIELS_WANDB_API_KEY", as_env_var="WANDB_API_KEY"),
    env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
    cache="auto",
    depends_on=[other_env],   # bundle multiple envs in one deploy
)

@env.task
async def train(x: int) -> int: ...
```

- **Resources**: `cpu` (number, `"500m"`, or `(request, limit)`), `memory`,
  `gpu` (strings `"T4:8"`, `"A10G:1"`, `"L4:1"`, `"L40s:4"`, `"A100 80G:4"`,
  etc. — full literal list verified in SDK), `disk`, `shm` (`"auto"`).
  Per-call override: `await task.override(resources=...)(...)`.
- **Images**: `flyte.Image.from_debian_base(python_version=(3,12))` +
  `.with_pip_packages(...)`, `.with_apt_packages(...)`,
  `.with_uv_project(...)`, `.with_env_vars({...})`, `.with_commands([...])`.
  Built remotely by the Union ImageBuilder (28s for base image, verified).
- **Secrets**: `flyte.Secret(key="NAME", as_env_var="ENV_VAR")` → read via
  `os.environ`. File mode: `mount=Path(...)`. Create with
  `flyte create secret NAME --value <v> [--project X --domain Y]`.
- **Caching**: off by default; `cache="auto"` hashes code;
  `flyte.Cache(behavior=..., ignored_inputs=(...))` for fine control.
- **Retries/timeouts**: `retries=flyte.RetryStrategy(count=4, backoff=...)`,
  `timeout=flyte.Timeout(max_runtime=..., deadline=...)`;
  `flyte.errors.NonRecoverableError` short-circuits retries.
- **Reusable containers**: `reusable=flyte.ReusePolicy(replicas=16,
  concurrency=1, idle_ttl=120, scaledown_ttl=120)`. Capacity = replicas ×
  concurrency. Requires `unionai-reuse` in the task image. This is the v2
  replacement for v1 `union.actor`.

## 2. Workflows, fanout, human-in-the-loop

A workflow is an async task that awaits other tasks; `asyncio.gather` fans
out; `flyte.map` for sync fanout. `flyte.ctx()` exposes action metadata,
`raw_data_path`, `custom_context`.

**HITL conditions** (verified in SDK — `flyte.new_condition`):

```python
condition = await flyte.new_condition.aio(
    "approval",
    prompt="## Approve dataset?\n\n...",
    prompt_type="markdown",
    data_type=bool,                    # bool | int | float | str
    timeout=timedelta(minutes=30),     # optional
)
try:
    approved = await condition.wait.aio()
except flyte.errors.ConditionTimedoutError:
    approved = False
```

Run pauses (PAUSED phase) until signaled from UI, CLI
(`flyte signal condition <run> <action> true`), or Python
(`flyte.remote.Condition.get(...).signal(True)`). Outbound webhook on
creation via `flyte.ConditionWebhook(url=..., payload={...})`.

## 3. Reports (`flyte.report`)

Enable with `@env.task(report=True)`. Live streaming supported — each flush
pushes HTML to the UI while the task runs.

```python
await flyte.report.log.aio("<h1>Training</h1>", do_flush=True)
tab = flyte.report.get_tab("Samples"); tab.log("<p>...</p>")
await flyte.report.replace.aio(full_html)
await flyte.report.flush.aio()
```

Arbitrary HTML/JS/CSS allowed (e.g. inline Chart.js-style canvases).

## 4. Traces (`@flyte.trace`)

Decorator for helper functions called inside tasks. Each call becomes a
recorded span (inputs/outputs/duration checkpointed). On retry/crash-resume,
successful traced calls are replayed from recorded outputs. Async functions
(and generators) only.

## 5. Artifacts and triggers

**SDK 2.6.0 has the full artifact + artifact-trigger API** (docs pages lag —
they still say "coming soon"; the installed SDK exports all of the below,
backend support must be verified empirically on the target cluster).

Publish (artifacts are offloaded assets only: `File`, `Dir`, `DataFrame`):

```python
import flyte.artifacts as artifacts
from flyte.io import File

@env.task
async def train() -> File:
    f = await File.from_local("weights.pt")
    return artifacts.new(f, artifacts.Metadata(name="policy-ckpt", kind="model"))
```

Consume: `flyte.remote.Artifact.get("name", version=...)`,
`Artifact.listall(name=...)`; pass as run inputs.

Artifact-event triggers:

```python
retrain = flyte.Trigger(
    name="retrain_on_new_model",
    automation=flyte.OnArtifact(name="policy-ckpt"),
    inputs={"model": flyte.TriggeredArtifact, "threshold": 0.5},
)

@env.task(triggers=[retrain])
async def evaluate(model: File, threshold: float) -> None: ...
```

Schedule triggers: `flyte.Trigger(name=..., automation=flyte.Cron("0 9 * * *"))`
or `flyte.FixedRate(...)`; `flyte.TriggerTime` sentinel injects timestamp.
Notifications: `flyte.notify.Slack/Email/Webhook` on phase changes.
Deploy triggers with `flyte.deploy(env)` / `flyte deploy file.py env`.

## 6. Apps (`flyte.app.AppEnvironment`)

```python
import flyte.app

app_env = flyte.app.AppEnvironment(
    name="lineage-app",
    image=flyte.Image.from_debian_base().with_pip_packages("fastapi", "uvicorn"),
    port=8080,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),  # scale-to-zero
    requires_auth=False,
    secrets=[...], env_vars={...},
    parameters=[flyte.app.Parameter(name="x", value=...)],
)
```

- Hooks: `@app_env.server` (starts the server), `@app_env.on_startup`.
- Deploy: `flyte.serve(app_env)` → `.url`; CLI `flyte serve/deploy file.py env`.
- FastAPI wrapper: `from flyte.app.extras import FastAPIAppEnvironment`.
- vLLM serving: `from flyteplugins.vllm import VLLMAppEnvironment`
  (`model_hf_path=...`, `stream_model=True`, OpenAI-compatible `/v1` API);
  `flyte.prefetch.hf_model(repo=...)` stages weights in the object store.
- Connect app to task outputs: `flyte.app.RunOutput(task_name=..., type="directory")`
  wrapped in `flyte.app.Parameter(..., download=True, env_var=...)`.

## 7. Observability

- **W&B — official plugin**: `pip install flyteplugins-wandb`; `@wandb_init`
  above `@env.task`; `get_wandb_run()` inside; `wandb_config(project=...)`.
  Manual alternative: `wandb.init()` in-task with secret as env var (works
  without the plugin; degrade gracefully with `WANDB_MODE=disabled`).
- **OpenTelemetry — official**: `pip install flyteplugins-otel`;
  `from flyteplugins.otel import init; init(service_name=...)` at module
  scope; tasks become spans, `@flyte.trace` calls child spans; standard
  `OTEL_EXPORTER_OTLP_ENDPOINT` env vars; exporters Tempo/Jaeger/Honeycomb.
- **Prometheus/Grafana**: platform-level (self-managed Union ships a
  Prometheus stack); no task-level SDK API.

## 8. Batch inference

`flyte.extras.TokenBatcher` — dynamic, token-budgeted batching, combined
with ReusePolicy + process-level singleton:

```python
from flyte.extras import TokenBatcher

@alru_cache(maxsize=1)
async def get_batcher():
    b = TokenBatcher(inference_fn=..., target_batch_tokens=32_000, max_batch_size=256)
    await b.start(); return b

@gpu_env.task  # gpu_env has reusable=ReusePolicy(replicas=2, concurrency=10)
async def infer_batch(prompts: list[str]) -> list[str]:
    b = await get_batcher()
    futs = [await b.submit(p) for p in prompts]
    return await asyncio.gather(*futs)
```

## 9. CLI

- `flyte --config <cfg> run [-p proj -d domain] [--local] file.py task --arg v`
- `flyte deploy file.py env` / `--all --recursive ./src`
- `flyte serve file.py app_env` (`--local --follow` for dev)
- `flyte get run|logs|io|secret`, `flyte signal condition <run> <action> <v>`
- Python: `flyte.init_from_config(path)`, `flyte.run(task, **kw)` → `Run`
  (`.name`, `.url`, `.wait()`), `flyte.with_runcontext(...)`.

## 10. Feature-existence summary

| Feature | v2 status |
|---|---|
| HITL gates | ✅ `flyte.new_condition` (+ webhook) |
| Reports (live) | ✅ `flyte.report` |
| Traces | ✅ `@flyte.trace` |
| Schedule triggers | ✅ `flyte.Trigger` + Cron/FixedRate |
| Artifact triggers | ✅ in SDK 2.6.0 (`OnArtifact`, `TriggeredArtifact`); docs lag; verify backend |
| Artifacts | ✅ `flyte.artifacts.new` / `flyte.remote.Artifact` |
| Reusable containers | ✅ `ReusePolicy` (v1 actors ❌) |
| Dynamic batcher | ✅ `flyte.extras.TokenBatcher` |
| W&B | ✅ `flyteplugins-wandb` (or manual `wandb.init`) |
| OTel | ✅ `flyteplugins-otel` |
| Prometheus/Grafana | deployment-level only |
| vLLM serving | ✅ `flyteplugins.vllm.VLLMAppEnvironment` |
