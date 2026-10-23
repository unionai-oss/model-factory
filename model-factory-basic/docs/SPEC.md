# Model Factory on Flyte v2 / Union — Specification

Status: v1 draft, 2026-08-12
Source context: `2026-08-11-dark-model-factory-patterns.md` (conference
abstract), poolside model-factory blog series, RLVR landscape research (see
`docs/research/`).

## 1. Goal

Build a **dark model factory**: a mostly-autonomous loop that trains agent
policies with RL from verifiable rewards, where each factory station is a
Flyte task/app, every produced asset is a versioned Flyte artifact, artifact
events drive the next station automatically, and humans are kept in the loop
only at the four bottleneck layers named in the conference abstract:

1. **Data validation** — approve/reject curated + synthetic task datasets.
2. **Training-loop observability** — live reports + W&B; auto-recovery
   (Flyte retries) with human escalation only on repeated failure.
3. **Inference/rollout output validation** — sample rollouts surfaced in
   reports before their data feeds back into the loop.
4. **Reward shaping** — reward components individually logged and
   inspectable per sample; promotion gated on eval + human vibe check.

Everything else goes dark: ingestion, filtering, synthetic generation,
rollout execution, reward computation, training, evaluation, packaging, and
inter-stage handoff are fully automated via artifact triggers.

## 2. The factory loop (reference architecture → Flyte mapping)

The poolside loop and its Flyte/Union equivalent, station by station:

| Factory station (poolside) | Flyte/Union building block |
|---|---|
| Orchestration spine (Dagster + K8s assets) | Flyte tasks + **artifacts** (versioned, lineage-tracked) + **OnArtifact triggers** |
| Data lake (Iceberg) | `flyte.io.DataFrame`/`Dir` artifacts in object store, versioned via `flyte.artifacts.Metadata` |
| Data delivery (Blender streaming) | Task I/O + `flyte.remote.Artifact.get` (small scope); streaming actors w/ `ReusePolicy` (large scope) |
| Training (Titan) | GPU `TaskEnvironment` (TRL GRPO small; torchtitan/verl large), retries + `Timeout` + checkpoint resume via `@flyte.trace` |
| Inference (Atlas / vLLM) | vLLM in-task (colocate, small); `ReusePolicy` worker fleet + `TokenBatcher` (medium); `VLLMAppEnvironment` real-time serving (promoted models) |
| Code execution env (Saucer/Task Engine) | Sandboxed subprocess executor inside rollout tasks (small); dedicated `ReusePolicy` executor fleet (medium/large) |
| Eval platform (Beacon) | Eval task auto-launched by `OnArtifact(policy-checkpoint)`; results as artifacts + HTML reports |
| Reward (RLCEF) | Pure-Python reward stack: format + anti-hack guards + all-tests-pass, executed in sandbox; every component logged per-sample |
| Observability (Neptune/Grafana/Sentry) | W&B (`WANDB_API_KEY`), live `flyte.report` streams, `@flyte.trace` spans, optional `flyteplugins-otel` → Grafana Tempo |
| Human layer (Podium / vibe check) | `flyte.new_condition` HITL gates + report tabs with inspectable samples + lineage **AppEnvironment** |

### Artifact graph (the "assets" of the factory)

```
seed dataset (HF)          synthetic generator (batch inference)
      │                            │
      ▼                            ▼
 [rl-tasks-dataset] ◄──── merge/curate ── [synthetic-tasks]
      │  (HITL gate 1: data validation)
      ▼  OnArtifact trigger
 GRPO training ──────► [policy-checkpoint]   (W&B + live report)
                              │  OnArtifact trigger
                              ▼
                        evaluation ────► [eval-report]
                              │  (auto gate: score ≥ threshold)
                              │  (HITL gate 2: promote? vibe check)
                              ▼
                       [promoted-model] ──► real-time serving app
                                            + lineage app (always on)
```

Names (artifact registry):

- `rl-tasks-dataset` — curated, verified RL task set (`DataFrame`).
- `synthetic-tasks` — synthetic problems w/ oracle-verified tests (`DataFrame`).
- `policy-checkpoint` — LoRA adapter + tokenizer + training manifest (`Dir`).
- `eval-report` — eval metrics + per-problem results (`File`, JSON).
- `promoted-model` — checkpoint that passed eval gate + human gate (`Dir`).

## 3. Small scope (POC, this repo): coding agent on one 24GB GPU

**Use case**: single-turn Python code generation with programmatically
verifiable rewards (unit tests). This is the smallest fully-closable loop:
prompt → completion → sandboxed test execution → scalar reward.

**Recipe** (from landscape research — DeepCoder/AceCoder/Oxen distillation):

- **Base model**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (smoke profile) /
  `Qwen/Qwen2.5-Coder-1.5B-Instruct` (default) + LoRA r=16.
- **Trainer**: TRL `GRPOTrainer`, vLLM colocate mode with sleep mode,
  `num_generations=4–8`, `beta=0.0` (no reference model),
  `loss_type="dapo"`, `scale_rewards="batch"`, `lr=1e-5` (LoRA),
  bf16, gradient checkpointing. Fits A10G:1 / L4:1 (24GB).
- **Seed data**: `KodCode/KodCode-Light-RL-10K` (HF) — question + reference
  solution + pytest-style tests. Curation: dedup, require ≥5 test
  assertions, length caps, drop tests that fail against their own reference
  solution (oracle verification), stratify by difficulty.
- **Reward function** (per completion, components logged individually):
  - `format` (0 or 0.1): exactly one fenced python block, non-empty.
  - `compile` (0 or 0.1): `ast.parse` succeeds on extracted code.
  - `tests_pass` (0 or 1.0): **all** hidden tests pass in a sandboxed
    subprocess (rlimits: CPU, memory, no network, tmp dir, 10s timeout).
    All-or-nothing per DeepCoder to prevent partial-credit hacking.
  - Anti-hack guards: reject solutions that import the test file, mock
    assertions, or print expected outputs (regex + AST checks) → reward 0.
  - Total ∈ [0, 1.2]; logged per-component to W&B and the report.
- **In-loop curation**: DAPO-style difficulty filter — drop prompts where
  all G completions pass or all fail (zero advantage).
- **Synthetic data subsystem** (batch inference): generator task prompts an
  instruct model to mutate seed problems (new edge-case tests, transformed
  problems AceCoder-style); the **sandbox executes the reference solution
  against generated tests** — only oracle-verified samples survive → emit
  `synthetic-tasks`. OnArtifact merge into `rl-tasks-dataset`. For the POC
  the generator uses the same small GPU env (vLLM offline batch); the
  interface allows swapping in a served endpoint later.
- **Eval**: HumanEval+ subset (or MBPP+ sanity slice) pass@1, candidate vs
  base model, plus held-out KodCode slice. Auto-gate: candidate ≥ base +
  configurable margin. Emits `eval-report`.
- **HITL gates** (`flyte.new_condition`, markdown prompts, timeouts):
  1. Dataset approval after curation (report shows sample tasks + stats).
  2. Checkpoint promotion after eval (report shows eval table + sample
     completions side-by-side). `auto_approve=True` flag exists for CI
     smoke runs; defaults to human gate.
- **Observability**: W&B run per training job (`WANDB_API_KEY`,
  degrade to `WANDB_MODE=disabled` if secret absent); live flyte report
  streaming reward/loss curves during training; `@flyte.trace` on rollout
  batch + reward computation helpers (crash-resume + span lineage).
- **Lineage app**: `AppEnvironment` (FastAPI, scale-to-zero) rendering the
  artifact graph — every artifact version, the run that produced it, its
  upstream artifact versions, HITL decisions, and eval scores. Reads
  `flyte.remote` APIs from inside the cluster.
- **GPU instances**: `A10G:1` (g5.2xlarge) or `L4:1` (g6.2xlarge) for
  training/generation; CPU-only for data, curation, eval-scoring, apps.

**Profiles**: `smoke` (0.5B, 64 tasks, ~10 GRPO steps, minutes),
`dev` (1.5B, 1–2K tasks, ~100 steps, ~1–2h), `full` (1.5B/3B, 10K tasks,
500+ steps).

### Pipeline entrypoints

- `factory.pipeline.run_factory` — one full loop iteration as a driver task
  (explicit chaining, easiest to demo/test end-to-end).
- Deployed triggers for continuous ("dark") operation:
  - `OnArtifact(rl-tasks-dataset)` → training
  - `OnArtifact(policy-checkpoint)` → evaluation
  - `OnArtifact(synthetic-tasks)` → merge/curation → new dataset version
  - `Cron` → nightly synthetic generation batch

## 4. Medium scope: multi-task coding agent / small SWE agent

Scale up along three axes (same artifact graph, bigger stations):

- **Tasks**: multi-turn tool-use episodes — repo-level bugfixing on
  SWE-Gym-lite/SWE-smith-generated instances. Reward = hidden test suite
  passes after agent edits, executed in per-episode containers.
- **Rollout/train split**: disaggregate. Rollout workers = `ReusePolicy`
  fleet (T4:8/L4:2 replicas) running vLLM, fed by `TokenBatcher`; trainer =
  single `L40s:4`/`A100 80G` task consuming trajectory batches (bounded
  staleness 1–2 steps, prime-rl style). Weight sync via checkpoint artifact
  versions (object store hop — acceptable at this scale).
- **Model**: Qwen2.5-Coder-7B / Qwen3-8B QLoRA; or 3B full-finetune.
- **Sandbox**: dedicated executor `TaskEnvironment` with `ReusePolicy`
  (warm containers, per-repo images), the "Task Engine" analog. Add
  container-level isolation (gVisor/Firecracker via cluster config) for
  untrusted code at scale.
- **Evals**: SWE-bench-lite slice + LiveCodeBench window + regression suite
  of previously-passed tasks; eval fleet fans out with `flyte.map`.
- **Data**: SWE-smith-style task synthesis (break tests in real repos) as
  the synthetic subsystem; artifact-triggered curation identical to POC.
- **Observability adds**: `flyteplugins-otel` → Grafana Tempo; W&B sweeps
  for hyperparameter exploration (`WandbSweep`); Slack notifications
  (`flyte.notify.Slack`) on failed runs and on HITL gates opening.

## 5. Large scope: frontier-style factory

- **Use cases** (favor programmatic verification): coding agents
  (SWE-bench-class), computer-use agents (DOM/pixel envs with checkable
  goal states), formal-math agents (Lean verification), science/AI-research
  agents (experiment DSLs where results are re-executable — e.g.
  auto-ML pipelines scored on held-out data). Legal/document agents only
  with rubric-as-code graders (weakest verification — keep more lights on).
- **Training**: multi-node clustered `TaskEnvironment`s (torchtitan or verl
  backend), full finetunes 32B+, GPU-to-GPU weight sync (checkpoint-engine
  pattern) replacing object-store hops; H100/H200/B200 pools.
- **Rollouts**: rollout-as-a-service — a permanent `AppEnvironment` fleet
  exposing an environment API (verifiers-compatible `load_environment()`
  packages), tens of thousands of concurrent episodes, preemptible/backfill
  queue classes so batch inference soaks idle GPUs (Volcano analog =
  cluster queues + `queue=` on triggers/envs).
- **Data**: continuous ingestion pipelines with lineage; every filter pass
  an artifact; proxy-model cluster scoring for automated pruning; the
  synthetic subsystem becomes a first-class generation service.
- **The factory improves itself**: agents trained by the factory get used
  as the build-fixer for sandbox images and as curation critics — the
  poolside "build agent trained in the factory" pattern.
- **Human layer**: the lineage app grows into Podium — dataset viewers,
  checkpoint diff/compare, rollout transcript browser, one-click gate
  signals (`ConditionWebhook` → Slack approval buttons).

## 6. Team decomposition and repo layout (POC implementation)

The factory is owned by four teams. **Artifacts are the only inter-team
interface**; downstream teams start work via `OnArtifact` triggers on the
upstream team's published artifact. Teams import only
`model_factory/contracts.py` (the interface) and `model_factory/shared/`
(platform libraries) — never each other's task modules.

| team | owns | consumes | publishes | deploy unit |
|---|---|---|---|---|
| data engineering | prep, validation, filtering, synthetic gen, data HITL gate | `synthetic-tasks` (own trigger) | `rl-tasks-dataset`, `synthetic-tasks` | `team_data.py` |
| model training | the RL training loop (GRPO, W&B, live reports) | `rl-tasks-dataset` (OnArtifact) | `policy-checkpoint` | `team_training.py` |
| model eval | eval runs, auto quality gate, promotion HITL gate | `policy-checkpoint` (OnArtifact) | `eval-report`, `promoted-model` | `team_eval.py` |
| inference | serving app for rollouts + evals, weight rollout ops | `policy-checkpoint` (OnArtifact) | `inference-endpoint` | `team_inference.py` |

The inference subsystem serves whatever checkpoint dropped last: its
`/generate` endpoint toggles the LoRA adapter per request, so eval compares
candidate (adapter on) vs base (adapter off) against one weights service;
the same endpoint is the rollout backend for disaggregated RL (dev+ scope —
the smoke trainer still colocates rollouts inside GRPOTrainer).

```
model_factory/
  contracts.py           # THE inter-team interface: artifact names, schemas, publish()
  config.py              # profiles (smoke/smoke-plus/dev/full), secret gating
  shared/                # platform libraries (any team may import)
    sandbox.py           #   sandboxed code execution (pure python)
    rewards.py           #   reward components + anti-hack guards
    reporting.py         #   HTML report builders
    assets.py            #   latest-asset resolution (artifact API + run-scan fallback)
    gates.py             #   HITL condition gates
    images.py            #   base images + secret wiring
    inference_client.py  #   HTTP client for the inference service
  data_engineering/      # envs, curation tasks, synthetic gen, release driver + triggers
  training/              # trainer env, train_grpo (+ OnArtifact(rl-tasks-dataset))
  evaluation/            # eval envs, evaluate/promote, eval_and_promote (+ OnArtifact(policy-checkpoint))
  inference/             # serving app (mf-inference) + refresh ops (+ OnArtifact(policy-checkpoint))
  lineage_app.py         # platform: global lineage AppEnvironment
team_*.py                # per-team deploy units (repo root)
integration.py           # cross-team E2E driver (stand-in event bus until artifact events ship)
tests/                   # pytest for sandbox, rewards, parsing, reporting
docs/, TODO.md
```

## 7. Secrets & external services

| Secret name | Used for | Required? |
|---|---|---|
| `HUGGINGFACE_TOKEN` | HF downloads (models/datasets are public → optional; needed for gated models + higher rate limits) | optional, graceful fallback |
| `WANDB_API_KEY` | W&B experiment tracking | optional → `WANDB_MODE=disabled` fallback |

Create with:

```
flyte create secret HUGGINGFACE_TOKEN --value <hf_...> --project model-factory --domain development
flyte create secret WANDB_API_KEY --value <key> --project model-factory --domain development
```

No other external services are required for the small scope. Medium/large
would add: Slack webhook (notifications), OTel endpoint (Grafana Tempo),
hosted sandbox provider (Modal/E2B) if untrusted-code isolation must exceed
subprocess rlimits.

## 8. Non-goals (POC)

- No multi-node training, no GPU-to-GPU weight sync (object-store artifact
  hop is the weight path).
- No microVM isolation: sandbox = subprocess + rlimits + no-network. The
  code being executed is model-generated against our own prompts, run
  inside the task container; risk accepted for POC, flagged for medium.
- No online/async RL: rollouts are on-policy inside GRPOTrainer (colocate).
  The off-policy actor/trainer split arrives in medium scope.
