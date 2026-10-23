# 2026-09-04 — round 5: last mile (tune service, @tune.resources, A/B)

## Context

Close the PRD's Phase-2 loop: a serving surface for proposals, the
one-line adoption decorator, and an auditable measurement of impact vs
hard-coded requests. Also this round: per-task observability upgrades
(traces, W&B, rich reports), typed OOM handling, and a reusable batched
generator for eval throughput.

## What was built

- **`rt-tune` service** (T4, scale-to-zero, pinned subdomain
  [rt-tune-resource-tuner-model-factory-development](https://rt-tune-resource-tuner-model-factory-development.apps.demo.hosted.unionai.cloud)):
  `POST /v1/propose` → schema-valid, bucketed, clamped kwargs from the
  trained policy; per-code-digest cache; `/reload`; `/records`; failures
  answer `source=fallback_prior` (visible degradation).
- **`@tune.resources`** (`resource_tuner/tune.py`): invocation-time
  interception — the wrapper requests a proposal (task's own source +
  env prior as context) and calls
  `task.override(resources=flyte.Resources(**proposal))`; DEGRADED
  fallback to the prior on any failure; explicit `.override()` outbids
  tuning. PRD adoption surface demoed verbatim in `tuned_demo.py`.
- **`tune_ab_experiment`** station: N held-out tasks × 2 real-pod arms
  (hard-coded prior vs tuned) → `tuning-ab-report` artifact (OOM
  prevention + overprovisioning reduction + per-task episodes).
- **Dashboard**: Tune-service impact panel (prior vs tuned,
  prevented/reduced badges) + A/B report station/Artifact Card.
- **Eval throughput**: generation moved off the eval task into a
  reusable GPU env (`rt-generator`, ReusePolicy replicas=(1,2)
  concurrency=8 idle_ttl=300) with a DynamicBatcher folding concurrent
  requests into batched generates; eval itself now runs CPU-only.
- **Observability**: W&B on by default (first run:
  [smoke-ullwh6kd4s727k5jvm59](https://wandb.ai/niels-bantilan/resource-tuner-model-factory/runs/spfzjg3j));
  trainer report rebuilt (reward curve, group-health chart —
  entropy + frac_reward_zero_std, dynamics chart, W&B link);
  `@flyte.trace` on the checkpoint upload; `flyte.group` on oracle
  subactions; typed `flyte.errors.OOMError` handling in episodes,
  eval generation, and training (pointed knob-fix message).

## Run ledger

| run | what | outcome |
|---|---|---|
| [ullwh6kd4s727k5jvm59](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ullwh6kd4s727k5jvm59) | validation smoke, first attempt | ❌ **@flyte.trace serializes inputs AND outputs as literals** — the traced checkpoint fn pickled the accelerate-wrapped trainer (PicklingError). W&B + training itself worked |
| [uk9nlfjmlgjkdqq5mfgs](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uk9nlfjmlgjkdqq5mfgs) | validation smoke, plain-data trace boundaries | ✅ full new path green (traced train → CPU eval → warm batched generator → typed episode OOM) |
| [uzbmfshw9bdkx5wkqrkb](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uzbmfshw9bdkx5wkqrkb) | grouped-oracle verification | ✅ oracle pods fold into flyte.group boxes |
| [utgc7xtglxmrxpkbfdd5](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/utgc7xtglxmrxpkbfdd5) | A/B attempt 1 | ❌ bundler import-graph rule again (`tune` imported in-body) |
| [uzcbdpjwcgx9jbvvqsmm](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uzcbdpjwcgx9jbvvqsmm) | A/B: 12 tasks, prior 2Gi | ✅ ran — and the tuned arm LOST (OOM 0%→17%, waste 64%→83%): the service served the **newest** checkpoint, a 10-step smoke train, not the dev-scale one. Recency ≠ quality — the promotion-gate gap, observed live |
| [u77jntkj2rxgvpk6c8kx](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u77jntkj2rxgvpk6c8kx) | A/B rerun: 15 tasks, prior 1Gi | still smoke-served (dev eval predated `checkpoint_path` linkage) — waste 54%→84%, though 1 OOM prevented |
| [urtf5kpc97bt889bp8c4](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/urtf5kpc97bt889bp8c4) | fresh dev-scale pipeline (150 composite steps) for linkage | train ✅; its eval showed **0% validity** — generator regression (below) |
| [u86mtkp9kgn8bh445f68](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u86mtkp9kgn8bh445f68) | standalone re-eval, batcher "fix" v1 | ❌ still 0% validity — `DynamicBatcher.start()` is a COROUTINE (annotation says `-> None`); un-awaited it's a silent no-op |
| [uwsl8sfzl2d2j8x9vmpm](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uwsl8sfzl2d2j8x9vmpm) | re-eval with `await batcher.start()` | ✅ generator path healthy: **validity 100%, fit 97% vs 62%, waste 51% vs 24%** — score 0.713 beats smoke's 0.430 |
| [u642czmkg6cnspbn9tdp](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u642czmkg6cnspbn9tdp) | **FINAL A/B**: 15 tasks, prior 1Gi, service serving the dev checkpoint | ✅ see below |

## Final A/B result (real pods, dev checkpoint served)

| metric | hard-coded prior (2 CPU / 1Gi) | tuned |
|---|---|---|
| OOM rate | 13% | **0%** |
| fit rate | 87% | **100%** |
| median overprovision | 54% | **47%** |

Both prior OOMs **prevented** (`data_engineering-1520172038` at 3.8GiB
peak, `ml_training-1357816123`); zero OOMs introduced. Honest caveat:
tuned memory requests are still uniform (4Gi) — the dev policy is
conservative-uniform on template tasks, so the waste win is modest;
per-task differentiation is the next training frontier (longer runs /
steeper waste weight / richer heldout).

## Findings

1. **Serve promoted, never latest.** The first A/B is the PRD's
   promotion-gate argument in one table: a fresh-but-worse checkpoint
   outranked a better one by recency and lost to a hard-coded prior.
   The service now scores checkpoints by their linked eval reports
   (validity × (fit − waste/200)) and serves the winner — the miniature
   of `promoted-tuner`.
2. **`@flyte.trace` boundaries must carry plain data only** — inputs and
   outputs are serialized as literals (trainer object → PicklingError;
   a traced model loader would pickle gigabytes).
3. The bundler import-graph rule caught its third victim (`tune`);
   the rule is now load-bearing enough to live in CLAUDE.md.
4. **`DynamicBatcher.start()` must be awaited** — it's a coroutine whose
   `-> None` annotation reads synchronous; un-awaited it no-ops and every
   `submit()` raises "not running".
5. **A DEGRADED fallback can mask total failure as a bad metric**: two
   evals reported 0% validity while every generator action failed; the
   run stayed green. Rule adopted: a green run with 0% validity is a red
   flag, not a pass — check metric content, not just phases.
6. Serve-best also needs the *linkage to exist*: the first serve-best
   deploy still picked a smoke checkpoint because the dev eval predated
   the `checkpoint_path` field. Selection, linkage, and metadata are one
   system.
