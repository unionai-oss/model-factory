# 2026-09-03 — round 1: first cluster experiments

Goal: prove tasks run and data flows on demo.hosted, then make reward go
up (stage A before anything fancier). All runs in project
`resource-tuner-model-factory` / `development`; URL base
`https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/`.

## Run ledger

| # | run | what | outcome |
|---|---|---|---|
| 1 | [uvcszrfvx5r87kzf7wjz](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uvcszrfvx5r87kzf7wjz) | `build_task_corpus` smoke | ✅ corpus artifact publishes |
| 2 | [uj9bstrlhljhgggd5zsm](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uj9bstrlhljhgggd5zsm) | `probe_episodes` (baseline proposals + deliberate 128Mi OOM probe) | ✅ 5/5 episodes fit; OOM probe correctly classified `ok=false, oom=true`; real RSS tracks analytic footprints (analytic 10–100% conservative) |
| 3 | [urctmsxgsj76xm69xdxk](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/urctmsxgsj76xm69xdxk) | `tuner_pipeline` smoke, attempt 1 | ❌ TRL: `generation_batch_size (4) must be divisible by num_generations (8)` |
| 4 | [utxxsdc7529ngc855rxg](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/utxxsdc7529ngc855rxg) | smoke, attempt 2 (batch=group size) | ✅ mechanically; ❌ scientifically: reward 0 at all 10 steps, `clipped_ratio=1.0`, `mean_terminated_length=0` |
| 5 | [ukcjkdwvh4pzmmvppk5f](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ukcjkdwvh4pzmmvppk5f) | smoke, attempt 3 (`/no_think`) | train ✅ (completions ~15 tok, reward 1.1 = stage-A max from step 1); eval hung 75+ min on an unschedulable episode pod → aborted |
| 6 | [ud8lwtzpjm764w6qcqdb](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ud8lwtzpjm764w6qcqdb) | `smoke-composite` (30 stage-B steps), attempt 1 | train ✅ (reward variance restored: 0.64–0.87, −0.4 OOM groups, real gradients); eval hit the same unschedulable hang → aborted |
| 7 | [us4xhjj2z48kshwkkdpd](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/us4xhjj2z48kshwkkdpd) | smoke-composite, attempt 2 (clamp + queue timeout) | ✅ full green: eval in 3.5 min, 5/6 real episodes ok, 1 policy-OOM (signal, not error) |

## Findings

1. **Qwen3 thinking eats the completion budget.** With default chat
   templating the model spends all 128 tokens inside `<think>` — no JSON,
   all-zero rewards, zero gradient (`frac_reward_zero_std=1`). Fix:
   `/no_think` prompt switch + `chat_template_kwargs={"enable_thinking":
   False}` in GRPOConfig + eval-side template kwarg.
2. **Stage A saturates instantly.** The base model's generous proposals
   fit most tasks → all-pass groups, no advantage. Exactly the cue the
   literature predicts for graduating to the composite reward (waste
   ranks the all-pass groups). Added the `smoke-composite` profile.
3. **Unclamped over-asks hang everything.** A policy proposal at the
   action-grid max (16 CPU / 64Gi) is unschedulable; the pod queues
   forever and the eval waits on it. This is the PRD's quota-clamp case
   observed live. Fix: clamp the POD request to `RT_EPISODE_MAX_*`
   (floor-to-grid — bucket-up would bounce above the ceiling) while the
   reward still scores the policy's original ask, plus
   `max_queued_time=300` on the harness so "can't schedule" becomes a
   failed episode.
4. **TRL divisibility**: `per_device_batch` must be a multiple of
   `num_generations`; profiles now use batch == group size.
5. Composite reward at 30 steps: mean ~0.65, no waste movement yet —
   consistent with "expect ~300 flat steps" from the literature; the
   dev-scale run in round 2 is the test.

## Code changes (all in PR [#7](https://github.com/unionai-oss/model-factory/pull/7))

- `config.py`: batch=group-size profiles; `smoke-composite` profile.
- `policy/prompts.py`, `training/grpo.py`, `training/evaluate.py`:
  no-think wiring.
- `environment/episodes.py`: `clamp_for_execution` (+`RT_EPISODE_MAX_*`);
  `environment/harness.py`: `max_queued_time=300`.
- `training/grpo.py`: live `flyte.report` reward curve via
  TrainerCallback; W&B behind cluster secret.
- `training/stations.py`: `probe_episodes` station.
