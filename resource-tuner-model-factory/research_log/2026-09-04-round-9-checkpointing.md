# 2026-09-04 — round 9: training checkpointing + the ambitious run

## Context

Two motivations: (1) round-8's arms revealed real long-run exposure — a
lost pod restarts training from step 0; (2) the next experiment (0.5M
corpus, 4B-class model, up to 3 days on a single T4) is only sane if
training is resumable at every layer. This round adds three checkpoint
layers and proves each on the cluster before the long run starts.

## What was built

- **Intra-task checkpoints** (`flyte.ctx().checkpoint`): TRL saves full
  trainer state every `profile.save_steps` steps; a callback uploads the
  newest `checkpoint-<N>` dir to the flyte checkpoint prefix. With
  `retries=3` on `train_tuner`, a failed attempt restores the tarball,
  finds the resumable dir (`find_trl_checkpoint` handles both restore
  layouts), and passes it to `trainer.train(resume_from_checkpoint=…)` —
  optimizer/scheduler/global_step intact.
- **Artifact checkpoints**: every `profile.artifact_checkpoint_every`
  steps the adapter is snapshotted and published by a SEPARATE child
  task (`publish_intermediate_checkpoint`) as
  `tuner-checkpoint-intermediate` — a distinct artifact name so the
  eval-on-new-checkpoint trigger stays quiet until the final checkpoint.
  publish() versions only on task return, hence the child task.
- **Warm-start parameterization**: `train_tuner(resume_from=<Dir>)` or
  `resume_from_artifact="<name>"` — the artifact form resolves the newest
  version imperatively via `flyte.remote.Artifact` (blob URI, never the
  console URL) and initializes the LoRA adapter from it.
- **Chaos hook** `fail_at_step`: the first attempt raises after step N;
  the retry must resume — makes the intra-task path testable end-to-end
  instead of waiting for a real preemption.
- Profiles: `smoke-ckpt` (10 steps, save every 4, publish at 5) and
  `ambitious` (Qwen3.5-4B QLoRA, single T4, 1200 steps, save every 25,
  publish every 100, 72h ceiling).
- Unit tests: restore-layout handling, incomplete-checkpoint rejection,
  resume resolution (explicit dir wins; unversioned artifact fails
  loudly), profile invariants (157 passing).

## Run ledger

Base URL: `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/`

| run | what | outcome |
|---|---|---|
| [u4mdvq689hln7pfvj99p](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u4mdvq689hln7pfvj99p) | smoke-ckpt + `fail_at_step=8` (chaos) | **Intra-task resume PROVEN**: attempt 1 died by injection after step 8; attempt 2 logged `[ckpt] resuming attempt from intra-task checkpoint` and its progress bar started at 9/10 — finished the last 2 steps in 86s instead of retraining. But zero intermediate artifacts published → bug 1 found. |
| [u4xjcf8rrg8fm5wlcjxj](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u4xjcf8rrg8fm5wlcjxj) | chaos rerun after bug-1 fix | Resume proven again; still no intermediate — in a chaos run the step-5 publish is killed ~15s later by the injected failure and the resumed attempt starts past step 5. Expected at smoke cadence; irrelevant at ambitious cadence (minute-steps, publish every 100). |
| [ujv65cq2pfdx2f76727w](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ujv65cq2pfdx2f76727w) | clean smoke run | Publish scheduled at step 5 but failed at drain: `Environment 'rt-driver' not found in image` → bug 2 found. |
| [uh4878qgccdwhwv9mj94](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uh4878qgccdwhwv9mj94) | clean smoke after bug-2 fix | **Artifact checkpoints PROVEN**: `tuner-checkpoint-intermediate` v1 published at step 5 (`tuner-inter-5-…`), final checkpoint also published. |
| [urgnxkvb8gk9w78tkxnj](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/urgnxkvb8gk9w78tkxnj) | warm start via `resume_from_artifact=tuner-checkpoint-intermediate` | **Warm start PROVEN**: `[ckpt] warm-starting adapter from artifact:…` — newest intermediate resolved imperatively, adapter loaded, trained to completion, checkpointed as usual. |
| [u2248fp4bgnpds44b24q](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u2248fp4bgnpds44b24q) | probe-qwen35 (5 QLoRA steps, T4) | **Qwen3.5 RL UNBLOCKED**: trl#5269's multimodal-arch failure no longer reproduces — Qwen3.5-4B trained 5/5 GRPO steps with real gradients (rewards 0.35–1.1) on one T4. |
| [uf7wrf4fmk8xdsbmkfhs](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uf7wrf4fmk8xdsbmkfhs) | 0.5M-row corpus release (440k archetypes + 60k templates, GPU tasks T4-capped) | RESULTS_PENDING |
| — | ambitious training (Qwen3.5-4B, single T4, ≤3 days) | RESULTS_PENDING |

## Findings

- **Smoke tests caught two real bugs before the 3-day run bet on them**:
  (1) artifact publishes gated inside `on_save` inherit the save-steps
  cadence (save 4 / publish 5 → LCM 20 → nothing at 10 steps) — moved to
  `on_step_end`; (2) a parent can only spawn children whose envs are in
  its dependency closure — the GPU trainer couldn't spawn a driver-env
  child (`Environment 'rt-driver' not found in image`), fixed with a
  dedicated 1-CPU `rt-ckpt` env in the trainer's `depends_on`.
- **Chaos hooks pay for themselves**: `fail_at_step` turned "resume
  probably works" into a reproducible two-attempt proof with the resumed
  step number visible in the log.
- **The Qwen3.5 blocker quietly expired** — worth re-probing "known"
  upstream blockers before designing around them.

## Code changes

Branch `flyte-2.7-metrics`: `training/grpo.py` (checkpoint layers,
`find_trl_checkpoint`, `_resolve_resume`, `publish_intermediate_checkpoint`,
72h/retries posture), `contracts.py` (intermediate artifact),
`config.py` (`save_steps`/`artifact_checkpoint_every` fields, smoke-ckpt +
ambitious profiles), `tests/test_checkpointing.py`.
