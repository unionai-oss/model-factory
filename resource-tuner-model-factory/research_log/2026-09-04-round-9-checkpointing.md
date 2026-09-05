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
| [u4mdvq689hln7pfvj99p](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u4mdvq689hln7pfvj99p) | smoke-ckpt + `fail_at_step=8` (chaos: attempt 1 dies, retry must resume mid-run) | RESULTS_PENDING |
| — | smoke-ckpt warm start via `resume_from_artifact=tuner-checkpoint-intermediate` | RESULTS_PENDING |
| [uf7wrf4fmk8xdsbmkfhs](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uf7wrf4fmk8xdsbmkfhs) | 0.5M-row corpus release (440k archetypes + 60k templates, GPU tasks T4-capped) | RESULTS_PENDING |
| — | ambitious training (Qwen3.5-4B-class, single T4, ≤3 days) | RESULTS_PENDING |

## Findings

RESULTS_PENDING

## Code changes

Branch `flyte-2.7-metrics`: `training/grpo.py` (checkpoint layers,
`find_trl_checkpoint`, `_resolve_resume`, `publish_intermediate_checkpoint`,
72h/retries posture), `contracts.py` (intermediate artifact),
`config.py` (`save_steps`/`artifact_checkpoint_every` fields, smoke-ckpt +
ambitious profiles), `tests/test_checkpointing.py`.
