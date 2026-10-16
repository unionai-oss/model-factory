# TODO — open items

## Secrets (done)

`HUGGINGFACE_TOKEN` and `WANDB_API_KEY` exist in `demo` org / project
`model-factory` / domain `development`. Attach them by deploying with
`MF_USE_SECRETS=1` (Flyte refuses to schedule a task whose declared secret is
missing on the tenant, hence the switch). To recreate on another tenant:

```bash
flyte --config <cfg> create secret HUGGINGFACE_TOKEN --value <hf_...> --project model-factory --domain development
flyte --config <cfg> create secret WANDB_API_KEY --value <wandb key> --project model-factory --domain development
```

## The nightly loop is armed

The factory's `nightly-release` cron and the `crown-champion` artifact trigger
are live on deploy. Once a champion exists for today, the cron trains a real
`daily` day (~1 A10G-hour) every night and crowns the next one, indefinitely.
That is the point of a dark factory, but it is a standing GPU cost on a shared
demo cluster: decide deliberately before seeding a champion for the current
date. To pause, set `auto_activate=False` on `crown_on_promotion`
(`model_factory/evaluation/tasks.py`) and redeploy `team_eval.py` — without
crowns, the next night stops at "mf-champion[D] not in the registry" before any
GPU work.

## Deferred integrations

- **vLLM rollouts**: GRPO generation is plain HF `generate`, which bounds the
  steps that fit in an hour. TRL's colocated vLLM needs `vllm` in `gpu_image`
  and an E2E pass.
- **OpenTelemetry / Grafana**: `flyteplugins-otel` is not wired (no OTLP
  endpoint for the demo). Traces (`@flyte.trace`) cover span-level lineage in
  the console meanwhile.
- **Slack notifications** on gate openings and failed nights: needs a webhook
  secret; `flyte.notify.Slack(...)`.

## Known POC limitations

- Sandbox = subprocess + rlimits inside the task container, not
  microVM-isolated (docs/SPEC.md §8).
- Held-out leak checks are exact-text (whitespace/case-normalized), not
  near-duplicate detection.
