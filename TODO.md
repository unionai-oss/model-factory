# TODO — human actions needed

## 1. Create secrets on the demo cluster (values needed from you)

Neither secret exists yet in `demo` org / project `model-factory`. The code
references them by these exact names but runs without them (public
models/datasets; W&B in disabled mode). To enable them:

```bash
flyte --config ~/.flyte/config-model-factory.yaml create secret \
    NIELS_HUGGINGFACE_TOKEN --value <hf_...> --project model-factory --domain development

flyte --config ~/.flyte/config-model-factory.yaml create secret \
    NIELS_WANDB_API_KEY --value <wandb key> --project model-factory --domain development
```

Then re-run/deploy with secrets attached:

```bash
MF_USE_SECRETS=1 uv run flyte --config ~/.flyte/config-model-factory.yaml run run.py factory ...
MF_USE_SECRETS=1 uv run flyte --config ~/.flyte/config-model-factory.yaml deploy run.py factory_env
```

(`MF_USE_SECRETS` gates secret attachment because Flyte refuses to schedule
tasks whose declared secrets don't exist — see `model_factory/config.py`.)

W&B runs land in project `model-factory` (change `WANDB_PROJECT` in
`config.py` if you want a different entity/project).

## 2. Stubs / deferred integrations

- **OpenTelemetry / Grafana**: the docs describe `flyteplugins-otel`, but no
  OTLP endpoint is available for this demo, so it is not wired. To enable:
  add `flyteplugins-otel` to the images, call
  `from flyteplugins.otel import init; init(service_name="model-factory")`
  at module scope in `envs.py`, and set `OTEL_EXPORTER_OTLP_ENDPOINT` (e.g.
  Grafana Tempo) as an env var/secret. Traces (`@flyte.trace`) already give
  span-level lineage in the Union console meanwhile.
- **Slack notifications on trigger failures / gate openings**: needs a Slack
  webhook URL secret; `flyte.notify.Slack(...)` is ready to attach to the
  triggers in `pipeline.py`.
- **flyteplugins-wandb** (decorator-style W&B integration): current code uses
  manual `wandb.init` (robust to the missing secret). Once the secret
  exists, optionally switch `train.py` to `@wandb_init` +
  `get_wandb_run()` for run-linking in the Union UI.
- **vLLM serving app for the promoted model**: `flyteplugins.vllm.
  VLLMAppEnvironment` serving the `promoted-model` artifact behind an
  OpenAI-compatible endpoint — next iteration (needs an L4/A10G app pool on
  the demo cluster).

## 3. Dark-mode trigger activation (deliberate switch)

Triggers deploy `auto_activate=False`. To turn the factory dark:

```bash
flyte --config ~/.flyte/config-model-factory.yaml update trigger train-on-new-dataset train_on_new_dataset --activate
flyte --config ~/.flyte/config-model-factory.yaml update trigger eval-on-new-checkpoint eval_on_new_checkpoint --activate
flyte --config ~/.flyte/config-model-factory.yaml update trigger merge-synthetic-tasks merge_synthetic_into_dataset --activate
flyte --config ~/.flyte/config-model-factory.yaml update trigger nightly-synthetic-generation nightly_synthetic_batch --activate
```

## 4. Known POC limitations (accepted, spec'd for medium scope)

- Sandbox = subprocess + rlimits inside the task container, not
  microVM-isolated (docs/SPEC.md §8).
- vLLM colocate rollouts disabled in the smoke profile (`use_vllm=False`);
  dev profile enables it — needs a vllm-enabled GPU image (add `vllm` to
  `gpu_image` in `envs.py`) and an E2E validation pass.
- DAPO-style online difficulty filtering not yet applied inside the GRPO
  loop (curation-time filters only).
