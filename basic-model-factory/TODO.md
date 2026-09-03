# TODO — human actions needed

## 1. ~~Create secrets on the demo cluster~~ — DONE (2026-09-02)

`HUGGINGFACE_TOKEN` and `WANDB_API_KEY` now exist in `demo` org / project
`model-factory` / domain `development`, and `model_factory/config.py`
references those names. CI deploys with `MF_USE_SECRETS=1`; for local
deploys/runs attach them the same way:

```bash
MF_USE_SECRETS=1 uv run flyte --config ~/.flyte/config-model-factory.yaml deploy team_data.py de_cpu_env
```

To recreate on another tenant:

```bash
flyte --config <cfg> create secret HUGGINGFACE_TOKEN --value <hf_...> --project model-factory --domain development
flyte --config <cfg> create secret WANDB_API_KEY --value <wandb key> --project model-factory --domain development
```

(`MF_USE_SECRETS` still gates secret attachment because Flyte refuses to
schedule tasks whose declared secrets don't exist on the target tenant — see
`model_factory/config.py`.)

W&B runs land in project `model-factory` (change `WANDB_PROJECT` in
`config.py` if you want a different entity/project).

## 2. Platform gap found during E2E: artifact service not live on demo backend

`flyte==2.6.0` ships the full artifact API (`flyte.artifacts.new`,
`flyte.remote.Artifact`, `OnArtifact` triggers) and the demo control plane
**accepts OnArtifact trigger definitions** (they deploy and register), but
its artifact CRUD service returns `Not Found` — so `artifacts.new()` versions
aren't queryable and artifact-event triggers can't actually fire yet.
Interestingly the produced-artifact intent IS recorded on action outputs
(`(produced artifact: rl-tasks-dataset)` in output literals).

Mitigation in place: `model_factory/assets.py` resolves "latest asset" by
scanning recent runs for producing station actions (used by dark-mode tasks
and the lineage app). It prefers the artifact API and falls back to run-scan,
so everything upgrades automatically when the backend ships artifacts.

**Empirically confirmed 2026-08-13**: with `train-on-new-dataset` ACTIVE, a
fresh `rl-tasks-dataset` publish (run `umstdqcbjv7vrfqbdb26`) spawned no
training run within 10+ minutes — artifact events are not delivered on this
backend. The cross-team cascade was validated by hand-cranking each trigger
task as its own run with the exact inputs the trigger would bind
(`integration.py` documents the pattern). **Action**: re-test after the next
demo-cluster upgrade — the wiring is already deployed and active.

Second networking finding: task pods get **HTTP 403** calling an app's
public `*.apps.demo.hosted.unionai.cloud` URL; in-cluster calls must use the
internal service DNS `http://<app>.<project>-<domain>.svc.cluster.local`
(handled automatically by `shared/inference_client.resolve_endpoint`).

## 3. Stubs / deferred integrations

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

## 4. Dark-mode trigger activation (deliberate switch)

Triggers deploy `auto_activate=False`. To turn the factory dark (one trigger
per team hand-off):

```bash
CFG=~/.flyte/config-model-factory.yaml
flyte --config $CFG update trigger train-on-new-dataset train_grpo --activate
flyte --config $CFG update trigger eval-on-new-checkpoint eval_and_promote --activate
flyte --config $CFG update trigger serve-new-checkpoint refresh_inference_service --activate
flyte --config $CFG update trigger merge-synthetic-tasks merge_synthetic_into_dataset --activate
flyte --config $CFG update trigger nightly-synthetic-generation nightly_synthetic_batch --activate
```

## 5. Known POC limitations (accepted, spec'd for medium scope)

- Sandbox = subprocess + rlimits inside the task container, not
  microVM-isolated (docs/SPEC.md §8).
- vLLM colocate rollouts disabled in the smoke profile (`use_vllm=False`);
  dev profile enables it — needs a vllm-enabled GPU image (add `vllm` to
  `gpu_image` in `envs.py`) and an E2E validation pass.
- DAPO-style online difficulty filtering not yet applied inside the GRPO
  loop (curation-time filters only).
