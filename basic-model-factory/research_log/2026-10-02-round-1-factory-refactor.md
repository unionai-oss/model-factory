# Round 1 — the trigger chain becomes a factory (2026-10-02)

## Context

The loop was wired by hand: each station published an artifact, an
`OnArtifact` trigger on that name fired the next station, and
`integration.py` re-implemented the whole chain a second time so it could be
tested in one run. That worked but gave no way to ask for a single target, no
reuse between runs, no backfill, and a test harness that duplicated
production.

Union factories (preview; needs `flyte>=2.10.0` +
`flyteplugins-union>=0.12.0`) replace exactly that layer: declare what each
artifact is made of, then materialize a target. This round ports the loop and
verifies it end to end on the demo tenant.

## Run ledger

Base: `https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

Probe runs first (a throwaway `probe-factory`, two trivial tasks) to learn the
API against the real control plane before touching the factory:

| run | what | result |
|---|---|---|
| [ukl5kxbs2448hghj6hbx](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/ukl5kxbs2448hghj6hbx) | probe materialize, `day: str` | **FAILED** — `Type conversion failed for variable 'day'. Expected str, got datetime` |
| [u9rt49g88qs5zkllfvff](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u9rt49g88qs5zkllfvff) | same, `day: datetime` | 1 built / 1 built |
| [u5ndwk47kx9vwn9xwn7w](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u5ndwk47kx9vwn9xwn7w) | re-materialize, unchanged | **all reused**, seconds |
| [uc24w4zgfvn82wck2nrl](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/uc24w4zgfvn82wck2nrl) | first `factory.serve` endpoint | endpoint built, app **CRASH_LOOP** (image never built) |
| [uqvsz8pb4kwhgm4vvpxn](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/uqvsz8pb4kwhgm4vvpxn) | after `flyte deploy` of the app env | still CRASH_LOOP — `Parameter(mount="/summaries")` |
| [ukj5x67br7cr7l5hfsg5](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/ukj5x67br7cr7l5hfsg5) | after dropping `mount=` | endpoint built; `/summary` returned the factory-built payload — but from the previous revision (see finding 10) |

Then the real factory:

| run | target | result |
|---|---|---|
| [bmf-smoke-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-smoke-r1) | `mf-promoted-model`, `date=2026-10-02` | **6 built**, ~12 min wall clock |
| [bmf-serve-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/bmf-serve-r1) | `mf-inference`, same partition | 6 reused + endpoint built |
| [u883f0f834236535d](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u883f0f834236535d) | **nightly cron**, `mf-promoted-model` — fired by itself at 06:00 UTC | all 6 cache hits, **9.1s** |
| [u7a78a1e42765c410](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/u7a78a1e42765c410) | **nightly cron**, `mf-inference` | 10.3s |

`bmf-smoke-r1` station timings: `ingest_and_curate` 2m0s,
`generate_synthetic_tasks` 3m33s, `assemble_dataset` 5s (gate auto-approved),
`train_grpo` → `evaluate_checkpoint` → `promote_checkpoint` the rest.

Serving verified live against `mf-inference`
(`https://young-sea-071c4.apps.demo.hosted.unionai.cloud`):
`/health` reported `loaded: true`, `base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct`,
and `/generate` with `use_adapter: true` returned a working `add_numbers`
implementation.

## Findings

**1. Factories work on this tenant, and the cache is per partition instance.**
A second materialization of an unchanged partition reports `reused` for every
node and finishes in seconds. That is the property the trigger chain never
had.

**2. An implicitly-injected time partition arrives as a `datetime`.** A task
parameter named like a `factory.Daily` dimension receives the partition value
automatically — typed `datetime`, not `str`. Annotating it `str` fails the
build. Cost one run (`ukl5kxbs2448hghj6hbx`).

**3. The factory publishes the artifacts; the task must not.** The driver
calls each build's task with `override(produces_artifacts=True)` inside a
`flyte.artifacts.produces(...)` block, declaring name / kind / partition
values / parent versions itself. So every station task now returns a plain
`File`/`Dir`, and `contracts.publish()` is gone. This deleted more code than
it added.

**4. `flyte factory deploy` does not build app images.** It builds only the
factory's own image. The `serve` app's spec references an image hash that must
already exist, so the app env has to be deployed once first. Cost two runs
(`uc24w4zgfvn82wck2nrl`, `uqvsz8pb4kwhgm4vvpxn`).

**5. `Parameter(mount=...)` at a filesystem-root path crash-loops the app.**
`mount="/summaries"` → exit 1, restart count 5+. Dropping `mount=` and reading
`flyte.app.get_parameter(name)` works; it returns a path relative to the app's
working directory, already downloaded.

**6. An artifact's partition schema is immutable once its first version
exists.** `factory deploy` rejected five of the six builds:

> `synthetic-tasks`: the registry already fixes this artifact's partitions and
> this build disagrees: declared `{date: Daily}` but the registry has no
> partitions. Match the registry or publish under a new artifact name

The pre-factory trigger chain had already published those names
unpartitioned. Chose the `mf-` prefix over deleting the old artifacts: it
keeps the daily dimension (and with it per-day reuse and backfill) and leaves
the earlier rounds' lineage intact. Recorded in `contracts.py` so the next
person does not rediscover it.

**7. `Artifact.listall` now works on the demo control plane.** The run-scanning
fallback in `shared/assets.py` — written because the artifact CRUD API
returned Not Found — is deleted. Partition values read back correctly, but
from `spec.timePartition` (`{key, granularity, value.timeValue}`), **not**
`spec.artifactId.partitions` where the first attempt looked.

**8. The factory stamps lineage automatically.** Every published version
carries `userMetadata`:

```
flyte.io/materialization: bmf-smoke-r1
flyte.io/factory:         model-factory@f8372ad9e7fe9052
flyte.io/consumed/checkpoint:  demo/model-factory/development/mf-policy-checkpoint@4121c7b5022f...
flyte.io/consumed/eval_report: demo/model-factory/development/mf-eval-report@3da68d0f7b23...
```

That is per-parameter provenance down to the artifact version, for free.

**9. Routing eval through the serving app was circular.** `evaluate_checkpoint`
preferred generating via `mf-inference`, but the factory deploys that app
*downstream* of the eval report (it serves `promoted-model`, which is gated on
the report). The build now passes `use_service=False`. The trigger chain hid
this because nothing forced the dependency order to be stated.

**10. A failed `serve` redeploy leaves the previous revision serving, which
can mask it.** The probe's last materialization redeployed the app from the
factory's *compiled* copy of the app env — which still carried
`mount="/summaries"`, because the factory had been deployed before that fix.
That revision crash-looped, while Knative kept the previously-good revision
answering: `/summary` returned 200 with the correct factory-built payload the
whole time. Checking the endpoint is therefore **not** sufficient to verify a
serve step; the app's `conditions` and `spec.inputs` are. Corollary: changing
an app env means redeploying the factory too, since the app spec is compiled
into the factory's graph.

(The two real endpoints were verified the right way: `mf-inference` reached
`DEPLOYMENT_STATUS_ACTIVE / RUNNING` with `spec.inputs` pinning
`mf-promoted-model@7fc0eca77cba`, and `/health` reporting
`loaded: true, base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct`.)

**11. The nightly trigger fired on its own, and the cache claim held.** The
`flyte.Cron("0 6 * * *")` trigger declared on the `Factory` fired at
06:00:00 UTC without being activated by hand, ~25 minutes after the factory
was deployed, and materialized the whole graph as **cache hits**: every child
action took 90ms-2.8s against 2min/3m33s when it actually built, for 9.1s
total. That is the "cheap to leave armed" property the docs promise, measured.

Also: one `factory.on(Cron, promoted, endpoint)` compiles into **one trigger
per target** — `nightly-release-mf-promoted-model` and
`nightly-release-mf-inference` ran as two separate runs, both at 06:00:00.

## Code changes

- **new** `model_factory/factory.py` — the graph:
  `mf-seed-tasks → mf-synthetic-tasks → mf-rl-tasks-dataset →
  mf-policy-checkpoint → mf-eval-report → mf-promoted-model → mf-inference`,
  all partitioned by `date: Daily`, plus a nightly `flyte.Cron` trigger.
- **new** `factory.py` (top-level deploy entrypoint) and `tests/test_factory.py`
  (12 offline graph tests).
- **deleted** `integration.py`, `data_engineering/release.py`,
  `inference/tasks.py`, `publish_dataset`, `eval_and_promote`, and all four
  `OnArtifact`/`Cron` triggers.
- Station tasks return plain `File`/`Dir`. `merge_datasets` became
  `assemble_dataset` and now carries the data-validation gate;
  `promote_checkpoint` carries the promotion gate and the auto-margin check.
  Both gates are still in the code path — `mf-promoted-model` cannot exist
  without them passing — now as build parameters rather than orchestrator
  logic.
- `inference/service.py`: the app takes a `model` parameter bound to
  `ArtifactValue(ARTIFACT_PROMOTED)` and preloads it in the background on
  startup; `/reload` kept for manual override. It no longer scans the registry
  for a checkpoint.
- `shared/assets.py`: registry-only, fixed partition parsing.
- `contracts.py`: `mf-`-prefixed names, `ENDPOINT_APP`, no `publish()`,
  no `InferenceEndpoint`.
- `pyproject.toml`: `flyte>=2.10.0`, `flyteplugins-union>=0.12.0`.

Tests: 68 pass (was 55).

## Bootstrap order (the part that is not obvious)

```
1. flyte deploy team_data.py de_cpu_env          # + trainer, eval envs
2. flyte deploy team_inference.py inference_app_env   # builds the app image;
                                                      # FAILS to deploy the app
                                                      # until a champion exists
3. flyte factory deploy factory.py
4. flyte factory materialize model-factory mf-promoted-model --partition date=<day>
5. flyte factory materialize model-factory mf-inference      --partition date=<day>
```

Step 2 failing with `Failed to materialize artifact mf-promoted-model@latest`
on a cold registry is expected: the image is built by then, which is all the
factory needs.

## Next

- Arm the gates (`AUTO_APPROVE = False`) and confirm a materialization parks
  on the condition rather than failing.
- Backfill a week (`--partition date=2026-09-25..2026-10-02`) to exercise
  one-action-per-partition fan-out.
- A `dev`-profile materialization, to put a real pass@1 number on the board.
