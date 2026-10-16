# model-factory

Monorepo of model-factory subprojects (`basic-model-factory/`,
`decision-model-factory/`, `resource-tuner-model-factory/`), each
self-contained (own `pyproject.toml`, `uv.lock`, `.flyte/config.yaml`, tests,
docs). Run commands from inside the subproject directory. Cluster:
demo.hosted.unionai.cloud.

## Research log — non-negotiable

Any experimentation (cluster runs, training, evals, synthetic data, dark
trigger runs, factory materializations) **requires updating
`<factory>/research_log/`** in the corresponding factory directory: run ledger
with Union console URLs, findings, code changes, index/standing-results
refresh. Failures are logged too, linked to their fixes. Use the
`run-experiments` skill for the full procedure; log entries ride the same
branch/PR as the experiment.

## Factories

All three projects declare their pipeline as a Union **factory** (preview;
needs `flyte>=2.10.0` + `flyteplugins-union>=0.12.0`). The graph lives in
`<pkg>/factory.py`, with a thin top-level `factory.py` as the deploy
entrypoint (the CLI loads a module by path, which a package-relative import
cannot satisfy).

- **Station tasks return plain `File`/`Dir`.** The factory declares and
  publishes each artifact — name, kind, partition values, parent versions — by
  calling the task with `produces_artifacts=True` inside a
  `flyte.artifacts.produces(...)` block. A task that wraps its own output in
  `flyte.artifacts.new()` is fighting that.
- **An artifact's partition schema is immutable** once its first version is
  published. Declaring new dimensions on a name that already exists
  unpartitioned is rejected at `factory deploy`; publish under a new name.
  This is why basic-model-factory uses `mf-*` and the resource-tuner's
  decision arm uses `rt-*`.
- **A partition value injected into a same-named task parameter is typed.** A
  `factory.Daily` dimension arrives as a `datetime`, not a `str`.
- **`flyte factory deploy` does not build app images.** Deploy the app env
  once first (`flyte deploy app.py <app_env>`); on a cold registry that deploy
  itself fails resolving the artifact its parameter is bound to, which is
  expected — the image is built by then. Bootstrap order is
  stations → app env → factory → materialize target → materialize endpoint.
- **An app env's spec is compiled into the factory's graph**, so changing the
  app env means redeploying the factory too. A failed serve redeploy leaves
  the previous revision serving, so verify a serve step via the app's
  `conditions`/`spec.inputs`, not by curling the endpoint.
- **`flyte.app.Parameter(mount=...)` pointing outside the app's writable
  working directory crash-loops the container.** Use
  `flyte.app.get_parameter(name)`, which returns the already-downloaded path.
- Task images must pin a `flyte` floor that **tracks the deploying SDK**; an
  image a minor behind cannot read a factory task spec.

## Operational notes

- resource-tuner triggers declare `auto_activate=True`: deploys leave dark
  mode LIVE (and rebind triggers to the new task version). Corollaries:
  a deploy also re-activates a trigger someone manually paused — pause by
  editing `auto_activate`, not just the console toggle; and any factory
  adding a trigger should set `auto_activate=True` unless it has a reason
  not to.
- `factory.on(...)` triggers are declared on the `Factory`, not on a task,
  and they are **LIVE as soon as the factory is deployed** — a declared
  `flyte.Cron` fired on schedule with no manual activation (run
  `u883f0f834236535d`, 2026-10-02 06:00 UTC). One `factory.on(event, a, b)`
  compiles into one trigger per target, so it shows up as several runs.
  A triggered materialization with nothing changed is all cache hits and
  finishes in seconds (9.1s for a 6-station graph), so it is cheap to leave
  armed. `factory.on(<source>)` needs a `factory.source` — an artifact
  published outside the factory; everything a factory builds itself is
  reached by materializing it, not by a trigger.
- The code bundler ships only statically-imported modules; import task
  dependencies at module top, not inside task bodies.
- `flyte.remote.Artifact.listall` works on the demo tenant. Partition values
  read back from `spec.timePartition` (one time partition) and
  `spec.partitions.value.<key>.staticValue` (string partitions) — two
  different places.
- A ~10^6-row corpus cannot be read with `pd.read_parquet`; stream it
  (`pyarrow.ParquetFile.iter_batches`) with an explicit column projection.
- Helper calls inside async flyte tasks only fail on the cluster, after
  everything upstream has already run. Pin cross-module call shapes with a
  local test instead (see
  `resource-tuner-model-factory/tests/test_decision_model.py`).
