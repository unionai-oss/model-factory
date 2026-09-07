# model-factory

Monorepo of model-factory subprojects (`basic-model-factory/`,
`resource-tuner-model-factory/`), each self-contained (own `pyproject.toml`,
`uv.lock`, `.flyte/config.yaml`, tests, docs). Run commands from inside the
subproject directory. Cluster: demo.hosted.unionai.cloud.

## Research log — non-negotiable

Any experimentation (cluster runs, training, evals, synthetic data, dark
trigger runs) **requires updating `<factory>/research_log/`** in the
corresponding factory directory: run ledger with Union console URLs,
findings, code changes, index/standing-results refresh. Failures are
logged too, linked to their fixes. Use the `run-experiments` skill for the
full procedure; log entries ride the same branch/PR as the experiment.

## Operational notes

- resource-tuner triggers declare `auto_activate=True`: deploys leave dark
  mode LIVE (and rebind triggers to the new task version). Corollaries:
  a deploy also re-activates a trigger someone manually paused — pause by
  editing `auto_activate`, not just the console toggle; and any factory
  adding a trigger should set `auto_activate=True` unless it has a reason
  not to.
- The code bundler ships only statically-imported modules; import task
  dependencies at module top, not inside task bodies.
- `publish()` versions an artifact only when the wrapped value is
  RETURNED from a task.
