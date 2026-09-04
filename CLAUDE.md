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

- Deploying a unit resets its OnArtifact triggers to inactive and rebinds
  them to the new task version — reactivate triggers after every deploy
  that dark mode should pick up.
- The code bundler ships only statically-imported modules; import task
  dependencies at module top, not inside task bodies.
- `publish()` versions an artifact only when the wrapped value is
  RETURNED from a task.
