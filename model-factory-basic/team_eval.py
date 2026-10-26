"""Model eval team deploy unit.

    flyte --config .flyte/config.yaml deploy --all team_eval.py

Deploys the factory's eval builds (`baseline-report`, `eval-report`,
`promoted-model`) and the two tasks that publish `mf-champion` from outside
the factory:

- `crown_champion` carries the `crown-champion` OnArtifact trigger: every new
  `promoted-model[D]` becomes `champion[D+1]`. Deploying this unit arms it.
- `seed_champion` starts the loop (the base model as champion zero):

    flyte --config .flyte/config.yaml run team_eval.py seed_champion --date 2026-10-13
"""

from model_factory.evaluation.envs import eval_cpu_env, eval_gpu_env  # noqa: F401
from model_factory.evaluation.tasks import (  # noqa: F401
    crown_champion,
    evaluate_baseline,
    evaluate_checkpoint,
    promote_checkpoint,
    seed_champion,
)
