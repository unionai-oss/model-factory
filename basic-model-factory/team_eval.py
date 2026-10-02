"""Model eval team deploy unit.

    flyte --config ~/.flyte/config-model-factory.yaml deploy team_eval.py eval_cpu_env

`evaluate_checkpoint` and `promote_checkpoint` are the factory's `eval-report`
and `promoted-model` builds; the promotion gates live inside
`promote_checkpoint`. See `model_factory/factory.py`.
"""

from model_factory.evaluation.envs import eval_cpu_env, eval_gpu_env  # noqa: F401
from model_factory.evaluation.tasks import evaluate_checkpoint, promote_checkpoint  # noqa: F401
