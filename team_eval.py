"""Model eval team deploy unit.

    flyte --config ~/.flyte/config-model-factory.yaml deploy team_eval.py eval_gpu_env
    # dark mode: eval-on-new-checkpoint OnArtifact trigger targets eval_and_promote
"""

from model_factory.evaluation.envs import eval_cpu_env, eval_gpu_env  # noqa: F401
from model_factory.evaluation.tasks import (  # noqa: F401
    eval_and_promote,
    evaluate_checkpoint,
    promote_checkpoint,
)
