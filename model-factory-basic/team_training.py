"""Model training team deploy unit.

    flyte --config .flyte/config.yaml deploy team_training.py trainer_env

`train_grpo` is the factory's `policy-checkpoint` build: it reads the day's
`rl-tasks-dataset` and warm-starts from the day's `champion`. See
`model_factory/factory.py`.
"""

from model_factory.training.envs import trainer_env  # noqa: F401
from model_factory.training.tasks import train_grpo  # noqa: F401
