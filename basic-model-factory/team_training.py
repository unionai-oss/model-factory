"""Model training team deploy unit.

    flyte --config ~/.flyte/config-model-factory.yaml deploy team_training.py trainer_env
    # manual (or dark-mode via the train-on-new-dataset OnArtifact trigger):
    flyte run team_training.py train_grpo --dataset <file> --profile_name smoke
"""

from model_factory.training.envs import trainer_env  # noqa: F401
from model_factory.training.tasks import train_grpo  # noqa: F401
