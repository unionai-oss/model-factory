"""resource-tuner-model-factory deploy unit.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy main.py driver_env

    # E2E smoke (corpus -> GRPO train on T4 -> eval + real cluster episodes):
    uv run flyte --config $CFG run main.py tuner_pipeline --profile_name smoke

    # stations individually:
    uv run flyte --config $CFG run main.py build_task_corpus --profile_name smoke
"""

from resource_tuner.environment.harness import harness_env, run_generated  # noqa: F401
from resource_tuner.training.envs import driver_env, trainer_env  # noqa: F401
from resource_tuner.training.evaluate import eval_tuner  # noqa: F401
from resource_tuner.training.grpo import train_tuner  # noqa: F401
from resource_tuner.training.stations import build_task_corpus, tuner_pipeline  # noqa: F401
