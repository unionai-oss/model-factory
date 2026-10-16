"""Data engineering team deploy unit.

    flyte --config ~/.flyte/config-model-factory.yaml deploy team_data.py de_cpu_env

The team's tasks are wired into the loop by the factory, not by this module:
see `model_factory/factory.py` for the `seed-tasks` -> `synthetic-tasks` ->
`rl-tasks-dataset` builds, and run them with

    flyte factory materialize model-factory rl-tasks-dataset --partition date=<day>
"""

from model_factory.data_engineering.envs import de_cpu_env, de_gpu_env  # noqa: F401
from model_factory.data_engineering.synthetic import generate_synthetic_tasks  # noqa: F401
from model_factory.data_engineering.tasks import assemble_dataset, ingest_and_curate  # noqa: F401
