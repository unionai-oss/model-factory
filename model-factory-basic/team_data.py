"""Data engineering team deploy unit.

    flyte --config .flyte/config.yaml deploy --all team_data.py

The team's tasks are wired into the loop by the factory, not by this module:
see `model_factory/factory.py` for the `heldout-tasks`, `seed-tasks`,
`synthetic-tasks` and `rl-tasks-dataset` builds.
"""

from model_factory.data_engineering.envs import de_cpu_env, de_gpu_env  # noqa: F401
from model_factory.data_engineering.synthetic import (  # noqa: F401
    generate_synthetic_tasks,
    generate_synthetic_tasks_served,
)
from model_factory.data_engineering.tasks import (  # noqa: F401
    assemble_dataset,
    curate_heldout,
    ingest_and_curate,
)
