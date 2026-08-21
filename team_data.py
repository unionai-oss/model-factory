"""Data engineering team deploy unit.

    flyte --config ~/.flyte/config-model-factory.yaml deploy team_data.py de_cpu_env
    flyte --config ~/.flyte/config-model-factory.yaml run team_data.py data_release --profile_name smoke
"""

from model_factory.data_engineering.envs import de_cpu_env, de_gpu_env  # noqa: F401
from model_factory.data_engineering.release import (  # noqa: F401
    data_release,
    merge_synthetic_into_dataset,
    nightly_synthetic_batch,
)
from model_factory.data_engineering.tasks import ingest_and_curate, publish_dataset  # noqa: F401
