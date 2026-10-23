"""Deploy unit for the factory's task environments.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy stations.py df_data_env
    uv run flyte --config $CFG deploy stations.py trainer_env
    uv run flyte --config $CFG deploy stations.py scorer_env
    uv run flyte --config $CFG deploy stations.py selector_env

The tasks are wired into the loop by the factory, not by this module; see
`decision_factory/factory.py`.
"""

from decision_factory.data.envs import data_env as df_data_env  # noqa: F401
from decision_factory.data.tasks import generate_episodes  # noqa: F401
from decision_factory.evaluation.tasks import score_model, select_champion  # noqa: F401
from decision_factory.evaluation.envs import scorer_env, selector_env  # noqa: F401
from decision_factory.training.envs import trainer_env  # noqa: F401
from decision_factory.training.tasks import finetune_decision_model  # noqa: F401
