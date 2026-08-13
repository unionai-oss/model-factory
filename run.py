"""CLI entrypoints for the model factory.

Run one full factory iteration (smoke profile, auto-approved gates):

    flyte --config ~/.flyte/config-model-factory.yaml run run.py factory \
        --profile_name smoke --auto_approve true

Run with human gates live (signal from the Union UI or
`flyte signal condition <run> <action> true`):

    flyte --config ~/.flyte/config-model-factory.yaml run run.py factory --profile_name smoke

Individual stations (debugging):

    flyte run run.py curate --profile_name smoke
    flyte run --local model_factory/data.py ...   # (won't work: use this file)

Deploy dark-mode triggers + envs:

    flyte --config ~/.flyte/config-model-factory.yaml deploy run.py factory_env
"""

from model_factory.data import ingest_and_curate as curate  # noqa: F401
from model_factory.envs import cpu_env, factory_env, gpu_env  # noqa: F401
from model_factory.pipeline import (  # noqa: F401
    eval_on_new_checkpoint,
    merge_synthetic_into_dataset,
    nightly_synthetic_batch,
    run_factory as factory,
    train_on_new_dataset,
)

if __name__ == "__main__":
    import flyte

    flyte.init_from_config()
    run = flyte.run(factory, profile_name="smoke", auto_approve=True)
    print(run.url)
