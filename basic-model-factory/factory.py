"""Entrypoint for deploying the factory.

    flyte --config ~/.flyte/config-model-factory.yaml factory deploy factory.py

    # one release day, end to end (builds what is missing, reuses the rest):
    flyte --config ~/.flyte/config-model-factory.yaml factory materialize \
        model-factory mf-promoted-model --partition date=2026-10-02 --wait

    # just the data release, or just the serving endpoint:
    flyte ... factory materialize model-factory mf-rl-tasks-dataset --partition date=2026-10-02
    flyte ... factory materialize model-factory mf-inference        --partition date=2026-10-02

    # backfill a week; one action per day, run in parallel
    flyte ... factory materialize model-factory mf-promoted-model \
        --partition date=2026-09-25..2026-10-02

The graph itself lives in `model_factory/factory.py` — this file only exists
because the deploy CLI loads a module by path, which a package-relative
import cannot satisfy.
"""

from model_factory.factory import model_factory  # noqa: F401
