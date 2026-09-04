"""Entrypoint for deploying the tune service.

    uv run flyte --config .flyte/config.yaml deploy tune_app.py tune_service_env
"""

from resource_tuner.tune_service import tune_service_env  # noqa: F401
