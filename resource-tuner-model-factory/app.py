"""Entrypoint for deploying the lineage dashboard.

    uv run flyte --config .flyte/config.yaml deploy app.py lineage_app_env
"""

from resource_tuner.lineage_app import lineage_app_env  # noqa: F401
