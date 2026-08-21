"""Entrypoint for deploying the lineage app.

    flyte --config ~/.flyte/config-model-factory.yaml deploy app.py lineage_app_env
"""

from model_factory.lineage_app import lineage_app_env  # noqa: F401
