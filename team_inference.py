"""Inference team deploy unit: serving app + checkpoint-refresh ops.

    flyte --config ~/.flyte/config-model-factory.yaml deploy team_inference.py inference_app_env
    flyte --config ~/.flyte/config-model-factory.yaml deploy team_inference.py inference_ops_env
"""

from model_factory.inference.service import inference_app_env  # noqa: F401
from model_factory.inference.tasks import inference_ops_env, refresh_inference_service  # noqa: F401
