"""Inference team deploy unit: the serving app.

    flyte --config ~/.flyte/config-model-factory.yaml deploy team_inference.py inference_app_env

Deploying this env is also what builds the app's image. `flyte factory deploy`
does NOT build it, so the app env has to be deployed at least once before the
factory's `mf-inference` endpoint is materialized — otherwise the app
crash-loops pulling an image that was never pushed.

The app no longer polls for checkpoints: the factory binds its `model`
parameter to the `promoted-model` version each materialization resolved.
"""

from model_factory.inference.service import inference_app_env  # noqa: F401
