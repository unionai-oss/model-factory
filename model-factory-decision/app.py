"""Deploy unit for the decision API app.

    uv run flyte --config .flyte/config.yaml deploy app.py decision_app_env

Deploying this env is also what BUILDS the app's image. `flyte factory deploy`
does not build app images, so this has to run at least once before the
factory's `decision-api` endpoint is materialized.

On a cold registry (no `decision-champion` published yet) this deploy fails at
"Failed to materialize artifact decision-champion@latest" — the image is built
by then, which is all the factory needs. Bootstrap order:

    1. flyte deploy stations.py <each env>
    2. flyte deploy app.py decision_app_env        # builds the image
    3. flyte factory deploy factory.py
    4. flyte factory materialize decision-factory decision-champion ...
    5. flyte factory materialize decision-factory decision-api ...
"""

from decision_factory.serving.service import decision_app_env  # noqa: F401
