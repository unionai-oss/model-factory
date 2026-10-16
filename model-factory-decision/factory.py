"""Entrypoint for deploying the factory.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG factory deploy factory.py

    # one full suite run: every model in the profile, for one day
    uv run flyte --config $CFG factory materialize decision-factory permission-champion \
        --partition date=2026-10-15 --partition model=qwen2.5-0.5b-pointer,qwen2.5-0.5b-sft --wait

    # then deploy the winner
    uv run flyte --config $CFG factory materialize decision-factory permission-gate \
        --partition date=2026-10-15 --partition model=qwen2.5-0.5b-pointer,qwen2.5-0.5b-sft --wait

    # just one model's scorecard
    uv run flyte --config $CFG factory materialize decision-factory decision-scorecard \
        --partition date=2026-10-15 --partition model=qwen2.5-0.5b

    # roll the endpoint back to an earlier champion
    uv run flyte --config $CFG factory materialize decision-factory permission-gate \
        --partition date=2026-10-15 --version permission-champion=<version>

The graph lives in `decision_factory/factory.py`; this file exists because the
deploy CLI loads a module by path, which a package-relative import cannot
satisfy.
"""

from decision_factory.factory import decision_factory  # noqa: F401
