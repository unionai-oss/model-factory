"""Entrypoint for deploying the decision-arm factory.

    CFG=.flyte/config.yaml
    uv run flyte --config $CFG deploy decision.py decision_env
    uv run flyte --config $CFG factory deploy factory.py

    # full suite over the cost-asymmetry frontier
    uv run flyte --config $CFG factory materialize resource-tuner rt-decision-champion \
        --partition arm=balanced,oom-averse,oom-paranoid --wait

    # one arm only
    uv run flyte --config $CFG factory materialize resource-tuner rt-decision-scorecard \
        --partition arm=oom-averse

    # pin the corpus version instead of taking the latest
    uv run flyte --config $CFG factory materialize resource-tuner rt-decision-champion \
        --partition arm=balanced,oom-averse,oom-paranoid \
        --version tuning-task-corpus=<version>

The graph lives in `resource_tuner/factory.py`; this file exists because the
deploy CLI loads a module by path, which a package-relative import cannot
satisfy.
"""

from resource_tuner.factory import resource_tuner_factory  # noqa: F401
