"""The factory graph: the fan-out over the suite and the collapse to a champion.

Runs with no cluster — `Task.get(..., auto_version="latest")` is a lazy
reference, so the graph declares offline. What cannot be checked here is what
needs the deployed task interfaces (parameter names, output arity, list-typed
parameters); `flyte factory deploy` validates those and is the real gate.
"""

import pytest

from decision_factory import contracts
from decision_factory.config import MODEL_SUITE, get_profile
from decision_factory.factory import FACTORY_NAME, build_graph, materialize_args


@pytest.fixture(scope="module")
def graph():
    return build_graph("smoke")


def test_structure_validates(graph):
    assert graph.validate() == []


def test_factory_name_matches_the_materialize_target(graph):
    assert graph.name == FACTORY_NAME


def test_it_declares_every_contract_artifact(graph):
    for name in (
        contracts.ARTIFACT_EPISODES,
        contracts.ARTIFACT_MODEL,
        contracts.ARTIFACT_SCORECARD,
        contracts.ARTIFACT_CHAMPION,
    ):
        assert name in graph._nodes


def test_it_produces_the_champion_and_the_endpoint(graph):
    assert {h.name for h in graph.produces} == {
        contracts.ARTIFACT_CHAMPION,
        contracts.ENDPOINT_APP,
    }


def test_the_suite_fans_out_over_a_model_partition(graph):
    # The per-model builds must carry BOTH dimensions, or there is only one
    # fine-tune instead of one per suite member.
    for name in (contracts.ARTIFACT_MODEL, contracts.ARTIFACT_SCORECARD):
        dims = graph._nodes[name].partitions
        assert contracts.PARTITION_DATE in dims, name
        assert contracts.PARTITION_MODEL in dims, name
        assert dims[contracts.PARTITION_MODEL] is str


def test_the_champion_collapses_the_model_dimension(graph):
    # This is the load-bearing property: the champion is ONE artifact per day,
    # chosen across the suite. If `model` survived here, there would be a
    # champion per model, which is not a choice at all.
    dims = graph._nodes[contracts.ARTIFACT_CHAMPION].partitions
    assert contracts.PARTITION_DATE in dims
    assert contracts.PARTITION_MODEL not in dims


def test_the_champion_build_reads_the_whole_suite(graph):
    build = graph._nodes[contracts.ARTIFACT_CHAMPION].producer
    queries = build.artifact_inputs
    # Both of its artifact inputs must be `.all(model)` list mappings.
    for param in ("candidates", "scorecards"):
        q = queries[param]
        assert q.kind == "all", f"{param} is {q.kind}, not all(model)"
        assert q.dim == contracts.PARTITION_MODEL
        assert q.is_list is True


def test_episodes_are_the_only_root(graph):
    roots = [n for n, h in graph._nodes.items() if h.producer is None or h.source]
    assert roots == [] or roots == [contracts.ARTIFACT_EPISODES]
    # Episodes are built (from the oracle), not sourced from outside.
    assert graph._nodes[contracts.ARTIFACT_EPISODES].source is False


def test_models_are_declared_as_models(graph):
    for name in (contracts.ARTIFACT_MODEL, contracts.ARTIFACT_CHAMPION):
        assert graph._nodes[name].kind == "model"


def test_the_endpoint_is_an_endpoint(graph):
    endpoint = graph._nodes[contracts.ENDPOINT_APP]
    assert endpoint.endpoint is True


def test_the_date_dimension_is_daily(graph):
    from flyteplugins.union import factory

    dims = graph._nodes[contracts.ARTIFACT_EPISODES].partitions
    assert dims[contracts.PARTITION_DATE] is factory.Daily


def test_nightly_trigger_targets_the_champion_and_the_endpoint(graph):
    (trigger,) = graph.triggers
    assert {t.name for t in trigger.targets} == {
        contracts.ARTIFACT_CHAMPION,
        contracts.ENDPOINT_APP,
    }


def test_materialize_args_names_the_profile_suite():
    args = materialize_args("smoke")
    assert args[contracts.PARTITION_MODEL] == list(get_profile("smoke").models)


def test_suite_slugs_are_valid_partition_values():
    # A partition value cannot contain '/', which every HF repo id does — the
    # whole reason the suite is keyed by slug rather than by repo id.
    for slug, candidate in MODEL_SUITE.items():
        assert "/" not in slug, slug
        assert slug == slug.lower()
        assert "/" in candidate.hf_id, f"{slug} should map to a HF repo id"


def test_every_profile_names_only_known_models():
    for name in ("smoke", "dev", "full"):
        profile = get_profile(name)
        assert profile.models, name
        for slug in profile.models:
            assert slug in MODEL_SUITE, f"{name} references unknown model {slug}"


def test_the_default_suite_is_ungated():
    # A gated model needs a licence acceptance plus a token; one in a default
    # profile makes the factory fail for anyone without it.
    for name in ("smoke", "dev", "full"):
        for candidate in get_profile(name).candidates:
            assert candidate.gated is False, f"{name} includes gated {candidate.slug}"


def test_smoke_trains_more_than_one_model():
    # A "suite" of one has nothing to compare, so champion selection would be
    # vacuous and the `.all` collapse untested by a smoke run.
    assert len(get_profile("smoke").models) >= 2


def test_graph_renders_as_mermaid(graph):
    mermaid = graph.graph()
    assert "flowchart" in mermaid
    assert "all(model)" in mermaid  # the collapse should be visible
