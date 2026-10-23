"""The factory graph: structure, partitioning, and what it produces.

These run with no cluster: `Task.get(..., auto_version="latest")` is a lazy
reference, so the graph can be declared and structurally validated offline.
What CANNOT be checked here is the part that needs the registry and the
deployed task interfaces (parameter names, output arity, source partition
schemas) — `flyte factory deploy` does that, and it is the real gate.
"""

import pytest

from model_factory import contracts
from model_factory.factory import FACTORY_NAME, build_graph


@pytest.fixture(scope="module")
def graph():
    return build_graph()


def test_factory_declares_every_contract_artifact(graph):
    declared = set(graph._nodes)
    for name in (
        contracts.ARTIFACT_SEED_TASKS,
        contracts.ARTIFACT_SYNTHETIC,
        contracts.ARTIFACT_RL_DATASET,
        contracts.ARTIFACT_CHECKPOINT,
        contracts.ARTIFACT_EVAL_REPORT,
        contracts.ARTIFACT_PROMOTED,
    ):
        assert name in declared, f"{name} is not a node of the factory"


def test_structure_validates(graph):
    assert graph.validate() == []


def test_factory_name_matches_the_materialize_target(graph):
    # `flyte factory materialize <name> ...` and the deployed task name
    # `<name>.materialize` both come from here.
    assert graph.name == FACTORY_NAME


def test_it_produces_the_promoted_model_and_the_endpoint(graph):
    produced = {h.name for h in graph.produces}
    assert produced == {contracts.ARTIFACT_PROMOTED, contracts.ENDPOINT_APP}


def test_the_endpoint_is_an_endpoint_not_an_artifact(graph):
    endpoint = graph._nodes[contracts.ENDPOINT_APP]
    assert endpoint.endpoint is True
    # An endpoint holds no value type and is never published.
    assert endpoint.source is False


def test_every_artifact_is_partitioned_by_date(graph):
    for name, handle in graph._nodes.items():
        if handle.endpoint:
            continue
        assert contracts.PARTITION_DATE in handle.partitions, (
            f"{name} lost the {contracts.PARTITION_DATE} dimension; it should be "
            "inherited from seed-tasks by identity mapping"
        )


def test_the_date_dimension_is_daily(graph):
    from flyteplugins.union import factory

    seed = graph._nodes[contracts.ARTIFACT_SEED_TASKS]
    assert seed.partitions[contracts.PARTITION_DATE] is factory.Daily


def test_models_are_declared_as_models(graph):
    # `kind` is what makes an artifact show up as a model in the registry, and
    # what `Artifact.kind` reads back.
    for name in (contracts.ARTIFACT_CHECKPOINT, contracts.ARTIFACT_PROMOTED):
        assert graph._nodes[name].kind == "model"


def test_nothing_is_a_source(graph):
    # Everything in this factory is built from the seed dataset download, so
    # there is no external artifact to source. If one appears, `factory.on`
    # becomes available as a trigger event and this test should change.
    assert [n for n, h in graph._nodes.items() if h.source] == []


def test_the_nightly_trigger_targets_the_whole_release(graph):
    (trigger,) = graph.triggers
    targets = {t.name for t in trigger.targets}
    assert targets == {contracts.ARTIFACT_PROMOTED, contracts.ENDPOINT_APP}


def test_graph_renders_as_mermaid(graph):
    # The lineage dashboard and `factory deploy` both print this.
    mermaid = graph.graph()
    assert "flowchart" in mermaid
    assert contracts.ARTIFACT_PROMOTED in mermaid
