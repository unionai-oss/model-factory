"""The resource-tuner decision-arm factory graph.

The property that matters most here is the one in
`test_the_corpus_is_a_source_not_a_build`: the corpus being a `factory.source`
is what let the whole arm be added without touching the LLM/GRPO path.
"""

import pytest

from resource_tuner.contracts import ARTIFACT_TASK_CORPUS
from resource_tuner.factory import (
    ARTIFACT_DECISION_CHAMPION,
    ARTIFACT_DECISION_MODEL,
    ARTIFACT_DECISION_SCORECARD,
    FACTORY_NAME,
    PARTITION_ARM,
    build_graph,
    materialize_args,
)
from resource_tuner.training.decision_model import DEFAULT_ARMS


@pytest.fixture(scope="module")
def graph():
    return build_graph()


def test_structure_validates(graph):
    assert graph.validate() == []


def test_factory_name_matches_the_materialize_target(graph):
    assert graph.name == FACTORY_NAME


def test_the_corpus_is_a_source_not_a_build(graph):
    # A source is an artifact made OUTSIDE the factory. That is what keeps the
    # existing build_task_corpus station, its OnArtifact triggers and the LLM
    # arm untouched by this factory.
    corpus = graph._nodes[ARTIFACT_TASK_CORPUS]
    assert corpus.source is True
    assert corpus.producer is None


def test_the_source_keeps_its_original_unpartitioned_name(graph):
    # Renaming it would orphan the three versions already in the registry and
    # break the LLM arm's triggers.
    assert ARTIFACT_TASK_CORPUS == "tuning-task-corpus"
    assert graph._nodes[ARTIFACT_TASK_CORPUS].partitions == {}


def test_the_suite_fans_out_over_an_arm_partition(graph):
    for name in (ARTIFACT_DECISION_MODEL, ARTIFACT_DECISION_SCORECARD):
        dims = graph._nodes[name].partitions
        assert PARTITION_ARM in dims, name
        assert dims[PARTITION_ARM] is str


def test_the_champion_collapses_the_arm_dimension(graph):
    # One champion across the suite, not one per arm.
    assert PARTITION_ARM not in graph._nodes[ARTIFACT_DECISION_CHAMPION].partitions


def test_the_champion_build_reads_the_whole_suite(graph):
    build = graph._nodes[ARTIFACT_DECISION_CHAMPION].producer
    for param in ("models", "scorecards"):
        q = build.artifact_inputs[param]
        assert q.kind == "all" and q.dim == PARTITION_ARM and q.is_list


def test_it_produces_only_the_champion(graph):
    assert {h.name for h in graph.produces} == {ARTIFACT_DECISION_CHAMPION}


def test_partitioned_names_avoid_the_pre_existing_unpartitioned_ones(graph):
    # An artifact's partition schema is fixed by its first version, and these
    # legacy names already exist unpartitioned from earlier rounds.
    legacy = {
        "tuner-checkpoint",
        "ml-baseline-model",
        "promoted-tuner",
        "tuner-eval-report",
        "synthetic-task-corpus",
    }
    partitioned = {n for n, h in graph._nodes.items() if h.partitions and not h.source}
    assert partitioned & legacy == set()


def test_models_are_declared_as_models(graph):
    for name in (ARTIFACT_DECISION_MODEL, ARTIFACT_DECISION_CHAMPION):
        assert graph._nodes[name].kind == "model"


def test_the_trigger_fires_on_a_new_corpus_version(graph):
    # The trigger shape only a source makes available: new corpus -> refit the
    # whole suite.
    (trigger,) = graph.triggers
    assert trigger.event is graph._nodes[ARTIFACT_TASK_CORPUS]
    assert {t.name for t in trigger.targets} == {ARTIFACT_DECISION_CHAMPION}


def test_materialize_args_names_the_default_suite():
    assert materialize_args()[PARTITION_ARM] == list(DEFAULT_ARMS)


def test_arm_names_are_valid_partition_values():
    for arm in DEFAULT_ARMS:
        assert "/" not in arm and arm == arm.lower()


def test_graph_renders_the_collapse(graph):
    mermaid = graph.graph()
    assert "all(arm)" in mermaid
    assert "source" in mermaid
