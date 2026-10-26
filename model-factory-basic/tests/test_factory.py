"""The factory graph: structure, partitioning, the loop wiring, held-out isolation.

These run with no cluster: `Task.get(..., auto_version="latest")` is a lazy
reference, so the graph can be declared and structurally validated offline.
What CANNOT be checked here is the part that needs the registry and the
deployed task interfaces (parameter names, output arity, source partition
schemas) — `flyte factory deploy` does that, and it is the real gate.
"""

import pytest

from model_factory import contracts as c
from model_factory.factory import FACTORY_NAME, build_graph

DAILY_ARTIFACTS = (
    c.ARTIFACT_SEED_TASKS,
    c.ARTIFACT_SYNTHETIC,
    c.ARTIFACT_RL_DATASET,
    c.ARTIFACT_CHECKPOINT,
    c.ARTIFACT_EVAL_REPORT,
    c.ARTIFACT_PROMOTED,
)
FIXED_ARTIFACTS = (c.ARTIFACT_HELDOUT, c.ARTIFACT_BASELINE)


@pytest.fixture(scope="module")
def graph():
    return build_graph()


def inputs_of(graph, artifact: str) -> dict[str, tuple[str, str]]:
    """param -> (upstream artifact, mapping kind) for the build that makes ``artifact``."""
    build = graph._nodes[artifact].producer
    return {k: (q.handle.name, q.kind) for k, q in build.artifact_inputs.items()}


def upstream_closure(graph, artifact: str) -> set[str]:
    seen, stack = set(), [artifact]
    while stack:
        h = graph._nodes[stack.pop()]
        if h.producer is None:
            continue
        for name, _ in inputs_of(graph, h.name).values():
            if name not in seen:
                seen.add(name)
                stack.append(name)
    return seen


# ── shape ───────────────────────────────────────────────────────────────


def test_factory_declares_every_contract_artifact(graph):
    for name in (*FIXED_ARTIFACTS, *DAILY_ARTIFACTS, c.ARTIFACT_CHAMPION, c.ENDPOINT_APP):
        assert name in graph._nodes, f"{name} is not a node of the factory"


def test_structure_validates(graph):
    # Includes the cycle check: the loop must close OUTSIDE the graph.
    assert graph.validate() == []


def test_factory_name_matches_the_materialize_target(graph):
    assert graph.name == FACTORY_NAME


def test_it_produces_the_release_and_the_endpoint(graph):
    assert {h.name for h in graph.produces} == {c.ARTIFACT_PROMOTED, c.ENDPOINT_APP}


def test_the_endpoint_serves_the_release(graph):
    endpoint = graph._nodes[c.ENDPOINT_APP]
    assert endpoint.endpoint is True
    assert inputs_of(graph, c.ENDPOINT_APP) == {"model": (c.ARTIFACT_PROMOTED, "identity")}


def test_daily_artifacts_are_partitioned_by_date(graph):
    from flyteplugins.union import factory

    for name in DAILY_ARTIFACTS:
        assert graph._nodes[name].partitions == {c.PARTITION_DATE: factory.Daily}, name


def test_the_yardstick_is_built_once(graph):
    # No partitions: every day reads the same version, so day-over-day pass@1
    # numbers are comparable. A date dimension here would re-draw the split daily.
    for name in FIXED_ARTIFACTS:
        assert graph._nodes[name].partitions == {}, name


def test_models_are_declared_as_models(graph):
    for name in (c.ARTIFACT_CHECKPOINT, c.ARTIFACT_PROMOTED, c.ARTIFACT_CHAMPION):
        assert graph._nodes[name].kind == "model"


# ── the closed loop ─────────────────────────────────────────────────────


def test_the_champion_is_the_only_source(graph):
    # The loop's state comes in from outside (crown_champion publishes it);
    # everything else is built here.
    assert [n for n, h in graph._nodes.items() if h.source] == [c.ARTIFACT_CHAMPION]


def test_the_champion_is_per_day(graph):
    from flyteplugins.union import factory

    assert graph._nodes[c.ARTIFACT_CHAMPION].partitions == {c.PARTITION_DATE: factory.Daily}


def test_training_warm_starts_from_the_same_days_champion(graph):
    assert inputs_of(graph, c.ARTIFACT_CHECKPOINT)["champion"] == (c.ARTIFACT_CHAMPION, "identity")


def test_the_curriculum_is_seeded_by_the_champion(graph):
    assert inputs_of(graph, c.ARTIFACT_SYNTHETIC)["champion"] == (c.ARTIFACT_CHAMPION, "identity")


def test_the_candidate_is_judged_against_the_champion_and_the_baseline(graph):
    ins = inputs_of(graph, c.ARTIFACT_EVAL_REPORT)
    assert ins["champion"] == (c.ARTIFACT_CHAMPION, "identity")
    assert ins["baseline"] == (c.ARTIFACT_BASELINE, "identity")
    assert ins["heldout"] == (c.ARTIFACT_HELDOUT, "identity")
    assert inputs_of(graph, c.ARTIFACT_PROMOTED)["champion"] == (c.ARTIFACT_CHAMPION, "identity")


def test_nothing_in_the_graph_reads_its_own_release(graph):
    # The release only flows forward through the crown, never back into a build.
    for name, h in graph._nodes.items():
        if h.producer is None or name == c.ENDPOINT_APP:
            continue
        assert c.ARTIFACT_PROMOTED not in {n for n, _ in inputs_of(graph, name).values()}, name


def test_the_crown_trigger_watches_the_release_and_publishes_the_champion():
    from model_factory.evaluation.tasks import crown_champion, crown_on_promotion

    assert crown_on_promotion.automation.name == c.ARTIFACT_PROMOTED
    assert crown_on_promotion.auto_activate is True
    assert crown_champion.produces_artifacts is True


def test_the_nightly_trigger_targets_the_whole_release(graph):
    (trigger,) = graph.triggers
    assert {t.name for t in trigger.targets} == {c.ARTIFACT_PROMOTED, c.ENDPOINT_APP}


# ── held-out isolation, structurally ────────────────────────────────────


def test_training_never_receives_the_heldout_split(graph):
    assert c.ARTIFACT_HELDOUT not in {n for n, _ in inputs_of(graph, c.ARTIFACT_CHECKPOINT).values()}


def test_every_data_build_that_could_leak_is_handed_the_heldout_split_to_exclude(graph):
    # seed excludes it, synthetic drops matches, assemble refuses to publish a leak.
    for name in (c.ARTIFACT_SEED_TASKS, c.ARTIFACT_SYNTHETIC, c.ARTIFACT_RL_DATASET):
        assert inputs_of(graph, name)["heldout"] == (c.ARTIFACT_HELDOUT, "identity"), name


def test_the_baseline_never_depends_on_daily_data(graph):
    assert upstream_closure(graph, c.ARTIFACT_BASELINE) == {c.ARTIFACT_HELDOUT}


def test_graph_renders_as_mermaid(graph):
    mermaid = graph.graph()
    assert "flowchart" in mermaid
    assert c.ARTIFACT_CHAMPION in mermaid
