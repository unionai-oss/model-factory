"""The inter-team contract: artifact names, payload schemas, partitioning.

Artifacts are the ONLY interface between teams, and the factory graph is
declared against these exact names — renaming one breaks a build's output
declaration without any import error.
"""

from model_factory import contracts


ARTIFACT_NAMES = [
    contracts.ARTIFACT_SEED_TASKS,
    contracts.ARTIFACT_SYNTHETIC,
    contracts.ARTIFACT_RL_DATASET,
    contracts.ARTIFACT_CHECKPOINT,
    contracts.ARTIFACT_EVAL_REPORT,
    contracts.ARTIFACT_PROMOTED,
]


def test_artifact_names_are_unique_and_url_safe():
    assert len(set(ARTIFACT_NAMES)) == len(ARTIFACT_NAMES)
    # Artifact names end up in URLs, k8s labels and factory build declarations.
    for n in ARTIFACT_NAMES:
        assert n == n.lower() and " " not in n


def test_endpoint_is_not_one_of_the_artifacts():
    # The serve node is an app, never published to the registry; treating it as
    # an artifact is what the old `inference-endpoint` artifact did.
    assert contracts.ENDPOINT_APP not in ARTIFACT_NAMES


def test_endpoint_name_is_a_dns_label():
    # factory.serve() requires the app's name to be lowercase letters, digits
    # and '-', because it becomes a k8s/DNS label.
    name = contracts.ENDPOINT_APP
    assert name == name.lower()
    assert all(c.isalnum() or c == "-" for c in name)
    assert name[0].isalnum() and name[-1].isalnum()


def test_dataset_schema_names_the_split_column():
    # Training filters on split == "train", eval on "heldout"; both teams
    # read these exact column names.
    for col in ("task_id", "question", "tests", "split"):
        assert col in contracts.DATASET_COLUMNS


def test_checkpoint_manifest_names_base_model():
    # The serving app loads manifest["base_model"] before the adapter.
    assert "base_model" in contracts.CHECKPOINT_MANIFEST_KEYS


def test_eval_report_names_the_gate_field():
    # promote_checkpoint refuses to promote on auto_gate_passed being False.
    assert "auto_gate_passed" in contracts.EVAL_REPORT_KEYS


def test_partition_dimension_is_an_identifier():
    # A factory partition dimension must be a valid Python identifier: the
    # value is injected into a task parameter of the same name.
    assert contracts.PARTITION_DATE.isidentifier()
