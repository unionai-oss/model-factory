"""The inter-team contract: artifact names and payload schemas.

Artifacts are the ONLY interface between teams, so the registry and payload
shapes are load-bearing — renaming an artifact or dropping a schema column
breaks another team's OnArtifact trigger without any import error.
"""

import pytest

from model_factory import contracts


def test_artifact_names_are_unique_and_url_safe():
    names = [
        contracts.ARTIFACT_RL_DATASET,
        contracts.ARTIFACT_SYNTHETIC,
        contracts.ARTIFACT_CHECKPOINT,
        contracts.ARTIFACT_EVAL_REPORT,
        contracts.ARTIFACT_PROMOTED,
        contracts.ARTIFACT_INFERENCE_ENDPOINT,
    ]
    assert len(set(names)) == len(names)
    # OnArtifact trigger names end up in URLs and label selectors.
    for n in names:
        assert n == n.lower() and " " not in n


def test_dataset_schema_names_the_split_column():
    # Training filters on split == "train", eval on "heldout"; both teams
    # read these exact column names.
    for col in ("task_id", "question", "tests", "split"):
        assert col in contracts.DATASET_COLUMNS


def test_checkpoint_manifest_names_base_model():
    # The serving app loads manifest["base_model"] before the adapter.
    assert "base_model" in contracts.CHECKPOINT_MANIFEST_KEYS


def test_inference_endpoint_round_trips_through_json():
    ep = contracts.InferenceEndpoint(
        url="https://app.example.com",
        base_model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        checkpoint_path="s3://bucket/ckpt",
        checkpoint_run="uwbwvdrsf2gzj27gmvgp",
    )
    assert contracts.InferenceEndpoint.from_json(ep.to_json()) == ep


def test_inference_endpoint_rejects_a_shape_drifted_payload():
    # A payload written by a newer producer with renamed fields must fail
    # loudly at the consumer, not half-populate.
    with pytest.raises(TypeError):
        contracts.InferenceEndpoint.from_json('{"url": "x", "endpoint": "y"}')
