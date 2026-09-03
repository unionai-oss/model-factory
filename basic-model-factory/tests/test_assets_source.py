"""Artifact source parsing.

`Artifact.source` is a pre-rendered display string, not an identifier. Using
it as a run name produced console links like
`/runs/run%20uwbw.../5jow...%20(attempt%201)` (404) and silently broke every
lookup keyed by run name (the lineage graph's run -> station attribution).
"""

from model_factory.shared.assets import _blob_uri, _source_ids

SPEC = {
    "spec": {
        "value": {"scalar": {"blob": {"uri": "s3://bucket/aa/rl_tasks_merged.parquet"}}},
        "source": {
            "taskAction": {
                "action": {
                    "run": {
                        "org": "demo",
                        "project": "model-factory",
                        "domain": "development",
                        "name": "uwbwvdrsf2gzj27gmvgp",
                    },
                    "name": "5jowh5dnk10xa0q8i50btyw8c",
                },
                "attempt": 1,
            }
        },
    }
}


class _FakeArtifact:
    """Mimics the SDK object: `.source` renders for humans, spec has the IDs."""

    source = "run uwbwvdrsf2gzj27gmvgp/5jowh5dnk10xa0q8i50btyw8c (attempt 1)"

    def __init__(self, d=None):
        self._d = SPEC if d is None else d

    def to_dict(self):
        return self._d


def test_source_ids_are_identifiers_not_the_display_string():
    run_name, action_name = _source_ids(_FakeArtifact())
    assert run_name == "uwbwvdrsf2gzj27gmvgp"
    assert action_name == "5jowh5dnk10xa0q8i50btyw8c"
    # The trap: the human-readable attribute must not leak into the IDs.
    assert " " not in run_name and "(" not in run_name
    assert run_name != _FakeArtifact.source


def test_source_ids_survive_a_missing_or_odd_source():
    assert _source_ids(_FakeArtifact({})) == ("", "")
    assert _source_ids(_FakeArtifact({"spec": {"source": {}}})) == ("", "")
    assert _source_ids(object()) == ("", "")


def test_blob_uri_reads_object_storage_not_the_console_url():
    assert _blob_uri(_FakeArtifact()) == "s3://bucket/aa/rl_tasks_merged.parquet"
    assert _blob_uri(_FakeArtifact({})) is None
