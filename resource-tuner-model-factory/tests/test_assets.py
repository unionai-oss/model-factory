"""Artifact spec parsing (the blob-URI / display-string lessons)."""

from resource_tuner.shared.assets import _blob_uri, _created_at, _source_ids

SPEC = {
    "metadata": {"createdAt": "2026-09-03T04:00:00Z"},
    "spec": {
        "value": {"scalar": {"blob": {"uri": "s3://bucket/aa/report.json"}}},
        "source": {
            "taskAction": {
                "action": {"run": {"name": "urun123"}, "name": "action9"},
                "attempt": 1,
            }
        },
    },
}


class FakeArtifact:
    source = "run urun123/action9 (attempt 1)"  # human string — never an ID

    def __init__(self, d=SPEC):
        self._d = d

    def to_dict(self):
        return self._d


def test_blob_uri_and_ids_come_from_the_spec():
    a = FakeArtifact()
    assert _blob_uri(a) == "s3://bucket/aa/report.json"
    assert _source_ids(a) == ("urun123", "action9")
    assert _created_at(a) == "2026-09-03T04:00:00Z"


def test_malformed_specs_degrade_to_nothing():
    assert _blob_uri(FakeArtifact({})) is None
    assert _source_ids(FakeArtifact({})) == ("", "")
    assert _created_at(object()) == ""
