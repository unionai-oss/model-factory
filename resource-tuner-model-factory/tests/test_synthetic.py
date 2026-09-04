"""Synthetic station: teacher parsing, safety screen, oracle curation."""

import pytest

from resource_tuner.contracts import CORPUS_COLUMNS
from resource_tuner.taskgen.synthetic import (
    RejectedTask,
    curate_measurement,
    parse_teacher_response,
    synthetic_record,
    validate_task_code,
)

GOOD_CODE = """\
import time
import numpy as np

def run() -> dict:
    deadline = time.monotonic() + 60
    total = 0
    while time.monotonic() < deadline:
        x = np.ones((1000, 1000))
        total += int(x.sum())
        del x
    return {"total": total}
"""


def test_parse_teacher_response_tolerates_wrapping():
    raw = '<think>hmm</think>Here:\n{"description": "matrix sums", "code": "import time\\ndef run():\\n    return {}"}'
    desc, code = parse_teacher_response(raw)
    assert desc == "matrix sums"
    assert "def run():" in code


@pytest.mark.parametrize("raw", ["no json", '{"description": "x"}', '{"code": 5}'])
def test_parse_teacher_response_rejects(raw):
    with pytest.raises(RejectedTask):
        parse_teacher_response(raw)


def test_validate_accepts_realistic_code():
    validate_task_code(GOOD_CODE)


@pytest.mark.parametrize(
    "code,reason",
    [
        ("import os\ndef run(): return {}", "forbidden import"),
        ("import requests\ndef run(): return {}", "forbidden import"),
        ("import cowsay\ndef run(): return {}", "not in allowlist"),
        ("def run(): return open('/etc/passwd')", "forbidden builtin"),
        ("def run(): return eval('1')", "forbidden builtin"),
        ("import numpy\nx = 1", "no top-level def run"),
        ("def run(: return", "syntax error"),
    ],
)
def test_validate_rejects_unsafe_or_malformed(code, reason):
    with pytest.raises(RejectedTask, match=reason):
        validate_task_code(code)


def test_curation_bounds():
    ok = {"ok": True, "peak_rss_mib": 800.0, "duration_s": 70.0, "cpu_avg_cores": 1.2}
    assert curate_measurement(ok) is None
    assert "failed" in curate_measurement({**ok, "ok": False, "error": "boom"})
    assert "out of bounds" in curate_measurement({**ok, "peak_rss_mib": 50})
    assert "out of bounds" in curate_measurement({**ok, "peak_rss_mib": 90000})
    assert "out of bounds" in curate_measurement({**ok, "duration_s": 5})


def test_synthetic_record_matches_corpus_schema():
    measured = {"ok": True, "peak_rss_mib": 812.5, "duration_s": 71.0, "cpu_avg_cores": 2.3}
    r = synthetic_record("synthetic-0-1", "batch_inference", GOOD_CODE, "desc", measured)
    assert set(r) == set(CORPUS_COLUMNS)
    assert r["true_peak_memory_mib"] == 812.5
    assert r["true_cpu_cores"] == 2.3
    assert r["split"] == "train"
    # labels come from the ORACLE, so policy view == harness view
    assert r["source_code"] == r["harness_code"]
