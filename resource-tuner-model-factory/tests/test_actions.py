"""Action space: parsing, bucketing, proposal validation."""

import pytest

from resource_tuner.policy.actions import (
    CPU_GRID,
    MEMORY_GRID_MIB,
    InvalidProposal,
    Proposal,
    bucket_cpu,
    bucket_memory_mib,
    format_memory,
    parse_memory_to_mib,
    validate_proposal,
)


def test_memory_parsing_units():
    assert parse_memory_to_mib("512Mi") == 512
    assert parse_memory_to_mib("4Gi") == 4096
    assert parse_memory_to_mib("1G") == pytest.approx(953.67, abs=0.1)
    assert parse_memory_to_mib(2048) == 2048  # bare numbers are MiB


@pytest.mark.parametrize("junk", ["", "lots", "-1Gi", "0", "4GiB extra", None, {}])
def test_memory_parsing_rejects_junk(junk):
    with pytest.raises((ValueError, TypeError)):
        parse_memory_to_mib(junk)


def test_bucketing_rounds_up_never_down():
    # Rounding down would under-provision — the failure direction that
    # costs a whole run.
    assert bucket_memory_mib(513) == 1024
    assert bucket_memory_mib(1024) == 1024
    assert bucket_memory_mib(10**9) == MEMORY_GRID_MIB[-1]
    assert bucket_cpu(1.1) == 2
    assert bucket_cpu(0.2) == 0.5
    assert bucket_cpu(999) == CPU_GRID[-1]


def test_format_memory_round_trips_the_grid():
    for step in MEMORY_GRID_MIB:
        assert parse_memory_to_mib(format_memory(step)) == step


def test_validate_proposal_happy_path():
    p = validate_proposal({"cpu": "500m", "memory": "900Mi"})
    assert p == Proposal(cpu=0.5, memory_mib=1024)
    assert p.to_kwargs() == {"cpu": "500m", "memory": "1Gi"}
    p2 = validate_proposal({"cpu": 2, "memory": "4Gi", "gpu": 1})
    assert p2.to_kwargs() == {"cpu": 2, "memory": "4Gi", "gpu": 1}


@pytest.mark.parametrize(
    "raw",
    [
        {"cpu": 1},  # missing memory
        {"memory": "1Gi"},  # missing cpu
        {"cpu": 1, "memory": "1Gi", "disk": "10Gi"},  # unknown key
        {"cpu": "fast", "memory": "1Gi"},
        {"cpu": 0, "memory": "1Gi"},
        {"cpu": 1, "memory": "1Gi", "gpu": 99},
        {"cpu": True, "memory": "1Gi"},  # bool is not a core count
        "cpu: 1",  # not a dict
    ],
)
def test_validate_proposal_rejects(raw):
    with pytest.raises(InvalidProposal):
        validate_proposal(raw)
