"""Episode failure classification: OOM vs ordinary failure.

The distinction is reward-bearing (stage B's oom_penalty), so the
heuristic gets its own tests.
"""

import flyte.errors
import pytest

from resource_tuner.environment.episodes import _looks_like_oom


@pytest.mark.parametrize(
    "err",
    [
        RuntimeError("pod terminated: OOMKilled"),
        RuntimeError("container exited with code 137"),
        RuntimeError("CUDA out of memory"),
    ],
)
def test_oom_shaped_errors(err):
    assert _looks_like_oom(err)


def test_flyte_oom_error_type_counts():
    oom_cls = getattr(flyte.errors, "OOMError", None)
    if oom_cls is None:
        pytest.skip("this flyte version has no OOMError")
    try:
        err = oom_cls("oom")
    except TypeError:
        err = oom_cls("oom", "task")
    assert _looks_like_oom(err)


@pytest.mark.parametrize(
    "err",
    [RuntimeError("ValueError: bad column"), TimeoutError("deadline"), RuntimeError("")],
)
def test_ordinary_failures_are_not_oom(err):
    assert not _looks_like_oom(err)
