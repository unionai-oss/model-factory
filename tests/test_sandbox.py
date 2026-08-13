from model_factory.sandbox import run_solution_against_tests

SOLUTION = """
def add(a, b):
    return a + b
"""

TESTS = """
from solution import add

def test_add():
    assert add(1, 2) == 3

def test_add_negative():
    assert add(-1, -2) == -3
"""


def test_passing_solution():
    result = run_solution_against_tests(SOLUTION, TESTS)
    assert result.passed
    assert not result.timed_out


def test_failing_solution():
    bad = "def add(a, b):\n    return a - b\n"
    result = run_solution_against_tests(bad, TESTS)
    assert not result.passed
    assert result.returncode != 0


def test_syntax_error_solution():
    result = run_solution_against_tests("def add(a, b:\n", TESTS)
    assert not result.passed


def test_missing_function():
    result = run_solution_against_tests("x = 1\n", TESTS)
    assert not result.passed


def test_timeout():
    slow = "def add(a, b):\n    while True: pass\n"
    result = run_solution_against_tests(slow, TESTS, timeout_s=3)
    assert not result.passed
    assert result.timed_out or result.returncode != 0


def test_isolated_env_no_secrets():
    """Env vars from the parent (tokens etc.) must not leak into the sandbox."""
    import os

    os.environ["MF_TEST_SECRET"] = "leaky"
    try:
        probe_tests = """
import os

def test_probe():
    assert os.environ.get("MF_TEST_SECRET") is None
"""
        result = run_solution_against_tests("x = 1\n", probe_tests)
        assert result.passed
    finally:
        del os.environ["MF_TEST_SECRET"]
