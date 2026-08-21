from model_factory.shared.rewards import (
    COMPILE_REWARD,
    FORMAT_REWARD,
    MAX_REWARD,
    build_prompt,
    check_guards,
    count_test_functions,
    extract_code,
    score_completion,
)

TESTS = """
from solution import add

def test_add():
    assert add(1, 2) == 3

def test_zero():
    assert add(0, 0) == 0
"""

GOOD_COMPLETION = """Here is my solution:

```python
def add(a, b):
    return a + b
```
"""


def test_extract_code_single_block():
    assert extract_code(GOOD_COMPLETION) == "def add(a, b):\n    return a + b"


def test_extract_code_no_block():
    assert extract_code("def add(a, b): return a + b") is None


def test_extract_code_two_blocks_rejected():
    two = "```python\nx = 1\n```\n```python\ny = 2\n```"
    assert extract_code(two) is None


def test_guards_catch_test_file_access():
    assert check_guards("import test_solution") is not None
    assert check_guards("open('test_solution.py')") is not None


def test_guards_catch_process_escape():
    assert check_guards("import subprocess") is not None
    assert check_guards("os.system('ls')") is not None
    assert check_guards("sys.exit(0)") is not None
    assert check_guards("__import__('os')") is not None


def test_guards_allow_normal_code():
    assert check_guards("import math\ndef f(x):\n    return math.sqrt(x)") is None


def test_score_full_reward():
    b = score_completion(GOOD_COMPLETION, TESTS)
    assert b.total == MAX_REWARD
    assert b.tests_passed


def test_score_failing_tests():
    completion = "```python\ndef add(a, b):\n    return a * b\n```"
    b = score_completion(completion, TESTS)
    assert b.total == FORMAT_REWARD + COMPILE_REWARD
    assert not b.tests_passed


def test_score_no_code_block():
    b = score_completion("I cannot solve this.", TESTS)
    assert b.total == 0.0


def test_score_syntax_error():
    b = score_completion("```python\ndef add(a, b:\n```", TESTS)
    assert b.total == FORMAT_REWARD


def test_score_guard_violation_zeroes_reward():
    hack = "```python\nimport subprocess\ndef add(a, b):\n    return a + b\n```"
    b = score_completion(hack, TESTS)
    assert b.total == 0.0
    assert b.guard_violation


def test_count_test_functions():
    assert count_test_functions(TESTS) == 2
    assert count_test_functions("not python !!!") == 0


def test_build_prompt_includes_signature():
    msgs = build_prompt("Add two numbers.", "def add(a, b):")
    assert msgs[0]["role"] == "system"
    assert "def add(a, b):" in msgs[1]["content"]
