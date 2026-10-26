from model_factory.data_engineering.synthetic import parse_generation

WELL_FORMED = """QUESTION:
Write a function that returns the maximum of a list.

SOLUTION:
```python
def find_max(nums):
    return max(nums)
```

TESTS:
```python
from solution import find_max

def test_max():
    assert find_max([1, 3, 2]) == 3
```
"""


def test_parse_well_formed():
    parsed = parse_generation(WELL_FORMED)
    assert parsed is not None
    assert "find_max" in parsed["solution"]
    assert "test_max" in parsed["tests"]
    assert parsed["question"].startswith("Write a function")


def test_parse_missing_tests_block():
    text = "QUESTION:\nDo a thing.\n\nSOLUTION:\n```python\nx = 1\n```\n"
    assert parse_generation(text) is None


def test_parse_garbage():
    assert parse_generation("I refuse to answer.") is None


# What a 1.5B teacher actually writes (round 2: the strict parser kept 0/48).
MARKDOWN_HEADERS = """### QUESTION:
Sum the even numbers in a list.

### SOLUTION:
```python
def sum_even(xs):
    return sum(x for x in xs if x % 2 == 0)
```

### TESTS:
```python
def test_sum_even():
    assert sum_even([1, 2, 4]) == 6
```
This solution is efficient.
"""


def test_parse_tolerates_markdown_headers():
    parsed = parse_generation(MARKDOWN_HEADERS)
    assert parsed is not None
    assert parsed["question"] == "Sum the even numbers in a list."


def test_tests_that_forget_the_import_get_one():
    assert parse_generation(MARKDOWN_HEADERS)["tests"].startswith("from solution import *")


def test_an_explicit_import_is_left_alone():
    assert parse_generation(WELL_FORMED)["tests"].startswith("from solution import find_max")


def test_parsed_output_passes_the_oracle_end_to_end():
    from model_factory.shared.sandbox import run_solution_against_tests

    parsed = parse_generation(MARKDOWN_HEADERS)
    assert run_solution_against_tests(parsed["solution"], parsed["tests"]).passed


def test_bare_asserts_are_wrapped_into_a_test_function():
    from model_factory.data_engineering.synthetic import normalize_tests
    from model_factory.shared.rewards import count_test_functions
    from model_factory.shared.sandbox import run_solution_against_tests

    tests = normalize_tests("assert sum_even([2, 3]) == 2\nassert sum_even([]) == 0\n")
    assert count_test_functions(tests) == 1
    assert run_solution_against_tests("def sum_even(xs):\n    return sum(x for x in xs if x % 2 == 0)\n", tests).passed
    assert not run_solution_against_tests("def sum_even(xs):\n    return 1\n", tests).passed


def test_solution_section_without_a_code_block_is_rejected():
    text = "QUESTION:\nq\n\nSOLUTION:\nsee below\n\nTESTS:\n```python\ndef test_x():\n    assert True\n```\n"
    assert parse_generation(text) is None
