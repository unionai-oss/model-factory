from model_factory.synthetic import parse_generation

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
