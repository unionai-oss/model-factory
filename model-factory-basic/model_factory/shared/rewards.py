"""Reward function for the coding agent — the factory's reward-shaping layer.

Design (docs/SPEC.md §3, informed by DeepCoder/AceCoder/R1):

- ``format``  (0 / 0.1): exactly one non-empty fenced python block.
- ``compile`` (0 / 0.1): extracted code parses (``ast.parse``).
- ``tests``   (0 / 1.0): ALL hidden tests pass in the sandbox. All-or-nothing
  — partial per-test credit teaches printing public-test answers.
- Anti-hack guards zero the entire reward: solutions may not touch the test
  file, pytest internals, the filesystem/process machinery, or dynamic
  import tricks.

Every component is returned separately so reward shaping stays inspectable
(per-sample breakdowns land in W&B and the task report).

No flyte imports here: unit-testable pure Python.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .sandbox import ExecutionResult, run_solution_against_tests

FORMAT_REWARD = 0.1
COMPILE_REWARD = 0.1
TESTS_REWARD = 1.0
MAX_REWARD = FORMAT_REWARD + COMPILE_REWARD + TESTS_REWARD

_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)

# Imports/attributes that have no business in a KodCode-style solution and
# are the classic reward-hacking vectors for a tests-based reward.
_BANNED_PATTERNS = (
    re.compile(r"\btest_solution\b"),  # reading/patching the test file
    re.compile(r"\bimport\s+pytest\b"),
    re.compile(r"\bpytest\b\s*\."),
    re.compile(r"\bimport\s+(subprocess|shutil|socket)\b"),
    re.compile(r"\bos\s*\.\s*(remove|unlink|rename|system|popen|kill|_exit)\b"),
    re.compile(r"\b(sys\s*\.\s*exit|exit|quit)\s*\("),
    re.compile(r"\b__import__\s*\("),
    re.compile(r"\bimportlib\b"),
    re.compile(r"\bbuiltins\s*\.\s*assert"),
    re.compile(r"conftest"),
)


@dataclass(frozen=True)
class RewardBreakdown:
    format_ok: bool
    compiles: bool
    guard_violation: str | None
    tests_passed: bool
    execution: ExecutionResult | None

    @property
    def total(self) -> float:
        if self.guard_violation:
            return 0.0
        r = 0.0
        if self.format_ok:
            r += FORMAT_REWARD
        if self.compiles:
            r += COMPILE_REWARD
        if self.tests_passed:
            r += TESTS_REWARD
        return r

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "format_ok": self.format_ok,
            "compiles": self.compiles,
            "guard_violation": self.guard_violation or "",
            "tests_passed": self.tests_passed,
            "execution_summary": self.execution.summary if self.execution else "not-run",
        }


def extract_code(completion: str) -> str | None:
    """Return the single fenced python block, or None if 0 or >1 blocks."""
    blocks = [b.strip() for b in _CODE_BLOCK.findall(completion) if b.strip()]
    if len(blocks) != 1:
        return None
    return blocks[0]


def check_guards(code: str) -> str | None:
    """Return the description of the first violated guard, else None."""
    for pat in _BANNED_PATTERNS:
        m = pat.search(code)
        if m:
            return f"banned pattern: {m.group(0)!r}"
    return None


def score_completion(
    completion: str,
    test_code: str,
    timeout_s: float = 10.0,
) -> RewardBreakdown:
    """Score one model completion against a hidden test suite."""
    code = extract_code(completion)
    if code is None:
        return RewardBreakdown(
            format_ok=False, compiles=False, guard_violation=None,
            tests_passed=False, execution=None,
        )

    violation = check_guards(code)
    if violation:
        return RewardBreakdown(
            format_ok=True, compiles=False, guard_violation=violation,
            tests_passed=False, execution=None,
        )

    try:
        ast.parse(code)
    except SyntaxError:
        return RewardBreakdown(
            format_ok=True, compiles=False, guard_violation=None,
            tests_passed=False, execution=None,
        )

    result = run_solution_against_tests(code, test_code, timeout_s=timeout_s)
    return RewardBreakdown(
        format_ok=True, compiles=True, guard_violation=None,
        tests_passed=result.passed, execution=result,
    )


def count_test_functions(test_code: str) -> int:
    """Number of ``def test_*`` functions — the curation floor."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return 0
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    )


SYSTEM_PROMPT = (
    "You are an expert Python programmer. Solve the problem below.\n"
    "Respond with exactly one fenced python code block containing the "
    "complete solution and nothing else. Do not include tests or example "
    "usage."
)


def build_prompt(question: str, function_declaration: str | None) -> list[dict]:
    """Chat-format prompt for the policy; tells the model the required API."""
    user = question.strip()
    if function_declaration:
        user += (
            "\n\nYour solution must define exactly this signature so the "
            f"grader can import it:\n`{function_declaration.strip()}`"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
