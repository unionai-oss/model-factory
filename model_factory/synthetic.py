"""Synthetic data station: batch inference generates new verified tasks.

AceCoder-style recipe: prompt an instruct model to *mutate* seed problems
(new constraints, edge cases, renamed domain), parse out
question/solution/tests, then let the sandbox act as the oracle — only
samples whose generated reference solution passes their own generated tests
survive. Garbage generations are filtered by execution, not by a judge model.

Batching: many per-seed coroutines submit prompts to a
``flyte.extras.DynamicBatcher`` whose process_fn runs batched HF generation
on the GPU — the POC-scale version of the factory's "batch inference soaks
the idle GPU" pattern.
"""

from __future__ import annotations

import asyncio
import hashlib
import re

import flyte
import flyte.io
import flyte.report
from flyte.extras import DynamicBatcher

from . import reporting
from .config import ARTIFACT_SYNTHETIC, get_profile
from .envs import gpu_env
from .rewards import count_test_functions
from .sandbox import run_solution_against_tests

_MUTATION_PROMPT = """You are generating a NEW Python programming exercise by mutating the one below. Change the problem meaningfully (different domain, constraint, or edge cases) but keep it self-contained and testable.

Original problem:
{question}

Original solution signature: {declaration}

Respond in EXACTLY this format:

QUESTION:
<one-paragraph problem statement>

SOLUTION:
```python
<complete reference solution>
```

TESTS:
```python
from solution import <function_name>

<at least {min_tests} pytest test functions named test_*, using assert>
```
"""

_CODE_BLOCKS = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_QUESTION = re.compile(r"QUESTION:\s*\n(.*?)(?:\n\s*SOLUTION:)", re.DOTALL)


def parse_generation(text: str) -> dict | None:
    """Extract {question, solution, tests} from the fixed response format."""
    qm = _QUESTION.search(text)
    blocks = _CODE_BLOCKS.findall(text)
    if not qm or len(blocks) < 2:
        return None
    question = qm.group(1).strip()
    solution, tests = blocks[0].strip(), blocks[1].strip()
    if not (question and solution and tests):
        return None
    return {"question": question, "solution": solution, "tests": tests}


@flyte.trace
async def _generate_batch(model_name: str, prompts: list[str], max_new_tokens: int) -> list[str]:
    """One batched GPU generation call (traced: replayed on task retry)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Module-level cache so the model loads once per container.
    global _MODEL, _TOK  # noqa: PLW0603
    if "_MODEL" not in globals():
        _TOK = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
    chats = [
        _TOK.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = _TOK(chats, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(_MODEL.device)
    with torch.no_grad():
        out = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=_TOK.pad_token_id or _TOK.eos_token_id,
        )
    gen = out[:, inputs["input_ids"].shape[1]:]
    return _TOK.batch_decode(gen, skip_special_tokens=True)


@gpu_env.task(report=True)
async def generate_synthetic_tasks(
    dataset: flyte.io.File, profile_name: str = "smoke"
) -> flyte.io.File:
    """Generate oracle-verified synthetic tasks; emit `synthetic-tasks` artifact."""
    import pandas as pd
    import flyte.artifacts as artifacts

    profile = get_profile(profile_name)
    df = pd.read_parquet(await dataset.download())
    seeds = df[df["split"] == "train"].sample(
        min(profile.synthetic_seeds, len(df)), random_state=11
    )

    batcher = DynamicBatcher(
        process_fn=lambda prompts: _generate_batch(
            profile.base_model, list(prompts), profile.synthetic_max_new_tokens
        ),
        max_batch_size=8,
        batch_timeout_s=0.5,
    )
    await batcher.start()

    async def one(seed) -> dict | None:
        prompt = _MUTATION_PROMPT.format(
            question=seed.question[:1500],
            declaration=seed.function_declaration or "any function",
            min_tests=max(3, profile.min_test_functions),
        )
        # submit() returns a future that resolves when its batch is processed
        future = await batcher.submit(prompt)
        text = await future
        parsed = parse_generation(text)
        if not parsed:
            return None
        if count_test_functions(parsed["tests"]) < 1:
            return None
        # Execution as the oracle: generated solution must pass generated tests.
        result = await asyncio.to_thread(
            run_solution_against_tests, parsed["solution"], parsed["tests"]
        )
        if not result.passed:
            return None
        qid = hashlib.sha1(parsed["question"].encode()).hexdigest()[:12]
        return {
            "task_id": f"syn_{qid}",
            "question": parsed["question"],
            "function_declaration": "",
            "tests": parsed["tests"],
            "reference_solution": parsed["solution"],
            "difficulty": "synthetic",
            "n_tests": count_test_functions(parsed["tests"]),
            "source": "synthetic",
            "split": "train",
        }

    results = await asyncio.gather(*(one(s) for s in seeds.itertuples()))
    await batcher.stop()
    kept = [r for r in results if r]

    out_df = pd.DataFrame(
        kept,
        columns=[
            "task_id", "question", "function_declaration", "tests",
            "reference_solution", "difficulty", "n_tests", "source", "split",
        ],
    )
    out = "/tmp/synthetic_tasks.parquet"
    out_df.to_parquet(out, index=False)

    body = reporting.stats_row(
        {
            "seeds mutated": len(seeds),
            "parsed ok": sum(1 for r in results if r is not None),
            "oracle-verified kept": len(kept),
            "yield": f"{len(kept) / max(len(seeds), 1):.0%}",
        }
    )
    if kept:
        body += "<h3>Verified synthetic samples (rollout-validation window)</h3>"
        body += reporting.table(
            ["task_id", "n_tests", "question", "solution (head)"],
            [[k["task_id"], k["n_tests"], k["question"][:250], k["reference_solution"][:250]] for k in kept[:10]],
        )
    await flyte.report.replace.aio(
        reporting.page("Synthetic task generation (batch inference)", body)
    )
    await flyte.report.flush.aio()

    f = await flyte.io.File.from_local(out)
    return artifacts.new(
        f,
        artifacts.Metadata(
            name=ARTIFACT_SYNTHETIC,
            description=f"Oracle-verified synthetic tasks ({len(kept)} kept from {len(seeds)} seeds)",
            kind="data",
        ),
    )
