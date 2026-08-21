"""Human-in-the-loop gates — flyte conditions with a CI bypass."""

from __future__ import annotations

from datetime import timedelta

import flyte

GATE_TIMEOUT = timedelta(hours=4)


async def gate(name: str, prompt_md: str, auto_approve: bool) -> bool:
    """Pause the run on a condition until a human signals; timeout = reject.

    ``auto_approve`` short-circuits for CI/smoke runs — the gate still exists
    in the code path, the light is just switched off deliberately.
    """
    if auto_approve:
        return True
    condition = await flyte.new_condition.aio(
        name,
        prompt=prompt_md,
        prompt_type="markdown",
        data_type=bool,
        timeout=GATE_TIMEOUT,
    )
    try:
        return bool(await condition.wait.aio())
    except flyte.errors.ConditionTimedoutError:
        return False
