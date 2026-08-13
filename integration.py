"""Cross-team integration driver — the E2E test of the artifact chain.

In production the teams are wired ONLY by artifacts + OnArtifact triggers:

    data_release → [rl-tasks-dataset] → train_grpo → [policy-checkpoint]
                                                   ├→ eval_and_promote → [eval-report], [promoted-model]
                                                   └→ refresh_inference_service → [inference-endpoint]

The demo control plane doesn't emit artifact events yet (TODO.md §2), so
this driver plays the event bus: it invokes each team's public entrypoint in
artifact order, passing exactly the values the triggers would bind. Nothing
here reaches into another team's internals — same contract, hand-cranked.

    flyte --config ~/.flyte/config-model-factory.yaml run integration.py factory_chain \
        --profile_name smoke --auto_approve
"""

from __future__ import annotations

import flyte
import flyte.io
import flyte.report

from model_factory.data_engineering.envs import de_cpu_env, de_gpu_env
from model_factory.data_engineering.release import data_release
from model_factory.evaluation.envs import eval_cpu_env, eval_gpu_env
from model_factory.evaluation.tasks import eval_and_promote
from model_factory.inference.tasks import inference_ops_env, refresh_inference_service
from model_factory.shared import reporting
from model_factory.shared.images import cpu_image
from model_factory.training.envs import trainer_env
from model_factory.training.tasks import train_grpo

integration_env = flyte.TaskEnvironment(
    name="integration",
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    image=cpu_image,
    depends_on=[de_cpu_env, de_gpu_env, trainer_env, eval_gpu_env, eval_cpu_env, inference_ops_env],
)


@integration_env.task(report=True)
async def factory_chain(
    profile_name: str = "smoke",
    auto_approve: bool = False,
    include_synthetic: bool = True,
    refresh_inference: bool = True,
) -> str:
    """One full pass through all four teams, in artifact order."""
    log: list[str] = []

    async def note(msg: str) -> None:
        log.append(msg)
        body = "<ol>" + "".join(f"<li>{reporting.esc(m)}</li>" for m in log) + "</ol>"
        await flyte.report.replace.aio(reporting.page("Factory chain (integration)", body))
        await flyte.report.flush.aio()

    await note("[data engineering] producing dataset release…")
    dataset = await data_release(
        profile_name=profile_name,
        auto_approve=auto_approve,
        include_synthetic=include_synthetic,
    )
    await note("[data engineering] published rl-tasks-dataset ✓  → would fire train trigger")

    await note("[training] GRPO training…")
    checkpoint = await train_grpo(dataset=dataset, profile_name=profile_name)
    await note("[training] published policy-checkpoint ✓  → would fire eval + inference triggers")

    if refresh_inference:
        await note("[inference] loading checkpoint into serving app…")
        try:
            await refresh_inference_service(checkpoint=checkpoint)
            await note("[inference] published inference-endpoint ✓")
        except Exception as e:
            await note(f"[inference] refresh failed ({str(e)[:150]}); eval will fall back to local generation")

    await note("[eval] candidate-vs-base evaluation + gates…")
    verdict = await eval_and_promote(
        checkpoint=checkpoint,
        profile_name=profile_name,
        auto_approve=auto_approve,
        dataset=dataset,
    )
    await note(f"[eval] {verdict}")
    return verdict
