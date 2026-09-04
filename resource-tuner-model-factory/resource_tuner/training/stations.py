"""Factory stations: corpus → train → eval, wired two ways.

- The `tuner_pipeline` driver chains the stations for one readable E2E run.
- OnArtifact triggers (train_tuner: on tuning-task-corpus; eval_tuner: on
  tuner-checkpoint) run the same stations dark — publishing a corpus IS the
  request to train, a checkpoint IS the request to evaluate. Triggers
  deploy `auto_activate=False`; activate them to go dark, and remember a
  trigger keeps firing the task version it was deployed with — re-deploy
  after any fix dark mode should pick up.

The synthetic station widens the corpus with teacher-LLM tasks whose
labels come from the execution oracle (a harness pod measures what the
code actually uses), never from the teacher's own guess.
"""

from __future__ import annotations

import tempfile

import flyte
import flyte.io

from ..config import get_profile
from ..contracts import ARTIFACT_SYNTHETIC, ARTIFACT_TASK_CORPUS, publish

# Static imports on purpose: the code bundler walks the import graph from
# the entrypoint, so a module imported only inside a task function is NOT
# bundled and the task dies with ImportError in the pod (hit for real with
# llm_client). Both modules are stdlib-only, so importing them here is free.
from ..shared import llm_client
from ..taskgen import synthetic as syn
from ..taskgen.corpus import build_corpus
from .envs import driver_env
from .evaluate import eval_tuner
from .grpo import train_tuner


@driver_env.task(produces_artifacts=True)
async def build_task_corpus(profile_name: str = "smoke", seed: int = 0) -> flyte.io.File:
    """Sample the task corpus and publish it as tuning-task-corpus."""
    import pandas as pd

    profile = get_profile(profile_name)
    records = build_corpus(profile.train_contexts, profile.eval_contexts, seed=seed)
    path = tempfile.mktemp(suffix=".parquet")
    pd.DataFrame(records).to_parquet(path, index=False)
    out = await flyte.io.File.from_local(path)
    return publish(
        out,
        ARTIFACT_TASK_CORPUS,
        description=f"{profile.train_contexts} train / {profile.eval_contexts} heldout, seed={seed}",
    )


@driver_env.task(produces_artifacts=True)
async def publish_synthetic_corpus(corpus_file: flyte.io.File, n: int, teacher: str) -> flyte.io.File:
    """Publish the oracle-verified synthetic rows as their own artifact.

    Split out because `publish()` only creates an artifact version when the
    wrapped value is RETURNED from a task — the release task returns the
    MERGED corpus, so the synthetic slice needs its own returning task
    (first attempt published nothing: 0 versions listed).
    """
    return publish(
        corpus_file,
        ARTIFACT_SYNTHETIC,
        description=f"{n} oracle-verified tasks from {teacher}",
    )


@driver_env.task(timeout=flyte.Timeout(max_runtime=2 * 3600), produces_artifacts=True)
async def synthetic_data_release(
    n_tasks: int = 10,
    teacher: str = "qwen38-27b",
    merge_with_templates: bool = True,
    profile_name: str = "smoke",
    seed: int = 0,
) -> flyte.io.File:
    """Teacher LLM → AST screen → execution oracle → curated corpus.

    Publishes `synthetic-task-corpus` (the oracle-verified teacher tasks)
    and, when `merge_with_templates`, a NEW `tuning-task-corpus` version
    merging them with the template corpus — which is what fires the
    train-on-new-corpus trigger in dark mode.
    """
    import asyncio

    import pandas as pd

    from ..environment.harness import run_generated

    base_url = llm_client.resolve_teacher(teacher)
    await asyncio.to_thread(llm_client.wait_until_ready, base_url)

    import random

    rng = random.Random(seed)
    prompts = []
    for i in range(n_tasks):
        family, hint = syn.FAMILY_HINTS[i % len(syn.FAMILY_HINTS)]
        prompts.append(
            (
                family,
                syn.GENERATION_PROMPT.format(
                    family_hint=hint,
                    duration_s=60,
                    allowed=", ".join(sorted(syn.ALLOWED_IMPORTS)),
                    target_mib=rng.choice([256, 512, 1024, 2048, 4096]),
                    target_cores=rng.choice([1, 1, 2, 4]),
                ),
            )
        )

    async def generate_one(idx: int, family: str, prompt: str) -> dict | None:
        try:
            text = await asyncio.to_thread(
                llm_client.chat, base_url, [{"role": "user", "content": prompt}]
            )
        except llm_client.TeacherError as e:
            print(f"[synthetic {idx}] teacher call failed: {e}")
            return None
        try:
            desc, code = syn.parse_teacher_response(text)
            syn.validate_task_code(code)
        except syn.RejectedTask as e:
            print(f"[synthetic {idx}] rejected pre-oracle: {e}; raw head: {text[:200]!r}")
            return None
        # The oracle: run it for real, generously provisioned, and measure.
        oracle = run_generated.override(
            resources=flyte.Resources(cpu=4, memory="14Gi", disk="10Gi")
        )
        try:
            measured = await oracle(harness_code=code, task_id=f"synthetic-{seed}-{idx}")
        except Exception as e:  # noqa: BLE001 — teacher code crashed its pod
            print(f"[synthetic {idx}] oracle pod failed: {e}")
            return None
        reason = syn.curate_measurement(measured)
        if reason:
            print(f"[synthetic {idx}] curated out: {reason}")
            return None
        return syn.synthetic_record(f"synthetic-{seed}-{idx}", family, code, desc, measured)

    results = await asyncio.gather(
        *(generate_one(i, fam, p) for i, (fam, p) in enumerate(prompts))
    )
    records = [r for r in results if r]
    print(f"synthetic yield: {len(records)}/{n_tasks} survived screen + oracle")
    if not records:
        raise RuntimeError(
            f"0/{n_tasks} synthetic tasks survived; teacher or oracle is broken"
        )

    path = tempfile.mktemp(suffix=".parquet")
    pd.DataFrame(records).to_parquet(path, index=False)
    synthetic_file = await publish_synthetic_corpus(
        corpus_file=await flyte.io.File.from_local(path), n=len(records), teacher=teacher
    )
    if not merge_with_templates:
        return synthetic_file

    profile = get_profile(profile_name)
    template_records = build_corpus(profile.train_contexts, profile.eval_contexts, seed=seed)
    merged = pd.concat(
        [pd.DataFrame(template_records), pd.DataFrame(records)], ignore_index=True
    )
    merged_path = tempfile.mktemp(suffix=".parquet")
    merged.to_parquet(merged_path, index=False)
    return publish(
        await flyte.io.File.from_local(merged_path),
        ARTIFACT_TASK_CORPUS,
        description=f"templates({len(template_records)}) + synthetic({len(records)}) "
        f"via {teacher}, seed={seed}",
    )


@driver_env.task(timeout=flyte.Timeout(max_runtime=3600))
async def probe_episodes(n_per_family: int = 1, include_oom_probe: bool = True) -> dict:
    """Environment smoke test: run REAL episodes with baseline proposals.

    Validates the whole episode path without a GPU — harness pods schedule
    with overridden resources, generated code executes, rusage flows back.
    `include_oom_probe` also runs one episode with a deliberately tiny
    memory request, proving underprovisioning surfaces as an OOM signal
    (a failed child action), not a crash of the driver.
    """
    import asyncio

    from ..environment.episodes import run_cluster_episode
    from ..policy.actions import Proposal
    from ..taskgen.corpus import build_corpus
    from ..training.baseline import baseline_proposal, fit_family_baseline

    families = 5
    records = build_corpus(n_train=families * 4, n_heldout=families * n_per_family, seed=11)
    train = [r for r in records if r["split"] == "train"]
    probes = [r for r in records if r["split"] == "heldout"]
    baselines = fit_family_baseline(train)

    jobs = [(r, baseline_proposal(baselines, r["family"])) for r in probes]
    if include_oom_probe:
        jobs.append((probes[0], Proposal(cpu=1, memory_mib=128)))

    results = await asyncio.gather(
        *(run_cluster_episode(r, p) for r, p in jobs), return_exceptions=True
    )
    episodes = []
    for (record, proposal), ep in zip(jobs, results):
        if isinstance(ep, BaseException):
            episodes.append({"task_id": record["task_id"], "driver_error": str(ep)[:500]})
            continue
        episodes.append(
            {
                "task_id": record["task_id"],
                "family": record["family"],
                "requested": proposal.to_kwargs(),
                "ok": ep.ok,
                "oom": ep.oom,
                "analytic_peak_mib": round(record["true_peak_memory_mib"], 1),
                "real_peak_rss_mib": round(ep.peak_memory_mib, 1),
                "duration_s": round(ep.duration_s, 1),
            }
        )
    scored = [e for e in episodes if "ok" in e]
    return {
        "episodes": episodes,
        "n_ok": sum(1 for e in scored if e["ok"]),
        "n_oom": sum(1 for e in scored if e["oom"]),
        "n_driver_errors": len(episodes) - len(scored),
    }


@driver_env.task(timeout=flyte.Timeout(max_runtime=8 * 3600))
async def tuner_pipeline(
    profile_name: str = "smoke", seed: int = 0, run_cluster_episodes: bool = True
) -> flyte.io.File:
    """E2E: corpus → GRPO train → eval report. Returns the eval report."""
    corpus = await build_task_corpus(profile_name=profile_name, seed=seed)
    checkpoint = await train_tuner(corpus=corpus, profile_name=profile_name)
    return await eval_tuner(
        corpus=corpus,
        checkpoint=checkpoint,
        profile_name=profile_name,
        run_cluster_episodes=run_cluster_episodes,
    )
