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
from ..shared.reporting import GOOD, MUTED, Reporter, esc, ok_pill, pill
from ..taskgen import synthetic as syn
from ..taskgen.corpus import build_corpus
from .envs import driver_env
from .evaluate import eval_tuner
from .grpo import train_tuner


@driver_env.task(produces_artifacts=True, report=True)
async def build_task_corpus(profile_name: str = "smoke", seed: int = 0) -> flyte.io.File:
    """Sample the task corpus and publish it as tuning-task-corpus."""
    import pandas as pd

    profile = get_profile(profile_name)
    rep = Reporter("Task corpus", f"profile={profile.name} seed={seed}")
    rep.kv({"train contexts": profile.train_contexts, "heldout contexts": profile.eval_contexts})
    await rep.flush()

    records = build_corpus(profile.train_contexts, profile.eval_contexts, seed=seed)
    df = pd.DataFrame(records)
    fam = df.groupby("family")["true_peak_memory_mib"]
    rep.h("Composition (analytic footprints)")
    rep.table(
        ["family", "tasks", "peak MiB min", "median", "max", "cpu cores max"],
        [
            [
                esc(name),
                esc(int(g.count())),
                esc(int(g.min())),
                esc(int(g.median())),
                esc(int(g.max())),
                esc(df[df.family == name]["true_cpu_cores"].max()),
            ]
            for name, g in fam
        ],
    )
    rep.h("Sample task (policy view)")
    rep.raw(
        f'<pre style="background:#131316;padding:10px;border-radius:8px;font-size:11px;'
        f'color:#c9c9cf;overflow-x:auto">{esc(records[0]["source_code"][:800])}</pre>'
    )
    await rep.flush()

    path = tempfile.mktemp(suffix=".parquet")
    df.to_parquet(path, index=False)
    out = await flyte.io.File.from_local(path)
    return publish(
        out,
        ARTIFACT_TASK_CORPUS,
        description=f"{profile.train_contexts} train / {profile.eval_contexts} heldout, seed={seed}",
    )


@driver_env.task(produces_artifacts=True, report=True)
async def publish_synthetic_corpus(corpus_file: flyte.io.File, n: int, teacher: str) -> flyte.io.File:
    """Publish the oracle-verified synthetic rows as their own artifact.

    Split out because `publish()` only creates an artifact version when the
    wrapped value is RETURNED from a task — the release task returns the
    MERGED corpus, so the synthetic slice needs its own returning task
    (first attempt published nothing: 0 versions listed).
    """
    rep = Reporter("Synthetic corpus publish", f"{n} oracle-verified tasks from {teacher}")
    rep.p("Returning the wrapped file versions the synthetic-task-corpus artifact.")
    await rep.flush()
    return publish(
        corpus_file,
        ARTIFACT_SYNTHETIC,
        description=f"{n} oracle-verified tasks from {teacher}",
    )


@driver_env.task(timeout=flyte.Timeout(max_runtime=2 * 3600), produces_artifacts=True, report=True)
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

    candidates = llm_client.resolve_teacher_candidates(teacher)
    base_url = candidates[0]
    # Live per-candidate pipeline status; re-rendered on every state change.
    status: list[dict] = [
        {"stage": "queued", "detail": "", "family": ""} for _ in range(n_tasks)
    ]
    rep = Reporter("Synthetic data release", f"teacher={teacher}")
    rep_lock = asyncio.Lock()

    async def render(phase: str) -> None:
        async with rep_lock:
            rep.reset_body()
            rep.kv({"endpoint": base_url, "phase": phase, "n_tasks": n_tasks})
            done = sum(1 for s in status if s["stage"] in ("kept", "rejected"))
            rep.progress(done, n_tasks, "candidates settled")
            rep.h("Candidate pipeline")
            colors = {"kept": GOOD, "rejected": "#F43B3E"}
            rep.table(
                ["#", "family", "stage", "detail"],
                [
                    [
                        esc(i),
                        esc(s["family"]),
                        pill(s["stage"], colors.get(s["stage"], MUTED)),
                        esc(s["detail"][:160]),
                    ]
                    for i, s in enumerate(status)
                ],
            )
            await rep.flush()

    await render("waking teacher (scale-from-zero can take 15+ min)")
    # Poll status streams into the report so a stuck wake is diagnosable
    # from the console (learned from a run that sat at "waking teacher"
    # with the actual per-poll HTTP statuses invisible).
    loop = asyncio.get_running_loop()

    def on_status(s: str) -> None:
        asyncio.run_coroutine_threadsafe(render(f"waking teacher — {s}"), loop)

    base_url = await asyncio.to_thread(
        llm_client.wait_until_ready, candidates, 1800, 15, on_status
    )
    await render("generating")

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

    async def set_stage(idx: int, stage: str, detail: str = "") -> None:
        status[idx].update(stage=stage, detail=detail)
        await render("generating")

    async def generate_one(idx: int, family: str, prompt: str) -> dict | None:
        status[idx]["family"] = family
        await set_stage(idx, "asking teacher")
        try:
            text = await asyncio.to_thread(
                llm_client.chat, base_url, [{"role": "user", "content": prompt}]
            )
        except llm_client.TeacherError as e:
            print(f"[synthetic {idx}] teacher call failed: {e}")
            await set_stage(idx, "rejected", f"teacher call failed: {e}")
            return None
        try:
            desc, code = syn.parse_teacher_response(text)
            syn.validate_task_code(code)
        except syn.RejectedTask as e:
            print(f"[synthetic {idx}] rejected pre-oracle: {e}; raw head: {text[:200]!r}")
            await set_stage(idx, "rejected", f"pre-oracle: {e}")
            return None
        # The oracle: run it for real, generously provisioned, and measure.
        await set_stage(idx, "oracle pod running", desc[:120])
        oracle = run_generated.override(
            resources=flyte.Resources(cpu=4, memory="14Gi", disk="10Gi")
        )
        try:
            measured = await oracle(harness_code=code, task_id=f"synthetic-{seed}-{idx}")
        except Exception as e:  # noqa: BLE001 — teacher code crashed its pod
            print(f"[synthetic {idx}] oracle pod failed: {e}")
            await set_stage(idx, "rejected", f"oracle pod failed: {str(e)[:140]}")
            return None
        reason = syn.curate_measurement(measured)
        if reason:
            print(f"[synthetic {idx}] curated out: {reason}")
            await set_stage(idx, "rejected", f"curated out: {reason}")
            return None
        await set_stage(
            idx,
            "kept",
            f"{desc[:80]} · peak {measured['peak_rss_mib']:.0f}MiB · "
            f"cpu {measured.get('cpu_avg_cores', 0):.1f} · {measured['duration_s']:.0f}s",
        )
        return syn.synthetic_record(f"synthetic-{seed}-{idx}", family, code, desc, measured)

    results = await asyncio.gather(
        *(generate_one(i, fam, p) for i, (fam, p) in enumerate(prompts))
    )
    records = [r for r in results if r]
    print(f"synthetic yield: {len(records)}/{n_tasks} survived screen + oracle")
    await render(f"done — yield {len(records)}/{n_tasks}")
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


@driver_env.task(timeout=flyte.Timeout(max_runtime=3600), report=True)
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

    rep = Reporter("Episode probe", "real harness pods under baseline proposals")
    rep.kv({"episodes": len(jobs), "oom probe": include_oom_probe})
    rep.p("Episode pods launching — each requests exactly its proposal.")
    await rep.flush()

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
    rep.reset_body()
    rep.kv(
        {
            "fit": sum(1 for e in scored if e["ok"]),
            "oom": sum(1 for e in scored if e["oom"]),
            "driver errors": len(episodes) - len(scored),
        }
    )
    rep.h("Episodes (requested vs analytic vs measured)")
    rep.table(
        ["task", "requested", "analytic MiB", "real RSS MiB", "duration", "outcome"],
        [
            [
                esc(e["task_id"]),
                esc(e.get("requested", "-")),
                esc(e.get("analytic_peak_mib", "-")),
                esc(e.get("real_peak_rss_mib", "-")),
                esc(f"{e.get('duration_s', 0):.0f}s"),
                pill("driver error", "#F43B3E")
                if "driver_error" in e
                else (pill("oom", "#e69812") if e["oom"] else ok_pill(e["ok"], "fit")),
            ]
            for e in episodes
        ],
    )
    await rep.flush()
    return {
        "episodes": episodes,
        "n_ok": sum(1 for e in scored if e["ok"]),
        "n_oom": sum(1 for e in scored if e["oom"]),
        "n_driver_errors": len(episodes) - len(scored),
    }


@driver_env.task(timeout=flyte.Timeout(max_runtime=8 * 3600), report=True)
async def tuner_pipeline(
    profile_name: str = "smoke", seed: int = 0, run_cluster_episodes: bool = True
) -> flyte.io.File:
    """E2E: corpus → GRPO train → eval report. Returns the eval report."""
    rep = Reporter("Tuner pipeline", f"profile={profile_name} seed={seed}")

    async def stage(name: str, state: str) -> None:
        rep.reset_body()
        stages = ["corpus", "train", "eval"]
        rep.table(
            ["stage", "state"],
            [
                [
                    esc(s),
                    pill(state, GOOD if state == "done" else "#4d65ff")
                    if s == name
                    else (
                        pill("done", GOOD)
                        if stages.index(s) < stages.index(name)
                        else pill("pending", MUTED)
                    ),
                ]
                for s in stages
            ],
        )
        rep.p("Per-stage detail lives on each child action's own report tab.")
        await rep.flush()

    await stage("corpus", "running")
    corpus = await build_task_corpus(profile_name=profile_name, seed=seed)
    await stage("train", "running")
    checkpoint = await train_tuner(corpus=corpus, profile_name=profile_name)
    await stage("eval", "running")
    report = await eval_tuner(
        corpus=corpus,
        checkpoint=checkpoint,
        profile_name=profile_name,
        run_cluster_episodes=run_cluster_episodes,
    )
    await stage("eval", "done")
    return report
