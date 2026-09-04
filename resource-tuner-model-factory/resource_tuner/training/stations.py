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

import json
import tempfile
import time

import flyte
import flyte.io

from ..config import get_profile
from ..contracts import (
    AB_REPORT_KEYS,
    ARTIFACT_AB_REPORT,
    ARTIFACT_SYNTHETIC,
    ARTIFACT_TASK_CORPUS,
    publish,
)

# Static imports on purpose: the code bundler walks the import graph from
# the entrypoint, so a module imported only inside a task function is NOT
# bundled and the task dies with ImportError in the pod (hit for real with
# llm_client). Both modules are stdlib-only, so importing them here is free.
from ..environment.harness import run_generated
from ..shared import llm_client
from .. import tune
from ..shared.reporting import GOOD, MUTED, Reporter, esc, ok_pill, pill
from ..taskgen import archetypes as arch
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
    import pandas as pd

    rep = Reporter("Synthetic corpus publish", f"{n} tasks from {teacher}")
    try:
        df = pd.read_parquet(await corpus_file.download())
        peaks = df["true_peak_memory_mib"]
        code_len = df["source_code"].str.len()
        rep.kv(
            {
                "rows": len(df),
                "archetypes": df["params_json"].apply(
                    lambda s: json.loads(s).get("archetype", "single")
                ).nunique(),
                "families": ", ".join(
                    f"{k}:{v}" for k, v in df["family"].value_counts().items()
                ),
                "label sources": ", ".join(
                    f"{k}:{v}"
                    for k, v in df["params_json"]
                    .apply(lambda s: json.loads(s).get("label_source", "measured"))
                    .value_counts()
                    .items()
                ),
            }
        )
        rep.h("Peak memory distribution (MiB)")
        rep.table(
            ["p5", "p25", "median", "p75", "p95", "max"],
            [[esc(int(peaks.quantile(q))) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0)]],
        )
        rep.h("Other stats")
        rep.kv(
            {
                "cpu cores (median / max)": f"{df['true_cpu_cores'].median():.1f} / "
                f"{df['true_cpu_cores'].max():.1f}",
                "duration s (median / max)": f"{df['duration_s'].median():.0f} / "
                f"{df['duration_s'].max():.0f}",
                "code length chars (median / p95)": f"{int(code_len.median())} / "
                f"{int(code_len.quantile(0.95))}",
            }
        )
    except Exception as e:  # noqa: BLE001 — stats must not block the publish
        rep.p(f"stats unavailable: {e}")
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
            # Group the oracle subactions so the run view shows one tidy
            # "execution-oracle" box instead of N loose run_generated rows.
            with flyte.group("execution-oracle"):
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


@driver_env.task(timeout=flyte.Timeout(max_runtime=8 * 3600), produces_artifacts=True, report=True)
async def archetype_data_release(
    total_tasks: int = 100_000,
    n_archetypes: int = 150,
    calibration_k: int = 3,
    teacher: str = "qwen38-27b",
    merge_with_templates: bool = True,
    profile_name: str = "smoke",
    seed: int = 0,
) -> flyte.io.File:
    """Scale synthetic generation: archetypes × instantiation → 10⁵ tasks.

    The teacher writes ~10² parameterized archetypes; the oracle calibrates
    each at `calibration_k` parameter points (real pods, measured RSS/CPU);
    a per-archetype fit labels sampled instantiations up to `total_tasks`.
    Labels are measurement-anchored — `label_source` marks measured
    (calibration rows) vs fitted (interpolated variants).

    The report shows SUMMARY stats + head/tail of the archetype table only
    (a 10⁵-row table in a report helps no one).
    """
    import asyncio
    import random

    import pandas as pd

    candidates = llm_client.resolve_teacher_candidates(teacher)
    base_url = candidates[0]
    rng = random.Random(seed)

    astatus: list[dict] = [
        {"stage": "queued", "detail": "", "family": syn.FAMILY_HINTS[i % len(syn.FAMILY_HINTS)][0]}
        for i in range(n_archetypes)
    ]
    counters = {"calib_done": 0, "calib_total": 0, "variants": 0}
    rep = Reporter("Archetype data release", f"teacher={teacher} target={total_tasks:,}")
    rep_lock = asyncio.Lock()
    last_flush = {"t": 0.0}

    async def render(phase: str, force: bool = False) -> None:
        # Throttled: hundreds of concurrent state changes must not turn
        # the report into a flush storm.
        if not force and time.monotonic() - last_flush["t"] < 3.0:
            return
        async with rep_lock:
            last_flush["t"] = time.monotonic()
            rep.reset_body()
            settled = sum(1 for s in astatus if s["stage"] in ("kept", "rejected"))
            kept = sum(1 for s in astatus if s["stage"] == "kept")
            rep.kv(
                {
                    "endpoint": base_url,
                    "phase": phase,
                    "archetypes kept / settled / total": f"{kept} / {settled} / {n_archetypes}",
                    "calibration pods": f"{counters['calib_done']}/{counters['calib_total']}",
                    "variants written": f"{counters['variants']:,}/{total_tasks:,}",
                }
            )
            rep.progress(settled, n_archetypes, "archetypes")
            if counters["calib_total"]:
                rep.progress(counters["calib_done"], counters["calib_total"], "calibrations")
            reasons: dict[str, int] = {}
            for s in astatus:
                if s["stage"] == "rejected":
                    key = s["detail"].split(":")[0][:60]
                    reasons[key] = reasons.get(key, 0) + 1
            if reasons:
                rep.h("Rejection reasons")
                rep.table(
                    ["reason", "count"],
                    [[esc(k), esc(v)] for k, v in sorted(reasons.items(), key=lambda x: -x[1])[:8]],
                )
            rep.h("Archetypes (head / tail)")
            shown = (
                list(enumerate(astatus))[:4] + list(enumerate(astatus))[-4:]
                if n_archetypes > 8
                else list(enumerate(astatus))
            )
            colors = {"kept": GOOD, "rejected": "#F43B3E"}
            rep.table(
                ["#", "family", "stage", "detail"],
                [
                    [esc(i), esc(s["family"]), pill(s["stage"], colors.get(s["stage"], MUTED)),
                     esc(s["detail"][:120])]
                    for i, s in shown
                ],
            )
            await rep.flush()

    await render("waking teacher", force=True)
    loop = asyncio.get_running_loop()

    def on_status(s: str) -> None:
        asyncio.run_coroutine_threadsafe(render(f"waking teacher — {s}", force=True), loop)

    base_url = await asyncio.to_thread(
        llm_client.wait_until_ready, candidates, 1800, 15, on_status
    )

    teacher_sem = asyncio.Semaphore(3)  # llama.cpp serializes anyway; keep a small queue
    oracle_sem = asyncio.Semaphore(24)

    async def build_archetype(idx: int) -> tuple[arch.Archetype, list[tuple[float, dict]], dict] | None:
        family, hint = syn.FAMILY_HINTS[idx % len(syn.FAMILY_HINTS)]
        prompt = arch.ARCHETYPE_PROMPT.format(
            family_hint=hint, duration_s=60, allowed=", ".join(sorted(syn.ALLOWED_IMPORTS))
        )
        astatus[idx].update(stage="asking teacher")
        await render("generating")
        try:
            async with teacher_sem:
                text = await asyncio.to_thread(
                    llm_client.chat, base_url, [{"role": "user", "content": prompt}], 6144
                )
            archetype = arch.parse_archetype_response(text)
        except (syn.RejectedTask, llm_client.TeacherError) as e:
            astatus[idx].update(stage="rejected", detail=f"pre-oracle: {e}")
            await render("generating")
            return None

        astatus[idx].update(stage="calibrating", detail=archetype.description[:100])
        counters["calib_total"] += calibration_k
        await render("generating")
        points: list[tuple[float, dict]] = []
        stats: dict = {"cpu": [], "dur": []}

        async def calibrate(point: dict) -> None:
            code = arch.instantiate(archetype, point)
            oracle = run_generated.override(
                resources=flyte.Resources(cpu=4, memory="14Gi", disk="10Gi")
            )
            try:
                async with oracle_sem:
                    # Per-archetype group: ~450 calibration pods fold into
                    # one box per archetype in the run view.
                    with flyte.group(f"calibrate-arch-{idx}"):
                        measured = await oracle(
                            harness_code=code, task_id=f"arch-{seed}-{idx}-c{len(points)}"
                        )
            except Exception as e:  # noqa: BLE001 — pod died (likely footprint > 14Gi)
                print(f"[arch {idx}] calibration pod failed: {e}")
                return
            finally:
                counters["calib_done"] += 1
            if syn.curate_measurement(measured) is None:
                points.append((measured["peak_rss_mib"], point))
                stats["cpu"].append(measured.get("cpu_avg_cores", 1.0))
                stats["dur"].append(measured["duration_s"])

        await asyncio.gather(*(calibrate(p) for p in arch.calibration_points(archetype, calibration_k)))
        if len(points) < 2:
            astatus[idx].update(
                stage="rejected", detail=f"calibration: only {len(points)}/{calibration_k} valid"
            )
            await render("generating")
            return None
        astatus[idx].update(
            stage="kept",
            detail=f"{archetype.description[:80]} · {len(points)} pts "
            f"{min(p[0] for p in points):.0f}–{max(p[0] for p in points):.0f}MiB",
        )
        await render("generating")
        return archetype, points, {"family": family, "cpu": stats["cpu"], "dur": stats["dur"]}

    built = await asyncio.gather(*(build_archetype(i) for i in range(n_archetypes)))
    kept = [b for b in built if b]
    await render(f"instantiating from {len(kept)} archetypes", force=True)
    if not kept:
        raise RuntimeError(f"0/{n_archetypes} archetypes survived; see report for reasons")

    records: list[dict] = []
    # Calibration rows first: measured labels, real instantiated code.
    for ai, (archetype, points, meta) in enumerate(kept):
        for pi, (peak, point) in enumerate(points):
            records.append(
                {
                    "task_id": f"arch-{seed}-{ai}-cal{pi}",
                    "family": meta["family"],
                    "source_code": arch.instantiate(archetype, point),
                    "harness_code": arch.instantiate(archetype, point),
                    "input_profile": archetype.description,
                    "params_json": json.dumps(
                        {"archetype": ai, "label_source": "measured", **point}
                    ),
                    "prior_json": "",
                    "history_json": "",
                    "true_peak_memory_mib": float(peak),
                    "true_cpu_cores": max(1.0, round(sum(meta["cpu"]) / len(meta["cpu"]), 1)),
                    "true_gpu_mem_mib": 0.0,  # oracle pods are CPU-only
                    "duration_s": int(sum(meta["dur"]) / len(meta["dur"])),
                    "split": "train",
                }
            )
    per = max((total_tasks - len(records)) // len(kept), 1)
    for ai, (archetype, points, meta) in enumerate(kept):
        fit = arch.FootprintFit(points)
        cpu_label = max(1.0, round(sum(meta["cpu"]) / len(meta["cpu"]), 1))
        dur_label = int(sum(meta["dur"]) / len(meta["dur"]))
        for vi in range(per):
            if len(records) >= total_tasks:
                break
            values = arch.sample_params(archetype, rng)
            peak = fit.predict(values[archetype.memory_param], archetype.memory_param)
            records.append(
                {
                    "task_id": f"arch-{seed}-{ai}-v{vi}",
                    "family": meta["family"],
                    "source_code": arch.instantiate(archetype, values),
                    "harness_code": arch.instantiate(archetype, values),
                    "input_profile": archetype.description,
                    "params_json": json.dumps(
                        {"archetype": ai, "label_source": "fitted", **values}
                    ),
                    "prior_json": "",
                    "history_json": "",
                    "true_peak_memory_mib": float(min(max(peak, 96.0), 12288.0)),
                    "true_cpu_cores": cpu_label,
                    "true_gpu_mem_mib": 0.0,
                    "duration_s": dur_label,
                    "split": "train",
                }
            )
        counters["variants"] = len(records)
        await render("instantiating")

    counters["variants"] = len(records)
    await render(f"writing parquet ({len(records):,} rows)", force=True)
    path = tempfile.mktemp(suffix=".parquet")
    pd.DataFrame(records).to_parquet(path, index=False)
    synthetic_file = await publish_synthetic_corpus(
        corpus_file=await flyte.io.File.from_local(path), n=len(records), teacher=teacher
    )
    if not merge_with_templates:
        await render("done", force=True)
        return synthetic_file

    profile = get_profile(profile_name)
    template_records = build_corpus(profile.train_contexts, profile.eval_contexts, seed=seed)
    merged_path = tempfile.mktemp(suffix=".parquet")
    pd.concat([pd.DataFrame(template_records), pd.DataFrame(records)], ignore_index=True).to_parquet(
        merged_path, index=False
    )
    await render("publishing merged corpus", force=True)
    return publish(
        await flyte.io.File.from_local(merged_path),
        ARTIFACT_TASK_CORPUS,
        description=f"templates({len(template_records)}) + archetypes({len(records)}) "
        f"via {teacher}, seed={seed}",
    )


@driver_env.task(timeout=flyte.Timeout(max_runtime=2 * 3600), produces_artifacts=True, report=True)
async def tune_ab_experiment(
    n_tasks: int = 12,
    prior_cpu: int = 2,
    prior_memory: str = "2Gi",
    seed: int = 23,
) -> flyte.io.File:
    """A/B on real pods: tune-service proposals vs a hard-coded prior.

    The last-mile evidence the PRD's pricing depends on: for each held-out
    task, run one episode with the one-size-fits-all prior (what authors
    hard-code today) and one with the tuned proposal. Report OOM
    prevention (prior OOMs the big tasks) and overprovisioning reduction
    (prior wastes the small ones) — published as tuning-ab-report.
    """
    import asyncio
    import statistics

    import pandas as pd

    from ..environment.episodes import run_cluster_episode
    from ..policy.actions import validate_proposal
    from ..rewards.rewards import overprovision_fraction
    from ..taskgen.corpus import build_corpus

    request_proposal, service_url = tune.request_proposal, tune.service_url

    prior_kwargs = {"cpu": prior_cpu, "memory": prior_memory}
    prior_proposal = validate_proposal(prior_kwargs)
    records = [
        r
        for r in build_corpus(n_train=10, n_heldout=n_tasks, seed=seed)
        if r["split"] == "heldout"
    ][:n_tasks]

    rep = Reporter("A/B: tuned vs hard-coded resources", f"prior={prior_kwargs}")
    rep.p("Warming the tune service (scale-from-zero + checkpoint load)…")
    await rep.flush()

    # Warm the service; the first propose triggers the checkpoint load.
    def warm() -> str:
        deadline = time.monotonic() + 1500
        last = ""
        while time.monotonic() < deadline:
            out = request_proposal("warmup", records[0]["source_code"],
                                   records[0]["input_profile"], prior_kwargs)
            if out is not None:
                return "warm"
            last = "waking"
            time.sleep(20)
        raise RuntimeError(f"tune service at {service_url()} never became ready ({last})")

    await asyncio.to_thread(warm)

    async def arm(record: dict, tuned: bool):
        if not tuned:
            return await run_cluster_episode(record, prior_proposal)
        kwargs = await asyncio.to_thread(
            request_proposal,
            record["task_id"],
            record["source_code"],
            record["input_profile"],
            prior_kwargs,
        )
        proposal = prior_proposal if kwargs is None else validate_proposal(kwargs)
        ep = await run_cluster_episode(record, proposal)
        return ep

    rep.reset_body().p(
        f"Running {n_tasks} tasks × 2 arms (prior vs tuned) on real pods…"
    )
    await rep.flush()
    tuned_eps, prior_eps = await asyncio.gather(
        asyncio.gather(*(arm(r, True) for r in records)),
        asyncio.gather(*(arm(r, False) for r in records)),
    )

    def summarize(eps):
        wastes = [
            100
            * (
                overprovision_fraction(e.requested_memory_mib, e.peak_memory_mib)
                + overprovision_fraction(e.requested_cpu, e.peak_cpu)
            )
            / 2
            for e in eps
            if e.ok
        ]
        n = len(eps) or 1
        return {
            "oom_rate": sum(1 for e in eps if e.oom) / n,
            "fit_rate": sum(1 for e in eps if e.ok) / n,
            "median_overprovision_pct": statistics.median(wastes) if wastes else None,
        }

    t, p = summarize(tuned_eps), summarize(prior_eps)
    episodes = [
        {
            "task_id": r["task_id"],
            "family": r["family"],
            "analytic_peak_mib": round(r["true_peak_memory_mib"], 1),
            "prior": {"requested_mib": pe.requested_memory_mib, "ok": pe.ok, "oom": pe.oom,
                      "peak_rss_mib": round(pe.peak_memory_mib, 1)},
            "tuned": {"requested_mib": te.requested_memory_mib, "ok": te.ok, "oom": te.oom,
                      "peak_rss_mib": round(te.peak_memory_mib, 1)},
        }
        for r, te, pe in zip(records, tuned_eps, prior_eps)
    ]
    report = {
        "n_tasks": n_tasks,
        "prior": prior_kwargs,
        "prior_oom_rate": p["oom_rate"],
        "tuned_oom_rate": t["oom_rate"],
        "prior_fit_rate": p["fit_rate"],
        "tuned_fit_rate": t["fit_rate"],
        "prior_median_overprovision_pct": p["median_overprovision_pct"],
        "tuned_median_overprovision_pct": t["median_overprovision_pct"],
        "episodes": episodes,
    }
    assert all(k in report for k in AB_REPORT_KEYS)

    rep.reset_body()
    rep.h("Result (real pods)")
    rep.table(
        ["metric", "hard-coded prior", "tuned"],
        [
            ["OOM rate", f"{p['oom_rate']:.0%}", f"{t['oom_rate']:.0%}"],
            ["fit rate", f"{p['fit_rate']:.0%}", f"{t['fit_rate']:.0%}"],
            [
                "median overprovision",
                "-" if p["median_overprovision_pct"] is None else f"{p['median_overprovision_pct']:.0f}%",
                "-" if t["median_overprovision_pct"] is None else f"{t['median_overprovision_pct']:.0f}%",
            ],
        ],
    )
    await rep.flush()

    path = tempfile.mktemp(suffix=".json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return publish(
        await flyte.io.File.from_local(path),
        ARTIFACT_AB_REPORT,
        description=f"{n_tasks} tasks: OOM {p['oom_rate']:.0%}→{t['oom_rate']:.0%}, "
        f"waste {p['median_overprovision_pct'] and round(p['median_overprovision_pct'])}%→"
        f"{t['median_overprovision_pct'] and round(t['median_overprovision_pct'])}%",
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
