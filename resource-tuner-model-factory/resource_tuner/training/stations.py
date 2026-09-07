"""Factory stations: corpus → train → eval, wired two ways.

- The `tuner_pipeline` driver chains the stations for one readable E2E run.
- OnArtifact triggers (train_tuner: on tuning-task-corpus; eval_tuner: on
  tuner-checkpoint) run the same stations dark — publishing a corpus IS the
  request to train, a checkpoint IS the request to evaluate. Triggers
  deploy `auto_activate=True` — dark mode is LIVE after every deploy —
  and a trigger fires the task version it was deployed with, so
  re-deploy after any fix dark mode should pick up.

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
async def build_task_corpus(
    profile_name: str = "smoke", seed: int = 0, gpu_max_vram_mib: float = 0
) -> flyte.io.File:
    """Sample the task corpus and publish it as tuning-task-corpus.
    `gpu_max_vram_mib` > 0 caps GPU families (14000 → single-T4-only)."""
    import pandas as pd

    profile = get_profile(profile_name)
    rep = Reporter("Task corpus", f"profile={profile.name} seed={seed}")
    rep.kv({"train contexts": profile.train_contexts, "heldout contexts": profile.eval_contexts})
    await rep.flush()

    records = build_corpus(
        profile.train_contexts,
        profile.eval_contexts,
        seed=seed,
        gpu_max_vram_mib=gpu_max_vram_mib or None,
    )
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


@driver_env.task(timeout=flyte.Timeout(max_runtime=24 * 3600), produces_artifacts=True, report=True)
async def archetype_data_release(
    total_tasks: int = 100_000,
    n_archetypes: int = 150,
    calibration_k: int = 3,
    # Comma-separated teacher list — style diversity by construction.
    teacher: str = "qwen38-27b",
    merge_with_templates: bool = True,
    profile_name: str = "smoke",
    seed: int = 0,
    # Template-side overrides for large releases: 0 = use the profile's
    # counts. gpu_max_vram_mib > 0 caps template GPU families' VRAM
    # (14000 → every GPU task fits, and should be proposed, ONE T4).
    template_train: int = 0,
    template_heldout: int = 0,
    gpu_max_vram_mib: float = 0,
    # Round-12 label-fidelity + quality gates:
    holdout_k: int = 2,          # extra oracle pods per archetype, fit-error check
    holdout_max_err: float = 0.25,  # reject archetypes whose labels miss by more
    enforce_quality: bool = True,   # fail the release on quality-gate misses
) -> flyte.io.File:
    """Scale synthetic generation: archetypes × instantiation → 10⁵-10⁶ tasks.

    Round-12 shape: teachers (plural) write scenario-grounded archetypes
    (domain × data-shape × structure sampled per call, with an avoid-list
    of prior descriptions); the oracle calibrates each at `calibration_k`
    log-spaced points PLUS 2 random full-param draws; power-law fits label
    peak/cpu/duration (and VRAM for GPU archetypes, calibrated on T4
    pods); `holdout_k` extra pods measure fit error and reject archetypes
    whose labels lie; every variant gets its own input_profile (sampled
    scale rendered in) and synthetic prior/history context; and the
    release fails loudly if quality gates (near-dup rate, footprint
    coverage, concentration, label error) miss.
    """
    import asyncio
    import random

    import pandas as pd

    teacher_names = [t.strip() for t in teacher.split(",") if t.strip()]
    base_url = "(waking)"
    rng = random.Random(seed)

    def _hint_for(idx: int) -> tuple[str, str, bool]:
        """(family, hint, is_gpu): 1-in-8 archetypes are GPU workloads,
        calibrated on T4 pods; the rest rotate the CPU families."""
        if idx % 8 == 7:
            fam, hint = syn.GPU_FAMILY_HINTS[(idx // 8) % len(syn.GPU_FAMILY_HINTS)]
            return fam, hint, True
        fam, hint = syn.FAMILY_HINTS[idx % len(syn.FAMILY_HINTS)]
        return fam, hint, False

    astatus: list[dict] = [
        {"stage": "queued", "detail": "", "family": _hint_for(i)[0]}
        for i in range(n_archetypes)
    ]
    counters = {"calib_done": 0, "calib_total": 0, "variants": 0, "holdout_rejects": 0}
    kept_descriptions: list[str] = []  # the avoid-list fed back to teachers
    label_errors: list[float] = []
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

    await render("waking teachers", force=True)
    loop = asyncio.get_running_loop()

    def on_status(s: str) -> None:
        asyncio.run_coroutine_threadsafe(render(f"waking teachers — {s}", force=True), loop)

    async def wake(name: str, deadline: float) -> str | None:
        cands = llm_client.resolve_teacher_candidates(name)
        try:
            return await asyncio.to_thread(
                llm_client.wait_until_ready, cands, deadline, 15, on_status
            )
        except Exception as e:  # noqa: BLE001 — one dead teacher ≠ dead release
            print(f"[teachers] {name} failed to wake: {e} — continuing without it")
            return None

    # Wake all teachers concurrently and start as soon as the FIRST one is
    # ready, then give the stragglers a short grace period. Order must not
    # matter: round 12 listed a still-DEPLOYING 397B first and the run sat
    # on it while an ACTIVE 27B waited idle. (A dead/absent teacher 404s
    # for its whole budget, which is why nobody gets to block the release.)
    wakes = [asyncio.create_task(wake(n, 1800)) for n in teacher_names]
    done, pending = await asyncio.wait(
        wakes, timeout=1800, return_when=asyncio.FIRST_COMPLETED
    )
    ready = [t.result() for t in done if t.result()]
    if pending:
        if ready:
            print(f"[teachers] {len(ready)} ready; grace period for {len(pending)} more")
            grace_done, still_pending = await asyncio.wait(pending, timeout=300)
        else:
            grace_done, still_pending = await asyncio.wait(pending, timeout=1500)
        ready += [t.result() for t in grace_done if t.result()]
        for t in still_pending:
            t.cancel()
    base_urls = ready
    if not base_urls:
        raise RuntimeError(f"no teacher woke up (tried {teacher_names})")
    base_url = " + ".join(base_urls)

    # Per-teacher small queues (llama.cpp serializes anyway).
    teacher_sems = [asyncio.Semaphore(3) for _ in base_urls]
    oracle_sem = asyncio.Semaphore(24)

    async def build_archetype(idx: int):
        family, hint, is_gpu = _hint_for(idx)
        arng = random.Random(seed * 1_000_003 + idx)
        prompt = arch.render_archetype_prompt(
            family_hint=hint,
            scenario=syn.build_scenario(arng),
            avoid=list(kept_descriptions),
            allowed=", ".join(sorted(syn.ALLOWED_IMPORTS)),
            duration_s=arng.choice([45, 60, 90, 150]),
            gpu=is_gpu,
        )
        astatus[idx].update(stage="asking teacher")
        await render("generating")
        t_i = idx % len(base_urls)
        text = ""
        try:
            async with teacher_sems[t_i]:
                text = await asyncio.to_thread(
                    llm_client.chat, base_urls[t_i], [{"role": "user", "content": prompt}], 8192
                )
            archetype = arch.parse_archetype_response(text)
        except (syn.RejectedTask, llm_client.TeacherError) as e:
            astatus[idx].update(stage="rejected", detail=f"pre-oracle: {e}")
            # Print it too: report-only rejection reasons made a 0/24
            # smoke undebuggable from logs (round-12 lesson).
            print(
                f"[arch {idx}] pre-oracle reject: {e} | resp[{len(text)}]: {text[:400]!r}"
            )
            await render("generating")
            return None

        astatus[idx].update(stage="calibrating", detail=archetype.description[:100])
        n_pods = calibration_k + 2 + holdout_k
        counters["calib_total"] += n_pods
        await render("generating")
        # measured: [(peak, cpu, dur, vram, params)]
        measured_pts: list[tuple[float, float, float, float, dict]] = []

        # Oracle sizing is a SCHEDULING decision, not a generosity one:
        # 3 CPU / 12Gi fits t3a.xlarge (3670m / 13.7Gi, 3–250 nodes), so
        # calibration fans out over the tenant's biggest pool. The old
        # 4 CPU / 14–20Gi only fit c5.4xlarge (0–5 nodes) — 30 pods then
        # queued past max_queued_time and timed out en masse (round 12).
        oracle_res = (
            flyte.Resources(cpu=4, memory="14Gi", gpu="T4:1", disk="20Gi")
            if is_gpu
            else flyte.Resources(cpu=3, memory="12Gi", disk="10Gi")
        )
        # Calibration is throughput work, not latency work: let pods wait
        # for a node rather than dying at the episode-tuned 300s.
        oracle_timeout = flyte.Timeout(max_runtime=900, max_queued_time=3600)

        async def run_point(point: dict, tag: str, sink: list) -> None:
            code = arch.instantiate(archetype, point)
            oracle = run_generated.override(
                resources=oracle_res, timeout=oracle_timeout
            )
            try:
                async with oracle_sem:
                    with flyte.group(f"calibrate-arch-{idx}"):
                        m = await oracle(harness_code=code, task_id=f"arch-{seed}-{idx}-{tag}")
            except Exception as e:  # noqa: BLE001 — pod died (likely over budget)
                print(f"[arch {idx}] {tag} pod failed: {e}")
                return
            finally:
                counters["calib_done"] += 1
            # Ceiling tracks the oracle pod's own budget (12Gi) with
            # headroom — a "measurement" at the pod limit is a truncated
            # workload, not a footprint.
            if syn.curate_measurement(m, max_mib=10240) is not None:
                return
            vram = float(m.get("gpu_peak_mib", 0.0) or 0.0)
            if is_gpu and (vram < 64 or vram > 14000):
                # A "GPU" archetype that never touched CUDA (or blew past a
                # T4) teaches the wrong lesson — drop the point.
                print(f"[arch {idx}] {tag}: gpu archetype vram {vram:.0f}MiB out of bounds")
                return
            sink.append(
                (m["peak_rss_mib"], m.get("cpu_avg_cores", 1.0), m["duration_s"], vram, point)
            )

        cal_points = arch.calibration_points(archetype, calibration_k, rng=arng, n_random=2)
        await asyncio.gather(
            *(run_point(p, f"c{i}", measured_pts) for i, p in enumerate(cal_points))
        )
        if len(measured_pts) < 3:
            astatus[idx].update(
                stage="rejected", detail=f"calibration: only {len(measured_pts)}/{len(cal_points)} valid"
            )
            await render("generating")
            return None

        mem_param = archetype.memory_param
        peak_fit = arch.ParamFit([(p[0], p[4]) for p in measured_pts], mem_param, floor=32.0)

        # Holdout: extra pods at RANDOM variant params measure how honest
        # the fitted labels will be; a lying fit rejects the archetype.
        holdout_pts: list[tuple[float, float, float, float, dict]] = []
        await asyncio.gather(
            *(
                run_point(arch.sample_params(archetype, arng), f"h{i}", holdout_pts)
                for i in range(holdout_k)
            )
        )
        err = arch.holdout_error(peak_fit, [(p[0], p[4]) for p in holdout_pts])
        if err is not None:
            label_errors.append(err)
            if err > holdout_max_err:
                counters["holdout_rejects"] += 1
                astatus[idx].update(
                    stage="rejected", detail=f"holdout label error {err:.0%} > {holdout_max_err:.0%}"
                )
                await render("generating")
                return None
        measured_pts += holdout_pts  # honest extra fit points once accepted

        astatus[idx].update(
            stage="kept",
            detail=f"{archetype.description[:70]} · {len(measured_pts)} pts "
            f"{min(p[0] for p in measured_pts):.0f}–{max(p[0] for p in measured_pts):.0f}MiB"
            + (f" · err {err:.0%}" if err is not None else ""),
        )
        kept_descriptions.append(archetype.description)
        await render("generating")
        return archetype, measured_pts, {"family": family, "gpu": is_gpu, "holdout_err": err}

    built = await asyncio.gather(*(build_archetype(i) for i in range(n_archetypes)))
    kept = [b for b in built if b]
    await render(f"instantiating from {len(kept)} archetypes", force=True)
    if not kept:
        raise RuntimeError(f"0/{n_archetypes} archetypes survived; see report for reasons")

    records: list[dict] = []
    import zlib as _zlib

    from ..taskgen import quality as qual
    from ..taskgen.corpus import context_fields_json

    def _row(ai, tag, archetype, meta, values, peak, cpu, dur, vram, source):
        crng = random.Random(_zlib.crc32(f"arch-{seed}-{ai}-{tag}".encode()))
        prior_json, history_json = context_fields_json(peak, cpu, vram, crng)
        return {
            "task_id": f"arch-{seed}-{ai}-{tag}",
            "family": meta["family"],
            "source_code": arch.instantiate(archetype, values),
            "harness_code": arch.instantiate(archetype, values),
            # Round-12 fix: every variant renders ITS OWN sampled scale —
            # a static per-archetype profile trained profile-blindness.
            "input_profile": arch.variant_profile(archetype.description, values),
            "params_json": json.dumps({"archetype": ai, "label_source": source, **values}),
            "prior_json": prior_json,
            "history_json": history_json,
            "true_peak_memory_mib": float(min(max(peak, 64.0), 16384.0)),
            "true_cpu_cores": max(0.5, round(float(cpu), 1)),
            "true_gpu_mem_mib": float(vram),
            "duration_s": int(min(max(dur, 30), 400)),
            "split": "train",
        }

    # Calibration/holdout rows first: measured labels, real code.
    for ai, (archetype, points, meta) in enumerate(kept):
        for pi, (peak, cpu, dur, vram, point) in enumerate(points):
            records.append(
                _row(ai, f"cal{pi}", archetype, meta, point, peak, cpu, dur, vram, "measured")
            )

    per = max((total_tasks - len(records)) // len(kept), 1)
    for ai, (archetype, points, meta) in enumerate(kept):
        mem_param = archetype.memory_param
        peak_fit = arch.ParamFit([(p[0], p[4]) for p in points], mem_param, floor=32.0)
        cpu_fit = arch.ParamFit([(p[1], p[4]) for p in points], mem_param, floor=0.5)
        dur_fit = arch.ParamFit([(p[2], p[4]) for p in points], mem_param, floor=20.0)
        vram_fit = (
            arch.ParamFit([(p[3], p[4]) for p in points], mem_param, floor=0.0)
            if meta["gpu"]
            else None
        )
        for vi in range(per):
            if len(records) >= total_tasks:
                break
            values = arch.sample_params(archetype, rng)
            x = values[mem_param]
            records.append(
                _row(
                    ai, f"v{vi}", archetype, meta, values,
                    peak_fit.predict(x),
                    cpu_fit.predict(x),
                    dur_fit.predict(x),
                    min(vram_fit.predict(x), 14000.0) if vram_fit else 0.0,
                    "fitted",
                )
            )
        counters["variants"] = len(records)
        await render("instantiating")

    counters["variants"] = len(records)

    # ── quality gates: measured, rendered, ENFORCED ─────────────────────
    await render("scoring corpus quality", force=True)
    qreport = qual.quality_report(
        [a.code for a, _, _ in kept], records, label_errors=label_errors
    )
    fails = qual.gate_failures(qreport)
    fmt = lambda v: "-" if v is None else (f"{v:.0%}" if isinstance(v, float) else str(v))  # noqa: E731
    rep.h("Quality gates")
    rep.table(
        ["metric", "value"],
        [
            [esc("archetype near-dup rate"), esc(fmt(qreport["near_dup_rate"]))],
            [esc("token entropy (bits)"), esc(qreport["token_entropy_bits"])],
            [esc("footprint grid coverage"), esc(fmt(qreport["coverage"]["coverage"]))],
            [esc("max archetype share of rows"), esc(fmt(qreport["max_archetype_share"]))],
            [esc("label error p50 / p90"),
             esc(f"{fmt(qreport['label_error_p50'])} / {fmt(qreport['label_error_p90'])}")],
            [esc("holdout-rejected archetypes"), esc(counters["holdout_rejects"])],
            [esc("family mixture"),
             esc(", ".join(f"{f}:{s:.0%}" for f, s in qreport["mixture"].items()))],
        ],
    )
    if fails:
        rep.h("GATE FAILURES")
        for f in fails:
            rep.p(f, color="#F43B3E")
    await rep.flush()
    if fails and enforce_quality:
        raise RuntimeError(f"corpus quality gates failed: {'; '.join(fails)}")

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
    template_records = build_corpus(
        template_train or profile.train_contexts,
        template_heldout or profile.eval_contexts,
        seed=seed,
        gpu_max_vram_mib=gpu_max_vram_mib or None,
    )
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
