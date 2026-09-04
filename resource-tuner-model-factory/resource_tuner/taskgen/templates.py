"""Task templates: realistic Flyte 2 code with a controllable footprint.

Each family renders Python that looks like in-distribution Flyte task code
(data engineering, data science, ML training, batch inference, ETL), while
its actual peak memory and CPU demand are functions of sampled parameters.
That duality is what makes the RL environment workable:

- the POLICY sees `render_for_policy()` — a plausible Flyte task module —
  and must propose `flyte.Resources` kwargs for it;
- the SIMULATOR scores proposals against `footprint()` — the analytic
  peak — without touching a cluster (the PRD's "cheap loops first");
- the HARNESS runs `render_for_harness()` — the same workload as a plain
  function — inside a real pod sized by the proposal, so the analytic
  model's error is measurable as a sim-vs-real gap rather than silent.

Workloads hold their peak allocation in a timed loop (default ≥60s):
the metrics pipeline scrapes pods every 15–30s and answers OutOfRange for
attempts under ~30s, so short bursts would be invisible to pod metrics.

Footprint formulas are deliberately coarse (constants fitted by reasoning,
not measurement). Their error is part of the environment: training reward
is computed against the same formulas the corpus was labeled with
(self-consistent), and the on-cluster eval quantifies the gap.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from random import Random
from typing import Callable

_MIB = 1024 * 1024

# Interpreter + import overhead, MiB. Torch-importing families pay more.
BASE_OVERHEAD_MIB = 220
TORCH_OVERHEAD_MIB = 650


@dataclass(frozen=True)
class GeneratedTask:
    """One sampled task: what the policy sees, what the harness runs, and
    the analytic ground truth used by the simulator."""

    task_id: str
    family: str
    source_code: str  # policy view: a realistic Flyte task module
    harness_code: str  # executable view: plain function + __main__ driver
    input_profile: str
    params: dict
    true_peak_memory_mib: float
    true_cpu_cores: float
    duration_s: int


@dataclass(frozen=True)
class Family:
    name: str
    task_name: str  # the def name used in rendered code
    imports: str  # top-of-module imports for the policy view
    body_template: str  # function body, formatted with params
    sample: Callable[[Random], dict]
    footprint: Callable[[dict], tuple[float, float]]  # → (peak MiB, cpu cores)
    profile: Callable[[dict], str]


def _render_policy(fam: Family, params: dict) -> str:
    """The Flyte module the policy is asked to size. No resources declared —
    the cold-start case (PRD UC2); the proposal IS the missing declaration."""
    body = textwrap.indent(fam.body_template.format(**params), "    ")
    return (
        f"{fam.imports}\n"
        "import flyte\n"
        "\n"
        f'env = flyte.TaskEnvironment(name="{fam.name.replace("_", "-")}")\n'
        "\n"
        "\n"
        "@env.task\n"
        f"async def {fam.task_name}() -> dict:\n"
        f"{body}\n"
    )


def _render_harness(fam: Family, params: dict) -> str:
    """The same workload as a plain sync function; the harness exec()s this
    and calls run() — no flyte imports, so it runs in any pod."""
    body = textwrap.indent(fam.body_template.format(**params), "    ")
    return f"{fam.imports}\n\n\ndef run() -> dict:\n{body}\n"


def _hold_loop(work_lines: str) -> str:
    """Wrap one unit of work in a loop that holds the footprint for
    `{duration_s}` seconds — the scrape-visibility floor."""
    return (
        "import time as _time\n"
        "_deadline = _time.monotonic() + {duration_s}\n"
        "_iters = 0\n"
        "while True:\n"
        + textwrap.indent(work_lines, "    ")
        + "\n    _iters += 1\n"
        "    if _time.monotonic() >= _deadline:\n"
        "        break\n"
    )


# ── data engineering: pandas ETL ────────────────────────────────────────

_DE_BODY = _hold_loop(
    """\
df = pd.DataFrame(
    np.random.default_rng(_iters).standard_normal(({rows}, {cols})),
    columns=[f"f{{i}}" for i in range({cols})],
)
df["segment"] = np.random.default_rng(_iters).integers(0, {segments}, {rows})
agg = df.groupby("segment").agg(["mean", "std", "max"])
enriched = df.merge(agg["f0"], on="segment", suffixes=("", "_seg"))
result = {{"rows": int(len(enriched)), "segments": int(agg.shape[0])}}
del df, agg, enriched"""
) + 'return result'


def _de_sample(rng: Random) -> dict:
    return {
        "rows": int(10 ** rng.uniform(5.0, 6.9)),
        "cols": rng.choice([4, 8, 16, 32, 48]),
        "segments": rng.choice([10, 100, 1000]),
        "duration_s": rng.randint(60, 120),
    }


def _de_footprint(p: dict) -> tuple[float, float]:
    # df + groupby temporaries + merged copy ≈ 3.5x the base frame.
    frame = p["rows"] * (p["cols"] + 1) * 8 / _MIB
    return BASE_OVERHEAD_MIB + 3.5 * frame, 1.0


# ── data science: sklearn model fit ─────────────────────────────────────

_DS_BODY = _hold_loop(
    """\
X = np.random.default_rng(_iters).standard_normal(({n_samples}, {n_features}))
y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
model = RandomForestClassifier(
    n_estimators={n_estimators}, max_depth=8, n_jobs={n_jobs}, random_state=0
)
model.fit(X, y)
result = {{"score": float(model.score(X, y)), "n_samples": {n_samples}}}
del X, y, model"""
) + 'return result'


def _ds_sample(rng: Random) -> dict:
    return {
        "n_samples": int(10 ** rng.uniform(4.0, 5.7)),
        "n_features": rng.choice([8, 16, 32, 64, 128]),
        "n_estimators": rng.choice([20, 50, 100]),
        "n_jobs": rng.choice([1, 2, 4]),
        "duration_s": rng.randint(60, 120),
    }


def _ds_footprint(p: dict) -> tuple[float, float]:
    # X + per-tree bootstrap/sort temporaries; forests parallelize per tree,
    # so n_jobs is the sustained CPU demand.
    x = p["n_samples"] * p["n_features"] * 8 / _MIB
    trees = p["n_estimators"] * p["n_samples"] * 24 / _MIB * 0.05
    return BASE_OVERHEAD_MIB + 2.5 * x + trees + 0.6 * x * p["n_jobs"], float(p["n_jobs"])


# ── ML training: torch MLP loop ─────────────────────────────────────────

_MLT_BODY = _hold_loop(
    """\
torch.manual_seed(_iters)
model = torch.nn.Sequential(*(
    layer
    for _ in range({depth})
    for layer in (torch.nn.Linear({hidden}, {hidden}), torch.nn.ReLU())
))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _step in range(4):
    x = torch.randn({batch_size}, {hidden})
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
result = {{"loss": float(loss.item())}}
del model, opt, x"""
) + 'return result'


def _mlt_sample(rng: Random) -> dict:
    return {
        "hidden": rng.choice([512, 1024, 2048, 3072]),
        "depth": rng.choice([2, 4, 6, 8]),
        "batch_size": rng.choice([64, 256, 1024]),
        "duration_s": rng.randint(60, 120),
    }


def _mlt_footprint(p: dict) -> tuple[float, float]:
    # fp32 weights + grads + two AdamW states = 4x params; activations per
    # layer kept for backward.
    params = p["depth"] * p["hidden"] ** 2 * 4 / _MIB
    acts = p["batch_size"] * p["hidden"] * p["depth"] * 4 * 2 / _MIB
    return TORCH_OVERHEAD_MIB + 4 * params + acts, 2.0


# ── batch inference: numpy embedding scoring ────────────────────────────

_BI_BODY = _hold_loop(
    """\
weights = np.random.default_rng(0).standard_normal(({dim}, {out_dim})).astype(np.float32)
scored = 0
for _b in range({n_batches}):
    batch = np.random.default_rng(_b).standard_normal(({batch_size}, {dim})).astype(np.float32)
    scores = batch @ weights
    scored += int((scores.max(axis=1) > 0).sum())
result = {{"scored": scored}}
del weights, batch, scores"""
) + 'return result'


def _bi_sample(rng: Random) -> dict:
    return {
        "dim": rng.choice([256, 768, 1536]),
        "out_dim": rng.choice([1024, 4096, 16384]),
        "batch_size": rng.choice([512, 2048, 8192]),
        "n_batches": 8,
        "duration_s": rng.randint(60, 120),
    }


def _bi_footprint(p: dict) -> tuple[float, float]:
    w = p["dim"] * p["out_dim"] * 4 / _MIB
    b = p["batch_size"] * (p["dim"] + p["out_dim"]) * 4 / _MIB
    return BASE_OVERHEAD_MIB + w + 2.2 * b, 1.0


# ── ETL: text/record processing ─────────────────────────────────────────

_ETL_BODY = _hold_loop(
    """\
records = [
    {{"id": i, "payload": ("x" * 96), "score": (i * 7919) % 1000}}
    for i in range({n_records})
]
by_bucket = {{}}
for r in records:
    by_bucket.setdefault(r["score"] // 100, []).append(r["id"])
result = {{"records": len(records), "buckets": len(by_bucket)}}
del records, by_bucket"""
) + 'return result'


def _etl_sample(rng: Random) -> dict:
    return {
        "n_records": int(10 ** rng.uniform(5.0, 6.8)),
        "duration_s": rng.randint(60, 120),
    }


def _etl_footprint(p: dict) -> tuple[float, float]:
    # Python dict-of-small-fields overhead dominates: ~500 bytes per record.
    return BASE_OVERHEAD_MIB + p["n_records"] * 500 / _MIB, 1.0


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in (
        Family(
            name="data_engineering",
            task_name="transform_events",
            imports="import numpy as np\nimport pandas as pd",
            body_template=_DE_BODY,
            sample=_de_sample,
            footprint=_de_footprint,
            profile=lambda p: (
                f"input: event table, ~{p['rows']:,} rows x {p['cols']} float columns "
                f"(~{p['rows'] * (p['cols'] + 1) * 8 / _MIB:.0f}MiB in memory), "
                f"{p['segments']} segments"
            ),
        ),
        Family(
            name="data_science",
            task_name="train_classifier",
            imports="import numpy as np\nfrom sklearn.ensemble import RandomForestClassifier",
            body_template=_DS_BODY,
            sample=_ds_sample,
            footprint=_ds_footprint,
            profile=lambda p: (
                f"input: feature matrix {p['n_samples']:,} x {p['n_features']} float64 "
                f"(~{p['n_samples'] * p['n_features'] * 8 / _MIB:.0f}MiB), "
                f"{p['n_estimators']} trees, n_jobs={p['n_jobs']}"
            ),
        ),
        Family(
            name="ml_training",
            task_name="train_mlp",
            imports="import torch",
            body_template=_MLT_BODY,
            sample=_mlt_sample,
            footprint=_mlt_footprint,
            profile=lambda p: (
                f"model: {p['depth']}-layer MLP, hidden {p['hidden']} "
                f"(~{p['depth'] * p['hidden'] ** 2 * 4 / _MIB:.0f}MiB fp32 params), "
                f"batch {p['batch_size']}, AdamW"
            ),
        ),
        Family(
            name="batch_inference",
            task_name="score_embeddings",
            imports="import numpy as np",
            body_template=_BI_BODY,
            sample=_bi_sample,
            footprint=_bi_footprint,
            profile=lambda p: (
                f"input: {p['n_batches']} batches of {p['batch_size']:,} vectors, "
                f"dim {p['dim']} -> {p['out_dim']} (fp32)"
            ),
        ),
        Family(
            name="etl",
            task_name="bucket_records",
            imports="",
            body_template=_ETL_BODY,
            sample=_etl_sample,
            footprint=_etl_footprint,
            profile=lambda p: f"input: {p['n_records']:,} JSON-ish records, ~100B payload each",
        ),
    )
}


def generate_task(family: str, seed: int) -> GeneratedTask:
    """Deterministically sample one task from a family.

    zlib.crc32, not hash(): str hashing is salted per process, which made
    "the same corpus" differ between runs (and made a test flake ~1 in 20).
    """
    import zlib

    fam = FAMILIES[family]
    rng = Random(zlib.crc32(family.encode()) ^ seed)
    params = fam.sample(rng)
    peak_mib, cpu = fam.footprint(params)
    return GeneratedTask(
        task_id=f"{family}-{seed}",
        family=family,
        source_code=_render_policy(fam, params),
        harness_code=_render_harness(fam, params),
        input_profile=fam.profile(params),
        params=params,
        true_peak_memory_mib=peak_mib,
        true_cpu_cores=cpu,
        duration_s=params["duration_s"],
    )
