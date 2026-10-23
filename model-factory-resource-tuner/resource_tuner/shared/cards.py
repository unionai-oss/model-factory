"""Artifact cards: what a published artifact CONTAINS, written at publish time.

Every station here hands its output to another station — or to a trigger,
or to a human three weeks later — as a versioned artifact. A name and a
one-line description are enough to route an artifact and nowhere near
enough to trust it: whoever picks up `tuning-task-corpus` needs the
schema, the split sizes, where the labels came from and which teacher
wrote the rows BEFORE they download a million rows to find out.

The lineage dashboard's `_artifact_card` answers those questions by
downloading the payload at VIEW time. That stays the right fallback for
versions published before this module existed, but it is the wrong
default: the producing task already holds every number, and re-deriving
them costs a download per viewer while still not recovering facts the
payload never carried (the seed, the profile, the teacher pool, the
hypothesis the run was testing).

So cards are rendered HERE, from the producer's own state, and uploaded
alongside the artifact. Renderers are pure (dict in, markdown out) so
they are testable without a cluster; `upload()` is the only function that
touches the network.

Markdown, not HTML: the console renders it, a human can read the raw
bytes, and a diff between two versions of an artifact is legible.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..contracts import CORPUS_COLUMN_DOCS

# ── formatting helpers ──────────────────────────────────────────────────


def _num(v: Any, fmt: str = ",.0f", dash: str = "–") -> str:
    """Format a number, or `dash` when it is missing — never 'None'."""
    if v is None or v == "":
        return dash
    try:
        return format(float(v), fmt)
    except (TypeError, ValueError):
        return str(v)


def _pct(v: Any, scale: float = 1.0, dash: str = "–") -> str:
    """`scale=1` for a 0-1 rate, `scale=0.01` for an already-percent value."""
    if v is None or v == "":
        return dash
    try:
        return f"{float(v) * scale * 100:.0f}%"
    except (TypeError, ValueError):
        return str(v)


def _usd(v: Any, dash: str = "–") -> str:
    return dash if v is None else f"${float(v):.4f}"


def _usd_signed(v: Any, dash: str = "–") -> str:
    """Savings read as a direction first: `+$40.52`, `-$3.30`."""
    if v is None or v == "":
        return dash
    return f"{'+' if float(v) >= 0 else '-'}${abs(float(v)):,.2f}"


def _cell(v: Any) -> str:
    """Table cells must not contain raw pipes — they would split the row."""
    return str(v).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: list[list[Any]]) -> str:
    """A markdown table. Empty rows render as nothing, not as a bare header."""
    if not rows:
        return ""
    head = "| " + " | ".join(_cell(h) for h in headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(_cell(c) for c in r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def kv(pairs: Mapping[str, Any]) -> str:
    """Bulleted key/value block, skipping keys whose value is empty."""
    return "\n".join(
        f"- **{k}**: {_cell(v)}" for k, v in pairs.items() if v not in (None, "", [])
    )


def _join(*blocks: str) -> str:
    """Stitch sections, dropping the ones that rendered empty."""
    return "\n\n".join(b.strip() for b in blocks if b and b.strip()) + "\n"


def _attrs(pairs: Mapping[str, Any]) -> dict[str, str]:
    """Artifact attrs are Mapping[str, str] — stringify and drop blanks."""
    return {k: str(v) for k, v in pairs.items() if v not in (None, "", [])}


# ── corpus stats (read from the parquet the task just wrote) ────────────

FACET_COLUMNS = ("family", "split", "generator")
NUMERIC_COLUMNS = (
    "true_peak_memory_mib",
    "true_cpu_cores",
    "true_gpu_mem_mib",
    "duration_s",
)


def corpus_stats(path: str, label_source_sample: int = 20_000) -> dict:
    """Facet counts + footprint quantiles for a corpus parquet.

    Reads only the columns it needs, and SAMPLES `params_json` for the
    label-source breakdown: at 1M rows materializing that column costs
    hundreds of MB to learn a two-way split, and the publishing task is
    already holding a corpus in memory.

    Every field is optional by design — a card is documentation, and a
    stats failure must never take down the publish that carries it.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    names = set(pf.schema_arrow.names)
    facets = [c for c in FACET_COLUMNS if c in names]
    numeric = [c for c in NUMERIC_COLUMNS if c in names]
    tbl = pq.read_table(path, columns=facets + numeric)

    stats: dict[str, Any] = {
        "rows": pf.metadata.num_rows,
        "row_groups": pf.num_row_groups,
        "columns": list(pf.schema_arrow.names),
    }
    for col in facets:
        stats[col] = {
            str(d["values"]): d["counts"] for d in pc.value_counts(tbl[col]).to_pylist()
        }
    for col in numeric:
        q = pc.quantile(tbl[col], q=[0.05, 0.5, 0.95]).to_pylist()
        stats.setdefault("quantiles", {})[col] = {
            "p5": q[0],
            "p50": q[1],
            "p95": q[2],
            "max": pc.max(tbl[col]).as_py(),
        }
    if "family" in facets and "true_peak_memory_mib" in numeric:
        g = tbl.group_by("family").aggregate(
            [
                ("family", "count"),
                ("true_peak_memory_mib", "min"),
                ("true_peak_memory_mib", "approximate_median"),
                ("true_peak_memory_mib", "max"),
            ]
            + ([("true_cpu_cores", "max")] if "true_cpu_cores" in numeric else [])
        ).to_pydict()
        stats["by_family"] = g
    if "true_gpu_mem_mib" in numeric:
        stats["gpu_rows"] = pc.sum(
            pc.greater(tbl["true_gpu_mem_mib"], 0)
        ).as_py() or 0

    # label_source lives inside params_json; sample rather than scan.
    if "params_json" in names:
        try:
            sample = pq.read_table(path, columns=["params_json"])["params_json"][
                :label_source_sample
            ].to_pylist()
            counts: dict[str, int] = {}
            for raw in sample:
                try:
                    src = json.loads(raw).get("label_source", "unrecorded")
                except (TypeError, ValueError):
                    src = "unparseable"
                counts[str(src)] = counts.get(str(src), 0) + 1
            stats["label_sources"] = counts
            stats["label_sources_sampled"] = len(sample)
        except Exception:  # noqa: BLE001 — a card must not break a publish
            pass
    return stats


def corpus_stats_safe(path: str) -> dict:
    """`corpus_stats` that degrades to an empty dict instead of raising.

    Card rendering tolerates missing keys, so a stats failure costs a
    thinner card — never the corpus release that was carrying it.
    """
    try:
        return corpus_stats(path)
    except Exception as e:  # noqa: BLE001
        print(f"[card] corpus stats unavailable ({type(e).__name__}: {e})")
        return {}


def _top(counts: Mapping[str, int] | None, limit: int = 8) -> str:
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda kv_: -kv_[1])
    shown = ", ".join(f"{k} {v:,}" for k, v in ordered[:limit])
    rest = len(ordered) - limit
    return shown + (f", +{rest} more" if rest > 0 else "")


# ── corpus card ─────────────────────────────────────────────────────────

_ROW_EXPLAINER = """\
## What one row is

One row is one **workload plus the resources it actually needs**. The
policy sees only the context fields (`source_code`, `input_profile`,
`prior_json`, `history_json`) and proposes CPU/memory/GPU; the `true_*`
columns are the answer key and are never shown to the policy at inference
time."""

_LABEL_EXPLAINER = """\
## Where the labels come from

`true_peak_memory_mib` / `true_cpu_cores` / `true_gpu_mem_mib` are
**measurement-anchored, never a model's guess** — that guess is the bias
this factory exists to remove:

- **template** rows carry analytic footprints computed from the generator
  parameters that produced the code.
- **teacher** rows (any non-`template` generator) are written by a teacher
  LLM, screened by AST for safety, then EXECUTED in a harness pod whose
  measured peak RSS and CPU become the label. Archetype rows are
  instantiated from a calibration fit over several measured points, and
  `params_json.label_source` records `measured` vs `fitted` per row."""


def corpus_card(
    *,
    artifact: str,
    stats: Mapping[str, Any],
    provenance: Mapping[str, Any],
    intended_use: str = "",
) -> str:
    """Dataset card for `tuning-task-corpus` / `synthetic-task-corpus`."""
    q = stats.get("quantiles", {}) or {}
    splits = stats.get("split") or {}
    at_a_glance = {
        "rows": f"{stats.get('rows', 0):,}",
        "splits": _top(splits) or "unsplit",
        "families": _top(stats.get("family")),
        "generators": _top(stats.get("generator")),
        "GPU rows": (
            f"{stats['gpu_rows']:,}" if stats.get("gpu_rows") is not None else None
        ),
        "label sources": (
            f"{_top(stats.get('label_sources'))} "
            f"(sampled {stats.get('label_sources_sampled', 0):,} rows)"
            if stats.get("label_sources")
            else None
        ),
        **{k.replace("_", " "): v for k, v in provenance.items()},
    }

    # Cores are fractional; MiB and seconds are not. Rounding 1.5 cores to
    # "2" in a card about right-sizing would be its own small lie.
    fmt = lambda col: ".1f" if col == "true_cpu_cores" else ",.0f"  # noqa: E731
    footprints = table(
        ["column", "p5", "median", "p95", "max"],
        [
            [
                col,
                _num(v["p5"], fmt(col)),
                _num(v["p50"], fmt(col)),
                _num(v["p95"], fmt(col)),
                _num(v["max"], fmt(col)),
            ]
            for col, v in q.items()
        ],
    )

    fam = stats.get("by_family") or {}
    by_family = table(
        ["family", "rows", "peak MiB min", "median≈", "max", "cpu cores max"],
        [
            [
                fam["family"][i],
                _num(fam["family_count"][i]),
                _num(fam["true_peak_memory_mib_min"][i]),
                _num(fam["true_peak_memory_mib_approximate_median"][i]),
                _num(fam["true_peak_memory_mib_max"][i]),
                _num(fam.get("true_cpu_cores_max", [None] * (i + 1))[i], ".1f"),
            ]
            for i in range(len(fam.get("family", [])))
        ],
    )

    present = set(stats.get("columns") or [])
    schema = table(
        ["column", "meaning"],
        [
            [c, doc]
            for c, doc in CORPUS_COLUMN_DOCS.items()
            if not present or c in present
        ],
    )
    extra = sorted(present - set(CORPUS_COLUMN_DOCS))

    return _join(
        f"# {artifact}",
        "Task contexts for the resource-tuning policy: workloads paired with "
        "the resources they actually consume.",
        "## At a glance",
        kv(at_a_glance),
        _ROW_EXPLAINER,
        "## Schema" + (f" ({len(present)} columns)" if present else ""),
        schema,
        f"Undocumented extra columns: {', '.join(extra)}." if extra else "",
        "## Composition by family" if by_family else "",
        by_family,
        "## Footprint distribution" if footprints else "",
        footprints,
        _LABEL_EXPLAINER,
        ("## Intended use\n\n" + intended_use) if intended_use else "",
    )


def corpus_attrs(
    stats: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, str]:
    splits = stats.get("split") or {}
    return _attrs(
        {
            "rows": stats.get("rows"),
            "train_rows": splits.get("train"),
            "heldout_rows": splits.get("heldout"),
            "families": len(stats.get("family") or {}) or None,
            "generators": ",".join(sorted(stats.get("generator") or {})),
            "gpu_rows": stats.get("gpu_rows"),
            "schema_columns": len(stats.get("columns") or []) or None,
            **provenance,
        }
    )


# ── checkpoint card ─────────────────────────────────────────────────────


def checkpoint_card(manifest: Mapping[str, Any], *, step: int | None = None) -> str:
    """Model card for `tuner-checkpoint` (and its intermediates)."""
    fm = manifest.get("final_metrics") or {}
    adapter = "LoRA adapter" if manifest.get("use_lora", True) else "full fine-tune"
    intermediate = step is not None

    glance = {
        "base model": manifest.get("base_model"),
        "adaptation": adapter + (", MLP layers" if manifest.get("lora_mlp") else ""),
        "training profile": manifest.get("profile"),
        "reward stage": manifest.get("reward_stage"),
        "steps": (
            f"{step:,}"
            + (f" of {_num(manifest['max_steps'])}" if manifest.get("max_steps") else "")
            if intermediate
            else _num(manifest.get("max_steps"))
        ),
        "trained on corpus": manifest.get("corpus_path"),
        "GBT hint in prompt": manifest.get("gbt_hint"),
        "warm-started from": manifest.get("resume") or manifest.get("intra_task_resume"),
    }

    usage = """\
## What it does

Given a Flyte task's source, an input profile and any prior/history, the
policy emits a JSON resource proposal (CPU, memory, and GPU memory when
the workload needs one). It is a GRPO-trained policy over the corpus
above, rewarded for fitting the measured footprint without overshooting.

## How to load

A PEFT adapter directory: `manifest.json`, `training_history.json`, the
adapter weights and the tokenizer. Load the base model named above, then
apply the adapter (`PeftModel.from_pretrained`). `resource_tuner.tune`
serves it; `resource_tuner.policy.prompts` holds the prompt it was
trained against — a different prompt is a different policy."""

    trajectory = kv(
        {
            "mean reward first → last": (
                f"{_num(fm.get('mean_reward_first'), '.3f')} → "
                f"{_num(fm.get('mean_reward_last'), '.3f')}"
                if fm
                else None
            ),
            "logged steps": fm.get("logged_steps"),
        }
    )

    shape = manifest.get("reward_shape")
    shape_block = (
        "## Reward shape\n\n"
        + table(
            ["component", "weight"],
            [[k, _cell(v)] for k, v in shape.items()],
        )
        if isinstance(shape, dict) and shape
        else ""
    )

    caveat = (
        "> **Intermediate checkpoint — not for promotion.** Published "
        f"mid-training at step {step:,} so a crashed run resumes from "
        "warm weights and so the reward trajectory is inspectable while "
        "training runs. It is deliberately published under a separate "
        "artifact name so the eval-on-new-checkpoint trigger stays quiet; "
        "the final `tuner-checkpoint` is the one that gets evaluated."
        if intermediate
        else "> **Unevaluated at publish time.** Publishing this artifact "
        "FIRES the eval trigger; the verdict lands in a `tuner-eval-report` "
        "that links back to this checkpoint's path. Do not read training "
        "reward as a quality gate — it is not comparable across reward "
        "stages."
    )

    return _join(
        "# tuner-checkpoint" + ("-intermediate" if intermediate else ""),
        f"Resource-proposal policy: {adapter} over "
        f"`{manifest.get('base_model', 'unknown base')}`, trained with GRPO.",
        caveat,
        "## At a glance",
        kv(glance),
        ("## Hypothesis\n\n" + manifest["hypothesis_description"])
        if manifest.get("hypothesis_description")
        else "",
        usage,
        "## Training trajectory" if trajectory else "",
        trajectory,
        shape_block,
    )


def checkpoint_attrs(
    manifest: Mapping[str, Any], *, step: int | None = None
) -> dict[str, str]:
    """Flat facets for filtering. The framework/model_type/architecture/task
    keys mirror `Metadata.create_model_metadata` so a checkpoint is
    identifiable as a model without opening its card."""
    fm = manifest.get("final_metrics") or {}
    return _attrs(
        {
            "framework": "peft" if manifest.get("use_lora", True) else "transformers",
            "model_type": "causal-lm",
            "architecture": manifest.get("base_model"),
            "task": "resource-proposal",
            "modality": "text",
            "serial_format": "safetensors",
            "profile": manifest.get("profile"),
            "reward_stage": manifest.get("reward_stage"),
            "steps": step if step is not None else manifest.get("max_steps"),
            "use_lora": manifest.get("use_lora"),
            "gbt_hint": manifest.get("gbt_hint"),
            "corpus_path": manifest.get("corpus_path"),
            "mean_reward_last": _num(fm.get("mean_reward_last"), ".4f", dash=""),
            "intermediate": "true" if step is not None else "",
        }
    )


# ── eval report card ────────────────────────────────────────────────────


def eval_report_card(report: Mapping[str, Any]) -> str:
    """Card for `tuner-eval-report`: the verdict, not just the numbers."""
    gate = bool(report.get("auto_gate_passed"))
    metrics = table(
        ["metric", "policy", "rule baseline", "ML baseline"],
        [
            [
                "fit rate (task fits the proposal)",
                _pct(report.get("success_rate")),
                _pct(report.get("baseline_success_rate")),
                _pct(report.get("ml_baseline_success_rate")),
            ],
            [
                "median overprovision (waste)",
                _pct(report.get("median_overprovision_pct"), 0.01),
                _pct(report.get("baseline_median_overprovision_pct"), 0.01),
                _pct(report.get("ml_baseline_median_overprovision_pct"), 0.01),
            ],
            [
                "$ / task-hr",
                _usd(report.get("policy_cost_per_task_hr")),
                _usd(report.get("baseline_cost_per_task_hr")),
                _usd(report.get("ml_baseline_cost_per_task_hr")),
            ],
        ],
    )
    return _join(
        "# tuner-eval-report",
        f"Held-out evaluation of one `tuner-checkpoint` against the "
        f"rule-based and quantile-GBT baselines. **Gate: "
        f"{'PASS' if gate else 'FAIL'}**.",
        "## At a glance",
        kv(
            {
                "base model": report.get("base_model"),
                "reward stage": report.get("reward_stage"),
                "held-out contexts": _num(report.get("n_contexts")),
                "schema validity": _pct(report.get("schema_validity")),
                "$ saved / 1k task-hrs": _usd_signed(
                    report.get("dollars_saved_per_1k_task_hrs"), dash=""
                ),
                "real cluster episodes": len(report.get("cluster_episodes") or []),
                "checkpoint scored": report.get("checkpoint_path"),
                "corpus behind it": report.get("train_corpus_path"),
            }
        ),
        ("## Hypothesis\n\n" + report["hypothesis_description"])
        if report.get("hypothesis_description")
        else "",
        "## Metrics",
        metrics,
        """\
## GPU estimation""",
        kv(
            {
                "GPU contexts": report.get("gpu_contexts"),
                "GPU fit rate": _pct(report.get("gpu_success_rate")),
                "GPU missed (needed, not proposed)": report.get("gpu_missing_count"),
                "GPU spurious (proposed, not needed)": report.get("gpu_spurious_count"),
            }
        ),
        """\
## Reading this

`success_rate` is fit on SIMULATED episodes over the held-out split;
`cluster_episodes` are the subset re-run on real pods, and they are what
sim-to-real claims rest on. The gate is an automatic check, not a
promotion decision — it compares the policy against the rule baseline on
fit AND waste, and a pass on one with a regression on the other still
reads FAIL. The full payload (per-family breakdown, invalid completions,
per-episode detail) is in the JSON this card describes.""",
    )


def eval_report_attrs(report: Mapping[str, Any]) -> dict[str, str]:
    return _attrs(
        {
            "gate": "PASS" if report.get("auto_gate_passed") else "FAIL",
            "base_model": report.get("base_model"),
            "reward_stage": report.get("reward_stage"),
            "n_contexts": report.get("n_contexts"),
            "success_rate": _num(report.get("success_rate"), ".4f", dash=""),
            "median_overprovision_pct": _num(
                report.get("median_overprovision_pct"), ".1f", dash=""
            ),
            "dollars_saved_per_1k_task_hrs": _num(
                report.get("dollars_saved_per_1k_task_hrs"), ".2f", dash=""
            ),
            "checkpoint_path": report.get("checkpoint_path"),
            "cluster_episodes": len(report.get("cluster_episodes") or []) or None,
        }
    )


# ── A/B report card ─────────────────────────────────────────────────────


def ab_report_card(report: Mapping[str, Any]) -> str:
    """Card for `tuning-ab-report`: tuned proposals vs a hard-coded prior."""
    rows = [
        [
            "OOM rate",
            _pct(report.get("prior_oom_rate")),
            _pct(report.get("tuned_oom_rate")),
        ],
        [
            "fit rate",
            _pct(report.get("prior_fit_rate")),
            _pct(report.get("tuned_fit_rate")),
        ],
        [
            "median overprovision",
            _pct(report.get("prior_median_overprovision_pct"), 0.01),
            _pct(report.get("tuned_median_overprovision_pct"), 0.01),
        ],
    ]
    return _join(
        "# tuning-ab-report",
        "Head-to-head on REAL pods: the tuned policy's proposals against a "
        "hard-coded resource prior, same tasks, same cluster.",
        "## At a glance",
        kv(
            {
                "tasks": report.get("n_tasks"),
                "episodes recorded": len(report.get("episodes") or []),
                "hard-coded prior": json.dumps(report.get("prior"), sort_keys=True)
                if report.get("prior")
                else None,
            }
        ),
        "## Result",
        table(["metric", "hard-coded prior", "tuned"], rows),
        """\
## Reading this

Every number here comes from a pod that ran, not a simulation: `oom_rate`
counts episodes the container was killed in, and `median_overprovision`
is measured against real peak RSS. This is the PRD's auditable savings
record at prototype size — small `n_tasks`, so read the direction, not
the third decimal place. Per-episode rows (requested vs peak, per arm)
are in the JSON.""",
    )


def ab_report_attrs(report: Mapping[str, Any]) -> dict[str, str]:
    return _attrs(
        {
            "n_tasks": report.get("n_tasks"),
            "prior_oom_rate": _num(report.get("prior_oom_rate"), ".3f", dash=""),
            "tuned_oom_rate": _num(report.get("tuned_oom_rate"), ".3f", dash=""),
            "prior_fit_rate": _num(report.get("prior_fit_rate"), ".3f", dash=""),
            "tuned_fit_rate": _num(report.get("tuned_fit_rate"), ".3f", dash=""),
        }
    )


# ── ML baseline card ────────────────────────────────────────────────────


def ml_baseline_card(manifest: Mapping[str, Any]) -> str:
    """Model card for `ml-baseline-model` (the quantile-GBT estimator)."""
    sim = manifest.get("sim_heldout") or {}
    ml, rule = sim.get("ml") or {}, sim.get("rule_baseline") or {}
    quant = manifest.get("quantiles") or {}
    return _join(
        "# ml-baseline-model",
        "The classical-ML floor the LLM policy has to beat: gradient-boosted "
        "quantile regressors that predict a task's footprint from features "
        "of its code and inputs.",
        "## At a glance",
        kv(
            {
                "estimator": manifest.get("estimator"),
                "target quantiles": ", ".join(f"{k} {v}" for k, v in quant.items()),
                "training rows": _num(manifest.get("n_train")),
                "features": manifest.get("n_features"),
                "trained on corpus": manifest.get("corpus"),
            }
        ),
        "## Held-out scoring (simulated)",
        table(
            ["metric", "this model", "rule baseline"],
            [
                ["fit", _pct(ml.get("fit")), _pct(rule.get("fit"))],
                [
                    "$ / task-hr",
                    _usd(ml.get("cost_per_task_hr")),
                    _usd(rule.get("cost_per_task_hr")),
                ],
                [
                    "median memory waste",
                    _pct(ml.get("median_mem_waste_pct"), 0.01),
                    _pct(rule.get("median_mem_waste_pct"), 0.01),
                ],
            ],
        ),
        """\
## How to load

`ml_baseline.joblib` + `manifest.json`. `resource_tuner.training.ml_baseline.MLBaseline.load()`
restores it; the tune service serves it as a fallback and as the A/B arm
against the LLM checkpoint. Predicting a high quantile rather than the
mean is deliberate — the cost of underestimating is an OOM, the cost of
overestimating is a few wasted MiB.""",
    )


def ml_baseline_attrs(manifest: Mapping[str, Any]) -> dict[str, str]:
    sim = (manifest.get("sim_heldout") or {}).get("ml") or {}
    return _attrs(
        {
            "framework": "scikit-learn",
            "model_type": "quantile-gbt",
            "architecture": "gradient-boosted regression trees",
            "task": "resource-regression",
            "modality": "tabular",
            "serial_format": "joblib",
            "estimator": manifest.get("estimator"),
            "n_train": manifest.get("n_train"),
            "n_features": manifest.get("n_features"),
            "heldout_fit": _num(sim.get("fit"), ".3f", dash=""),
            "cost_per_task_hr": _num(sim.get("cost_per_task_hr"), ".4f", dash=""),
            "corpus": manifest.get("corpus"),
        }
    )


# ── upload ──────────────────────────────────────────────────────────────


async def upload(content: str, card_type: str = "generic"):
    """Upload a rendered card and return the `Card` to attach to Metadata.

    Returns None on failure: a card is documentation ABOUT an artifact,
    and losing the documentation is never a reason to lose the artifact —
    the publish that carries it is usually the tail end of a multi-hour
    training or corpus run.
    """
    import flyte.artifacts as artifacts

    try:
        return await artifacts.Card.create_from.aio(
            content=content, format="md", card_type=card_type
        )
    except Exception as e:  # noqa: BLE001
        print(f"[card] upload failed ({type(e).__name__}: {e}) — publishing without it")
        return None
