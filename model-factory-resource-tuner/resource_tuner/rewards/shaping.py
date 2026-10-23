"""Composable reward shaping — stage C of the curriculum.

Stage B's linear waste penalty plateaued: it saturates near 1.0 for very
padded requests (4Gi vs 1Gi for a 200MiB task score almost identically),
so the policy settled on a uniform safe answer. This module implements the
full menu of shaping approaches behind ONE config, `RewardShape`:

Mutually exclusive (one per experiment — the reward-shaping judgment):
- `waste_form`: how per-axis overprovisioning is scored.
    linear     (requested−peak)/requested — stage B's shape, the control
    quadratic  frac² — punishes gross padding hardest
    sqrt       √frac — strongest gradient near right-sized
    log_ratio  log2(requested/target)/CAP — scale-invariant: 2× vs 40×
               padding stay distinguishable where linear saturates
    bucket     grid-steps above the minimal fitting bucket — dense,
               integer-interpretable distance-to-oracle

Composable (any subset, on top of any form):
- `headroom`: (lo, hi) target band of requested/peak. Waste is measured
  from the band's top instead of from peak — "right-sized WITH margin" is
  the optimum, not the knife edge — and fitting below `lo` costs a small
  knife-edge penalty.
- `cost_weighted`: aggregate the cpu/mem/gpu axis penalties by their $
  share of the request (pricing.py) instead of a plain mean — the reward
  literally optimizes dollars.
- `baseline_relative`: potential-based bonus for costing less than the
  rule-based baseline's proposal on the same context, clipped to [-1, 1].
- `w_waste_final`: anneal the waste weight from `w_waste` to this value
  over training (success saturates early; waste is the remaining signal).
- `group_tiebreak`: bonus split among the cheapest SUCCESSFUL completions
  of each GRPO group — a zero-variance group of identical safe answers
  yields no gradient, so daring to go smaller and surviving must pay.
- `robustness_samples`: score against k jittered simulator draws and
  average — knife-edge proposals lose across the jitter band.

The success/OOM asymmetry and "waste only counts on successful episodes"
invariants from rewards.py hold throughout.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, fields, replace

from .. import pricing
from ..environment.simulator import EpisodeResult
from ..policy.actions import CPU_GRID, GPU_TYPES, GPU_VRAM_MIB, MEMORY_GRID_MIB
from .rewards import FORMAT_REWARD, SUCCESS_REWARD

LOG2_CAP = 6.0  # log_ratio saturates at 64× overprovisioned


@dataclass(frozen=True)
class RewardShape:
    """One reward configuration = one experiment arm."""

    name: str
    waste_form: str = "linear"  # linear|quadratic|sqrt|log_ratio|bucket
    headroom: tuple[float, float] | None = None  # (lo, hi) requested/peak band
    cost_weighted: bool = False
    baseline_relative: bool = False
    w_success: float = 1.0
    w_waste: float = 0.5
    w_waste_final: float | None = None  # anneal target; None = constant
    w_oom: float = 0.5
    w_gpu_missing: float = 0.5  # needed a GPU, proposed none
    w_throttle: float = 0.05
    w_knife_edge: float = 0.1  # fits, but under the headroom band
    w_savings: float = 0.3  # baseline_relative bonus weight
    group_tiebreak: float = 0.0
    robustness_samples: int = 1

    def waste_weight(self, step_frac: float) -> float:
        if self.w_waste_final is None:
            return self.w_waste
        return self.w_waste + (self.w_waste_final - self.w_waste) * min(
            max(step_frac, 0.0), 1.0
        )


@dataclass(frozen=True)
class ShapedBreakdown:
    """Per-component scores for one (possibly robustness-averaged) episode."""

    total: float
    components: dict = field(default_factory=dict)


# ── per-axis waste forms ────────────────────────────────────────────────


def _grid_index(grid: tuple, value: float) -> int:
    """Index of the smallest grid step covering `value` (clamped to max)."""
    for i, step in enumerate(grid):
        if value <= step:
            return i
    return len(grid) - 1


def _gpu_cost_rank(gpu_type: str | None, count: int) -> float:
    """GPU 'grid position' for the bucket form: 0 = no GPU, then types by
    $/hr; extra count adds whole ranks."""
    if not count or not gpu_type:
        return 0.0
    order = sorted(GPU_TYPES, key=pricing.GPU_DOLLARS_PER_HR.__getitem__)
    return 1.0 + order.index(gpu_type) + (count - 1) * len(order)


def axis_waste(form: str, requested: float, target: float, grid: tuple | None) -> float:
    """One axis's waste penalty in [0, 1]. `target` is what the request
    SHOULD be (peak, or the headroom band top); `requested >= target` is
    the waste direction, below-target is someone else's problem (OOM /
    knife-edge handle it)."""
    if requested <= 0 or requested <= target:
        return 0.0
    frac = min(max((requested - target) / requested, 0.0), 1.0)
    if form == "linear":
        return frac
    if form == "quadratic":
        return frac * frac
    if form == "sqrt":
        return math.sqrt(frac)
    if form == "log_ratio":
        return min(math.log2(requested / max(target, 1e-9)), LOG2_CAP) / LOG2_CAP
    if form == "bucket":
        if grid is None:
            return frac  # axes without a grid fall back to linear
        span = len(grid) - 1
        dist = _grid_index(grid, requested) - _grid_index(grid, target)
        return min(max(dist, 0), span) / span if span else 0.0
    raise ValueError(f"unknown waste_form {form!r}")


def _episode_waste(shape: RewardShape, ep: EpisodeResult) -> float:
    """Aggregate cpu/mem/gpu waste for one SUCCESSFUL episode, in [0, 1]."""
    band_hi = shape.headroom[1] if shape.headroom else 1.0
    mem_target = ep.peak_memory_mib * band_hi
    cpu_target = ep.peak_cpu * band_hi

    axes: list[tuple[str, float]] = [
        ("memory", axis_waste(shape.waste_form, ep.requested_memory_mib, mem_target, MEMORY_GRID_MIB)),
        ("cpu", axis_waste(shape.waste_form, ep.requested_cpu, cpu_target, CPU_GRID)),
    ]
    gpu_requested = bool(ep.requested_gpu)
    gpu_needed = ep.true_gpu_mem_mib > 0
    if gpu_requested or gpu_needed:
        if gpu_requested and not gpu_needed:
            gpu_waste = 1.0  # a whole idle accelerator
        elif gpu_requested:
            vram = GPU_VRAM_MIB.get(ep.requested_gpu_type or "T4", 0) * ep.requested_gpu
            if shape.waste_form == "bucket":
                need_rank = _gpu_cost_rank(
                    pricing.cheapest_gpu_for(ep.true_gpu_mem_mib) or ep.requested_gpu_type, 1
                )
                have_rank = _gpu_cost_rank(ep.requested_gpu_type, ep.requested_gpu)
                span = float(len(GPU_TYPES))
                gpu_waste = min(max(have_rank - need_rank, 0.0), span) / span
            else:
                gpu_waste = axis_waste(
                    shape.waste_form, vram, ep.true_gpu_mem_mib * band_hi, None
                )
        else:  # needed but missing — a failure, not waste (w_gpu_missing)
            gpu_waste = 0.0
        axes.append(("gpu", gpu_waste))

    if not shape.cost_weighted:
        return sum(w for _, w in axes) / len(axes)

    # $-share aggregation: each axis's penalty weighted by what that axis
    # of the REQUEST costs — wasting a GPU-hour outweighs wasting a core.
    rates = {
        "cpu": ep.requested_cpu * pricing.CPU_DOLLARS_PER_CORE_HR,
        "memory": (ep.requested_memory_mib / 1024) * pricing.MEM_DOLLARS_PER_GIB_HR,
    }
    if any(name == "gpu" for name, _ in axes):
        rates["gpu"] = ep.requested_gpu * pricing.GPU_DOLLARS_PER_HR.get(
            ep.requested_gpu_type or "T4", 0.0
        )
        # a needed-but-unrequested GPU still deserves weight in the mean
        rates["gpu"] = rates["gpu"] or pricing.GPU_DOLLARS_PER_HR["T4"]
    total_rate = sum(rates.get(name, 0.0) for name, _ in axes) or 1.0
    return sum(w * rates.get(name, 0.0) for name, w in axes) / total_rate


def episode_dollars_per_hr(ep: EpisodeResult) -> float:
    """$/hr of the episode's REQUEST — the thing the customer pays for."""
    return pricing.dollars_per_hr(
        ep.requested_cpu,
        ep.requested_memory_mib,
        ep.requested_gpu_type,
        ep.requested_gpu,
    )


# ── scoring ─────────────────────────────────────────────────────────────


def score_shaped(
    shape: RewardShape,
    episodes: list[EpisodeResult],
    step_frac: float = 1.0,
    baseline_cost_per_hr: float | None = None,
) -> ShapedBreakdown:
    """Score one proposal against its (robustness-sampled) episodes.

    All episodes share the same proposal; averaging their component scores
    is the robustness-averaged reward.
    """
    if not episodes:
        return ShapedBreakdown(total=0.0, components={"invalid": 1.0})
    w_waste = shape.waste_weight(step_frac)
    comp_sums: dict[str, float] = {
        "format": 0.0, "success": 0.0, "waste": 0.0, "oom": 0.0,
        "gpu_missing": 0.0, "throttle": 0.0, "knife_edge": 0.0, "savings": 0.0,
    }
    for ep in episodes:
        comp_sums["format"] += FORMAT_REWARD
        if ep.gpu_missing:
            comp_sums["gpu_missing"] -= shape.w_gpu_missing
        elif ep.oom:
            comp_sums["oom"] -= shape.w_oom
        if not ep.ok:
            continue
        comp_sums["success"] += shape.w_success * SUCCESS_REWARD
        comp_sums["waste"] -= w_waste * _episode_waste(shape, ep)
        if ep.throttled:
            comp_sums["throttle"] -= shape.w_throttle
        if shape.headroom is not None:
            lo = shape.headroom[0]
            if ep.peak_memory_mib > 0 and ep.requested_memory_mib < ep.peak_memory_mib * lo:
                comp_sums["knife_edge"] -= shape.w_knife_edge
        if shape.baseline_relative and baseline_cost_per_hr:
            saved = (baseline_cost_per_hr - episode_dollars_per_hr(ep)) / baseline_cost_per_hr
            comp_sums["savings"] += shape.w_savings * min(max(saved, -1.0), 1.0)
    n = len(episodes)
    components = {k: v / n for k, v in comp_sums.items()}
    return ShapedBreakdown(total=sum(components.values()), components=components)


def apply_group_tiebreak(
    shape: RewardShape,
    totals: list[float],
    episodes_ok: list[bool],
    costs_per_hr: list[float | None],
    group_size: int,
) -> list[float]:
    """Within each GRPO group, split a bonus among the cheapest successful
    completions. No-op unless the shape asks for it and the batch divides
    into whole groups."""
    if not shape.group_tiebreak or group_size <= 1 or len(totals) % group_size:
        return totals
    out = list(totals)
    for start in range(0, len(totals), group_size):
        seg = range(start, start + group_size)
        priced = [
            (costs_per_hr[i], i)
            for i in seg
            if episodes_ok[i] and costs_per_hr[i] is not None
        ]
        if not priced:
            continue
        best = min(c for c, _ in priced)
        winners = [i for c, i in priced if c <= best * 1.0001]
        for i in winners:
            out[i] += shape.group_tiebreak / len(winners)
    return out


# ── the named judgment calls (experiment arms) ──────────────────────────

SHAPES: dict[str, RewardShape] = {
    s.name: s
    for s in (
        # Control: stage B's exact economics through the new machinery,
        # plus GPU-awareness and $-share aggregation.
        RewardShape(name="c-linear", waste_form="linear", cost_weighted=True),
        RewardShape(name="c-quadratic", waste_form="quadratic", cost_weighted=True),
        RewardShape(name="c-sqrt", waste_form="sqrt", cost_weighted=True),
        # The plateau-breaker: scale-invariant waste + annealed weight.
        RewardShape(
            name="c-log",
            waste_form="log_ratio",
            cost_weighted=True,
            w_waste=0.4,
            w_waste_final=0.9,
        ),
        # The recommended bundle: distance-to-oracle waste, headroom band,
        # $-weighting, cheapest-survivor tie-break, robustness averaging,
        # annealed pressure.
        RewardShape(
            name="c-bucket",
            waste_form="bucket",
            headroom=(1.1, 1.4),
            cost_weighted=True,
            w_waste=0.5,
            w_waste_final=1.0,
            group_tiebreak=0.15,
            robustness_samples=3,
        ),
        # Business-metric-native: reward IS dollars saved vs the baseline.
        RewardShape(
            name="c-cost",
            waste_form="linear",
            cost_weighted=True,
            baseline_relative=True,
            group_tiebreak=0.15,
        ),
    )
}


def get_shape(name: str) -> RewardShape:
    """Resolve a shape by name; RT_REWARD_SHAPE (a JSON dict of RewardShape
    field overrides) lets one experiment tweak knobs without a code change:
        RT_REWARD_SHAPE='{"w_waste_final": 1.5}' → applied on top.
    """
    try:
        shape = SHAPES[name]
    except KeyError:
        raise ValueError(f"unknown reward shape {name!r}; choose from {sorted(SHAPES)}")
    override = os.environ.get("RT_REWARD_SHAPE")
    if override:
        raw = json.loads(override)
        valid = {f.name for f in fields(RewardShape)}
        unknown = set(raw) - valid
        if unknown:
            raise ValueError(f"RT_REWARD_SHAPE has unknown fields {sorted(unknown)}")
        if "headroom" in raw and raw["headroom"] is not None:
            raw["headroom"] = tuple(raw["headroom"])
        shape = replace(shape, **raw)
    return shape


def is_shaped_stage(stage: str) -> bool:
    return stage in SHAPES
