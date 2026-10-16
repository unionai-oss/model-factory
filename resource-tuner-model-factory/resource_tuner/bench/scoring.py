"""Score a set of decisions: what would it cost to get every run to SUCCEEDED?

The unit is **dollars per 1,000 task-hours**, the same unit rounds 6-16
reported, so a number here can be read next to theirs. Every run is
treated as one nominal task-hour of work (bench workloads are scaled-down
kernels of real tasks; their absolute seconds are not the point), and the
request decides the price of that hour:

- **memory fits** (limit >= what the run needs): pay the request's $/hr for
  the run, stretched if CPU was under-asked (CPU is compressible -- the
  run is slower, not dead);
- **memory does not fit**: the pod is OOM-killed after burning the part of
  the run that came before the peak (``t_peak_frac``, measured), and the
  platform retries at double the memory -- the standard escalation -- until
  it fits. Every attempt is paid for.

Against that, the **oracle** asks for the cheapest cell that fits, which no
arm can beat; ``regret`` = cost / oracle cost - 1 is what an arm leaves on
the table. Static and rule baselines bracket the useful range.

Pure numpy; the decision-theoretic helper ``bayes_mem_index`` is shared by
every arm that produces a distribution, so they are all decoded the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import grid

N_MEM = len(grid.MEMORY_GRID_MIB)
N_CPU = len(grid.CPU_GRID)
#: Fixed cost of an attempt beyond its run time: scheduling, image pull,
#: the task process starting. As a fraction of the nominal task-hour.
ATTEMPT_OVERHEAD = 0.02
#: Attempts after which a run is declared failed (escalation reached the top).
MAX_ATTEMPTS = 8
#: What an OOM costs beyond the memory it burned, in multiples of the run's
#: oracle GiB-hours: the delay (a late result, a blocked downstream task, a
#: page). 1.0 = "an OOM costs one extra run". Without it, the cheapest
#: policy under a pure reserved-memory objective tolerates ~37% OOMs (the
#: size rule did, tuned on v1-partial), which no operator would accept.
#: The risk frontier sweeps it from 0 to 16.
OOM_PENALTY = 1.0


@dataclass
class RunOutcome:
    cost: float  # $ for this run, all attempts, nominal 1 task-hour of work
    attempts: int
    gib_hours: float  # memory reserved, all attempts -- the price-free waste number
    oom_first: bool
    failed: bool
    hours: float  # wall-clock in nominal hours, all attempts
    over_provision: float  # first-fitting attempt's limit / need - 1


def simulate_run(mem_idx: int, cpu_idx: int, required_mib: float, avg_cores: float,
                 t_peak_frac: float, pricing: str = "packed") -> RunOutcome:
    cpu = grid.CPU_GRID[cpu_idx]
    stretch = max(1.0, max(avg_cores, 0.1) / cpu)
    cost, hours, attempts, gib_h = 0.0, 0.0, 0, 0.0
    m = mem_idx
    oom_first = False
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        mem = grid.MEMORY_GRID_MIB[m]
        rate = grid.dollars_per_hr(mem, cpu, pricing)
        if mem >= required_mib:
            run_h = stretch + ATTEMPT_OVERHEAD
            return RunOutcome(cost + rate * run_h, attempts, gib_h + mem / 1024 * run_h, oom_first, False,
                              hours + run_h, mem / required_mib - 1.0)
        if attempts == 1:
            oom_first = True
        burned = stretch * max(t_peak_frac, 0.05) + ATTEMPT_OVERHEAD
        cost += rate * burned
        hours += burned
        gib_h += mem / 1024 * burned
        if m == N_MEM - 1:
            break
        m = grid.mem_index_covering(mem * 2)
    return RunOutcome(cost, attempts, gib_h, oom_first, True, hours, float("nan"))


def oracle(row: dict) -> tuple[int, int]:
    return int(row["label_mem_idx"]), int(row["label_cpu_idx"])


# ── decision theory for distributional arms ─────────────────────────────


def cost_matrix(t_peak_frac: float = 0.5, cpu_idx: int = 1, avg_cores: float = 1.0,
                oom_penalty: float = OOM_PENALTY) -> np.ndarray:
    """C[m, j]: memory GiB-hours (the headline objective) spent getting a run
    to SUCCEEDED when it requests cell m and needs cell j.

    The need is represented by its cell's interior -- the label is the
    cheapest cell that fits, so `needs cell j` means `fits in j, not j-1`.
    `oom_penalty` adds, per OOM, that many multiples of the run's oracle
    cost: the price of the delay (an SLA, a blocked downstream team) on
    top of the memory an OOM burns (default: the headline's OOM_PENALTY).
    """
    C = np.zeros((N_MEM, N_MEM))
    for j in range(N_MEM):
        need = grid.MEMORY_GRID_MIB[j] * (0.85 if j else 0.5)  # interior of cell j
        base = simulate_run(j, cpu_idx, need, avg_cores, t_peak_frac).gib_hours
        for m in range(N_MEM):
            o = simulate_run(m, cpu_idx, need, avg_cores, t_peak_frac)
            C[m, j] = o.gib_hours + oom_penalty * base * (o.attempts - 1)
    return C


_DEFAULT_C: np.ndarray | None = None


def default_cost_matrix() -> np.ndarray:
    global _DEFAULT_C
    if _DEFAULT_C is None:
        _DEFAULT_C = cost_matrix()
    return _DEFAULT_C


def bayes_mem_index(probs, C: np.ndarray | None = None) -> int:
    """The memory cell minimizing expected cost under a predicted
    distribution over the needed cell -- one matrix-vector product."""
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    C = default_cost_matrix() if C is None else C
    return int(np.argmin(C @ p))


def p_oom(probs, mem_idx: int) -> float:
    """P(the run needs more than cell mem_idx) under the predicted distribution."""
    p = np.asarray(probs, dtype=float)
    return float(p[mem_idx + 1 :].sum() / p.sum())


# ── scoring a prediction table ──────────────────────────────────────────


def score(rows: list[dict], preds: dict[str, dict]) -> dict:
    """Aggregate metrics for one arm over `rows` (which carry labels).

    `preds[run_id]` = {"mem_idx", "cpu_idx", optional "mem_probs"}.

    **Headline: `mem_regret`** -- memory GiB-hours reserved to get every run
    to SUCCEEDED (OOM-killed attempts included, plus OOM_PENALTY oracle-runs
    of delay per OOM), over what the oracle reserves, minus 1. Price-free, and it is the decision the OOM cliff
    sits on. CPU is held at the oracle's cell for every arm here so the
    headline isolates memory (nearly every bench workload wants ~1 core; a
    CPU coin-flip would otherwise dominate any dollar figure). Each arm's
    own CPU calls are scored separately (`cpu_*`), and dollars are reported
    under both price models as secondary columns.
    """
    out: dict = {"n": 0}
    acc = {k: [] for k in ("gibh", "raw_gibh", "ogibh", "cost", "ocost", "fcost", "focost", "own_cost",
                           "oom", "fail", "att", "hours", "ohours", "cpu_exact", "cpu_under")}
    overs, briers, nlls, p_ooms, real_ooms = [], [], [], [], []
    missing = 0
    for r in rows:
        p = preds.get(r["run_id"])
        if p is None:
            missing += 1
            continue
        om, oc = oracle(r)
        args = (r["required_mib"], r["avg_cores"], r["t_peak_frac"])
        m = int(p["mem_idx"])
        o = simulate_run(m, oc, *args)
        orc = simulate_run(om, oc, *args)
        acc["gibh"].append(o.gib_hours + OOM_PENALTY * orc.gib_hours * (o.attempts - 1))
        acc["ogibh"].append(orc.gib_hours)
        acc["raw_gibh"].append(o.gib_hours)
        acc["cost"].append(o.cost), acc["ocost"].append(orc.cost)
        acc["fcost"].append(simulate_run(m, oc, *args, "fargate").cost)
        acc["focost"].append(simulate_run(om, oc, *args, "fargate").cost)
        acc["own_cost"].append(simulate_run(m, int(p["cpu_idx"]), *args).cost)
        acc["oom"].append(o.oom_first), acc["fail"].append(o.failed), acc["att"].append(o.attempts)
        acc["hours"].append(o.hours), acc["ohours"].append(orc.hours)
        acc["cpu_exact"].append(int(p["cpu_idx"]) == oc), acc["cpu_under"].append(int(p["cpu_idx"]) < oc)
        if not o.oom_first:
            overs.append(o.over_provision)
        probs = p.get("mem_probs")
        if probs is not None:
            pr = np.asarray(probs, dtype=float)
            pr = pr / pr.sum()
            onehot = np.zeros(N_MEM)
            onehot[om] = 1
            briers.append(float(((pr - onehot) ** 2).sum()))
            nlls.append(float(-math.log(max(pr[om], 1e-9))))
            p_ooms.append(p_oom(pr, m))
            real_ooms.append(o.oom_first)
    n = len(acc["gibh"])
    if not n:
        return {"n": 0, "missing": missing}
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    out.update({
        "n": n,
        "missing": missing,
        "mem_regret": float(np.sum(acc["gibh"]) / np.sum(acc["ogibh"]) - 1.0),
        "gib_hours_per_run": mean["gibh"],
        # the same without the OOM delay penalty: memory actually reserved
        "reserved_mem_regret": float(np.sum(acc["raw_gibh"]) / np.sum(acc["ogibh"]) - 1.0),
        "oom_rate": mean["oom"],
        "fail_rate": mean["fail"],
        "mean_attempts": mean["att"],
        "latency_vs_oracle": mean["hours"] / mean["ohours"],
        "median_over_provision": float(np.median(overs)) if overs else float("nan"),
        "p90_over_provision": float(np.quantile(overs, 0.9)) if overs else float("nan"),
        "cost_per_1k_task_hr": 1000 * mean["cost"],
        "oracle_cost_per_1k_task_hr": 1000 * mean["ocost"],
        "regret": mean["cost"] / mean["ocost"] - 1.0,
        "fargate_cost_per_1k_task_hr": 1000 * mean["fcost"],
        "fargate_regret": mean["fcost"] / mean["focost"] - 1.0,
        "own_cpu_regret": mean["own_cost"] / mean["ocost"] - 1.0,
        "cpu_exact": mean["cpu_exact"],
        "cpu_under": mean["cpu_under"],
    })
    if briers:
        out.update({
            "brier": float(np.mean(briers)),
            "nll": float(np.mean(nlls)),
            "ece_oom": expected_calibration_error(p_ooms, real_ooms),
            "mean_p_oom": float(np.mean(p_ooms)),
        })
    return out


def expected_calibration_error(p: list[float], y: list[bool], bins: int = 10) -> float:
    """ECE of the predicted P(OOM) at the chosen cell against what happened."""
    p_arr, y_arr = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    if not len(p_arr):
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_arr >= lo) & (p_arr < hi if hi < 1 else p_arr <= hi)
        if m.any():
            ece += m.mean() * abs(p_arr[m].mean() - y_arr[m].mean())
    return float(ece)


#: OOM penalties swept for the risk frontier (multiples of a run's oracle cost).
FRONTIER_PENALTIES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
#: Margins (memory cells added) swept for point-estimate arms.
FRONTIER_SHIFTS = (-1, 0, 1, 2, 3, 4)


def frontier(rows: list[dict], preds: dict[str, dict]) -> list[dict]:
    """OOM rate vs memory regret as the arm's risk knob moves.

    A distributional arm re-decodes the SAME predictions under a range of
    OOM penalties -- no retraining, one matrix product per decision. A
    point-estimate arm can only add or remove margin.
    """
    distributional = all(p.get("mem_probs") is not None for p in preds.values())
    points = []
    if distributional:
        for lam in FRONTIER_PENALTIES:
            C = cost_matrix(oom_penalty=lam)
            dec = {rid: {**p, "mem_idx": bayes_mem_index(p["mem_probs"], C)} for rid, p in preds.items()}
            s = score(rows, dec)
            points.append({"knob": f"penalty={lam:g}", "oom_rate": s["oom_rate"],
                           "mem_regret": s["mem_regret"], "regret": s["regret"]})
    else:
        for k in FRONTIER_SHIFTS:
            dec = {rid: {**p, "mem_idx": int(min(N_MEM - 1, max(0, p["mem_idx"] + k)))} for rid, p in preds.items()}
            s = score(rows, dec)
            points.append({"knob": f"shift={k:+d}", "oom_rate": s["oom_rate"],
                           "mem_regret": s["mem_regret"], "regret": s["regret"]})
    return points


def score_slices(rows: list[dict], preds: dict[str, dict]) -> dict[str, dict]:
    """The headline split plus the slices that say *why* an arm wins."""
    slices: dict[str, list[dict]] = {}
    for r in rows:
        keys = [
            f"split={r['split']}",
            f"driver={r['memory_driver']}",
            f"history={'cold' if r['n_history'] == 0 else 'warm'}",
            f"family={r['family']}",
        ]
        if r["split"] in ("test", "ood"):
            keys.append(f"{r['split']}/history={'cold' if r['n_history'] == 0 else 'warm'}")
        for k in keys:
            slices.setdefault(k, []).append(r)
    return {k: score(v, preds) for k, v in sorted(slices.items())}
