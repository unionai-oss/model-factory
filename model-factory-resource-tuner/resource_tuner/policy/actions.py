"""The action space: bucketed `flyte.Resources` kwargs.

The PRD (§8, label derivation) buckets each metric so the policy chooses
from a small grid instead of free-form numbers: memory on a log-scale grid,
CPU on fixed increments, GPUs as small integers. Bucketing does three jobs
at once — it shrinks the action space for RL, makes proposals directly
comparable across runs, and turns the safety margin into part of the action
rather than a post-hoc patch.

Everything here is pure logic (no flyte import) so it is unit-testable and
usable inside both the trainer's reward function and the episode runner.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── grids ───────────────────────────────────────────────────────────────
# Memory: powers-of-two style log grid, 128Mi → 64Gi (PRD's proposed range).
MEMORY_GRID_MIB: tuple[int, ...] = (
    128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
)
# CPU: fixed increments. Fractional CPUs below 1 are a single "500m" step —
# the demo tenant's smallest schedulable request that still gets a task
# through admission quickly.
CPU_GRID: tuple[float, ...] = (0.5, 1, 2, 4, 6, 8, 12, 16)
# GPU: the tenant's proposable accelerators, cheapest-$ first (see
# pricing.py for the derivation). Only pools that actually schedule on
# demo.hosted: T4 (g4dn.metal), L4 (g6.2xlarge), L40S (g6e.2xlarge).
GPU_TYPES: tuple[str, ...] = ("T4", "L4", "L40S")
GPU_VRAM_MIB: dict[str, int] = {"T4": 16 * 1024, "L4": 24 * 1024, "L40S": 48 * 1024}

_MIB = 1024 * 1024
_UNITS = {
    "ki": 1024, "mi": _MIB, "gi": 1024 * _MIB, "ti": 1024 * 1024 * _MIB,
    "k": 1000, "m": 1000 * 1000, "g": 1000 * 1000 * 1000,
    "b": 1, "": _MIB,  # bare numbers are interpreted as MiB
}


def parse_memory_to_mib(value: str | int | float) -> float:
    """'4Gi' / '512Mi' / 2048 → MiB. Raises ValueError on junk."""
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"memory must be positive, got {value!r}")
        return float(value)
    s = str(value).strip()
    for suffix in sorted(_UNITS, key=len, reverse=True):
        if suffix and s.lower().endswith(suffix):
            num = s[: -len(suffix)]
            break
    else:
        num, suffix = s, ""
    try:
        mib = float(num) * _UNITS[suffix] / _MIB
    except (ValueError, KeyError):
        raise ValueError(f"unparseable memory value {value!r}")
    if mib <= 0:
        raise ValueError(f"memory must be positive, got {value!r}")
    return mib


def format_memory(mib: int) -> str:
    """MiB → the canonical k8s string ('512Mi', '4Gi')."""
    if mib % 1024 == 0:
        return f"{mib // 1024}Gi"
    return f"{mib}Mi"


def bucket_memory_mib(mib: float) -> int:
    """Smallest grid step that covers `mib`; above the grid clamps to max."""
    for step in MEMORY_GRID_MIB:
        if mib <= step:
            return step
    return MEMORY_GRID_MIB[-1]


def bucket_cpu(cores: float) -> float:
    """Smallest CPU grid step that covers `cores`; clamps to grid max."""
    for step in CPU_GRID:
        if cores <= step:
            return step
    return CPU_GRID[-1]


# ── proposals ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Proposal:
    """A validated, bucketed resource proposal — the policy's action.

    GPU is part of the action: `gpu` is a count and `gpu_type` one of
    GPU_TYPES (a count without a type defaults to the tenant's cheapest,
    T4). A GPU request for a CPU task is pure waste the reward sees; a
    missing GPU on a task that needs one is a failure, like an OOM.
    """

    cpu: float
    memory_mib: int
    gpu: int = 0
    gpu_type: str | None = None

    def to_kwargs(self) -> dict:
        kwargs: dict = {
            # flyte.Resources accepts int cores or k8s millicore strings.
            "cpu": int(self.cpu) if float(self.cpu).is_integer() else f"{int(self.cpu * 1000)}m",
            "memory": format_memory(self.memory_mib),
        }
        if self.gpu:
            kwargs["gpu"] = f"{self.gpu_type or GPU_TYPES[0]}:{self.gpu}"
        return kwargs


class InvalidProposal(ValueError):
    """The model's output could not be turned into a schedulable request."""


def validate_proposal(raw: dict) -> Proposal:
    """Model-emitted kwargs dict → bucketed Proposal.

    Mirrors the PRD's validation step (§9.3 step 4): schema-check, then
    snap onto the action grid. Unknown keys are rejected rather than
    dropped — a model inventing keys is a formatting failure the reward
    should see, not something to silently launder.
    """
    if not isinstance(raw, dict):
        raise InvalidProposal(f"proposal must be a dict, got {type(raw).__name__}")
    unknown = set(raw) - {"cpu", "memory", "gpu"}
    if unknown:
        raise InvalidProposal(f"unknown resource keys {sorted(unknown)}")
    if "cpu" not in raw or "memory" not in raw:
        raise InvalidProposal(f"proposal needs cpu and memory, got {sorted(raw)}")

    cpu_raw = raw["cpu"]
    if isinstance(cpu_raw, str):
        s = cpu_raw.strip().lower()
        try:
            cores = float(s[:-1]) / 1000 if s.endswith("m") else float(s)
        except ValueError:
            raise InvalidProposal(f"unparseable cpu {cpu_raw!r}")
    elif isinstance(cpu_raw, (int, float)) and not isinstance(cpu_raw, bool):
        cores = float(cpu_raw)
    else:
        raise InvalidProposal(f"unparseable cpu {cpu_raw!r}")
    if cores <= 0:
        raise InvalidProposal(f"cpu must be positive, got {cpu_raw!r}")

    try:
        mem_mib = parse_memory_to_mib(raw["memory"])
    except ValueError as e:
        raise InvalidProposal(str(e))

    gpu_count, gpu_type = _parse_gpu(raw.get("gpu", 0))

    return Proposal(
        cpu=bucket_cpu(cores),
        memory_mib=bucket_memory_mib(mem_mib),
        gpu=gpu_count,
        gpu_type=gpu_type,
    )


def _parse_gpu(gpu_raw) -> tuple[int, str | None]:
    """Model-emitted gpu value → (count, type). Accepts 0/N (typed T4),
    "T4", or "T4:1" — only the tenant's GPU_TYPES are valid."""
    if gpu_raw in (0, None, ""):
        return 0, None
    if isinstance(gpu_raw, str):
        type_part, _, count_part = gpu_raw.strip().partition(":")
        gpu_type = type_part.strip().upper()  # tolerate "t4" / "l40s"
        matches = [t for t in GPU_TYPES if t.upper() == gpu_type]
        if not matches:
            raise InvalidProposal(
                f"gpu type must be one of {list(GPU_TYPES)}, got {gpu_raw!r}"
            )
        try:
            count = int(count_part) if count_part.strip() else 1
        except ValueError:
            raise InvalidProposal(f"unparseable gpu count in {gpu_raw!r}")
        if count < 1 or count > 8:
            raise InvalidProposal(f"gpu count must be in [1, 8], got {gpu_raw!r}")
        return count, matches[0]
    if isinstance(gpu_raw, int) and not isinstance(gpu_raw, bool):
        if gpu_raw < 0 or gpu_raw > 8:
            raise InvalidProposal(f"gpu must be an int in [0, 8], got {gpu_raw!r}")
        return gpu_raw, GPU_TYPES[0] if gpu_raw else None
    raise InvalidProposal(f"unparseable gpu {gpu_raw!r}")
