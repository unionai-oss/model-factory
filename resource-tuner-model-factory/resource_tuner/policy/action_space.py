"""The discretized action space, as an explicit, configurable object.

`actions.py` already buckets a *proposal* onto grids (memory on a log grid,
CPU on fixed increments, GPUs as typed counts). That is enough for the LLM
arm, which emits `{"cpu": ..., "memory": ..., "gpu": ...}` as text and gets
snapped onto the grid afterwards.

A **decision model** needs the grid up front, because it does not emit numbers
at all — it *chooses a cell*. So this module turns the same grids into a
finite, indexable action space with three heads:

    memory  -> one of len(memory_grid) buckets   (10 by default)
    cpu     -> one of len(cpu_grid) buckets      (8 by default)
    gpu     -> one of len(gpu_options) choices   (4 by default: none/T4/L4/L40S)

Three heads rather than one joint head on purpose. The joint space is
10 x 8 x 4 = 320 classes, most of which never appear in any corpus, so a
joint classifier spends its capacity learning which combinations exist. The
factored version shares all the features and keeps each head's label
distribution dense.

The grids are **hyperparameters**: `ActionSpace` is a frozen dataclass, and
`ACTION_SPACES` registers named variants (`default`, `coarse`, `fine`) that a
training arm can select. Coarsening trades achievable efficiency for a
smaller, better-estimated action space — exactly the kind of thing the suite
should settle by measurement rather than argument.

Decoding always produces a `Proposal`, so everything downstream (the reward
function, the episode runner, the tune service, `Proposal.to_kwargs()` ->
`flyte.Resources(cpu=..., memory=..., gpu=...)`) is reused unchanged. The
decision model is a new way to *pick* an action, not a new action type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .actions import (
    CPU_GRID,
    GPU_TYPES,
    GPU_VRAM_MIB,
    MEMORY_GRID_MIB,
    Proposal,
)

#: One GPU choice: ``None`` for a CPU-only task, or an accelerator and a count.
GpuOption = tuple[str, int] | None


def default_gpu_options() -> tuple[GpuOption, ...]:
    """CPU-only plus one of each accelerator, cheapest first.

    Single-card only by default: no corpus workload needs more than one, and
    every added count is a class the heads have to learn with no examples.
    """
    return (None, *((t, 1) for t in GPU_TYPES))


@dataclass(frozen=True)
class ActionSpace:
    """A finite, indexable set of resource requests.

    ``name`` is what a training arm selects, and it is recorded in the model
    manifest — a checkpoint trained on one grid cannot be decoded with another,
    so the space is part of the model's identity, not a runtime flag.
    """

    name: str = "default"
    memory_grid_mib: tuple[int, ...] = MEMORY_GRID_MIB
    cpu_grid: tuple[float, ...] = CPU_GRID
    gpu_options: tuple[GpuOption, ...] = field(default_factory=default_gpu_options)

    def __post_init__(self) -> None:
        if not self.memory_grid_mib or not self.cpu_grid or not self.gpu_options:
            raise ValueError(f"action space {self.name!r} has an empty dimension")
        if list(self.memory_grid_mib) != sorted(self.memory_grid_mib):
            raise ValueError("memory_grid_mib must be ascending")
        if list(self.cpu_grid) != sorted(self.cpu_grid):
            raise ValueError("cpu_grid must be ascending")
        if self.gpu_options[0] is not None:
            # Index 0 is the CPU-only choice by convention: it is the default
            # prediction and the one a CPU-only corpus collapses onto.
            raise ValueError("gpu_options[0] must be None (the CPU-only choice)")

    # ── shape ───────────────────────────────────────────────────────────

    @property
    def head_sizes(self) -> tuple[int, int, int]:
        """Output width of the (memory, cpu, gpu) heads."""
        return (len(self.memory_grid_mib), len(self.cpu_grid), len(self.gpu_options))

    @property
    def n_joint(self) -> int:
        """Size of the joint space, for the docstring's argument about factoring."""
        m, c, g = self.head_sizes
        return m * c * g

    # ── encoding: a request -> head indices ─────────────────────────────

    def memory_index(self, mib: float) -> int:
        """Smallest memory bucket that covers ``mib``; clamps to the top."""
        for i, step in enumerate(self.memory_grid_mib):
            if mib <= step:
                return i
        return len(self.memory_grid_mib) - 1

    def cpu_index(self, cores: float) -> int:
        """Smallest CPU bucket that covers ``cores``; clamps to the top."""
        for i, step in enumerate(self.cpu_grid):
            if cores <= step:
                return i
        return len(self.cpu_grid) - 1

    def gpu_index(self, gpu_mem_mib: float) -> int:
        """Cheapest accelerator whose VRAM covers ``gpu_mem_mib``.

        0 (or less) means a CPU task -> index 0. A requirement bigger than
        every card clamps to the largest rather than falling back to CPU: an
        under-provisioned GPU request fails the task, which is the expensive
        direction.
        """
        if gpu_mem_mib is None or gpu_mem_mib <= 0:
            return 0
        best: int | None = None
        for i, option in enumerate(self.gpu_options):
            if option is None:
                continue
            gpu_type, count = option
            if GPU_VRAM_MIB.get(gpu_type, 0) * count >= gpu_mem_mib:
                best = i if best is None else best
                break
        if best is not None:
            return best
        # Nothing fits: take the largest total VRAM on offer.
        return max(
            range(len(self.gpu_options)),
            key=lambda i: (
                0
                if self.gpu_options[i] is None
                else GPU_VRAM_MIB.get(self.gpu_options[i][0], 0) * self.gpu_options[i][1]
            ),
        )

    def encode_labels(
        self, peak_memory_mib: float, cpu_cores: float, gpu_mem_mib: float
    ) -> tuple[int, int, int]:
        """The training target for one workload: the smallest cell that fits.

        This is the whole labelling rule. The corpus carries measured/analytic
        ground truth (`true_peak_memory_mib`, `true_cpu_cores`,
        `true_gpu_mem_mib`), and the correct *action* is the cheapest grid cell
        that covers it — so the decision model is trained directly on the
        decision, with no regression-then-round step in between.
        """
        return (
            self.memory_index(peak_memory_mib),
            self.cpu_index(cpu_cores),
            self.gpu_index(gpu_mem_mib),
        )

    def encode_proposal(self, proposal: Proposal) -> tuple[int, int, int]:
        """Head indices for an existing Proposal (e.g. an author's prior)."""
        gpu_mem = (
            GPU_VRAM_MIB.get(proposal.gpu_type or GPU_TYPES[0], 0) * proposal.gpu
            if proposal.gpu
            else 0
        )
        return self.encode_labels(proposal.memory_mib, proposal.cpu, gpu_mem)

    # ── decoding: head indices -> a schedulable request ─────────────────

    def decode(self, memory_idx: int, cpu_idx: int, gpu_idx: int) -> Proposal:
        """Head indices -> a `Proposal`, i.e. flyte cpu/memory/gpu kwargs.

        Indices are clamped rather than validated: an argmax over a head can
        only land in range, and a clamp keeps a corrupt checkpoint serving
        something schedulable instead of raising inside the tune service.
        """
        memory_idx = _clamp(memory_idx, len(self.memory_grid_mib))
        cpu_idx = _clamp(cpu_idx, len(self.cpu_grid))
        gpu_idx = _clamp(gpu_idx, len(self.gpu_options))
        option = self.gpu_options[gpu_idx]
        return Proposal(
            cpu=self.cpu_grid[cpu_idx],
            memory_mib=self.memory_grid_mib[memory_idx],
            gpu=option[1] if option else 0,
            gpu_type=option[0] if option else None,
        )

    # ── cost structure, for the loss ────────────────────────────────────

    def memory_step_ratios(self) -> list[float]:
        """Each memory bucket as a multiple of the smallest, for cost weighting."""
        base = float(self.memory_grid_mib[0])
        return [step / base for step in self.memory_grid_mib]

    def to_dict(self) -> dict:
        """Serialized into the model manifest: a checkpoint is only decodable
        with the grid it was trained on."""
        return {
            "name": self.name,
            "memory_grid_mib": list(self.memory_grid_mib),
            "cpu_grid": list(self.cpu_grid),
            "gpu_options": [list(o) if o else None for o in self.gpu_options],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSpace":
        return cls(
            name=d.get("name", "default"),
            memory_grid_mib=tuple(int(x) for x in d["memory_grid_mib"]),
            cpu_grid=tuple(float(x) for x in d["cpu_grid"]),
            gpu_options=tuple(
                None if o is None else (str(o[0]), int(o[1])) for o in d["gpu_options"]
            ),
        )


def _clamp(i: int, n: int) -> int:
    return 0 if i < 0 else (n - 1 if i >= n else int(i))


# ── named variants (the hyperparameter) ─────────────────────────────────

DEFAULT_SPACE = ActionSpace(name="default")

#: Half the memory resolution and a coarse CPU ladder. Fewer, better-estimated
#: classes at the cost of granularity: the top memory step is 64Gi either way,
#: but a workload needing 3Gi now pays for 4Gi.
COARSE_SPACE = ActionSpace(
    name="coarse",
    memory_grid_mib=(256, 1024, 4096, 16384, 65536),
    cpu_grid=(1, 4, 8, 16),
)

#: Half-steps between the powers of two, for workloads whose footprint sits
#: awkwardly between buckets. More classes, each with fewer examples.
FINE_SPACE = ActionSpace(
    name="fine",
    memory_grid_mib=(
        128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144,
        8192, 12288, 16384, 24576, 32768, 49152, 65536,
    ),
    cpu_grid=(0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16),
)

ACTION_SPACES: dict[str, ActionSpace] = {
    s.name: s for s in (DEFAULT_SPACE, COARSE_SPACE, FINE_SPACE)
}


def get_action_space(name: str) -> ActionSpace:
    try:
        return ACTION_SPACES[name]
    except KeyError:
        raise ValueError(f"unknown action space {name!r}; choose from {sorted(ACTION_SPACES)}")


def grid_headroom(space: ActionSpace, true_mib: Sequence[float]) -> float:
    """Median over-provisioning forced by the grid alone, as a fraction.

    The floor no model on this grid can beat: even a perfect decision pays for
    the bucket it rounds up into. Reported on the scorecard so a 20% waste
    number can be read against the ~15% the grid itself costs, rather than
    against zero.
    """
    if not true_mib:
        return 0.0
    ratios = []
    for mib in true_mib:
        if mib <= 0:
            continue
        chosen = space.memory_grid_mib[space.memory_index(mib)]
        ratios.append(chosen / mib - 1.0)
    if not ratios:
        return 0.0
    ratios.sort()
    mid = len(ratios) // 2
    return ratios[mid] if len(ratios) % 2 else (ratios[mid - 1] + ratios[mid]) / 2
