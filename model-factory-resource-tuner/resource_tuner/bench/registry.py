"""Every RightSize bench by name, for code shared across them (the LLM arms,
the GPU training station). Imported at module top everywhere: the code
bundler ships only statically-imported modules."""

from __future__ import annotations

from ..bench_cpu.spec import CPU
from ..bench_gpu.spec import GPU
from .mem_spec import MEM
from .spec import BenchSpec

SPECS: dict[str, BenchSpec] = {s.name: s for s in (MEM, CPU, GPU)}


def get_spec(name: str = "mem") -> BenchSpec:
    try:
        return SPECS[name]
    except KeyError:
        raise ValueError(f"unknown bench {name!r}; choose from {sorted(SPECS)}") from None
