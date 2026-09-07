"""Corpus quality: diversity/representativeness measured, then GATED.

A synthetic release that ships skewed data silently is worse than one
that fails loudly. This module computes the numbers the release report
shows and the gates it enforces:

- near-duplicate rate between archetypes (MinHash over token shingles of
  PARAMS-normalized code — variants of one archetype are duplicates BY
  DESIGN, so dedup is measured at the archetype level);
- token entropy (lexical variety of the archetype pool);
- footprint grid coverage (are the mem × cpu buckets the policy must
  learn actually populated?);
- family mixture vs a declared target;
- concentration (max share of rows any one archetype contributes).

Pure functions over strings/records — unit-testable without a cluster.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter

# ── normalization + shingles ────────────────────────────────────────────

_PARAMS_LINE = re.compile(r"^PARAMS\s*=.*$", re.MULTILINE)
_NUM = re.compile(r"\b\d[\d_.]*\b")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z_]\w*|[^\sA-Za-z_\d]")


def normalize_code(code: str) -> str:
    """Strip the parts that differ between variants of the same idea:
    the PARAMS line, numeric literals, whitespace."""
    code = _PARAMS_LINE.sub("PARAMS=X", code)
    code = _NUM.sub("N", code)
    return _WS.sub(" ", code).strip()


def _shingles(code: str, k: int = 5) -> set[str]:
    toks = _TOKEN.findall(normalize_code(code))
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _minhash(shingles: set[str], n_hashes: int = 64) -> tuple[int, ...]:
    if not shingles:
        return tuple([0] * n_hashes)
    return tuple(
        min(zlib.crc32(f"{salt}:{s}".encode()) for s in shingles)
        for salt in range(n_hashes)
    )


def near_duplicate_rate(codes: list[str], threshold: float = 0.7) -> float:
    """Fraction of codes whose estimated Jaccard similarity to some OTHER
    code exceeds `threshold` (MinHash agreement rate ≈ Jaccard)."""
    if len(codes) < 2:
        return 0.0
    sigs = [_minhash(_shingles(c)) for c in codes]
    n_hashes = len(sigs[0])
    dup = [False] * len(codes)
    for i in range(len(sigs)):
        if dup[i]:
            continue
        for j in range(i + 1, len(sigs)):
            agree = sum(1 for a, b in zip(sigs[i], sigs[j]) if a == b) / n_hashes
            if agree >= threshold:
                dup[i] = dup[j] = True
    return sum(dup) / len(codes)


def token_entropy(codes: list[str]) -> float:
    """Shannon entropy (bits) of the identifier distribution across the
    pool — a coarse lexical-variety number (higher = more varied)."""
    counts = Counter(
        t for c in codes for t in _TOKEN.findall(normalize_code(c)) if t[0].isalpha()
    )
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


# ── coverage + mixture ──────────────────────────────────────────────────

MEM_BUCKETS_MIB = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
CPU_BUCKETS = (0.5, 1, 2, 4, 8)


def _bucket(value: float, edges: tuple) -> int:
    for i, e in enumerate(edges):
        if value <= e:
            return i
    return len(edges) - 1


def footprint_coverage(records: list[dict]) -> dict:
    """Occupancy of the log mem-grid × cpu-grid the policy must learn."""
    cells = {
        (_bucket(float(r["true_peak_memory_mib"]), MEM_BUCKETS_MIB),
         _bucket(float(r["true_cpu_cores"]), CPU_BUCKETS))
        for r in records
    }
    total = len(MEM_BUCKETS_MIB) * len(CPU_BUCKETS)
    return {"cells_occupied": len(cells), "cells_total": total,
            "coverage": len(cells) / total}


def mixture(records: list[dict]) -> dict[str, float]:
    counts = Counter(r.get("family", "?") for r in records)
    n = len(records) or 1
    return {f: c / n for f, c in sorted(counts.items())}


def concentration(records: list[dict]) -> float:
    """Max share of rows contributed by a single archetype (task_id prefix
    'arch-<seed>-<idx>' groups variants)."""
    groups = Counter(
        r["task_id"].rsplit("-", 1)[0] if str(r.get("task_id", "")).startswith("arch-")
        else r.get("task_id", "?")
        for r in records
    )
    n = len(records) or 1
    return max(groups.values()) / n if groups else 0.0


# ── the gate ────────────────────────────────────────────────────────────


def by_generator(codes_by_generator: dict[str, list[str]], records: list[dict]) -> dict:
    """Per-teacher diversity/coverage — the tier comparison. Answers "is
    the frontier model actually writing more varied archetypes than the
    small one?" with numbers instead of vibes."""
    rows_by_gen: dict[str, list[dict]] = {}
    for r in records:
        rows_by_gen.setdefault(str(r.get("generator", "?")), []).append(r)
    out = {}
    for gen, codes in codes_by_generator.items():
        rows = rows_by_gen.get(gen, [])
        out[gen] = {
            "archetypes": len(codes),
            "rows": len(rows),
            "near_dup_rate": near_duplicate_rate(codes),
            "token_entropy_bits": round(token_entropy(codes), 2),
            "coverage": footprint_coverage(rows)["coverage"] if rows else 0.0,
            "median_peak_mib": (
                sorted(float(r["true_peak_memory_mib"]) for r in rows)[len(rows) // 2]
                if rows
                else None
            ),
        }
    return out


def quality_report(
    archetype_codes: list[str],
    records: list[dict],
    label_errors: list[float] | None = None,
) -> dict:
    errs = sorted(label_errors or [])
    pct = lambda q: errs[min(int(q * len(errs)), len(errs) - 1)] if errs else None  # noqa: E731
    return {
        "n_archetypes": len(archetype_codes),
        "n_records": len(records),
        "near_dup_rate": near_duplicate_rate(archetype_codes),
        "token_entropy_bits": round(token_entropy(archetype_codes), 2),
        "coverage": footprint_coverage(records),
        "mixture": mixture(records),
        "max_archetype_share": concentration(records),
        "label_error_p50": pct(0.5),
        "label_error_p90": pct(0.9),
        "label_error_n": len(errs),
    }


def gate_failures(report: dict) -> list[str]:
    """Human-readable reasons the release should NOT ship; empty = pass."""
    fails = []
    if report["n_archetypes"] >= 20 and report["near_dup_rate"] > 0.15:
        fails.append(f"near-dup rate {report['near_dup_rate']:.0%} > 15%")
    if report["coverage"]["coverage"] < 0.45:
        fails.append(f"footprint coverage {report['coverage']['coverage']:.0%} < 45%")
    # Concentration is about SKEW, not the floor implied by the archetype
    # count: with N archetypes every one holds ≥1/N of the rows, so a flat
    # 2% is unreachable below 50 archetypes (run umpfpp92 failed at 3.0%
    # with 33 kept — perfectly even). Flag an archetype holding more than
    # twice its fair share, never less than the 2% target.
    n_arch = max(report.get("n_archetypes", 0), 1)
    share_limit = max(0.02, 2.0 / n_arch)
    if report["max_archetype_share"] > share_limit:
        fails.append(
            f"one archetype contributes {report['max_archetype_share']:.1%} of rows "
            f"(> {share_limit:.1%} = 2x fair share of {n_arch} archetypes)"
        )
    if report["label_error_p50"] is not None and report["label_error_p50"] > 0.25:
        fails.append(f"median label error {report['label_error_p50']:.0%} > 25%")
    return fails
