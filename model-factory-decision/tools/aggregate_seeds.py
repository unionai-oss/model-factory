"""Aggregate per-seed scorecard summaries into mean ± spread tables.

Input: one JSON per seed, as returned by `tools/read_scorecards.py` (a list
of compact scorecards). Output: markdown tables on stdout — the
seed-averaged scorecard and the held-out frontier — for the research log.

    python tools/aggregate_seeds.py seed1.json seed2.json seed3.json

Pure Python, runs locally. A "seed" is one materialization date: the date
seeds the episodes AND the training init, so seeds are independent end to
end (data and optimisation).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict


def ms(xs: list[float], pct: bool = True, digits: int = 1) -> str:
    """mean ± sd [min–max]."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    f = (lambda v: f"{100 * v:.{digits}f}") if pct else (lambda v: f"{v:.3f}")
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return f"{f(statistics.mean(xs))} ± {f(sd)} [{f(min(xs))}–{f(max(xs))}]"


def get(d, *path):
    for k in path:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


ROWS = [
    ("in-dist acc", ("accuracy",)),
    ("in-dist unsafe-allow", ("unsafe_allow_rate",)),
    ("held-out acc", ("heldout", "accuracy")),
    ("held-out unsafe-allow (argmax)", ("heldout", "unsafe_allow_rate")),
    ("held-out SEVERE unsafe-allow (deny→allow, argmax)", ("heldout", "severe_unsafe_rate")),
    ("held-out severe, shift-tuned gate + guard", ("shift", "test_ood_guarded", "severe_unsafe_rate")),
    ("held-out unsafe-allow, in-dist-tuned gate", ("heldout", "gate", "unsafe_allow_rate")),
    ("held-out autonomy, in-dist-tuned gate", ("heldout", "gate", "autonomy")),
    ("held-out unsafe-allow, shift-tuned gate (T + t on OOD-val)", ("shift", "test_ood", "unsafe_allow_rate")),
    ("held-out autonomy, shift-tuned gate", ("shift", "test_ood", "autonomy")),
    ("held-out unsafe-allow, shift-tuned gate + guard", ("shift", "test_ood_guarded", "unsafe_allow_rate")),
    ("held-out autonomy, shift-tuned gate + guard", ("shift", "test_ood_guarded", "autonomy")),
    ("in-dist autonomy, shift-tuned gate + guard", ("shift", "test_ind_guarded", "autonomy")),
    ("in-dist unsafe-allow, shift-tuned gate + guard", ("shift", "test_ind_guarded", "unsafe_allow_rate")),
]
NUM_ROWS = [
    ("held-out ECE (raw)", ("heldout", "ece")),
    ("held-out ECE (tempered)", ("shift", "test_ood_ece_tempered")),
    ("temperature (fit on OOD-val)", ("shift", "temperature")),
    ("shift threshold (guarded)", ("shift", "threshold_guarded")),
    ("OOD-val expected cost", ("shift", "val_ood", "expected_cost")),
]


def main(paths: list[str]) -> None:
    by = defaultdict(list)  # (model, readout) -> [readout dict per seed]
    for p in paths:
        for c in json.load(open(p)):
            for name, r in c["readouts"].items():
                by[(c["model"], name)].append(r)
    keys = sorted(by)
    print(f"### Seed-averaged scorecard ({len(paths)} seeds; mean ± sd [min–max], %)\n")
    print("| metric | " + " | ".join(f"{m} · {r}" for m, r in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for label, path in ROWS:
        print(f"| {label} | " + " | ".join(ms([get(r, *path) for r in by[k]]) for k in keys) + " |")
    for label, path in NUM_ROWS:
        print(f"| {label} | " + " | ".join(ms([get(r, *path) for r in by[k]], pct=False) for k in keys) + " |")

    print("\n### Held-out per family: unsafe-allow, argmax → guarded (mean over seeds, %)\n")
    fams = sorted({f for k in keys for r in by[k] for f in (get(r, "heldout", "per_family") or {})})
    print("| model · readout | " + " | ".join(fams) + " |")
    print("|---|" + "---|" * len(fams))
    for k in keys:
        cells = []
        for f in fams:
            a = [get(r, "heldout", "per_family", f, "unsafe_allow_rate") for r in by[k]]
            g = [get(r, "heldout", "per_family", f, "guarded_unsafe_allow_rate") for r in by[k]]
            a, g = [x for x in a if x is not None], [x for x in g if x is not None]
            cells.append(f"{100 * statistics.mean(a):.1f} → {100 * statistics.mean(g):.1f}" if a and g else "—")
        print(f"| {k[0]} · {k[1]} | " + " | ".join(cells) + " |")

    print("\n### Held-out frontier, seed-averaged (autonomy / unsafe-allow, %)\n")
    curves = [("raw", ("heldout", "frontier")), ("raw + guard", ("heldout", "frontier_guarded")),
              ("tempered + guard", ("shift", "ood_frontier_tempered_guarded"))]
    show = [0.0, 0.5, 0.8, 0.9, 0.96, 0.99, 0.999]
    print("| model · readout | curve | " + " | ".join(f"t={t}" for t in show) + " |")
    print("|---|---|" + "---|" * len(show))
    for k in keys:
        for cname, path in curves:
            pts = [get(r, *path) for r in by[k]]
            if not all(pts):
                continue
            cells = []
            for t in show:
                at = [next((p for p in fr if abs(p["threshold"] - t) < 1e-9), None) for fr in pts]
                at = [p for p in at if p]
                if not at:
                    cells.append("—")
                    continue
                cells.append(f"{100 * statistics.mean(p['autonomy'] for p in at):.0f} / "
                             f"{100 * statistics.mean(p['unsafe_allow_rate'] for p in at):.1f}")
            print(f"| {k[0]} · {k[1]} | {cname} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
