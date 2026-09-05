"""Classical-ML baseline: quantile gradient-boosted trees, no LLM.

The "non-fancy deep learning" bar the policy must clear. Design follows
the HPC/job-scheduler resource-prediction literature (Tanash et al.'s
Slurm memory predictors, Microsoft Resource Central, Google Autopilot's
percentile framing): tree ensembles over engineered features, predicting
a HIGH QUANTILE of peak usage per resource so the safety margin is part
of the loss, not a post-hoc fudge factor.

Four models, all sklearn HistGradientBoosting (fast, CPU-only, handles
mixed features natively):
- memory:  quantile regression on log2(peak MiB) at MEM_QUANTILE
- cpu:     quantile regression on cores at CPU_QUANTILE
- gpu:     classifier over {none, T4, L4, L40S} (the cheapest fitting card)
- gpu_mem: quantile regression on log2(VRAM MiB), gated on the classifier
           (picks the card via pricing.cheapest_gpu_for at predict time)

Features come from what any scheduler could see WITHOUT a language model:
sampled params (params_json), numbers/units mentioned in the input
profile, lexical flags of the source code (imports/keywords), the
author-declared prior, and run history when present. Alternatives
considered and rejected for the first cut: Bayesian ridge (mean + k·sigma
— interpretable but linear, and the corpus's param->footprint curves are
not), Autopilot-style percentile-of-history (no cold-start answer), and
k-NN over code similarity (needs embeddings — that's the fancy path).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from ..policy.actions import (
    GPU_TYPES,
    Proposal,
    bucket_cpu,
    bucket_memory_mib,
)
from ..pricing import cheapest_gpu_for

MEM_QUANTILE = 0.9
CPU_QUANTILE = 0.75
GPU_MEM_QUANTILE = 0.9
# Bucketing already rounds up; a thin multiplicative margin covers the
# simulator's OOM jitter band on top of the quantile.
MEM_MARGIN = 1.1

_CODE_FLAGS = (
    "torch", "cuda", "pandas", "numpy", "sklearn", "groupby", "merge",
    "DataFrame", "float32", "float16", "Linear", "matmul", "@", "fit(",
    "generate", "backward", "optim",
)
_NUM_RE = re.compile(r"(\d[\d_,]*\.?\d*)\s*(GiB|Gi|MiB|Mi|GB|MB|[KkMB]?\b)")


def _profile_numbers(profile: str) -> list[float]:
    """Numbers in the input profile, normalized to a common scale (bytes-ish
    for unit-suffixed, raw otherwise), largest first."""
    out = []
    for num, unit in _NUM_RE.findall(profile or ""):
        try:
            v = float(num.replace(",", "").replace("_", ""))
        except ValueError:
            continue
        unit = unit.strip()
        mult = {
            "GiB": 1024.0, "Gi": 1024.0, "GB": 1000.0,
            "MiB": 1.0, "Mi": 1.0, "MB": 1.0,
            "k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9,
        }.get(unit, 1.0)
        out.append(v * mult)
    return sorted(out, reverse=True)


def extract_features(record: dict) -> dict[str, float]:
    """One corpus row → flat numeric feature dict. Pure and total: any
    row shape degrades to zeros, never raises."""
    f: dict[str, float] = {}
    # sampled params (ground-truth generator state — a scheduler would have
    # the task's submitted arguments, which these stand in for)
    try:
        params = json.loads(record.get("params_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        params = {}
    for key in ("rows", "cols", "segments", "n_samples", "n_features",
                "n_estimators", "n_jobs", "hidden", "depth", "batch_size",
                "seq_len", "lora_r", "dim", "out_dim", "n_batches",
                "n_records", "duration_s"):
        v = params.get(key)
        f[f"p_{key}"] = math.log1p(float(v)) if isinstance(v, (int, float)) else 0.0
    # input-profile numbers (top 3, log-scaled)
    nums = _profile_numbers(record.get("input_profile") or "")
    for i in range(3):
        f[f"prof_num_{i}"] = math.log1p(nums[i]) if i < len(nums) else 0.0
    # lexical code flags
    code = record.get("source_code") or ""
    for flag in _CODE_FLAGS:
        f[f"code_{flag}"] = float(flag in code)
    f["code_len"] = math.log1p(len(code))
    # author prior
    prior_mem = prior_cpu = prior_gpu = 0.0
    try:
        prior = json.loads(record.get("prior_json") or "{}")
        from ..policy.actions import parse_memory_to_mib

        if "memory" in prior:
            prior_mem = math.log1p(parse_memory_to_mib(prior["memory"]))
        if "cpu" in prior:
            prior_cpu = float(str(prior["cpu"]).rstrip("m") or 0)
        prior_gpu = float(bool(prior.get("gpu")))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    f["prior_mem"], f["prior_cpu"], f["prior_gpu"] = prior_mem, prior_cpu, prior_gpu
    # run history: max observed peak (the Autopilot signal, when present)
    hist_peak = 0.0
    try:
        for h in json.loads(record.get("history_json") or "[]"):
            m = re.match(r"([\d.]+)MiB", str(h.get("peak", "")))
            if m:
                hist_peak = max(hist_peak, float(m.group(1)))
    except (json.JSONDecodeError, TypeError):
        pass
    f["hist_peak"] = math.log1p(hist_peak)
    f["has_history"] = float(hist_peak > 0)
    return f


@dataclass
class MLBaseline:
    """Quantile-GBT resource estimator. fit() on corpus train rows,
    propose() per row → a grid-snapped Proposal."""

    feature_names: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)
    _mem = None
    _cpu = None
    _gpu_cls = None
    _gpu_mem = None

    def _vec(self, record: dict) -> list[float]:
        f = extract_features(record)
        fam = record.get("family", "")
        return [f.get(name, 0.0) for name in self.feature_names] + [
            float(fam == known) for known in self.families
        ]

    def fit(self, records: list[dict]) -> "MLBaseline":
        import numpy as np
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        if not records:
            raise ValueError("MLBaseline.fit needs a non-empty train split")
        self.feature_names = sorted(extract_features(records[0]))
        self.families = sorted({r.get("family", "") for r in records})
        X = np.array([self._vec(r) for r in records])
        y_mem = np.log2(np.maximum([float(r["true_peak_memory_mib"]) for r in records], 64.0))
        y_cpu = np.array([float(r["true_cpu_cores"]) for r in records])
        gpu_mem = np.array(
            [float(r.get("true_gpu_mem_mib", 0.0) or 0.0) for r in records]
        )
        y_gpu = np.array(
            [cheapest_gpu_for(g) or "none" if g > 0 else "none" for g in gpu_mem],
            dtype=object,
        )

        def q_reg(quantile):
            return HistGradientBoostingRegressor(
                loss="quantile", quantile=quantile, max_iter=200, random_state=0
            )

        self._mem = q_reg(MEM_QUANTILE).fit(X, y_mem)
        self._cpu = q_reg(CPU_QUANTILE).fit(X, y_cpu)
        self._gpu_cls = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(
            X, y_gpu
        )
        gpu_rows = gpu_mem > 0
        if gpu_rows.sum() >= 8:
            self._gpu_mem = q_reg(GPU_MEM_QUANTILE).fit(
                X[gpu_rows], np.log2(gpu_mem[gpu_rows])
            )
        return self

    def propose(self, record: dict) -> Proposal:
        import numpy as np

        X = np.array([self._vec(record)])
        mem_mib = (2.0 ** float(self._mem.predict(X)[0])) * MEM_MARGIN
        cpu = max(float(self._cpu.predict(X)[0]), 0.5)
        gpu_label = str(self._gpu_cls.predict(X)[0])
        gpu, gpu_type = 0, None
        if gpu_label != "none":
            # Re-derive the card from predicted VRAM when the regressor
            # exists — classification picks IF a GPU is needed, the
            # quantile picks WHICH ONE safely.
            if self._gpu_mem is not None:
                vram = 2.0 ** float(self._gpu_mem.predict(X)[0])
                gpu_type = cheapest_gpu_for(vram) or GPU_TYPES[-1]
            else:
                gpu_type = gpu_label if gpu_label in GPU_TYPES else GPU_TYPES[0]
            gpu = 1
        return Proposal(
            cpu=bucket_cpu(cpu),
            memory_mib=bucket_memory_mib(mem_mib),
            gpu=gpu,
            gpu_type=gpu_type,
        )
