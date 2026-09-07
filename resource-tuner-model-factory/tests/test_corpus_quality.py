"""Round-12 synthetic-data overhaul: quality metrics/gates, power-law
fits, holdout gating, scenario prompts, per-variant context."""

import math
import random

import pytest

from resource_tuner.taskgen import quality as qual
from resource_tuner.taskgen.archetypes import (
    ParamFit,
    holdout_error,
    render_archetype_prompt,
    variant_profile,
)
from resource_tuner.taskgen.corpus import context_fields_json
from resource_tuner.taskgen.synthetic import build_scenario, validate_task_code

CODE_A = "import pandas as pd\nPARAMS = {'rows': 10}\ndef run():\n    df = pd.DataFrame({'a': range(PARAMS['rows'])})\n    return {'n': len(df)}\n"
CODE_A2 = CODE_A.replace("10", "99999").replace("'a'", "'b'")  # same idea
CODE_B = "import numpy as np\nPARAMS = {'dim': 4}\ndef run():\n    w = np.linalg.svd(np.random.rand(PARAMS['dim'], PARAMS['dim']))\n    return {'s': float(w[1][0])}\n"


def test_near_duplicate_rate_finds_rewrites_not_strangers():
    assert qual.near_duplicate_rate([CODE_A, CODE_A2]) == 1.0  # same skeleton
    assert qual.near_duplicate_rate([CODE_A, CODE_B]) == 0.0
    assert qual.near_duplicate_rate([CODE_A]) == 0.0


def test_footprint_coverage_and_concentration():
    spread = [
        {"task_id": f"arch-0-{i}-v0", "true_peak_memory_mib": 2 ** (7 + i % 9) + 1,
         "true_cpu_cores": [0.5, 1, 2, 4, 8][i % 5], "family": "etl"}
        for i in range(45)
    ]
    cov = qual.footprint_coverage(spread)
    assert cov["coverage"] > 0.5
    narrow = [
        {"task_id": "arch-0-7-v%d" % i, "true_peak_memory_mib": 512,
         "true_cpu_cores": 1, "family": "etl"}
        for i in range(100)
    ]
    assert qual.footprint_coverage(narrow)["cells_occupied"] == 1
    assert qual.concentration(narrow) == 1.0  # one archetype owns everything
    assert qual.concentration(spread) < 0.1


def test_gate_failures_fire_on_bad_corpora():
    bad = qual.quality_report(
        [CODE_A, CODE_A2] * 15,
        [{"task_id": "arch-0-1-v%d" % i, "true_peak_memory_mib": 512,
          "true_cpu_cores": 1, "family": "etl"} for i in range(50)],
        label_errors=[0.4, 0.5, 0.6],
    )
    reasons = qual.gate_failures(bad)
    assert any("near-dup" in r for r in reasons)
    assert any("coverage" in r for r in reasons)
    assert any("archetype contributes" in r for r in reasons)
    assert any("label error" in r for r in reasons)


def test_param_fit_recovers_power_laws():
    rng = random.Random(1)
    # y = 3 * x^2 — the old linear fit's structural miss
    pts = [(3 * x * x, {"rows": x}) for x in (10, 100, 1000, rng.randint(20, 800))]
    fit = ParamFit(pts, "rows")
    assert fit.predict(300) == pytest.approx(3 * 300 * 300, rel=0.05)
    # holdout error near zero for the true law, large for a wrong one
    holdout = [(3 * 500 * 500, {"rows": 500})]
    assert holdout_error(fit, holdout) < 0.05
    wrong = ParamFit([(50.0, {"rows": x}) for x in (10, 1000)], "rows")
    assert holdout_error(wrong, holdout) > 0.5


def test_scenario_prompt_carries_grid_and_avoid_list():
    rng = random.Random(7)
    scenarios = {build_scenario(rng) for _ in range(30)}
    assert len(scenarios) > 20  # the grid actually varies
    prompt = render_archetype_prompt(
        "tabular ETL", build_scenario(rng), ["old archetype one", "old two"],
        allowed="numpy, pandas", gpu=False,
    )
    assert "old archetype one" in prompt and "Domain:" in prompt
    gpu_prompt = render_archetype_prompt(
        "CUDA inference", build_scenario(rng), [], allowed="torch", gpu=True
    )
    assert "torch.cuda.is_available()" in gpu_prompt
    assert "torch.cuda" not in prompt


def test_variant_profile_renders_sampled_scale():
    p = variant_profile("ad-click sessionization", {"rows": 2_400_000, "cols": 16})
    assert "2,400,000" in p and "ad-click sessionization" in p
    q = variant_profile("same", {"rows": 10})
    assert p != q  # variants no longer share one profile string


def test_context_fields_json_label_based():
    rng = random.Random(3)
    outputs = {context_fields_json(800.0, 2.0, 0.0, random.Random(s)) for s in range(40)}
    priors = [p for p, _ in outputs if p]
    hists = [h for _, h in outputs if h]
    assert priors and hists  # both regimes occur
    assert any(p == "" for p, _ in outputs)  # cold start survives
    gpu_priors = [
        context_fields_json(800.0, 2.0, 9000.0, random.Random(s))[0] for s in range(60)
    ]
    assert any("T4:1" in p for p in gpu_priors)


def test_widened_allowlist_accepts_io_phase_code():
    validate_task_code(
        "import tempfile\nimport csv\nimport io\n"
        "def run():\n"
        "    buf = io.StringIO()\n"
        "    w = csv.writer(buf)\n"
        "    w.writerow([1, 2])\n"
        "    return {'n': 1}\n"
    )
    with pytest.raises(Exception):
        validate_task_code("import os\ndef run():\n    return {}\n")


def test_open_allowed_for_tempfile_staging_but_os_still_banned():
    """Round 12: file-staged phases need open(); os stays forbidden."""
    validate_task_code(
        "import tempfile\n"
        "def run():\n"
        "    with tempfile.TemporaryDirectory() as d:\n"
        "        with open(d + '/part.csv', 'w') as fh:\n"
        "            fh.write('a,b\\n1,2\\n')\n"
        "    return {'n': 1}\n"
    )
    for bad in ("import os", "import sys", "import shutil"):
        with pytest.raises(Exception):
            validate_task_code(f"{bad}\ndef run():\n    return {{}}\n")
    with pytest.raises(Exception):
        validate_task_code("def run():\n    eval('1')\n    return {}\n")


def test_generator_provenance_on_every_row():
    """Every corpus row names who wrote it (round 12)."""
    from resource_tuner.contracts import CORPUS_COLUMNS
    from resource_tuner.taskgen.corpus import build_corpus
    from resource_tuner.taskgen.synthetic import synthetic_record

    assert "generator" in CORPUS_COLUMNS
    rows = build_corpus(10, 5, seed=2)
    assert {r["generator"] for r in rows} == {"template"}
    syn_row = synthetic_record(
        "t1", "etl", "def run(): return {}", "desc",
        {"ok": True, "peak_rss_mib": 200.0, "duration_s": 60, "cpu_avg_cores": 1.0},
        generator="qwen35-397b",
    )
    assert syn_row["generator"] == "qwen35-397b"
    assert set(syn_row) == set(CORPUS_COLUMNS)


def test_by_generator_separates_tiers():
    from resource_tuner.taskgen import quality as q

    rows = [
        {"generator": "big", "task_id": "arch-0-1-v0", "true_peak_memory_mib": 900,
         "true_cpu_cores": 2, "family": "etl"},
        {"generator": "small", "task_id": "arch-0-2-v0", "true_peak_memory_mib": 300,
         "true_cpu_cores": 1, "family": "etl"},
    ]
    out = q.by_generator({"big": [CODE_A, CODE_B], "small": [CODE_A, CODE_A2]}, rows)
    assert out["big"]["near_dup_rate"] == 0.0      # varied author
    assert out["small"]["near_dup_rate"] == 1.0    # repetitive author
    assert out["big"]["rows"] == 1 and out["small"]["archetypes"] == 2
