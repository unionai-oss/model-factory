"""The Kubernetes twin (round 22): every Flyte run as a plain job.py in a Job
manifest, labelled from its Flyte measurement plus the staging's cost.

What the K8s release's numbers rest on: the twin's code is the workload's
code (it parses, it keeps the body), its manifest is a Job whose only
undecided field is the request, the labels move exactly as the accounting
says (no Flyte runtime; boto3's footprint; tmpfs bytes charged), twins stay
in their task's split and crossfit fold, and the prompt still ends the way
the pointer head's offsets assume.
"""

from __future__ import annotations

import ast
import json

import numpy as np
import pytest
import yaml

from resource_tuner.bench import hybrid
from resource_tuner.bench import workloads as mem_w
from resource_tuner.bench.grid import LABEL_HEADROOM
from resource_tuner.bench_all.k8s import FLYTE_RUNTIME_MIB as FLYTE_RUNTIME_OVERHEAD_MIB
from resource_tuner.bench_all import arms, dataset, features, gating, grid, k8s, prompting, scoring
from resource_tuner.bench_cpu import workloads as cpu_w
from resource_tuner.bench_gpu import workloads as gpu_w

from test_bench_all import corpus  # noqa: F401 -- the combined bench's fake-measured corpus


@pytest.fixture(scope="module")
def twinned(corpus):  # noqa: F811
    return dataset.with_k8s(corpus)


@pytest.mark.parametrize("mod", [mem_w, cpu_w, gpu_w], ids=["mem", "cpu", "gpu"])
def test_every_family_renders_a_job(mod):
    for fam in sorted(mod.FAMILIES):
        t = mod.make_task(fam, 3, n_runs=2)
        code = mem_w.render_task(t)
        for st in k8s.STAGINGS:
            src = k8s.module_code(code, t.inputs, st)
            tree = ast.parse(src)
            assert "flyte" not in src and "await" not in src
            fns = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
            assert t.func_name in fns
            # the body is the harness body: every line of it survives
            body = mem_w.render_harness(t).split("def main(", 1)[1].split("\n", 1)[1]
            assert body.strip() in src
            assert ("import boto3" in src) == k8s.is_boto3(st)
            spec = yaml.safe_load(k8s.manifest(code, t.inputs, st, "abcd1234", 1))["spec"]["template"]["spec"]
            main = spec["containers"][0]
            assert main["resources"] == {}
            assert (spec["volumes"][0]["emptyDir"] == {"medium": "Memory"}) == k8s.is_tmpfs(st)
            assert ("initContainers" in spec) == (not k8s.is_boto3(st))
            env = {e["name"] for e in main["env"]}
            assert set(k8s.THREAD_ENV) <= env  # the measurement's thread floor


def test_disk_and_tmpfs_differ_only_in_the_volume():
    """...and, for an init container staging onto the tmpfs, its limit: it is
    charged every byte it writes there (rt-k8spod-pilot-2)."""
    t = mem_w.make_task("pandas_join", 5, n_runs=2)
    code = mem_w.render_task(t)
    sizes = {"orders": 300 * 2**20, "catalog": 20 * 2**20}
    for who in ("init", "boto3"):
        disk, tmp = (k8s.manifest(code, t.inputs, f"{who}-{w}", "k", 0, sizes) for w in ("disk", "tmpfs"))
        diff = [(a, b) for a, b in zip(disk.splitlines(), tmp.splitlines()) if a != b]
        volume = ("          emptyDir: {}", "          emptyDir: {medium: Memory}")
        if who == "boto3":
            assert diff == [volume]
        else:
            assert diff[-1] == volume and len(diff) == 2
            staged = k8s.init_limit_mib(sizes, "init-tmpfs")
            assert staged == k8s.INIT_BASE_MIB + 321  # every staged byte (and the result), on top of aws-cli's own
            assert diff[0] == (diff[0][0], diff[0][0].replace(f"memory: {k8s.INIT_BASE_MIB}Mi}}}}",
                                                               f"memory: {staged}Mi}}}}"))
        assert k8s.module_code(code, t.inputs, f"{who}-disk") == k8s.module_code(code, t.inputs, f"{who}-tmpfs")


def test_twin_memory_accounting():
    sizes = {"a": 3 * 2**30, "b": 10_000}
    flyte_need = 4096.0
    peak = flyte_need / LABEL_HEADROOM - FLYTE_RUNTIME_OVERHEAD_MIB
    staged = k8s.staged_bytes(sizes) / 2**20
    assert k8s.twin_need_mib(flyte_need, sizes, "init-disk") == pytest.approx(peak * LABEL_HEADROOM)
    assert k8s.twin_need_mib(flyte_need, sizes, "init-tmpfs") == pytest.approx((peak + staged) * LABEL_HEADROOM)
    assert k8s.twin_need_mib(flyte_need, sizes, "boto3-disk") == pytest.approx((peak + k8s.BOTO3_MIB) * LABEL_HEADROOM)
    assert k8s.twin_need_mib(flyte_need, sizes, "boto3-tmpfs") == pytest.approx(
        (peak + k8s.BOTO3_MIB + staged) * LABEL_HEADROOM)
    assert staged >= 3 * 1024 + 10_000 / 2**20  # page-rounded, plus the result file
    # no Flyte runtime in a plain container: an init-disk twin needs less
    assert k8s.twin_need_mib(flyte_need, sizes, "init-disk") < flyte_need
    # history in container units: a K8s peak is the need / headroom, as is
    assert k8s.need_from_peak(1000.0, "k8s") == pytest.approx(1000 * LABEL_HEADROOM)
    assert k8s.need_from_peak(1000.0, "flyte") == pytest.approx((1000 + FLYTE_RUNTIME_OVERHEAD_MIB) * LABEL_HEADROOM)


def test_twins_pair_up_and_keep_their_split(corpus, twinned):  # noqa: F811
    assert len(twinned) == 2 * len(corpus)
    by_id = {r["run_id"]: r for r in twinned}
    tw = [r for r in twinned if r["platform"] == "k8s"]
    assert len({r["run_id"] for r in twinned}) == len(twinned)
    for t in tw:
        f = by_id[t["twin_of"]]
        assert t["split"] == f["split"] and t["source"] == f["source"] and t["family"] == f["family"]
        assert hybrid.fold_of(t) == hybrid.fold_of(f)  # crossfit never puts a twin on the other side
        assert t["staging"] == k8s.staging_for(f["task_id"])
        if t["source"] == "gpu":
            assert t["label_gpu"] == f["label_gpu"] and t["label_mem"] == -1
        else:
            assert t["label_mem"] == dataset.grid.mem_cell(t["need_mib"])
    # every staging is drawn
    assert {t["staging"] for t in tw} == set(k8s.STAGINGS)
    # a task's runs share their staging (one manifest per task)
    per_task = {}
    for t in tw:
        per_task.setdefault(t["task_id"], set()).add(t["staging"])
    assert all(len(v) == 1 for v in per_task.values())


def test_boto3_twins_download_inside_the_job(twinned):
    for t in twinned:
        if t["platform"] != "k8s" or t["source"] != "cpu" or not k8s.is_boto3(t["staging"]):
            continue
        series = json.loads(t["cpu_series_json"])
        d = k8s.download_s(json.loads(t["input_bytes_json"]))
        assert series[: max(1, round(d))] == [round(k8s.DL_CORES, 4)] * max(1, round(d))
        assert t["need_cores"] >= k8s.DL_CORES * min(1.0, max(1, round(d)) / 15) - 1e-9
        return
    pytest.skip("no boto3 CPU twin in the fixture")


def test_twin_history_is_in_container_units(twinned):
    for t in twinned:
        hist = json.loads(t["history_json"])
        if t["platform"] != "k8s" or t["source"] == "gpu" or not hist or t["staging"] != "init-disk":
            continue
        text = prompting.format_history(hist, "k8s")
        assert f"peak memory {hist[0]['peak_mem_mib']:,.0f} MiB" in text
        return
    pytest.skip("no init-disk twin with history in the fixture")


def test_prompt_ends_the_same_on_both_platforms(twinned):
    tail = f"### Options\n{prompting.options_text()}\n\n{prompting.ANSWER}"
    for p in ("flyte", "k8s"):
        r = next(r for r in twinned if r["platform"] == p)
        u = prompting.user_prompt(r)
        assert u.endswith(tail)
        assert ("### Job manifest" in u) == (p == "k8s")


def test_trees_can_see_the_staging(twinned):
    tw = [r for r in twinned if r["platform"] == "k8s"]
    for st in k8s.STAGINGS:
        f = features.row_features(next(r for r in tw if r["staging"] == st))
        assert f["k8s_medium: Memory"] == float(k8s.is_tmpfs(st))
        assert f["k8s_import boto3"] == float(k8s.is_boto3(st))
        assert f["k8s_kind: Job"] == 1.0
    flyte = features.row_features(next(r for r in twinned if r["platform"] == "flyte"))
    assert flyte["k8s_kind: Job"] == 0.0
    assert set(features.feature_names()) == set(flyte)


def test_rules_read_history_in_its_platforms_units(twinned):
    state = arms.fit_rules([r for r in twinned if r["split"] == "train"], [r for r in twinned if r["split"] == "val"])
    preds = arms.predict_rules(state, twinned)
    assert len(preds) == len(twinned)


def test_slices_and_pairs(twinned):
    rows = [r for r in twinned if dataset.usable(r)]
    preds = {r["run_id"]: dict(zip(("mem_idx", "cpu_idx", "gpu_idx"), scoring.oracle_action(r))) for r in rows}
    sl = scoring.score_slices(rows, preds)
    for split in ("test", "ood"):
        assert sl[f"{split}/platform=k8s"]["n"] and sl[f"{split}/platform=flyte"]["n"]
        pairs = sl[f"{split}/pairs"]
        if pairs["n"]:  # the oracle moves exactly as the label moved
            # (its regret is ~0, not 0: a twin's 1.8-core download second can exceed its busiest-window cell)
            assert pairs["mem_shift_exact"] == 1.0 and pairs["regret_gap"] == pytest.approx(0.0, abs=1e-2)


def test_balanced_draws_per_source_and_platform(twinned):
    out = dataset.balanced(twinned, 3)
    groups = {(r["source"], r["platform"]) for r in out}
    assert len(groups) == 6 and len(out) == 18


def _onehot(i, n):
    p = np.full(n, 1e-6)
    p[i] = 1.0
    return p / p.sum()


def test_adapter_moves_a_flyte_label_onto_its_twins(twinned):
    by_id = {r["run_id"]: r for r in twinned}
    tw = [r for r in twinned if r["platform"] == "k8s" and r["source"] != "gpu" and dataset.usable(r)]
    assert {gating.staging_seen(t) for t in tw} == set(k8s.STAGINGS)
    assert all(gating.staging_seen(t) == t["staging"] for t in tw)  # what the manifest and job.py say
    flyte = {f["run_id"]: [_onehot(f["label_mem"], grid.N_MEM), _onehot(f["label_cpu"], grid.N_CPU),
                           _onehot(grid.NO_GPU, grid.N_GPU)] for f in (by_id[t["twin_of"]] for t in tw)}
    own = {t["run_id"]: [_onehot(0, grid.N_MEM), _onehot(t["label_cpu"], grid.N_CPU), _onehot(grid.NO_GPU, grid.N_GPU)]
           for t in tw}
    ad = gating.adapted(tw, {**flyte, **own})
    assert all(ad[t["run_id"]][1].argmax() == t["label_cpu"] for t in tw if not k8s.is_boto3(t["staging"]))
    d = np.array([int(ad[t["run_id"]][0].argmax()) - t["label_mem"] for t in tw])
    # a cell's middle stands in for the need, so a twin may land one cell off, never more
    assert (np.abs(d) <= 1).all() and (d == 0).mean() > 0.6
    unadapted = np.array([by_id[t["twin_of"]]["label_mem"] - t["label_mem"] for t in tw])
    assert (d == 0).mean() > (unadapted == 0).mean()
    for t in tw:  # mass is moved, never made or lost
        assert ad[t["run_id"]][0].sum() == pytest.approx(1.0) and ad[t["run_id"]][1].sum() == pytest.approx(1.0)


def test_gate_routes_a_val_quantile_out():
    rng = np.random.default_rng(0)
    rows = [{"run_id": f"r{i}", "split": "val" if i < 1000 else "ood"} for i in range(2000)]
    signal = {r["run_id"]: float(rng.random() + (r["split"] == "ood")) for r in rows}
    inner = {r["run_id"]: {"mem_idx": 1, "cpu_idx": 1, "gpu_idx": 0} for r in rows}
    outer = {r["run_id"]: {"mem_idx": 2, "cpu_idx": 2, "gpu_idx": 0} for r in rows}
    g, tau = gating.gate(rows, signal, inner, outer, 0.9)
    val_out = np.mean([g[r["run_id"]]["mem_idx"] == 2 for r in rows if r["split"] == "val"])
    assert val_out == pytest.approx(0.1, abs=0.01)
    assert np.mean([g[r["run_id"]]["gated_out"] for r in rows if r["split"] == "ood"]) > 0.9
    assert gating.auroc([signal[r["run_id"]] for r in rows[:1000]], [signal[r["run_id"]] for r in rows[1000:]]) > 0.9
    assert gating.auroc([0, 1], [0, 1]) == pytest.approx(0.5)
    assert gating.js(np.array([1.0, 0]), np.array([1.0, 0])) == pytest.approx(0.0, abs=1e-9)
    assert gating.js(np.array([1.0, 0]), np.array([0, 1.0])) == pytest.approx(np.log(2), abs=1e-6)


def test_gate_all_shapes(twinned):
    from resource_tuner.bench_all import stations
    rows = [r for r in twinned if dataset.usable(r)]
    rng = np.random.default_rng(1)

    def dist():
        return [rng.dirichlet(np.ones(n)) for n in (grid.N_MEM, grid.N_CPU, grid.N_GPU)]

    probs = {v: {r["run_id"]: dist() for r in rows} for v in ("gbt-dist", "rl-pointer", "jev-pointer-flyte")}
    decoded = {v: {r["run_id"]: {"mem_idx": 3, "cpu_idx": 1, "gpu_idx": 0} for r in rows}
               for v in ("pool-rl-pointer-argmax", "rl-pointer")}
    metas = {"jev-pointer-flyte": {"gpu_threshold": 0.5}, "jev-pointer": {}}
    table, knobs = stations.gate_all(rows, probs, decoded, metas, "pool-rl-pointer-argmax", "rl-pointer", "rl-pointer")
    names = {t["variant"] for t in table}
    assert {"gate-rl-pointer", "gate-knn-q95", "gate-js-q90", "gate-knn-numeric-q95", "jev-pointer-flyte-adapt",
            "jev-pointer-flyte-adapt-argmax",
            "jev-pointer-flyte-adapt-prompt"} <= names
    assert all(0 <= x["auroc_val_vs_ood"] <= 1 for x in knobs["signals"].values())


def test_counterfactual_renders_one_run_under_every_staging(corpus):  # noqa: F811
    from resource_tuner.bench_all import counterfactual

    for r in corpus:  # the fake corpus has no test/ood split of its own to spare: borrow every CPU-node run
        r.setdefault("platform", "flyte")
    picked = [r for r in corpus if r["source"] != "gpu" and dataset.usable(r)][:6]
    rends = counterfactual.renditions(corpus, picked)
    by_base: dict[str, dict[str, dict]] = {}
    for x in rends:
        base, st = x["run_id"].rsplit(counterfactual.SEP, 1)
        by_base.setdefault(base, {})[st] = x
    assert len(by_base) == len(picked)
    for p in picked:
        s = by_base[p["run_id"] + k8s.SUFFIX]
        assert set(s) == set(k8s.STAGINGS)
        # only the staging differs: same function, same inputs, same split
        assert {x["inputs_desc"] for x in s.values()} == {p["inputs_desc"]}
        assert {x["split"] for x in s.values()} == {p["split"]}
        assert {x["twin_of"] for x in s.values()} == {p["run_id"]}
        # the label moves exactly as the accounting says
        sizes = json.loads(p["input_bytes_json"])
        for st, x in s.items():
            assert x["need_mib"] == pytest.approx(k8s.twin_need_mib(p["need_mib"], sizes, st))
            assert ("medium: Memory" in x["manifest"]) == k8s.is_tmpfs(st)
        assert s["init-tmpfs"]["label_mem"] >= s["init-disk"]["label_mem"]
        # the task's own staging is its real twin
        own = k8s.twin(p, None)
        assert s[own["staging"]]["task_code"] == own["task_code"]
    # a request that ignores the manifest keeps every pick: direction right on no moved pair
    flat = {x["run_id"]: {"mem_idx": 3, "cpu_idx": 1, "gpu_idx": 0} for x in rends}
    rec = counterfactual.report([{**x, "split": "test"} for x in rends], {"flat": flat})["flat"]["split"]["test"]
    assert rec["medium"]["unmoved_kept"] in (1.0,) or rec["medium"]["unmoved_kept"] != rec["medium"]["unmoved_kept"]
    if rec["medium"]["label_moved"]:
        assert rec["medium"]["moved_direction"] == 0.0


def test_hosted_answer_schema_lists_every_option():
    from resource_tuner.bench_all import stations

    sch = stations.answer_schema()
    assert sch["required"] == list(prompting.KEYS)
    for k, opts in zip(prompting.KEYS, prompting.OPTIONS):
        assert sch["properties"][k]["enum"] == list(opts)
        for o in opts:  # every option a constrained answer can give parses back to itself
            pass
    ans = json.dumps({k: o[2] for k, o in zip(prompting.KEYS, prompting.OPTIONS)})
    assert prompting.parse_answer(ans) == (2, 2, 2)
    assert set(stations.HOSTED_KNOBS) == set(stations.HOSTED_ARMS)


def test_trained_arms_do_not_key_on_post_hoc_code():
    """rsk-k8s-1-gate retrained every LLM arm because the gate's code was in the one
    cache key every station shared. The trained arms' key now names only what decides
    their predictions; the post-hoc stations (pool, gate, counterfactual, scoring) keep theirs."""
    from resource_tuner.bench import versioning as v

    assert not {"bench_all/gating", "bench_all/counterfactual", "bench_all/paper"} & set(v._ALL_TRAIN)
    v._hash_defs("bench_all/stations", v._ALL_TRAIN_DEFS)  # every name resolves
    post = {"pool_all_arms", "gate_all_arms", "staging_counterfactual", "score_all_arm", "all_leaderboard"}
    assert not post & set(v._ALL_TRAIN_DEFS)
    assert v.all_train_cache().version_override != v.all_arms_cache().version_override
