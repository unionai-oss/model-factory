"""Publish RightSize Bench and its models to the Hugging Face Hub (round 24).

Everything a release needs lives in the cluster's object store (the corpus,
every arm's predictions, the trained adapters), which the laptop cannot read.
So staging and pushing are Flyte tasks; ``hub_release.py`` resolves a finished
factory run's outputs and launches ``release``. See docs/huggingface-release.md.

What gets published, under https://huggingface.co/unionai:

    dataset  unionai/rightsize-bench                 configs: pods (default), jobs, flyte_tasks,
                                                     features, counterfactual, predictions, leaderboard
    model    unionai/rightsize-decision-qwen3-1.7b   supervised pointer model; seeds 2, 3 on branches
    model    unionai/rightsize-rl-qwen3-1.7b         the same, trained further with GRPO
    model    unionai/rightsize-trees                 gradient-boosted quantile trees (skops)
    model    unionai/rightsize-decision-qwen3-1.7b-{job,flyte}-only   the training-data ablations
    collection  "RightSize Bench" grouping all of the above

``stage_*`` build a repository's files in a ``Dir`` and push nothing;
``push`` uploads a staged ``Dir`` with the write token in the
``HF_UNIONAI_WRITE_TOKEN`` secret. Run ``release`` with ``push=False`` first,
read what it staged, then run it again with ``push=True`` (the staging tasks
cache-hit).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections import defaultdict

import flyte
import flyte.io
import numpy as np
import pandas as pd

from ..bench.common import read_rows
from . import arms as all_arms
from . import counterfactual, dataset, features, grid, hub_cards, hub_pointer, k8s, prompting, scoring
from .envs import allbench_env
from ..bench.envs import bench_gpu_image, bench_image

ORG = os.environ.get("RT_HUB_ORG", "unionai")
DATASET_REPO = f"{ORG}/rightsize-bench"
#: arm -> model repo. The trees repo holds a refit of ``gbt-dist`` (its arm keeps no fitted state).
MODEL_REPOS = {
    "jev-pointer": f"{ORG}/rightsize-decision-qwen3-1.7b",
    "rl-pointer": f"{ORG}/rightsize-rl-qwen3-1.7b",
    "gbt-dist": f"{ORG}/rightsize-trees",
    "jev-pointer-k8s": f"{ORG}/rightsize-decision-qwen3-1.7b-job-only",
    "jev-pointer-flyte": f"{ORG}/rightsize-decision-qwen3-1.7b-flyte-only",
}
COLLECTION_TITLE = "RightSize Bench"
#: The write token's secret: a fine-grained token with write access to the org's repos.
WRITE_TOKEN_SECRET = "HF_UNIONAI_WRITE_TOKEN"

#: Corpus split -> Hub split.
HUB_SPLITS = {"train": "train", "val": "validation", "test": "test", "ood": "unseen"}
PLATFORM_DIR = {"flyte": "flyte", "k8s": "job"}
#: The approaches the paper reports, in its order (``paper/scripts/tables.py``).
REPORTED = ("static", "rules", "rules-sized", "zeroshot-27b", "zeroshot-27b-json", "gbt-dist", "gbt-dist-k8s",
            "gbt-dist-flyte", "jev-pointer", "jev-pointer-argmax", "jev-pointer-k8s", "jev-pointer-k8s-argmax",
            "jev-pointer-flyte", "jev-pointer-flyte-argmax", "jev-pointer-flyte-adapt", "rl-pointer",
            "rl-pointer-argmax", "pool-jev-pointer", "pool-jev-pointer-argmax", "pool-jev-pointer-k8s",
            "pool-jev-pointer-flyte", "pool-jev-pointer-flyte-adapt", "pool-rl-pointer", "pool-rl-pointer-argmax",
            "gate-rl-pointer", "gate-knn-raw-q95", "gate-knn-numeric-q95", "gate-js-q50")

hub_image = bench_image.with_pip_packages("huggingface_hub>=1.1,<2", "skops>=0.13", "safetensors>=0.6")
hub_gpu_image = bench_gpu_image.with_pip_packages("huggingface_hub>=1.1,<2", "safetensors>=0.6")
#: interruptible: the on-demand CPU pools sit at their limits on demo.hosted (CLAUDE.md); spot admits.
hub_env = flyte.TaskEnvironment(
    name="rt-hub",
    image=hub_image,
    resources=flyte.Resources(cpu=4, memory="24Gi", disk="60Gi"),
    interruptible=True,
)
#: CPU only: it loads each decision model once to check that the published inference code reproduces its predictions.
hub_model_env = flyte.TaskEnvironment(
    name="rt-hub-model",
    image=hub_gpu_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", disk="40Gi"),  # Qwen3-1.7B in fp32 is ~7 GiB
    secrets=[flyte.Secret(key="HUGGINGFACE_TOKEN", as_env_var="HF_TOKEN")],  # read-only: the base model
    interruptible=True,
)
hub_push_env = flyte.TaskEnvironment(
    name="rt-hub-push",
    image=hub_image,
    resources=flyte.Resources(cpu=2, memory="4Gi", disk="60Gi"),
    secrets=[flyte.Secret(key=WRITE_TOKEN_SECRET, as_env_var="HF_TOKEN")],
    interruptible=True,  # a reclaimed upload retries (retries=2); upload_folder is idempotent per commit
)
hub_release_env = flyte.TaskEnvironment(
    name="rt-hub-release",
    image=hub_image,
    resources=flyte.Resources(cpu=2, memory="4Gi", disk="60Gi"),  # it reads every arm's meta.json
    interruptible=True,
    depends_on=[hub_env, hub_model_env, hub_push_env, allbench_env],
)


# ── the dataset's tables (pure) ─────────────────────────────────────────


def option(axis: int, idx) -> str | None:
    opts = prompting.OPTIONS[axis]
    return opts[int(idx)] if idx is not None and 0 <= int(idx) < len(opts) else None


def pod_record(row: dict) -> dict:
    """One pod as the dataset publishes it. Column names follow the corpus, so
    ``scoring.score(rows, preds)`` runs on the published rows unchanged."""
    msgs = prompting.messages(row)
    platform = dataset.platform_of(row)
    m, c, g = scoring.cheapest_action(row)
    return {
        # identity
        "run_id": row["run_id"], "task_id": row["task_id"], "twin_of": row.get("twin_of") or "",
        "measured_run_id": row.get("twin_of") or row["run_id"],
        "split": HUB_SPLITS[row["split"]], "platform": PLATFORM_DIR[platform], "staging": row.get("staging") or "",
        "run_idx": int(row.get("run_idx") or 0), "n_history": int(row.get("n_history") or 0),
        # what an approach may read
        "task_code": row["task_code"], "manifest": row.get("manifest") or "", "inputs_desc": row["inputs_desc"],
        "input_bytes_json": row.get("input_bytes_json") or "{}", "history_json": row.get("history_json") or "[]",
        "prompt_system": msgs[0]["content"], "prompt_user": msgs[1]["content"],
        # labels: grid cells (-1 where the axis is not scored) and their option strings
        "label_mem": int(row["label_mem"]), "label_cpu": int(row["label_cpu"]), "label_gpu": int(row["label_gpu"]),
        "label_memory": option(0, row["label_mem"]), "label_cpu_cores": option(1, row["label_cpu"]),
        "label_gpu_option": option(2, row["label_gpu"]),
        # the cheapest request under the cost model, and its cost (the regret denominator)
        "cheapest_mem": m, "cheapest_cpu": c, "cheapest_gpu": g,
        "cheapest_cost": scoring.min_cost(row), "label_cost": scoring.oracle_cost(row),
        # the cost simulator's inputs (measured; never shown to an approach)
        "source": row["source"], "family": row["family"], "driver": row.get("driver") or "",
        "unseen_family": row["split"] == "ood",
        "need_mib": _f(row.get("need_mib")), "t_peak_frac": _f(row.get("t_peak_frac")),
        "need_cores": _f(row.get("need_cores")), "mean_cores": _f(row.get("mean_cores")),
        "cpu_series_json": row.get("cpu_series_json") or "[]", "peak_gpu_gib": _f(row.get("peak_gpu_gib")),
        "sharded": bool(row.get("sharded")), "gpu_imbalance": _f(row.get("gpu_imbalance")),
        "wall_s": _f(row.get("wall_s")),
    }


def _f(v) -> float | None:
    if v is None:
        return None
    v = float(v)
    return None if v != v else v


def features_record(row: dict) -> dict:
    f = features.row_features(row)
    return {"run_id": row["run_id"], "split": HUB_SPLITS[row["split"]],
            "platform": PLATFORM_DIR[dataset.platform_of(row)],
            **{n: float(f.get(n, 0.0)) for n in features.feature_names()}}


def counterfactual_rows(rows: list[dict], n_per_split: int = 600) -> list[dict]:
    """The staging counterfactual's renditions (``counterfactual``): each sampled
    first run as a Job under all four stagings."""
    picked = counterfactual.sample_runs(rows, n_per_split)
    out = []
    for r in counterfactual.renditions(rows, picked):
        rec = pod_record(r)
        rec["counterfactual_of"] = r["twin_of"]
        out.append(rec)
    return out


def predictions_table(rows: list[dict], picks: dict[int, dict[str, dict[str, tuple]]]) -> pd.DataFrame:
    """Every reported approach's request per evaluated pod, per seed, with its cost.

    ``picks[seed][variant][run_id] = (mem_idx, cpu_idx, gpu_idx)``."""
    by_id = {r["run_id"]: r for r in rows}
    recs = []
    for seed, variants in sorted(picks.items()):
        for v in REPORTED:
            for rid, p in (variants.get(v) or {}).items():
                r = by_id.get(rid)
                if r is None:
                    continue
                m, c, g = (int(x) for x in p[:3])
                o = scoring.cost(r, m, c, g)
                mc = scoring.min_cost(r)
                recs.append({"seed": seed, "approach": v, "run_id": rid, "split": HUB_SPLITS[r["split"]],
                             "platform": PLATFORM_DIR[dataset.platform_of(r)], "source": r["source"],
                             "mem_idx": m, "cpu_idx": c, "gpu_idx": g,
                             "memory": option(0, m), "cpu": option(1, c), "gpu": option(2, g),
                             "cost": o.cost, "cheapest_cost": mc, "label_cost": scoring.oracle_cost(r),
                             "regret": o.cost / max(mc, 1e-12) - 1.0,
                             "first_attempt_ok": bool(o.first_ok), "attempts": int(o.attempts)})
    return pd.DataFrame(recs)


def leaderboard_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Mean regret and dollar regret per seed, approach, split and pod type (against the cheapest request)."""
    g = preds.groupby(["seed", "approach", "split", "platform"])
    out = g.agg(n=("regret", "size"), regret=("regret", "mean"), cost=("cost", "sum"),
                cheapest=("cheapest_cost", "sum"), first_attempt_ok=("first_attempt_ok", "mean")).reset_index()
    out["dollar_regret"] = out["cost"] / out["cheapest"] - 1.0
    return out.drop(columns=["cost", "cheapest"])


def write_split_files(df: pd.DataFrame, root: str, sub: str, by_platform: bool = True) -> list[str]:
    """``data/<sub>/[<platform>/]<split>-00000-of-00001.parquet``."""
    written = []
    keys = ["platform", "split"] if by_platform else ["split"]
    for key, part in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        d = os.path.join(root, "data", sub, *key[:-1])
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{key[-1]}-00000-of-00001.parquet")
        part.reset_index(drop=True).to_parquet(p, index=False, row_group_size=2000)
        written.append(os.path.relpath(p, root))
    return written


async def _arm(d: flyte.io.Dir) -> tuple[str, pd.DataFrame, dict]:
    local = await d.download()
    with open(os.path.join(local, "meta.json")) as f:
        meta = json.load(f)
    return local, pd.read_parquet(os.path.join(local, "predictions.parquet")), meta


def _picks_of(preds: pd.DataFrame) -> dict[str, dict[str, tuple]]:
    out: dict[str, dict[str, tuple]] = defaultdict(dict)
    for v, rid, m, c, g in preds[["variant", "run_id", "mem_idx", "cpu_idx", "gpu_idx"]].itertuples(index=False):
        out[v][rid] = (int(m), int(c), int(g))
    return out


# ── staging tasks ───────────────────────────────────────────────────────


@hub_env.task(cache="auto", report=False, timeout=flyte.Timeout(max_runtime=3 * 3600))
async def stage_dataset(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir], arm_seeds: list[int],
                        release: str) -> flyte.io.Dir:
    """The dataset repo: every usable pod, its tree features, the counterfactual
    renditions, every reported approach's request and cost per seed, the
    leaderboard, each approach's tuned settings, and the card."""
    rows = [r for r in await read_rows(corpus) if dataset.usable(r)]
    root = tempfile.mkdtemp(prefix="rt-hub-ds-")
    files: dict[str, list[str]] = {}
    pods = pd.DataFrame([pod_record(r) for r in rows])
    files["pods"] = write_split_files(pods, root, "pods")
    files["features"] = write_split_files(pd.DataFrame([features_record(r) for r in rows]), root, "features")
    cf = pd.DataFrame(counterfactual_rows(rows))
    files["counterfactual"] = write_split_files(cf, root, "counterfactual", by_platform=False)

    picks: dict[int, dict[str, dict[str, tuple]]] = defaultdict(dict)
    os.makedirs(os.path.join(root, "approaches"), exist_ok=True)
    for d, seed in zip(arm_dirs, arm_seeds):
        _local, preds, meta = await _arm(d)
        picks[seed].update(_picks_of(preds))
        with open(os.path.join(root, "approaches", f"seed{seed}-{meta.get('arm', 'arm')}.json"), "w") as f:
            json.dump(meta, f, indent=2, default=float)
    eval_rows = [r for r in rows if r["split"] in ("val", "test", "ood")]
    preds = predictions_table(eval_rows, picks)
    files["predictions"] = write_split_files(preds, root, "predictions", by_platform=False)
    board = leaderboard_table(preds)
    os.makedirs(os.path.join(root, "data", "leaderboard"), exist_ok=True)
    board.to_parquet(os.path.join(root, "data", "leaderboard", "results-00000-of-00001.parquet"), index=False)

    fam = pods["source"] + ":" + pods["family"]  # duckdb_sql is a family in two corpora
    unseen = sorted(fam[pods["unseen_family"]].unique())
    stats = {"release": release, "pods": len(pods), "runs": int((pods["platform"] == "flyte").sum()),
             "families": int(fam.nunique()),
             "unseen_families": [f"{f.split(':')[1]} ({f.split(':')[0]})" for f in unseen],
             "by_split": {f"{p}/{s}": int(n) for (p, s), n in pods.groupby(["platform", "split"]).size().items()},
             "counterfactual_pods": len(cf), "prediction_rows": len(preds), "seeds": sorted(picks),
             "files": files}
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(hub_cards.dataset_card(stats, board))
    with open(os.path.join(root, "release.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return await flyte.io.Dir.from_local(root)


def _metrics(rows: list[dict], preds: pd.DataFrame, variants: tuple[str, ...]) -> dict:
    """Regret against the cheapest request, by variant, split and pod type."""
    by_id = {r["run_id"]: r for r in rows}
    out: dict = {}
    for v, part in preds[preds["variant"].isin(variants)].groupby("variant"):
        acc: dict = defaultdict(list)
        for rid, m, c, g in part[["run_id", "mem_idx", "cpu_idx", "gpu_idx"]].itertuples(index=False):
            r = by_id.get(rid)
            if r is None or r["split"] not in ("test", "ood"):
                continue
            acc[(HUB_SPLITS[r["split"]], PLATFORM_DIR[dataset.platform_of(r)])].append(
                scoring.cost(r, int(m), int(c), int(g)).cost / max(scoring.min_cost(r), 1e-12) - 1.0)
        out[v] = {f"{s}/{p}": {"regret": float(np.mean(x)), "n": len(x)} for (s, p), x in sorted(acc.items())}
    return out


@hub_env.task(cache="auto", report=False, timeout=flyte.Timeout(max_runtime=3 * 3600))
async def stage_trees(corpus: flyte.io.File, published: flyte.io.Dir, release: str) -> flyte.io.Dir:
    """The trees repo: ``gbt-dist`` refit on the same rows (it is deterministic),
    saved with skops, and checked request-for-request against the published arm."""
    import skops.io

    rows = await read_rows(corpus)
    splits = dataset.by_split(rows)
    fit, predict = all_arms.ARMS["gbt-dist"]
    state = fit(splits["train"], splits["val"])
    eval_rows = [r for s in ("val", "test", "ood") for r in splits[s]]
    refit = predict(state, eval_rows)
    _local, preds, meta = await _arm(published)
    pub = _picks_of(preds)["gbt-dist"]
    same = sum(tuple(int(refit[k][x]) for x in ("mem_idx", "cpu_idx", "gpu_idx")) == pub.get(k) for k in refit)
    root = tempfile.mkdtemp(prefix="rt-hub-trees-")
    skops.io.dump({k: state[k] for k in ("clf", "mem", "cpu", "gpu")}, os.path.join(root, "model.skops"))
    cfg = {"release": release, "approach": "gbt-dist", "levels": list(state["levels"]),
           "gpu_threshold": float(state["threshold"]), "features": features.feature_names(),
           "targets": {"mem": "log2(memory need, MiB)", "cpu": "log2(busiest-15 s cores, floor 0.05)",
                       "gpu": "log2(peak GPU memory, GiB, floor 0.1)"},
           "options": {k: list(o) for k, o in zip(prompting.KEYS, prompting.OPTIONS)},
           "n_train": len(splits["train"]), "refit_matches_published": same / max(1, len(refit))}
    with open(os.path.join(root, "rightsize_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    metrics = _metrics(eval_rows, preds, ("gbt-dist",))
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(hub_cards.trees_card(cfg, metrics))
    if same != len(refit):
        print(f"WARNING: the refit matches {same}/{len(refit)} published requests")
    return await flyte.io.Dir.from_local(root)


@hub_model_env.task(cache="auto", report=False, timeout=flyte.Timeout(max_runtime=3 * 3600))
async def stage_pointer(corpus: flyte.io.File, arm_dir: flyte.io.Dir, release: str, seed: int,
                        verify_n: int = 6) -> flyte.io.Dir:
    """A decision-model repo: the LoRA adapter, the pointer heads as safetensors,
    the calibration, the standalone inference file, and the card. The inference
    file is run on ``verify_n`` validation prompts and compared with the arm's
    own argmax requests (on CPU in fp32, so a near-tie may flip)."""
    import shutil

    import torch
    from safetensors.torch import save_file

    local, preds, meta = await _arm(arm_dir)
    arm = meta["arm"]
    root = tempfile.mkdtemp(prefix="rt-hub-model-")
    src = os.path.join(local, "model")
    for name in os.listdir(src):
        if name.startswith("adapter_"):
            shutil.copy(os.path.join(src, name), root)
    state = torch.load(os.path.join(src, "decision_head.pt"), map_location="cpu")
    save_file({k: v.contiguous() for k, v in state.items()}, os.path.join(root, hub_pointer.HEAD_FILE))
    shutil.copy(hub_pointer.__file__, os.path.join(root, "rightsize_pointer.py"))
    c = meta["config"]
    cfg = {"release": release, "approach": arm, "seed": seed, "base_model": c["model_name"], "max_len": c["max_len"],
           "head_dim": 256, "keys": list(prompting.KEYS), "options": [list(o) for o in prompting.OPTIONS],
           "temperatures": [float(t) for t in meta["temperatures"]], "gpu_threshold": float(meta["gpu_threshold"]),
           "train_platforms": meta.get("train_platforms", "all"), "n_train": meta.get("n_train"),
           "base_arm": meta.get("base_arm"), "rl": meta.get("rl"), "lora": {"r": c.get("lora_r"), "alpha": c.get("lora_alpha")}}
    with open(os.path.join(root, hub_pointer.CONFIG_FILE), "w") as f:
        json.dump(cfg, f, indent=2)

    rows = await read_rows(corpus)
    by_id = {r["run_id"]: r for r in rows}
    argmax = _picks_of(preds).get(f"{arm}-argmax", {})
    val_ids = sorted(rid for rid in argmax if by_id.get(rid, {}).get("split") == "val")[:: max(1, len(argmax) // 2000)]
    agree, checked = 0, 0
    if verify_n:
        model = hub_pointer.RightSizePointer.from_pretrained(root, device="cpu")
        for rid in val_ids[:verify_n]:
            req = model.request(prompting.messages(by_id[rid]))
            want = argmax[rid]
            got = tuple(prompting.OPTIONS[a].index(req[k]) for a, k in enumerate(prompting.KEYS))
            agree += got == want
            checked += 1
    cfg_card = {**cfg, "verify": {"checked": checked, "agree": agree}}
    metrics = _metrics(rows, preds, (arm, f"{arm}-argmax"))
    with open(os.path.join(root, "README.md"), "w") as f:
        f.write(hub_cards.pointer_card(cfg_card, metrics))
    return await flyte.io.Dir.from_local(root)


# ── pushing ─────────────────────────────────────────────────────────────


@hub_push_env.task(report=False, retries=2, timeout=flyte.Timeout(max_runtime=2 * 3600))
async def push(staged: flyte.io.Dir, repo_id: str, repo_type: str, revision: str = "main", private: bool = True,
               message: str = "", tag: str = "") -> str:
    """Upload a staged repo. Creates the repo (private by default) and the branch
    if needed, and replaces the files a previous push of the same layout wrote.
    ``tag`` (the release, e.g. ``k8s-2``) marks the commit, so a later release
    can move ``main`` while this one stays loadable with ``revision="k8s-2"``."""
    from huggingface_hub import HfApi

    local = await staged.download()
    api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    if revision != "main":
        api.create_branch(repo_id, repo_type=repo_type, branch=revision, exist_ok=True)
    info = api.upload_folder(repo_id=repo_id, repo_type=repo_type, folder_path=local, revision=revision,
                             commit_message=message or f"RightSize Bench {revision}",
                             delete_patterns=["data/**", "approaches/**"] if repo_type == "dataset" else None)
    if tag:
        name = tag if revision == "main" else f"{tag}-{revision}"
        api.create_tag(repo_id, repo_type=repo_type, tag=name, revision=info.oid, exist_ok=True)
    return str(info.commit_url)


@hub_push_env.task(report=False, retries=2)
async def make_collection(items: list[str], private: bool = True) -> str:
    """The "RightSize Bench" collection: the dataset first, then the models. ``items`` are ``<type>:<repo_id>``."""
    from huggingface_hub import HfApi

    api = HfApi()
    col = api.create_collection(title=COLLECTION_TITLE, namespace=ORG, private=private, exists_ok=True,
                                description=hub_cards.COLLECTION_DESCRIPTION)
    for it in items:
        kind, repo = it.split(":", 1)
        api.add_collection_item(col.slug, item_id=repo, item_type=kind, note=hub_cards.collection_note(repo),
                                exists_ok=True)
    return f"https://huggingface.co/collections/{col.slug}"


# ── the release ─────────────────────────────────────────────────────────


@hub_release_env.task(report=False)
async def release(corpus: flyte.io.File, arm_dirs: list[flyte.io.Dir], arm_seeds: list[int], release: str = "k8s-2",
                  push_to_hub: bool = False, private: bool = True, ablations: bool = True) -> str:
    """Stage every repo of a release, and push them when ``push_to_hub``.

    ``arm_dirs`` are every ``fit_all_arm``, ``rl_all_policy``, ``pool_all_arms``
    and ``gate_all_arms`` output of the release's trainings; ``arm_seeds`` gives
    each one's seed. Seed 1 goes to each model repo's ``main``; other seeds to
    branch ``seed-<k>``. Returns a JSON summary: staged paths and, when pushed, commit URLs."""
    metas = await asyncio.gather(*(_arm(d) for d in arm_dirs))
    by_arm = {(meta.get("arm"), seed): d for d, seed, (_l, _p, meta) in zip(arm_dirs, arm_seeds, metas)}
    jobs: dict[str, tuple] = {"dataset": ("dataset", DATASET_REPO, "main",
                                          stage_dataset(corpus, arm_dirs, arm_seeds, release))}
    if ("gbt-dist", 1) in by_arm:
        jobs["trees"] = ("model", MODEL_REPOS["gbt-dist"], "main", stage_trees(corpus, by_arm[("gbt-dist", 1)], release))
    for arm, repo in MODEL_REPOS.items():
        if arm == "gbt-dist" or (not ablations and arm in ("jev-pointer-k8s", "jev-pointer-flyte")):
            continue
        for seed in sorted({s for a, s in by_arm if a == arm}):
            rev = "main" if seed == 1 else f"seed-{seed}"
            jobs[f"{arm}@{rev}"] = ("model", repo, rev, stage_pointer(corpus, by_arm[(arm, seed)], release, seed))
    staged = dict(zip(jobs, await asyncio.gather(*(j[3] for j in jobs.values()))))
    out = {k: {"repo": j[1], "type": j[0], "revision": j[2], "staged": staged[k].path} for k, j in jobs.items()}
    if push_to_hub:
        # main before branches: a branch is cut from the repo's main
        order = sorted(jobs, key=lambda k: jobs[k][2] != "main")
        mains = [k for k in order if jobs[k][2] == "main"]
        urls = await asyncio.gather(*(push(staged[k], jobs[k][1], jobs[k][0], "main", private,
                                           f"RightSize Bench {release}", release) for k in mains))
        rest = [k for k in order if k not in mains]
        urls += await asyncio.gather(*(push(staged[k], jobs[k][1], jobs[k][0], jobs[k][2], private,
                                            f"RightSize Bench {release}, {jobs[k][2]}", release) for k in rest))
        for k, u in zip(mains + rest, urls):
            out[k]["commit"] = u
        items = [f"dataset:{DATASET_REPO}"] + [f"model:{r}" for r in dict.fromkeys(
            j[1] for j in jobs.values() if j[0] == "model")]
        out["collection"] = await make_collection(items, private)
    return json.dumps(out, indent=2)


@hub_env.task(report=False)
async def peek(staged: flyte.io.Dir, files: list[str]) -> str:
    """Some of a staged repo's text files, and its file listing, inline (JSON): for reviewing a dry run from the laptop."""
    local = await staged.download()
    out = {"listing": sorted(os.path.relpath(os.path.join(d, f), local) for d, _, fs in os.walk(local) for f in fs)}
    for f in files:
        p = os.path.join(local, f)
        out[f] = open(p).read()[:200_000] if os.path.exists(p) else None
    return json.dumps(out)
