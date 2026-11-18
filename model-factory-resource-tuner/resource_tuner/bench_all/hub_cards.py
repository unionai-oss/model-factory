"""Hugging Face cards for RightSize Bench's dataset and models (``hub``).

Pure text: each function takes what the staging task measured and returns
a README.md with the Hub's YAML metadata block. Numbers in the cards come
from the staged files themselves, so a card cannot disagree with its repo.
"""

from __future__ import annotations

import json

import pandas as pd

GITHUB = "https://github.com/unionai-oss/model-factory"
CODE = f"{GITHUB}/tree/main/model-factory-resource-tuner"
PAPER = f"{CODE}/paper"
DATASET = "unionai/rightsize-bench"
CITATION = """@misc{bantilan2026rightsize,
  title  = {RightSize Bench: Learning Kubernetes Resource Requests from Code},
  author = {Bantilan, Niels},
  year   = {2026},
  note   = {Union.ai},
  url    = {https://huggingface.co/datasets/unionai/rightsize-bench}
}"""
COLLECTION_DESCRIPTION = ("A benchmark for sizing Kubernetes batch pods (memory, CPU, GPU) from their code, "
                          "scored in dollars, and the baseline models trained on it.")
NOTES = {
    "unionai/rightsize-bench": "64k pods: every measured run as a raw Job and as a Flyte task",
    "unionai/rightsize-decision-qwen3-1.7b": "Qwen3-1.7B + LoRA, pointer heads over the prompt's options",
    "unionai/rightsize-rl-qwen3-1.7b": "The decision model trained further with GRPO against the cost model",
    "unionai/rightsize-trees": "Gradient-boosted quantile trees over 209 scheduler-visible features",
    "unionai/rightsize-decision-qwen3-1.7b-job-only": "Ablation: trained on Job pods only",
    "unionai/rightsize-decision-qwen3-1.7b-flyte-only": "Ablation: trained on Flyte pods only",
}
#: Display names for the approaches (the paper's).
NAMES = {
    "static": "static", "rules": "rules", "rules-sized": "rules-sized", "zeroshot-27b": "zero-shot 27B",
    "zeroshot-27b-json": "zero-shot 27B, JSON", "gbt-dist": "trees", "jev-pointer": "decision",
    "jev-pointer-argmax": "decision, argmax", "rl-pointer": "RL", "rl-pointer-argmax": "RL, argmax",
    "pool-jev-pointer": "pool × decision", "pool-rl-pointer": "pool × RL", "pool-rl-pointer-argmax": "pool × RL, argmax",
    "gate-rl-pointer": "gate (kNN, q 0.95)",
}


def collection_note(repo: str) -> str:
    return NOTES.get(repo, "")


def _yaml(meta: dict) -> str:
    """A YAML front-matter block. JSON is valid YAML, so nested values are written as JSON."""
    lines = ["---"]
    for k, v in meta.items():
        if isinstance(v, (list, dict)):
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {json.dumps(v) if isinstance(v, str) else v}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _configs(files: dict[str, list[str]]) -> list[dict]:
    split_order = ["train", "validation", "test", "unseen", "results"]

    def splits(sub: str, platform: str = "*") -> list[dict]:
        names = sorted({p.rsplit("/", 1)[-1].split("-")[0] for p in files.get(sub, [])}, key=split_order.index)
        mid = f"{platform}/" if platform else ""
        return [{"split": s, "path": f"data/{sub}/{mid}{s}-*.parquet"} for s in names]

    return [
        {"config_name": "pods", "default": True, "data_files": splits("pods")},
        {"config_name": "jobs", "data_files": splits("pods", "job")},
        {"config_name": "flyte_tasks", "data_files": splits("pods", "flyte")},
        {"config_name": "features", "data_files": splits("features")},
        {"config_name": "counterfactual", "data_files": splits("counterfactual", "")},
        {"config_name": "predictions", "data_files": splits("predictions", "")},
        {"config_name": "leaderboard", "data_files": [{"split": "results", "path": "data/leaderboard/results-*.parquet"}]},
    ]


def _board_md(board: pd.DataFrame, seed: int = 1) -> str:
    b = board[(board["seed"] == seed) & (board["approach"].isin(NAMES))]
    piv = b.pivot_table(index="approach", columns=["platform", "split"], values="regret")
    order = [a for a in NAMES if a in piv.index]
    lines = ["| approach | Job, test | Job, unseen | Flyte, test | Flyte, unseen |", "|---|---:|---:|---:|---:|"]
    for a in order:
        cells = [piv.get((p, s), pd.Series()).get(a) for p, s in
                 (("job", "test"), ("job", "unseen"), ("flyte", "test"), ("flyte", "unseen"))]
        lines.append(f"| {NAMES[a]} | " + " | ".join("" if c is None or c != c else f"{100 * c:.1f}" for c in cells) + " |")
    return "\n".join(lines)


def dataset_card(stats: dict, board: pd.DataFrame) -> str:
    meta = {
        "pretty_name": "RightSize Bench",
        "license": "cc-by-4.0",
        "language": ["en"],
        "task_categories": ["tabular-classification", "text-classification"],
        "tags": ["kubernetes", "resource-allocation", "scheduling", "cost-aware", "code", "flyte", "benchmark"],
        "size_categories": ["10K<n<100K"],
        "configs": _configs(stats["files"]),
    }
    s = stats["by_split"]
    rows = "\n".join(f"| {p} | {s.get(f'{p}/train', 0):,} | {s.get(f'{p}/validation', 0):,} | "
                     f"{s.get(f'{p}/test', 0):,} | {s.get(f'{p}/unseen', 0):,} |" for p in ("job", "flyte"))
    return _yaml(meta) + f"""
# RightSize Bench

Before a batch pod runs on Kubernetes, someone has to decide how much memory and CPU it gets, and whether it
gets a GPU. RightSize Bench measures how well an approach makes that decision from what a scheduler can read
before the pod starts: the task's code, the pod spec, the size and format of its inputs, and what earlier runs
of the same task used. A request is scored by what it costs to get the pod to finish, failed attempts and
throttling included, against the cheapest request that would have run it.

This is release **{stats['release']}**: {stats['runs']:,} measured runs of {stats['families']} families of
generated Python tasks (dataframes, SQL, linear algebra, classical ML, PyTorch training and inference, sharded
language models). Every run appears twice, as a raw `batch/v1` Job with its own manifest and as a Flyte task,
so there are {stats['pods']:,} pods. Code, paper and the factory that builds the benchmark: [{GITHUB}]({CODE}).

## Configurations

| config | rows | what |
|---|---|---|
| `pods` (default) | {stats['pods']:,} | every pod: what an approach may read, its labels, and the cost simulator's inputs |
| `jobs`, `flyte_tasks` | | the same rows, one pod type each |
| `features` | {stats['pods']:,} | the 209 scheduler-visible features the trees read, by `run_id` |
| `counterfactual` | {stats['counterfactual_pods']:,} | first runs rendered as Jobs under all four stagings, to test whether an approach reads the manifest |
| `predictions` | {stats['prediction_rows']:,} | every reported approach's request per evaluated pod, for seeds {', '.join(map(str, stats['seeds']))}, with its cost |
| `leaderboard` | | regret by seed, approach, split and pod type, computed from `predictions` |

```python
from datasets import load_dataset

jobs = load_dataset("{DATASET}", "jobs", split="test")
row = jobs[0]
print(row["prompt_user"])               # code, manifest, inputs, history, options
print(row["label_memory"], row["label_cpu_cores"], row["label_gpu_option"])
```

## Splits

Splits are by task, never by run: a hash of the task id sends 70% of each family's tasks to `train`, 10% to
`validation` and 20% to `test`. Seven families are held out entirely, in `unseen`
({', '.join(stats['unseen_families'])}). Tune on `validation`, which has familiar families only.

| pods | train | validation | test | unseen |
|---|---:|---:|---:|---:|
{rows}

A run's Job pod and Flyte pod share a workload and a split; resample them together (by `task_id` without its
`@k8s` suffix) when you compute intervals.

## What an approach may read

`task_code`, `manifest` (Job pods), `inputs_desc`, `input_bytes_json`, `history_json`, and the rendered
`prompt_system` / `prompt_user`, which contain the same information. Everything else in a row is a label,
the cost simulator's input, or metadata for slicing results (`family`, `driver`, `source`, `unseen_family`).
The code and the manifest name the task after its family (for example `pandas-join`); the paper's name test
re-scores with neutral names.

## Labels and the cost model

`label_mem`, `label_cpu` and `label_gpu` are indices into the request grid (17 memory cells from 256Mi to 64Gi,
16 CPU cells from 0.25 to 8 cores, and no GPU or 17 GPU-memory cells that land on the cheapest of seven AWS GPU
shapes); `-1` marks an axis that is not scored (memory and CPU on a GPU node). The string columns give the same
labels as option text. Memory labels are the measured peak plus what the pod type adds (the Flyte runtime, the
boto3 client, bytes staged on a memory-backed volume), with 5% headroom. CPU labels are the busiest 15 seconds.
GPU labels are the cheapest machine the run's peak GPU memory fits on.

The simulator (`resource_tuner/bench_all/scoring.py` in the repo) replays each run under a request: an OOM
kills the container at its peak and retries with double the memory, a CPU limit throttles the measured
per-second series, a missing GPU fails at start. `cheapest_cost` is the lowest cost over the grid and is the
regret denominator; `cheapest_mem/cpu/gpu` is that request. `label_cost` is the cost of requesting the label.

```python
# score your own requests with the benchmark's simulator
#   git clone {GITHUB} && cd model-factory/model-factory-resource-tuner
from resource_tuner.bench_all import scoring
rows = jobs.to_list()
preds = {{r["run_id"]: {{"mem_idx": 8, "cpu_idx": 3, "gpu_idx": 0}} for r in rows}}
print(scoring.score(rows, preds, oracle="cheapest")["regret"])
```

## Leaderboard

Mean regret (%) against the cheapest request, seed 1. Lower is better. Every seed and the dollar-weighted
regret are in the `leaderboard` config; intervals and the full analysis are in the paper.

{_board_md(board)}

## How it was made

Each family is a Python generator that writes a task and draws hidden settings (thread counts, dtypes,
batch sizes) that decide how input bytes turn into resource use. Runs were measured in Kubernetes pods on
`demo.hosted.unionai.cloud`: peak memory from `ru_maxrss`, CPU sampled every second, GPU memory from NVML every
half second while moving up a ladder of GPU machines. The inputs are synthetic (random numbers, strings and
bytes); there is no personal data. The Job labels were checked in about 480 real pods. The whole benchmark is a
Union factory: re-running it rebuilds only what changed.

## Known biases

- The workloads are generated and short (seconds to minutes). Production tasks are longer and messier.
- Many test tasks share their exact code with a training task and differ only in input size. `unseen` is the
  split for new code.
- Memory labels are about one cell generous for a third of runs (`ru_maxrss` includes the measuring parent's
  high-water mark; mapped files count). Process pools are labelled by their largest process.
- GPU runs that ran out of memory on the largest machine they reached are dropped, and many of those were
  multi-GPU runs that never got capacity. Fits on machines a run was not measured on are modelled.
- Prices are on-demand `us-east-1` list prices.

## Citation

```bibtex
{CITATION}
```
"""


def _model_index(name: str, metrics: dict) -> list[dict]:
    results = []
    for variant, by in metrics.items():
        for key, m in by.items():
            split, platform = key.split("/")
            results.append({
                "task": {"type": "tabular-classification", "name": "Kubernetes resource request sizing"},
                "dataset": {"name": "RightSize Bench", "type": DATASET,
                            "config": "jobs" if platform == "job" else "flyte_tasks", "split": split},
                "metrics": [{"type": "regret", "name": f"mean regret, {variant}", "value": round(100 * m["regret"], 2)}],
            })
    return [{"name": name, "results": results}]


def _metrics_md(metrics: dict) -> str:
    lines = ["| read-out | Job, test | Job, unseen | Flyte, test | Flyte, unseen |", "|---|---:|---:|---:|---:|"]
    for v, by in metrics.items():
        cells = [by.get(k, {}).get("regret") for k in ("test/job", "unseen/job", "test/flyte", "unseen/flyte")]
        lines.append(f"| `{v}` | " + " | ".join("" if c is None else f"{100 * c:.1f}" for c in cells) + " |")
    return "\n".join(lines)


_POINTER_ABOUT = {
    "jev-pointer": "the supervised decision model: trained with cross-entropy on the label cells",
    "rl-pointer": ("the decision model trained further with group-relative policy optimization (GRPO) against the "
                   "benchmark's cost simulator; a group of 16 joint requests is sampled from one forward pass"),
    "jev-pointer-k8s": "an ablation of the decision model trained on Job pods only",
    "jev-pointer-flyte": "an ablation of the decision model trained on Flyte pods only",
}


def pointer_card(cfg: dict, metrics: dict) -> str:
    arm = cfg["approach"]
    name = f"rightsize-{arm}"
    meta = {
        "license": "apache-2.0",
        "library_name": "peft",
        "base_model": cfg["base_model"],
        "datasets": [DATASET],
        "language": ["en"],
        "pipeline_tag": "text-classification",
        "tags": ["kubernetes", "resource-allocation", "lora", "pointer-network", "rightsize-bench"]
                + (["grpo", "reinforcement-learning"] if arm.startswith("rl") else []),
        "model-index": _model_index(name, metrics),
    }
    v = cfg.get("verify") or {}
    verify = (f"The inference file in this repo reproduced the training run's argmax request on "
              f"{v['agree']} of {v['checked']} validation prompts (fp32 on CPU; the run used bf16 on an L4)."
              if v.get("checked") else "")
    seeds = "This branch is seed {}.".format(cfg["seed"]) if cfg["seed"] != 1 else (
        "`main` is seed 1; branches `seed-2` and `seed-3` hold the other two trainings.")
    return _yaml(meta) + f"""
# RightSize decision model: `{arm}`

This is {_POINTER_ABOUT.get(arm, arm)}, from [RightSize Bench](https://huggingface.co/datasets/{DATASET}),
release {cfg['release']}. Given a batch task's code, its pod spec, its inputs and what earlier runs used, it
picks a memory request, a CPU request and a GPU (or none) for the pod, in one forward pass. {seeds}

## How it works

A Qwen3-1.7B torso ({cfg['base_model']}, language-model head removed) with a LoRA adapter (rank
{cfg['lora']['r']}, alpha {cfg['lora']['alpha']}, every projection) reads the prompt. The prompt ends with the
options for each axis. For each axis a pointer head compares the hidden state at the prompt's last token with
the hidden state at each option's last token (LayerNorm, then 2,048 to 256 query and key projections), so the
model scores the candidates written in its context. Temperatures fitted on the validation split calibrate each
axis: {', '.join(f'{k} {t:.2f}' for k, t in zip(cfg['keys'], cfg['temperatures']))}. Prompts longer than
{cfg['max_len']:,} tokens are cut from the left; no benchmark prompt is that long.

## Use

```python
# pip install torch transformers peft safetensors huggingface_hub datasets
from huggingface_hub import hf_hub_download
import importlib.util, datasets

spec = importlib.util.spec_from_file_location("rightsize_pointer",
                                              hf_hub_download("{{repo_id}}", "rightsize_pointer.py"))
rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)

model = rp.RightSizePointer.from_pretrained("{{repo_id}}")
row = datasets.load_dataset("{DATASET}", "jobs", split="test")[0]
messages = [{{"role": "system", "content": row["prompt_system"]}}, {{"role": "user", "content": row["prompt_user"]}}]
print(model.request(messages))   # e.g. {{"memory": "1.5Gi", "cpu": "1", "gpu": "none"}}
```

`request` is the argmax read-out. The paper's cost-aware read-out, which picks the memory cell with the lowest
expected cost under the calibrated distribution, is `scoring.decode` in [the repo]({CODE}). Write your own
prompts with the same layout (`resource_tuner/bench_all/prompting.py`); the options block and the final answer
line must stay the same, because the heads find the options by their offset from the end.
{verify}

## Results

Mean regret (%) against the cheapest request that runs each pod, on the benchmark's test split (new tasks
from familiar families) and unseen split (seven held-out families). `{arm}` is the cost-aware read-out and
`{arm}-argmax` the argmax read-out.

{_metrics_md(metrics)}

## Training

Train split of RightSize Bench {cfg['release']}, {cfg.get('n_train') or 'the'} rows (at most 5,000 per corpus and
pod type), pod types: {cfg['train_platforms']}. One NVIDIA L4. Seed {cfg['seed']}.
{"GRPO from the supervised model: " + json.dumps(cfg['rl']) + "." if cfg.get('rl') else ""}

## Limitations

The model was trained on generated tasks that run for seconds to minutes, on one cluster's machine shapes and
prices. It is much better on code like its training families than on new kinds of code, where a rule that
scales the nearest earlier run by input size does about as well. It has not been tested on production
workloads. Do not use it to set limits without a fallback for out-of-memory retries.

## Citation

```bibtex
{CITATION}
```
""".replace("{repo_id}", f"unionai/rightsize-{_suffix(arm)}")


def _suffix(arm: str) -> str:
    return {"jev-pointer": "decision-qwen3-1.7b", "rl-pointer": "rl-qwen3-1.7b",
            "jev-pointer-k8s": "decision-qwen3-1.7b-job-only",
            "jev-pointer-flyte": "decision-qwen3-1.7b-flyte-only"}.get(arm, arm)


def trees_card(cfg: dict, metrics: dict) -> str:
    meta = {
        "license": "apache-2.0",
        "library_name": "sklearn",
        "datasets": [DATASET],
        "pipeline_tag": "tabular-classification",
        "tags": ["kubernetes", "resource-allocation", "quantile-regression", "skops", "tabular", "rightsize-bench"],
        "model-index": _model_index("rightsize-trees", metrics),
    }
    match = cfg.get("refit_matches_published", 0)
    return _yaml(meta) + f"""
# RightSize trees: gradient-boosted quantile regression

The strongest classical approach in [RightSize Bench](https://huggingface.co/datasets/{DATASET}), release
{cfg['release']}. For each axis (memory, CPU, GPU memory), 19 `HistGradientBoostingRegressor` models predict the
quantiles {cfg['levels'][0]:.2f} to {cfg['levels'][-1]:.2f} of the log2 need from 209 features a scheduler can
compute before the pod starts, and a classifier predicts whether the task needs a GPU. The quantiles define a
piecewise-linear CDF, and each request cell's probability is the CDF mass between its edges.

This is a refit on the same rows as the benchmark run (the fit is deterministic): it reproduces
{100 * match:.2f}% of the published requests exactly.

## Use

```python
# pip install skops scikit-learn datasets huggingface_hub
import skops.io, datasets, numpy as np
from huggingface_hub import hf_hub_download

path = hf_hub_download("unionai/rightsize-trees", "model.skops")
models = skops.io.load(path, trusted=skops.io.get_untrusted_types(file=path))
feats = datasets.load_dataset("{DATASET}", "features", split="test").to_pandas()
X = feats[models_features := __import__("json").load(open(hf_hub_download("unionai/rightsize-trees",
                                                                           "rightsize_config.json")))["features"]]
X = X.to_numpy(np.float32)
p_gpu = models["clf"].predict_proba(X)[:, 1]                              # P(task needs a GPU)
mem_q = np.stack([m.predict(X) for m in models["mem"]], axis=1)          # log2 MiB at 19 quantiles
```

Review the types `get_untrusted_types` lists before trusting them: they should all be scikit-learn and NumPy.
The features are `resource_tuner.bench_all.features.row_features` in [the repo]({CODE}); the `features`
config of the dataset has them precomputed for every pod. The read-out that turns quantiles into a request
(cost-aware memory, GPU threshold {cfg['gpu_threshold']}) is `resource_tuner.bench_all.arms.decode_all`.

## Results

Mean regret (%) against the cheapest request that runs each pod.

{_metrics_md(metrics)}

The trees are among the best approaches on familiar code and among the worst on the unseen families: they
have nothing to go on for code tokens they never saw in training.

## Citation

```bibtex
{CITATION}
```
"""
