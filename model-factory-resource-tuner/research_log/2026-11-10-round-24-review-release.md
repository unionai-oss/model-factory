# Round 24: a third review, the cheapest-request oracle, and the Hugging Face release

## Context

Round 23 produced the k8s-2 release and an arXiv draft. This round:

1. Runs a third methodology review.
   - Three independent referee agents, one each on evaluation and statistics, dataset and labels, and approaches.
   - Each reads the paper and the code and checks claims against the exports.
2. Fixes what the review found, offline where possible, plus one small audit run on the cluster.
3. Rewrites the paper and the HTML report.
   - Plain technical prose, with the AI tells cut (the humanizer patterns).
   - New figures, built around what the benchmark and the models contribute.
4. Adds a runbook and tasks for publishing the dataset and models to huggingface.co/unionai.

No model was retrained. Every number below comes from round 23's exports (`rt-paper-k2s*`), re-analysed.

## Run ledger

| run | what | outcome |
|---|---|---|
| [rt-paper-k2s1-audit](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s1-audit) | `paper.paper_audit` on rsk-k8s-2's corpus: for each test and unseen run, is its `task_code` identical to a training task's; distinct programs per family; the modules each family imports | succeeded: Findings 2 and 3 |
| [rt-hub-k8s-2-stage-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-hub-k8s-2-stage-1) | first `hub_release.py` dry run | never created a run. The driver sat for about 45 min with its output piped through `tail`, so it was invisible; killed and relaunched unbuffered |
| [rt-hub-k8s-2-stage-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-hub-k8s-2-stage-2) | `hub.release` with `push_to_hub=False` on on-demand pools | aborted: every task sat in `waiting_for_resources` ("Insufficient cpu/memory"; the on-demand pools were at their limits). Fixed by making every hub environment `interruptible` with smaller requests |
| [rt-hub-k8s-2-stage-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-hub-k8s-2-stage-3) | the same dry run on spot: stage the dataset repo, the trees repo (a refit, checked against the published requests), and every decision-model repo, all seeds, with ablations | succeeded (3 h 41 min, set by the trees refit; the trees task ran on its third attempt after spot reclaims): Finding 9 |
| [rt-hub-k8s-2-peek-1-*](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-hub-k8s-2-peek-1-0) | `hub.peek` on each staged folder: listing, cards, configs | succeeded: Finding 9 |
| [rt-hub-k8s-2-peek-trees-0](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-hub-k8s-2-peek-trees-0) | `hub.peek` on the staged trees folder | succeeded: `refit_matches_published` = 1.0 |

## Findings

### 1. The oracle was the label, and the label is not the cheapest request

The review searched every memory x CPU cell on 800 CPU-node rows. In 46.5% of them, some cell beats the label.
The reason is that one CPU cell below the busiest 15 seconds only throttles that window. Against the label,
RL and pool x decision had negative mean regret on RightSizeCPU.

- **The new oracle.** `scoring.candidate_actions` / `cheapest_action` / `min_cost` define regret against the
  cheapest request under the same cost model:
  - a CPU task searches 272 cells (memory x CPU, no GPU);
  - a GPU task searches every GPU cell.
- **What stays the same.** The penalties keep their unit, the label request's cost, so a request's cost does
  not change, only the denominator. `analyze.py` recomputes the cheapest request under every penalty setting.
  `scoring.score(..., oracle="cheapest")` exposes it; the stations' scorecards are unchanged.
- **No request beats it.** `below_cheapest` is 0.000 for every approach.
- **Regret rises.**

| approach | Job test | Job unseen |
|---|---|---|
| pool x decision | 6.6 -> 11.7 | |
| gate | 8.1 -> 13.4 | |
| rules-sized | | 39.2 -> 42.7 |

- **RL's reward.** RL trained against cost / label cost. That denominator is a constant per run, so the best
  request per run is the same whichever denominator is used, and RL does not need retraining.

### 2. Most familiar-family test tasks repeat a training program (rt-paper-k2s1-audit)

- **Seen code.** 4,135 of 5,095 test Job pods (81%) have `task_code` identical to some training task's, and
  differ only in sizes and history.
- **Families with few programs.** `python_grouping` has 3 distinct programs in 400 tasks, `jsonl_records` 4,
  and `pandas_join` 8. `torch_train` has 327 and `gpt_train` 444.
- **Regret on seen versus new programs:**

| approach | seen | new (960 pods) |
|---|---|---|
| decision | 14.0 | 34.3 |
| pool x RL argmax | 12.4 | 13.2 |
| gate | 12.6 | 16.8 |
| pool x decision | 10.3 | 17.9 |
| trees | 15.7 | 21.4 |
| rules-sized | 44.5 | 44.4 |

### 3. "Six of seven held-out families bring a new library" was false

Only DuckDB (two of the seven families) imports a package no training family imports. pyarrow's
`csv`/`compute` submodules are new; the three unseen GPU families use torch, numpy and pandas only. The paper
used the false claim to explain the gate's AUROC of 0.999. Code tokens still separate the families (without
them the AUROC is 0.67), but by the code, not by imports.

### 4. Read-outs are now chosen on validation

Nothing used to choose between argmax and decoded. Validation (Job and Flyte) now chooses, and it chooses the
same way on every seed:

- decoded for the decision models and pool x decision;
- argmax for RL and pool x RL.

So RL's unseen number in the leaderboard is 50.6% (argmax), not the 42.8% of its decoded read-out. The gate's
outer member is RL decoded, chosen in round 22 with the unseen families in view. The paper now says so, and
drops "pre-registered" for "fixed before k8s-2". RL's temperatures sit at the search bound (5.0), so its
decoded read-out is set by that bound.

### 5. Family-clustered intervals on the unseen split

Clustering by the 7 families widens unseen intervals four to ten times:

- rules-sized [40.6, 44.7] -> [27.8, 60.4];
- RL argmax [48.3, 53.0] -> [22.0, 83.5].

Of the leaderboard approaches, only the static request is clearly worse than rules-sized on unseen families.

### 6. Retry penalties: the claim holds by task, not by family

At penalty 5 and 20, rules-sized has the lowest unseen regret (116.1% and 391.4%, against 140.1% and 505.1% for
RL decoded and the gate).

| | rules-sized minus gate, penalty 20 (pp) |
|---|---|
| clustered by task | -113.7 [-129.8, -97.4] |
| clustered by family | [-263.7, +15.9] |

Every approach was tuned at penalty 1 (decoders, theta, w), so this re-prices fixed requests. The paper says
both.

### 7. Zero-shot on a matched sample

The zero-shot sample is 150 runs per split (130 test / 150 unseen Job pods), not "about 300". On those pods:

| | zero-shot | rules | rules-sized | RL argmax |
|---|---|---|---|---|
| test | 58.5 | 39.1 | 34.0 | 12.0 |
| unseen | 45.8 | | 40.4 | |

On the unseen pods it is level with the best (the gate is 43.5).

### 8. Corrections to the methods text, checked in code

- **CPU loss weight.** `policy.train_supervised` sums the three cross-entropies unweighted; the 0.3 CPU weight
  exists only in the older `train_jev`.
- **GPU cost constants.** The GPU path uses 0.05 h per attempt and 4 attempts (`bench_gpu/scoring.py`), not
  0.02 h and 8.
- **The RL update.** It is one on-policy step per batch with a group-normalized advantage and an exact KL:
  REINFORCE with a group baseline, no clipped ratio. The text no longer calls it GRPO without that qualifier.
- **The adapter.** The headline adapter (`ADAPT_SOURCES[""] = "rendition"`) reads the run's Flyte task. The
  deployable `-prompt` variant does not help:

| seed | unadapted | rendition | `-prompt` | `-prompt` pooled with Flyte-trained trees |
|---|---|---|---|---|
| 1 | 43.6 | 34.6 | 44.9 | 29.6 |
| 2 | 40.1 | 28.1 | 40.2 | 26.1 |
| 3 | 44.2 | 34.9 | 43.4 | 26.9 |

- **Rules-sized** uses the label generator's own constants (runtime, boto3, headroom). The paper says so.
- **Job history** is computed with the label's accounting (the review's finding 10). The paper says so.
- **Init stagings** do not pay for the init container's download. The paper says so.
- **GPU censoring.** It was mostly capacity, per round 21: every 4xL4 shard and half the 8xT4 shards timed out.
  The corpus caption says so.

### 9. The Hugging Face release, staged (rt-hub-k8s-2-stage-3)

Nothing was pushed. The dry run wrote every repository's folder to the run's blob store, ready for `push`.

- **Dataset** (`unionai/rightsize-bench`):
  - 64,164 pods across train, validation, test and unseen;
  - 2,189,644 prediction rows (every reported approach, seeds 1-3);
  - its leaderboard config reproduces the paper's leaderboard exactly.
  - The card's family count was computed per family name, which merges same-named families from different corpora.
    `stage_dataset` now counts `source:family` and lists the unseen families as "name (source)". The fix lands
    on the next staging; this run's card still has the old count.
- **Trees** (`unionai/rightsize-trees`): the refit on 36,106 training rows, saved with skops, reproduces the
  published gbt-dist requests on every pod (`refit_matches_published` = 1.0). The card's model-index has test
  regret of 16.77 (Job) and 19.79 (Flyte), and unseen regret of 73.76 / 73.92.
- **Decision and RL models**: 12 folders, staged as 4 repositories × 3 seeds, with seeds 2 and 3 as branches.
  - The repositories: decision, RL, job-only, flyte-only.
  - Each folder holds the LoRA adapter, the heads in safetensors, the calibration and `rightsize_pointer.py`.
  - Each reloaded through `rightsize_pointer.py` and reproduced 6 of 6 validation prompts.
- **Capacity.** On-demand pools could not admit the stage (stage-2). Every hub environment is now `interruptible`.

### Not done this round (needs new cluster work)

- Leave-family-out validation to choose gate members, read-outs, q and w without the unseen families.
- Re-decode and re-tune every approach at penalties 5 and 20 (needs `probs` re-exported).
- Re-temper RL over a wider range (needs `probs`).
- Re-measure the censored multi-GPU runs when 4xL4 and 4xL40S capacity exists, and validate the modelled L4 and
  4xL4 fits.
- Run Flyte pods with their limit at the label.
- Fix the `process_pool` labels (the cgroup counts every worker).
- Keep the 432 device-filling GPU runs, with the measured device as the label.
- Add baselines: kNN on history, the trees with an argmax read-out, a gate with rules-sized as its outer member.
- Add a few-shot zero-shot baseline.

## Code changes

- `bench_all/scoring.py`: `candidate_actions`, `cheapest_action`, `min_cost`; `score(..., oracle=)`.
- `paper/scripts/analyze.py`:
  - regret against the cheapest request, under every penalty;
  - family-clustered intervals;
  - read-out selection on validation;
  - the zero-shot sample;
  - per-corpus and seen/new-code slices;
  - paired tests against each column's best and under the penalties.
- `paper/scripts/tables.py`:
  - the leaderboard shows the selected read-outs, and bold marks every approach inseparable from the best;
  - new tables: read-outs, seen/new code, by corpus, the zero-shot sample;
  - `num_main.tex` holds every number the prose quotes, as macros.
- `paper/scripts/figures.py` (new): the trade-off, penalty, counterfactual and coverage charts, as PDF and as SVG.
- `paper/figures/diagrams/` (new): TikZ overview, pointer model and factory diagrams, which replace the ASCII
  listings.
- `paper/scripts/to_html.py`:
  - inlines every figure as SVG, with colors mapped to the report's CSS variables (light and dark);
  - reads `num_main.tex`.
- `paper/scripts/audit.py`, `bench_all/paper.py`: `code_audit` / `paper_audit`.
- `paper/sections/*`:
  - rewritten: contributions refocused on the benchmark and the models;
  - every review finding above reflected;
  - a new release section with a datasheet;
  - the AI-writing tells cut.
- `bench_all/hub.py`, `hub_cards.py`, `hub_pointer.py`, `hub_release.py`, `docs/huggingface-release.md`,
  `tests/test_hub.py` (new): the Hugging Face release.
