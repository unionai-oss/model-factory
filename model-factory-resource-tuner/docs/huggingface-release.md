# Runbook: publishing RightSize Bench to the Hugging Face Hub

This publishes the benchmark dataset and its baseline models under
[huggingface.co/unionai](https://huggingface.co/unionai), grouped in one
collection. Everything runs on `demo.hosted.unionai.cloud`, because the
corpus, the predictions and the trained adapters live in the cluster's object
store and the laptop cannot read it. The code is in
`resource_tuner/bench_all/hub.py` (tasks and tables), `hub_cards.py` (the
README cards), `hub_pointer.py` (the inference file each model repo ships)
and `hub_release.py` (the driver). Tests: `tests/test_hub.py`.

## What gets published

| repo | type | contents |
|---|---|---|
| `unionai/rightsize-bench` | dataset | Parquet, one file per split; configs `pods` (default), `jobs`, `flyte_tasks`, `features`, `counterfactual`, `predictions`, `leaderboard`; each approach's tuned settings in `approaches/`; `release.json` |
| `unionai/rightsize-decision-qwen3-1.7b` | model | LoRA adapter (`adapter_model.safetensors`), pointer heads (`decision_head.safetensors`), `rightsize_config.json` (calibration, options), `rightsize_pointer.py`; seed 1 on `main`, seeds 2 and 3 on branches `seed-2`, `seed-3` |
| `unionai/rightsize-rl-qwen3-1.7b` | model | the same layout, for the GRPO-trained model |
| `unionai/rightsize-trees` | model | `model.skops` (the quantile trees and the GPU classifier), `rightsize_config.json` (feature names, quantile levels, threshold) |
| `unionai/rightsize-decision-qwen3-1.7b-job-only`, `-flyte-only` | model | the training-data ablations (skip with `--no-ablations`) |
| collection "RightSize Bench" | collection | the dataset first, then the models |

Splits are `train`, `validation`, `test` and `unseen` (the seven held-out
families). Each push to `main` is also tagged with the release (`k8s-2`), and
each branch push with `k8s-2-seed-2` and so on, so a later release can move
`main` while this one stays loadable with `revision="k8s-2"`.

Conventions this follows, from the Hub's dataset and model card guides:

- Parquet data files named `<split>-00000-of-00001.parquet`, declared under
  `configs:` in the card's YAML. The dataset viewer and `load_dataset` read
  them without a loading script.
- One concept per repo, and related repos grouped in a collection. Seeds are
  revisions of one model (branches), not separate repos. Ablations are separate
  repos because they are trained on different data.
- Weights in `safetensors`; scikit-learn models in `skops`, never pickle.
- Model cards declare `base_model`, `library_name`, `datasets`, `license`,
  `pipeline_tag` and a `model-index` with the benchmark's regret, so the Hub
  links each model to its base model and to the dataset.
- Repos are created private, reviewed, then made public.

## Decisions to confirm before the first public push

- **Licenses.** The cards say CC-BY-4.0 for the dataset and Apache-2.0 for the
  models (the base model, Qwen3-1.7B, is Apache-2.0). Change `license:` in
  `hub_cards.py` if Union wants something else.
- **Ablations.** Two extra model repos. Drop them with `--no-ablations`.
- **arXiv.** When the paper has an ID, add it to the cards (see below).

## Prerequisites (once)

1. Be a member of the `unionai` org on huggingface.co with write access.
2. Create a fine-grained access token at
   <https://huggingface.co/settings/tokens>: repository permissions "write"
   for the `unionai` org (and "manage collections" if offered). A read token,
   such as the cluster's existing `HUGGINGFACE_TOKEN`, cannot push.
3. Store it as a Flyte secret in the project:

   ```bash
   cd model-factory-resource-tuner
   uv run flyte --config .flyte/config.yaml create secret HF_UNIONAI_WRITE_TOKEN \
       --project resource-tuner-model-factory --domain development
   # paste the token when prompted
   ```

   Only the `rt-hub-push` tasks mount it. Staging never sees it.

## 1. Stage (dry run)

```bash
cd model-factory-resource-tuner
uv run python hub_release.py --release k8s-2 --name rt-hub-k8s-2-stage-<n>
```

The driver finds the release's three training runs (`RUNS` in
`hub_release.py`: `rsk-k8s-2`, `-s2`, `-s3`), collects their corpus and arm
outputs, and launches `hub.release` with `push_to_hub=False`. That fans out:

- `stage_dataset`: every usable pod, the 209 tree features, the counterfactual
  renditions, every reported approach's request and cost per seed, the
  leaderboard and the card. About an hour on 8 CPUs.
- `stage_trees`: refits the trees on the same rows (the fit is deterministic)
  and checks the refit against the published requests. The card reports the
  match rate, which should be 100%.
- `stage_pointer`, once per decision model and seed: converts the head to
  safetensors, writes the config and card, then loads the repo with its own
  `rightsize_pointer.py` on CPU and checks the argmax request on 6 validation
  prompts against the training run's. The run used bf16 on an L4, so a near-tie
  can flip; 5 or 6 of 6 is expected.

It prints a JSON summary with each repo's staged directory.

## 2. Review what was staged

From the run's page in the Union console, open each `stage_*` action's
output and check:

- `README.md` renders, and its numbers look like the paper's (the leaderboard
  is regret against the cheapest request, seed 1).
- `release.json` (dataset): pod counts per split and pod type match the
  paper's split table (36,106 / 5,036 / 10,190 / 12,832 pods).
- `rightsize_config.json` (trees): `refit_matches_published` is 1.0.
- The pointer cards' verification line.

## 3. Push, private

```bash
uv run python hub_release.py --release k8s-2 --push --name rt-hub-k8s-2-push-<n>
```

The staging tasks cache-hit, so this only uploads. `main` is pushed before the
seed branches. The summary has each commit URL and the collection URL.

Check on the Hub, signed in as an org member:

- The dataset viewer shows every config and split.
- `load_dataset("unionai/rightsize-bench", "jobs", split="test")` works.
- Each model card's metadata panel shows the base model and the dataset, and
  the evaluation results come from the `model-index`.
- `rightsize_pointer.py` runs from the card's snippet on one prompt.

## 4. Make it public

```bash
uv run python hub_release.py --release k8s-2 --push --public --name rt-hub-k8s-2-public-<n>
```

`create_repo(..., exist_ok=True)` does not change an existing repo's
visibility, so for repos that already exist, flip them in each repo's settings,
or run:

```python
from huggingface_hub import HfApi
api = HfApi()
for repo, kind in [("unionai/rightsize-bench", "dataset"), ("unionai/rightsize-decision-qwen3-1.7b", "model"),
                   ("unionai/rightsize-rl-qwen3-1.7b", "model"), ("unionai/rightsize-trees", "model")]:
    api.update_repo_settings(repo, repo_type=kind, private=False)
```

and make the collection public from its page.

## 5. After the paper is on arXiv

Add `arxiv:<id>` to each card's `tags` and the paper's URL to the citation in
`hub_cards.py`, then add the paper to the collection
(`add_collection_item(slug, "<id>", "paper")`). Re-push. The Hub then links
the dataset and models from the paper's page.

## A new release

Add the release's runs to `RUNS`, then repeat the steps above. Each release is
tagged, so `main` moves and older releases stay loadable by tag. If a
release's schema changes, say so in the dataset card's changelog.

## Rolling back

Every push is a commit, and every release is a tag. The Hub has no revert
call: to roll `main` back, download the previous tag
(`snapshot_download(repo, revision="<tag>")`) and upload that folder to `main`
with `upload_folder(..., delete_patterns="*")`. Delete a bad seed branch with
`HfApi().delete_branch`. Deleting a repo breaks every link to it; make it
private instead.

## Research log

A staging or push run is an experiment like any other: log it in
`research_log/` with its console URL and what it published.
