# 2026-09-05 — round 10: the classical-ML baseline (quantile GBTs)

## Context

"What does a non-fancy, no-LLM approach buy?" Options surveyed (HPC
memory-requirement predictors, Slurm ensemble predictors, Google
Autopilot's percentile framing, Bayesian-regression uncertainty):

1. **Quantile gradient-boosted trees** — chosen: dominant in the job
   -resource literature, nonlinear, mixed-feature-native, CPU-cheap, and
   the quantile IS the safety margin.
2. Bayesian ridge (μ + k·σ) — interpretable closed-form uncertainty, but
   linear; the corpus's param→footprint curves aren't.
3. Autopilot-style percentile-of-history — strongest with history, no
   cold-start answer (the ledger-history path covers that regime).
4. k-NN over code similarity — needs embeddings to compete; that's the
   fancy path this baseline exists to contrast.

## What was built (PR #10, branch worktree-ml-baseline)

`resource_tuner/training/ml_baseline.py`: four sklearn
HistGradientBoosting models — memory quantile(0.9) on log2(peak MiB),
CPU quantile(0.75), a GPU-type classifier {none, T4, L4, L40S}, and a
VRAM quantile(0.9) that picks the cheapest fitting card. Features are
strictly non-LLM: sampled params, input-profile numbers, lexical code
flags, author prior, run history. `eval_tuner` renders a three-way table
(LLM policy vs rule baseline vs ML baseline) and reports
`ml_baseline_*` keys; the lineage comparison shows the ML $/task-hr.
Tests: 5 new (162 total), incl. GPU recall >90% with <5% spurious and
GBT ≥ rule-baseline fit with no extra waste in sim.

## Run ledger

| run | what | outcome |
|---|---|---|
| [uvl4sb6j7ttlwvqktwz5](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uvl4sb6j7ttlwvqktwz5) | three-way eval (r8-c-cost checkpoint, 256 heldout, sim scoring) | see table |

| estimator | fit | $/task-hr (requested) | $/SUCCESSFUL task-hr |
|---|---|---|---|
| LLM policy (r8-c-cost) | 55% | **$0.1188** | $0.216 |
| rule baseline (family median) | 43% | $0.1717 | $0.399 |
| ML baseline (quantile GBT) | **99%** | $0.2163 | $0.219 |

## Findings

- **The GBT baseline nearly never fails** (99% fit on the heldout where
  the LLM policy manages 55%) by paying ~82% more per requested
  task-hour than the policy. Divide by success and the two are
  effectively TIED (~$0.22/successful task-hr), both ~45% cheaper than
  the rule baseline.
- Honest implication: **the LLM policy has not yet earned its GPU** — a
  CPU-seconds GBT matches its effective cost with far better
  reliability. The LLM's path to justification: close the fit gap (the
  ambitious 4B run is the live test) and exploit what trees can't — raw
  unseen code without engineered features, and prior/history reasoning.
- Caveat in the GBT's favor is also its limit: its features include the
  sampled `params_json` (the generator's ground-truth state). Real
  cold-start tasks won't hand a scheduler such clean numbers; the
  lexical/profile features carry most but not all of that signal. A
  fair fight on truly unseen task text still favors code-reading models.
- The three-way table now renders in every eval report and the lineage
  dashboard, so this comparison is standing, not one-off.
