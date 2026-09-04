# 2026-09-04 — round 7: reward shaping, $ pricing, GPU estimation

## Context

Round 5's plateau (uniform ~4Gi proposals, waste stuck at 47% vs the 25%
baseline gate) is a reward-shape problem: the linear waste fraction
saturates for padded requests, so 4Gi and 1Gi for a 200MiB task score
almost identically. This round implements the full shaping menu behind
one config and judges every arm by the business metric — **dollars saved
per task-hour** — priced from this tenant's actual node groups. GPU
estimation joins the action space (per the brief: T4 / L4 / L40S pools
only, the ones that schedule on demo.hosted).

## What was built

- **`pricing.py`** — $/hr = Fargate unit rates (vCPU $0.04048, GiB
  $0.004445) + per-GPU residuals derived from the tenant's node groups:
  T4 $0.279 (g4dn.metal), L4 $0.512 (g6.2xlarge), L40S $1.634
  (g6e.2xlarge). Used by training rewards, the eval gate, and the
  rt-tune value ledger (now charts cumulative $/hr saved).
- **GPU estimation end-to-end** — proposals carry `gpu: "TYPE:count"`
  (validated against the tenant ladder); simulator adds VRAM truth with
  memory-like semantics (missing GPU → failure; undersized VRAM → OOM;
  unneeded GPU → a whole idle accelerator of waste); two new task
  families (`gpu_batch_inference`, `gpu_finetune`) span ~1.5–40GiB VRAM
  so the right answer walks the whole ladder; the family baseline
  proposes the cheapest fitting GPU.
- **`rewards/shaping.py`** — `RewardShape`: the mutually exclusive
  `waste_form` (linear | quadratic | sqrt | log_ratio | bucket) composed
  with headroom band, $-weighted axis aggregation, baseline-relative
  savings bonus, annealed waste weight, in-group cheapest-survivor
  tie-break, and robustness-averaged episodes. `RT_REWARD_SHAPE` (JSON)
  overrides knobs per experiment without code changes.
- **Named arms** (profiles `dev-c-*`; identical training, reward the
  only free variable):
  | arm | waste form | composed knobs |
  |---|---|---|
  | c-linear | linear | $-weighted (stage-B economics + GPU — the control) |
  | c-log | log_ratio | $-weighted, anneal 0.4→0.9 |
  | c-bucket | bucket (grid distance) | headroom (1.1,1.4), $-weighted, tie-break 0.15, robustness ×3, anneal 0.5→1.0 |
  | c-cost | linear | $-weighted, baseline-relative savings, tie-break 0.15 |
- **Human-inspectable eval** — the report opens with the shape config,
  then a Dollars section ($/task-hr policy vs baseline, $ saved per
  1,000 task-hours), per-axis waste medians, a GPU-estimation table
  (missing/spurious/success), and a per-family breakdown. The gate now
  requires *policy cost ≤ baseline cost*. The lineage dashboard grew a
  $-per-task-hour chart and a reward-shape comparison table (one row
  per eval: shape, fit, waste, $/task-hr, $ saved, GPU fit, gate).

## Run ledger

Base URL: `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/`

All arms: `tuner_pipeline --seed 7` (identical corpus incl. GPU
families), dev scale (512 contexts, 150 steps, Qwen3-1.7B LoRA, T4).

| run | arm | outcome |
|---|---|---|
| [uz2l5vkpmtrlvzmj955x](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uz2l5vkpmtrlvzmj955x) | dev-c-linear (control) | RESULTS_PENDING |
| [ubhxqvrd6jx75bp57kb7](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ubhxqvrd6jx75bp57kb7) | dev-c-log | RESULTS_PENDING |
| [u9h88pgmnsqxthk94zt7](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u9h88pgmnsqxthk94zt7) | dev-c-bucket | RESULTS_PENDING |
| [unncwgrzhc4tjrcc8xwn](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/unncwgrzhc4tjrcc8xwn) | dev-c-cost | RESULTS_PENDING |

## Findings

RESULTS_PENDING

## Code changes

Branch `flyte-2.7-metrics` (commit `5998b65`): `resource_tuner/pricing.py`
(new), `rewards/shaping.py` (new), actions/simulator/templates/corpus/
baseline/prompts (GPU), grpo (shaped reward path + manifest),
evaluate (comparison report + $ gate), lineage_app ($-chart + shape
table), tune_store/tune_service ($ ledger), config (dev-c-* profiles),
tests (146 passing).
