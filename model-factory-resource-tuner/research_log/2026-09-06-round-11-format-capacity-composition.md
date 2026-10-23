# 2026-09-06 — round 11: graded format reward, capacity, GBT composition, full FT

## Context

Round 10 left the LLM policy trailing a quantile-GBT on reliability
(55% fit vs 99%) while winning on cost. Four changes, one variable each,
all at round-8 scale (250k corpus / 4,096 contexts / 300 steps /
`c-cost`) so the rows drop into the standing comparison:

1. **Graded parseable-JSON reward** (`format_credit`, all arms): invalid
   completions score on a ladder (no JSON 0 → braces-not-JSON .25 →
   wrong schema .5 → gpu-field-only invalid .75 → valid 1.0) instead of
   a flat zero.
2. **`r11-r64-mlp`**: LoRA r=64 + MLP projections — the capacity test.
3. **`r11-gbt-hint`**: out-of-fold GBT estimates in the prompt AND the
   savings reward referenced to the GBT's cost.
4. **`r11-fullft-4b`**: full fine-tune (no adapters, 8-bit Adam).

## Run ledger

Base URL: `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/`

| arm | training | eval |
|---|---|---|
| r11-r64-mlp | [uf46cjqzz77vknn7zgt5](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uf46cjqzz77vknn7zgt5) | [ubnpsnskbw5n8pgxxkn7](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ubnpsnskbw5n8pgxxkn7) |
| r11-gbt-hint | [ugtcw8zxzw5w5864mbxx](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ugtcw8zxzw5w5864mbxx) | [udsxgxtv42djsplhshq8](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/udsxgxtv42djsplhshq8) |
| r11-fullft-4b (Qwen3-4B, 1x L40S) | [unv2hq4csvqzqv4tnpg2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/unv2hq4csvqzqv4tnpg2) | [udwhdwvn7r75czrp45gd](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/udwhdwvn7r75czrp45gd) |
| ambitious (Qwen3.5-4B QLoRA, 500k corpus) | [uhjwfqmws5qrf7jkrb6s](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uhjwfqmws5qrf7jkrb6s) | [ub5lqdvxfsl8ngkgcsjp](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ub5lqdvxfsl8ngkgcsjp) |

Full-FT provisioning ladder (aborted, kept for retry): 32B/`L40s:8`
[u69bnsmj4scx7hfvp985] · 14B/`L40s:4` [ul6nqttdcjkmc6zdtngm] and retry
[uk9pj886sxmw9sqnhgrt] · 8B/`L4:4` [un7n94jwjgtbqzzbs5qc] — none of the
multi-GPU node groups provisioned (Karpenter nominated ten g6.12xlarge
nodeclaims without producing a node); `L40s:1` scheduled in ~4 minutes.

## Results

The 250k arms share a heldout (baseline: fit 43%, $0.1717/task-hr,
GBT $0.2163 at 99% fit). `ambitious` is scored on the 500k heldout
(baseline fit 56%, $0.1931; GBT $0.1994 at 100%) — its gate is
self-consistent but its numbers are NOT comparable to the other rows.

| arm | validity | fit | waste | $/task-hr | **$ saved / 1k hrs** | GPU fit | native reward | gate |
|---|---|---|---|---|---|---|---|---|
| r11-r64-mlp | 100% | 53% | 34% | **$0.0382** | **+$133.48** | 0% | 0.41 → 1.17 | fail |
| r11-gbt-hint | 100% | **93%** | 48% | $0.2168 | −$45.05 | **86%** | 0.86 → 1.06 | fail |
| r11-fullft-4b | 100% | 79% | 31% | $0.1321 | +$39.67 | 65% | 0.87 → 1.18 | fail |
| **ambitious** (Qwen3.5-4B) | 100% | 87% | 33% | $0.1526 | +$40.52 | 77% | 0.45 → −0.40 | **PASS** |

## Findings

- **The graded format reward fixed schema validity outright: 100% on
  every arm** (72% in round 7, 90% in round 8). Turning the format cliff
  into a ladder removed the entire invalid-completion failure class, and
  with it the GPU-proposal blindness — GPU-task fit went 0% → 65–86% on
  the arms that kept context.
- **FIRST GATE PASS of the project**: the ambitious Qwen3.5-4B QLoRA run
  on the 500k corpus — validity 100%, fit 87% vs baseline 56%, waste 33%,
  **$40.52 saved per 1,000 task-hours**, GPU fit 77%. Scale (500k corpus,
  1,200 steps) plus the format reward did it.
- **The GBT-hint composition did exactly what was predicted — including
  the failure mode.** Reliability jumped to 93% fit (best LLM arm) and
  86% GPU fit, but cost landed at $0.2168 ≈ the GBT's own $0.2163: the
  policy learned to COPY the anchor, inheriting its padding, and lost
  money. Anchoring buys reliability; the savings term as weighted did not
  buy the "beat the anchor" behavior. Next iteration: penalize
  equal-to-anchor proposals, or reference savings to the anchor while
  keeping an absolute waste term.
- **Full FT beat LoRA on reliability at equal-ish budget**: 79% vs 53%
  fit with LOWER waste (31% vs 34%) — but it is one model size and one
  seed, so treat it as a signal, not a law. The 4B full-FT arm needed a
  single L40S; multi-GPU rungs never provisioned.
- **`r11-r64-mlp` is the cautionary arm**: the cheapest proposals in
  project history ($0.0382/task-hr, $133 saved/1k) at 53% fit and 0% GPU
  fit — more adapter capacity chased the cost term into unreliability.
  The $ metric alone would have crowned it; the gate's reliability
  clauses are what stop that.
- Native reward is confirmed non-comparable across arms: `ambitious`
  improved on every eval axis while its own curve fell 0.45 → −0.40.

## Code changes

Branch `capacity-and-composition-arms` (PR #11): `format_credit` ladder;
`lora_mlp`/`use_lora`/`gbt_hint` profile knobs; out-of-fold GBT hints
(`out_of_fold_hints`) + hint plumbing through prompts/eval/serving;
full-model checkpoint loading in generator and tune service;
`hypothesis_description` (markdown, rides manifest → eval → dashboard);
per-component reward trajectories; wide interactive report charts.

### Operational fix found here
Four concurrent evals with four different checkpoints timed out at 2h:
the reusable generator cached one FULL base model per checkpoint on a
single T4 (2× 4B-class + 2× 1.7B ≫ 16GB), thrashing the OOM-retry path
with 1,280 queued calls. Replicas now evict engines/batchers on
checkpoint switch; the same four evals then took 9–13 minutes each,
run sequentially.
