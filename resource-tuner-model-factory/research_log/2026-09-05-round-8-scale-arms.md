# 2026-09-05 — round 8: shape arms at 250k scale (+ GPU + context)

## Context

Round 7's arms all lost money at dev scale (512 ctx / 150 steps) and
couldn't emit valid GPU proposals at all. Round 8 re-runs the same four
reward shapes with everything scaled: a 250k mixed corpus (208k teacher
archetypes + 42k template rows incl. GPU families capped to single-T4
VRAM and prior/history context fields), 4,096 train contexts (fixed-seed
random sample — a head slice would have dropped every archetype row),
300 steps. Judged on each arm's native reward improvement AND the shared
business metric ($ saved / 1k task-hours vs the rule baseline).

## Run ledger

Base URL: `https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/`

- Corpus (250k): [u69648h4d4mczxnt9n6s](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u69648h4d4mczxnt9n6s)
- Trainings (~5h each on T4; queued up to 3h behind each other; ~60s/step
  from the longer archetype+context prompts):
  [u6h55bkcd2w54cqmqckj](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/u6h55bkcd2w54cqmqckj) linear ·
  [unzs5xwh7rxvg85xppvw](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/unzs5xwh7rxvg85xppvw) log ·
  [ugq84t7mp2827k4strqd](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ugq84t7mp2827k4strqd) bucket ·
  [ujzkqd7npgwpdm6w9slv](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ujzkqd7npgwpdm6w9slv) cost
- Evals (256 shared heldout ctx):
  [ubm5w5mx5jz2p44zvkb2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/ubm5w5mx5jz2p44zvkb2) ·
  [upkxws5z85vjc96pt9w5](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/upkxws5z85vjc96pt9w5) ·
  [udrjgqh5p6hrpfnqfntr](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/udrjgqh5p6hrpfnqfntr) ·
  [uwdd94rv75wmsbfdc29g](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/uwdd94rv75wmsbfdc29g)

## Results

Baseline on this (much harder) heldout: fit 43%, waste 26%,
$0.1717/task-hr.

| arm | native reward | validity | fit | waste | $/task-hr | **$ saved / 1k hrs** | gpu fit | gate |
|---|---|---|---|---|---|---|---|---|
| r8-c-linear | 0.70 → 0.98 | 90% | **59%** | 44% | $0.1187 | +$53.06 | 17% | fail |
| r8-c-log | 0.89 → 0.95 | 90% | 57% | 44% | $0.1196 | +$52.10 | 15% | fail |
| r8-c-bucket | 0.97 → 1.05 | 90% | 54% | 44% | $0.1167 | **+$55.03** | 13% | fail |
| r8-c-cost | **0.41 → 1.01** | 90% | 55% | 44% | $0.1191 | +$52.60 | **22%** | fail |

## Findings

- **The business metric flipped positive at scale**: every arm saves
  $52–55 per 1,000 task-hours (~31% cheaper requests) where round 7 lost
  $19–47. Scale (8× contexts, 2× steps) + the GPU/prior/history context
  did what the shape changes alone could not.
- **Shapes barely separate on dollars** (±3% band). Native rewards all
  improved (c-cost's Δ+0.60 largest, but its start is lowest — the
  baseline-relative term recenters its scale, which is exactly why
  cross-arm absolute rewards are not comparable). At this scale the
  reward-shape choice matters less than data/steps.
- **Every arm still fails the gate — on the waste clause** (44% vs the
  baseline's 26%), and fit is 54–59% vs 43%. Both absolute numbers are
  low because this heldout is genuinely harder: T4-capped GPU tasks and
  archetype-scale footprints. The $ metric and the fit/waste clauses
  disagree — cheap-but-often-failing is exactly the regime the gate's
  reliability clauses exist to catch. Honest verdict: not promotable yet.
- **GPU estimation went from zero to partial**: validity 90% (vs 72%),
  gpu-task success 13–22% (vs 0), zero spurious GPUs. The remaining GPU
  failures need inspection via the new `invalid_completion_examples` +
  GPU tables in the eval reports.
- The single aggregated view works: the lineage dashboard's comparison
  table (reward shape × native Δ × $/task-hr × $ saved × GPU fit × gate)
  is where these rows live from now on.

## Next

The ambitious run (round 9: 500k corpus, Qwen3.5-4B, ≤3 days on one T4,
c-cost reward) is the direct test of the "scale dominates" finding.
