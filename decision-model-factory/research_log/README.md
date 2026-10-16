# Research log

Audit trail of decision-model-factory experiments. Same conventions as
[resource-tuner-model-factory/research_log](../../resource-tuner-model-factory/research_log/README.md):
one entry per experiment round (`YYYY-MM-DD-<slug>.md`) with a run ledger
(Union console URLs, failures included and linked to fixes), findings, and
code changes; keep the tables below current.

This factory deploys into the shared `model-factory` project, so run URLs
share a base with basic-model-factory:
`https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

## Standing results

Suite scorecards on the held-out split (120 episodes, 54 near-boundary).
Majority-class baseline is **20.0%** (`lookup_order`). `exact` = right tool
**and** right arguments, which is what the champion is selected on.

| run | model | params | exact | tool | JSON valid | near-boundary exact | untuned base exact | lift vs base |
|---|---|---|---|---|---|---|---|---|
| [df-smoke-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df-smoke-r1) | **qwen2.5-0.5b** (champion) | 494M | **82.5%** | 82.5% | 100% | 85.2% | 32.5% | **+50.0pp** |
| [df-smoke-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df-smoke-r1) | smollm2-135m | 135M | 31.7% | 35.8% | 97.5% | 29.6% | 0.0% | +31.7pp |

Reading:

- **Fine-tuning does two different jobs, and the metrics separate them.** For
  the 135M model it bought *format*: 0% → 97.5% valid JSON (the untuned model
  cannot emit a parseable tool call at all), while the decision stays near
  chance. For the 494M model it bought the *decision*: 32.5% → 82.5% exact.
- **The suite spans a real capability gradient** on one task: 3.6x the
  parameters is 2.6x the exact accuracy. That is the comparison the factory
  exists to produce.
- **The champion's errors concentrate in one place**:
  `escalate_to_human → issue_refund`, 17 of 120. It approves refunds that
  should go to a human — the three-condition conjunction (delivered, inside
  the 30-day window, at or under 10,000 cents), and the *expensive* direction
  of the error.
- Near-boundary accuracy (85.2%) is slightly **above** overall (82.5%) for the
  champion, so the near-boundary sampling is not where it fails; the refund
  over-approval is not purely a threshold-proximity effect. Worth a round of
  its own.

## Entries

| date | entry | tl;dr |
|---|---|---|
| 2026-10-02 | [round-1-suite-and-serve](2026-10-02-round-1-suite-and-serve.md) | Project built and green end to end: oracle-labelled synthetic episodes, a 2-model suite fanned out over a `model` partition, `.all("model")` collapse to a champion, CPU-served decision API. qwen2.5-0.5b wins at 82.5% exact (+50pp over untuned) |

## Key links

- Sibling factories: [basic-model-factory](../../basic-model-factory),
  [resource-tuner-model-factory](../../resource-tuner-model-factory)
- Factories docs: https://www.union.ai/docs/v2/union/user-guide/factories/
- Artifacts docs: https://www.union.ai/docs/v2/union/user-guide/artifacts/
