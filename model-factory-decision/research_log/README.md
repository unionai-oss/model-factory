# Research log

Audit trail of model-factory-decision experiments. Same conventions as
[model-factory-resource-tuner/research_log](../../model-factory-resource-tuner/research_log/README.md):
one entry per experiment round (`YYYY-MM-DD-<slug>.md`) with a run ledger
(Union console URLs, failures included and linked to fixes), findings, and
code changes; keep the tables below current.

This factory deploys into the shared `model-factory` project, so run URLs
share a base with model-factory-basic:
`https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/<run>`

## Standing results — coding-agent permission gate (round 2 →)

Latest suite: [df2-dev-r3](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df2-dev-r3)
(permission-* artifacts, date 2026-10-17). Test split: 1,600 in-distribution
calls (45% adversarial) + 600 calls from four families never seen in training.
Majority baseline 37.0% (`ask_user`). Unsafe-allow = P(model allows | oracle
says ask or deny). p50 = one decision, batch size 1, A10G. Champion chosen on
val expected cost (2-SE ties → lowest latency).

| model · readout | acc | unsafe-allow | Brier | ECE | held-out acc | held-out unsafe | p50 | output tokens | head params |
|---|---|---|---|---|---|---|---|---|---|
| **qwen2.5-0.5b · pointer** (served) | **99.44%** | 0.09% | 0.011 | 0.004 | 62.7% | 31.7% | **25 ms** | **0** | 0.92M |
| qwen2.5-0.5b · sft json | 98.31% | 1.21% | 0.034 | 0.017 | 64.3% | 25.4% | 163 ms | 7.4 | 136.1M |
| qwen3-0.6b · pointer | 98.81% | 0.37% | 0.021 | 0.008 | **79.7%** | **12.7%** | 36 ms | 0 | 1.05M |
| qwen3-0.6b · sft json | 98.88% | 0.28% | 0.023 | 0.011 | 57.7% | 25.8% | 237 ms | 7.4 | 155.6M |
| smollm2-360m · pointer | 97.25% | 0.93% | 0.043 | 0.005 | 62.7% | 18.6% | 33 ms | 0 | 0.99M |
| smollm2-360m · sft json | 83.25% | 7.18% | 0.335 | 0.168 | 57.3% | 21.1% | 235 ms | 8.1 | 47.2M |
| ref: qwen3-0.6b untuned (option likelihood) | 44.0% | 70.2% | 0.894 | 0.373 | 30.2% | 78.9% | — | — | — |
| ref: majority (`ask_user`) | 37.0% | 0.0% | 1.26 | 0.63 | — | — | — | — | — |

Reading:

- **In distribution, the gate is solved at ≥0.5B by either head** (98.3–99.4%,
  ECE ≤ 0.017), reproducibly across three days of data. SFT's *token
  log-probs* are as well calibrated as the pointer head; its JSON answer is
  one-hot, so it has no threshold to turn.
- **At 360M the head matters**: pointer 95.8–98.4% vs SFT 82.0–83.3%,
  unsafe-allow 0.4–1.6% vs 7.2–8.9%, on each of three days of data.
- **The pointer head is 6.5× faster with zero output tokens** and a ~1M-param
  head instead of a vocabulary-sized one.
- **Out of distribution nobody is calibrated**: 55–80% accuracy, 13–32%
  unsafe-allow at confidence ~0.95, and the val-tuned ≤1% threshold does not
  transfer. These numbers swing ±10–30pp between seeds; treat single runs as
  anecdotes.
- Served: `https://solitary-moon-91169.apps.demo.hosted.unionai.cloud`
  (CPU, 1.7–2.0 s/decision with prefix KV caching; 25 ms on an A10G).

## Round 1 results — support triage (retired problem, `decision-*` artifacts)

Suite scorecards on the held-out split (120 episodes, 54 near-boundary).
Majority-class baseline is **20.0%** (`lookup_order`). `exact` = right tool
**and** right arguments.

| run | model | params | exact | tool | JSON valid | near-boundary exact | untuned base exact | lift vs base |
|---|---|---|---|---|---|---|---|---|
| [df-smoke-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df-smoke-r1) | **qwen2.5-0.5b** (champion) | 494M | **82.5%** | 82.5% | 100% | 85.2% | 32.5% | **+50.0pp** |
| [df-smoke-r1](https://demo.hosted.unionai.cloud/v2/domain/development/project/model-factory/runs/df-smoke-r1) | smollm2-135m | 135M | 31.7% | 35.8% | 97.5% | 29.6% | 0.0% | +31.7pp |

## Entries

| date | entry | tl;dr |
|---|---|---|
| 2026-10-02 | [round-1-suite-and-serve](2026-10-02-round-1-suite-and-serve.md) | Project built and green end to end: oracle-labelled synthetic episodes, a 2-model suite fanned out over a `model` partition, `.all("model")` collapse to a champion, CPU-served decision API. qwen2.5-0.5b wins at 82.5% exact (+50pp over untuned) |
| 2026-10-15 | [round-2-coding-agent-permission-gate](2026-10-15-round-2-coding-agent-permission-gate.md) | Re-targeted to a coding-agent permission gate (5-rule oracle over shell effects, 45% adversarial) and a real decision head (Strands-style pointer, ~1M params) vs generative SFT on the same torso, as a torso×head `model` partition. In-distribution solved at ≥0.5B by both heads (99%+); pointer wins clearly at 360M (98% vs 83%) and is 6.5× faster with zero output tokens. Held-out families: everyone 55–80%, overconfident, threshold doesn't transfer. Also: stations had no `cache=` (factory re-ran everything), round 1's nightly trigger ran every GPU job twice nightly |

## Key links

- Sibling factories: [model-factory-basic](../../model-factory-basic),
  [model-factory-resource-tuner](../../model-factory-resource-tuner)
- Factories docs: https://www.union.ai/docs/v2/union/user-guide/factories/
- Artifacts docs: https://www.union.ai/docs/v2/union/user-guide/artifacts/
- Strands Decider: https://strandsagents.com/blog/introducing-strands-decider/
