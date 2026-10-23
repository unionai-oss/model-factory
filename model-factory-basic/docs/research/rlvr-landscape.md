# Research notes: RLVR / model-factory landscape (2024–2026)

Compiled 2026-08-12. Focus: infrastructure and recipes for training coding
agents with RL from verifiable rewards, at scales from one 24GB GPU up.

## 1. Model-factory-style system architectures

**The converged architecture.** Nearly every serious 2025–2026 writeup
describes the same decomposition: **rollout workers** (inference engines +
environments generating trajectories), **trainer workers** (GPU processes
consuming a replay/trajectory buffer), and a **weight-sync path** between
them, with asynchrony as the central design tension. AReaL pioneered the
production-scale version; RollArt (https://arxiv.org/pdf/2512.22560) and
Polar (https://arxiv.org/pdf/2605.24220) argue for "rollout-as-a-service":
sandbox setup, agent execution, and reward computation split behind a service
boundary, separate from training. The recurring systems problem is
**long-tailed trajectory lengths** causing GPU idle under synchronous
batching; fixes are trajectory-level async rollouts and bounded-staleness
(off-policy by 1–2 steps) training.

**Cursor** — Composer 2 report (https://cursor.com/resources/Composer2.pdf):
four decoupled services — async Ray+PyTorch training, and an **environment
server using microVMs** running "mini versions of Cursor" per rollout.
Tab RL post (https://cursor.com/blog/tab-rl): policy-gradient training on
live accept/reject signals, **1.5–2 hour cycle** per checkpoint.

**Kimi (Moonshot)** — K2 report (https://arxiv.org/html/2507.20534): hybrid
colocated architecture; checkpoint-engine cut weight-update for 1T params
from ~10 min to ~20 s.

**Prime Intellect** — the most complete open factory stack: prime-rl (fully
async trainer), SHARDCAST (weight broadcast), TOPLOC (rollout verification),
and the Environments Hub built on **verifiers**
(https://github.com/PrimeIntellect-ai/verifiers) — environments as
pip-installable packages exposing `load_environment()`, decoupled from any
trainer. INTELLECT-2 ran two-step off-policy async RL across a permissionless
GPU swarm.

**Framework landscape** (see
https://www.anyscale.com/blog/open-source-rl-libraries-for-llms):

- **verl** — HybridFlow; colocated "HybridEngine", Megatron TP+PP for 70B+,
  default for big open runs.
- **OpenRLHF** — Ray+vLLM, disaggregates actor/critic/reward/reference.
- **NeMo-RL** — clean interfaces, environment-first, less mature.
- **SkyRL** — sync or async pipelining, colocated or disaggregated.
- **TRL** — single-node-friendly, the small-scale entry point.
- **rLLM/Agentica** — atop verl.

**Labs.** DeepSeek-R1 (https://arxiv.org/pdf/2501.12948) established the
canonical RLVR pipeline: GRPO with **rule-based accuracy + format rewards**
(deliberately no neural reward model, to avoid reward hacking), multi-stage
(cold-start SFT → reasoning RL → rejection-sampling SFT → all-scenario RL).
OpenAI RFT productized the loop with programmable graders. The "dark factory"
term is currently used mostly for autonomous *coding* pipelines, not
autonomous *training* loops — the training-loop sense is novel as a brand.

## 2. RLVR for coding agents: environments, datasets, models, rewards

**Two task genres.** (a) competitive-programming / function-level: reward =
unit tests pass. (b) repo-level agentic (SWE): reward = hidden test suite
passes after the agent edits a real codebase in a container.

**Standard datasets/environments:**

- **SWE-Gym** (https://github.com/SWE-Gym/SWE-Gym): 2,438 real Python task
  instances with executable runtimes + unit tests.
- **R2E-Gym**: ~8K procedurally curated executable SWE tasks; used by
  DeepSWE (Qwen3-32B, pure RL, 59% SWE-bench Verified).
- **SWE-smith** (https://arxiv.org/pdf/2504.21798): environment-first —
  build the execution env, then synthesize task instances that break
  existing tests; 50K instances from 128 repos at 295 GB.
- **Function-level RL corpora**: TACO (25.4K problems w/ I/O tests), APPS,
  CodeContests, **KodCode-V1**
  (https://huggingface.co/datasets/KodCode/KodCode-V1; largest fully
  synthetic verifiable set) with **KodCode-Light-RL-10K**
  (https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K) as the
  ready-made RL slice, PrimeIntellect SYNTHETIC-1, LiveCodeBench.
  Caveats: KodCode/LeetCode too easy for strong bases; APPS/TACO noisy
  (flawed/missing tests) — filtering is mandatory.
- **DeepCoder** (https://www.together.ai/blog/deepcoder) curation recipe is
  the reference: 24K problems programmatically verified against official
  solutions, **≥5 unit tests per problem** (fewer invites hacking), dedup.

**Base models.** Qwen dominates: **Qwen2.5-Coder-Instruct 1.5B/3B/7B** is the
standard small RLVR base; Qwen3-4B/8B and R1-Distill-Qwen for reasoning-style
code RL; SmolLM3-3B as the low-memory option.

**Reward designs that work:**

- **Sparse binary pass/fail on all sampled tests** (DeepCoder: 1 iff all of
  the 15 hardest tests pass; explicitly avoided partial per-test credit
  because it teaches printing public-test answers).
- **Partial/tiered rewards at small scale**: Oxen.ai Rust 1.5B GRPO run
  (https://www.oxen.ai/blog/training-a-rust-1-5b-coder-lm-with-reinforcement-learning-grpo)
  stacked ~5 rewards — format guards, non-empty/assertion checks, build
  pass, lint pass, tests pass — as plain Python functions shelling out via
  subprocess. Build rate 61%→80%, test pass 22%→37%, on one H100, <$100,
  15K examples, num_generations=4.
- **Similarity rewards without execution**: Meta SWE-RL
  (https://github.com/facebookresearch/swe-rl) — sequence similarity between
  predicted and ground-truth patches; 41.0% SWE-bench Verified at 70B.
- **Synthesized tests as rewards**: AceCoder
  (https://tiger-ai-lab.github.io/AceCoder/) — LLM-generated test cases,
  filtered, pass-rate as reward — +25% HumanEval+ on Qwen2.5-Coder base
  **within 80 RL steps**. Cheapest path to self-instruct for verifiable
  code tasks.
- **Format rewards** as small auxiliary term, following R1.

**Sandboxes.** DeepCoder: Together Code Interpreter + local subprocess
sandbox, 6–12 s/test timeouts. Hosted: E2B (Firecracker), Modal (gVisor),
Runloop, Daytona. Small runs mostly use subprocess-in-Docker with resource
limits.

## 3. Practical GRPO on a 24GB GPU (A10G/L4)

**TRL GRPOTrainer** (https://huggingface.co/docs/trl/grpo_trainer):

- **vLLM colocate mode**: `use_vllm=True, vllm_mode="colocate",
  vllm_gpu_memory_utilization=0.2–0.3, vllm_enable_sleep_mode=True` — sleep
  mode offloads vLLM weights during optimizer step.
- **Feasible sizes on 24GB**: full finetune ≈ 0.5–1.5B; LoRA/QLoRA 1.5B–3B
  comfortably, 7B possible with QLoRA + gradient checkpointing. Sweet spot:
  Qwen2.5-Coder-1.5B/3B-Instruct with LoRA r=16–32.
- **Hyperparameters**: `learning_rate=1e-6` (LoRA runs 1e-5–2e-5),
  `num_generations=8` (4 on tight memory), `beta=0.0` (no KL/reference model
  — saves a full model of memory), `loss_type="dapo"`, `epsilon=0.2`,
  `scale_rewards="batch"`, temperature 1.0. Truncated Importance Sampling on
  by default — leave it on.
- **Pitfalls**: (a) reward hacking — length/partial-credit rewards get gamed;
  mitigations: ≥5 hidden tests, all-or-nothing test reward, anti-hack guards,
  DAPO difficulty filtering (drop prompts where all G completions pass or
  all fail — zero advantage wastes batch). (b) entropy/policy collapse —
  watch reward↑ + KL↑ simultaneously; DeepCoder removed entropy+KL losses
  but raised the clip upper bound; opposing school keeps beta≈0.001.
  (c) format collapse — small format reward + good system prompt.
  (d) overlong truncation poisoning — mask loss on truncated sequences.

## 4. The factory automation loop

- **Automated curation**: verify-against-official-solutions + min-test-count
  + dedup gate; DAPO-style online difficulty filtering is the standard
  in-loop curator.
- **Synthetic task generation**: AceCoder (LLM-imagined tests → filtered →
  verifiable rewards), KodCode (447K synthetic triples w/ self-verification),
  SWE-smith (break existing tests in real repos, ~381 tasks/repo).
- **Eval gating before promotion**: run candidate checkpoints through
  held-out suites (HumanEval+/MBPP+/LiveCodeBench window) and block
  promotion below thresholds. Human vibe-check gates persist as a last-mile
  check behind automated rubrics.
- **Cross-lab lessons** (philschmid synthesis): train in the production
  harness ("train where you deploy"), keep rollouts near-on-policy via
  async, expect reward hacking and patch incentives, treat context
  management as learnable behavior.

## Actionable synthesis for our 24GB POC loop

Qwen2.5-Coder-1.5B-Instruct + LoRA (r=16) + TRL GRPOTrainer with vLLM
colocate/sleep-mode; data = KodCode-Light-RL-10K (filtered to ≥5 tests,
mixed difficulty) plus AceCoder-style self-generated tasks; reward = binary
all-tests-pass + small format reward + anti-hack guards, executed in
subprocess sandboxes with per-test timeouts; lr=1e-6–1e-5,
num_generations=4–8, beta=0, loss_type="dapo", scale_rewards="batch"; gate
checkpoint promotion on HumanEval+/MBPP+ subset; expect AceCoder-scale gains
(measurable in ~80–200 steps).
