# 2026-09-02 — design + scaffold

## Context

Prototype the [AI Resource Tuning PRD](https://app.notion.com/p/AI-Resource-Tuning-3cb8cc06513d81f4b381c8294419a920)'s
Phase-3 RL track: GRPO + LoRA fine-tune a small LLM that reads a Flyte
task's source + input profile and emits right-sized `flyte.Resources`
kwargs. Reward = task succeeds without heavy overprovisioning.

## Research inputs

- PRD (reward formula, bucketed action space, sim-first "cheap loops
  first", quota clamp, rule-based baseline as the bar).
- [flyteplugins-union@niels/get-metrics](https://github.com/unionai/flyteplugins-union/tree/niels/get-metrics):
  `Metrics.get_for_action(run_name, ...)` → dataplane Prometheus series.
  Caveats that shaped the design: attempts <30s answer OutOfRange (so
  corpus workloads hold their footprint ≥60s); pod "used memory" includes
  page cache; `used_cpu_avg` is a 5m irate. Private repo → optional
  dependency, lazy import, rusage fallback.
- unionai fine-tuning examples (workshops `llm-fine-tuning-grpo-code`,
  unionai-examples `rl_grpo_lora`): TRL GRPOTrainer + LoRA recipe, eager
  attention (SDPA nan logits on left-padded GRPO), `with_pip_packages`
  not `with_requirements`, `depends_on` for cross-env calls.
- 2026 RLVR literature sweep: `beta=0`, `loss_type="dapo"`,
  `scale_rewards="batch"`, num_generations 8–16, waste penalty only on
  successful episodes, sim threshold randomization vs Simulation
  Optimization Bias, staged rewards (Kimi k1.5 precedent).

## Key design decisions

1. **Sim-first environment**: template task families with analytic
   footprints; training rewards from the simulator, real cluster episodes
   for validation only (`EpisodeResult` is the seam for later on-policy
   canaries).
2. **Model default `Qwen/Qwen3-1.7B` (text-only), not Qwen3.5**: every
   Qwen3.5 checkpoint is multimodal (`Qwen3_5ForConditionalGeneration`)
   and TRL GRPO fails on the arch — [trl#5269](https://github.com/huggingface/trl/issues/5269),
   [vllm#39993](https://github.com/vllm-project/vllm/issues/39993).
   `MODEL_LADDER` keeps `*-qwen35` rungs for when upstream closes.
3. **GPU: `T4:1` default** (smallest that trains 1.7B fp16 LoRA; keeps
   the tuner off basic-model-factory's scarce A10G pool).
4. **Reward curriculum**: stage A binary success first (prove reward
   moves), stage B composite with asymmetric OOM penalty (worst wasteful
   success must outscore any OOM) and waste counted only on successes.

## Artifacts

- Design doc: [../docs/DESIGN.md](../docs/DESIGN.md)
- PR: [#6 — project scaffold + monorepo reorg](https://github.com/unionai-oss/model-factory/pull/6)
- Cluster project `resource-tuner-model-factory` verified on
  demo.hosted; own `.flyte/config.yaml`.
