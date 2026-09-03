# resource-tuner-model-factory — system design

Status: v1, 2026-09-02.
Product context: the **AI Resource Tuning** PRD (Notion, `unionai/prds` →
`product_prds/ai_resource_tuning/prd.md`). This repo is a prototype of the
PRD's **Phase 3 RL track**: fine-tune a small LLM that proposes
`flyte.Resources` kwargs for Flyte tasks such that the task succeeds
without heavy overprovisioning.

## 1. The problem as an MDP

| Element | Definition |
|---|---|
| State | Estimation context: task source code + input profile (+ prior/history later) |
| Action | A `flyte.Resources` kwargs dict, bucketed (`{"cpu": 2, "memory": "4Gi"}`) |
| Episode | One task run reaching a terminal phase |
| Reward | `w1·run_success − w2·overprovision − w3·oom_penalty − w4·throttle` |

Memory is incompressible (undershoot → OOMKilled → failed run); CPU is
compressible (undershoot → throttling → slower run). The reward encodes
that asymmetry: an OOM must always score below the most wasteful success,
or the policy learns that failing cheaply beats fitting generously.

## 2. Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │            tuner_pipeline (driver, CPU)      │
                    └──┬─────────────────┬──────────────────┬──────┘
                       ▼                 ▼                  ▼
              build_task_corpus     train_tuner         eval_tuner
              (CPU, taskgen)        (GPU, TRL GRPO      (GPU: greedy gen,
                       │             + LoRA, sim         sim scoring,
              [tuning-task-corpus]   rewards)            + REAL episodes)
                                        │                   │     │
                                 [tuner-checkpoint]         │  fan-out
                                                            │     ▼
                                              [tuner-eval-report] run_generated
                                                                  (harness pods,
                                                                   resources =
                                                                   the proposal)
```

Five subsystems, mirroring basic-model-factory's station layout:

- **taskgen/** — synthetic task corpus. Each template family renders
  realistic Flyte 2 code (pandas ETL, sklearn fits, torch training loops,
  batch inference, ETL) whose true peak memory / CPU demand is an analytic
  function of sampled parameters. One record carries two renderings: the
  policy view (a Flyte task module) and the harness view (a plain function
  the episode pod executes).
- **environment/** — two interchangeable episode backends returning the
  same `EpisodeResult` shape:
  - `simulator.py` — the cheap loop: score a proposal against the analytic
    footprint, sub-millisecond. Training runs ~100% here.
  - `harness.py` + `episodes.py` — the real loop: a single generic
    `run_generated` task, deployed once; every episode calls it with
    `.override(resources=flyte.Resources(**proposal))` so the proposal is
    the pod's actual request. OOM → failed child action → negative signal
    (retries=0, deliberately). The harness self-reports peak RSS via
    getrusage.
  - `metrics.py` — pod-truth cross-check via **flyteplugins-union @
    niels/get-metrics** (`Metrics.get_for_action`): dataplane Prometheus
    series per action, fetched post-hoc for finished runs.
- **policy/** — the I/O contract: prompt rendering (context → chat
  messages), completion parsing (lenient wrapping / strict content), and
  the bucketed action grid (memory on a log2 grid 128Mi–64Gi, CPU on fixed
  increments — PRD §8 label derivation).
- **rewards/** — the staged curriculum (§4).
- **training/** — TRL `GRPOTrainer` + LoRA, the rule-based baseline, and
  evaluation.

### Why sim-first (and what keeps the sim honest)

Real episodes cost minutes and dollars; GRPO wants
`num_generations × batch` episodes per step. A synchronous trainer blocked
on cluster pods would idle the GPU ~100% of the time (Red Hat's Async-GRPO
measured 11–12.5× losses from in-loop slow rewards). So: **train against
the simulator, validate against the cluster.** The known failure mode is
Simulation Optimization Bias — the policy exploiting simulator quirks — and
three defenses are built in:

1. **Threshold randomization**: the sim's OOM boundary jitters within
   `JITTER_BAND` during training, so there is no deterministic edge to ride.
2. **Sim-to-real probes**: every eval runs a batch of real episodes under
   the policy's proposals and reports analytic-vs-measured peak RSS
   side by side; a growing gap means the footprint model needs refitting.
3. **Self-consistency**: rewards are computed against the same formulas
   that labeled the corpus, so footprint-model error cannot corrupt the
   learning signal — it only shifts what "success" means, which the real
   episodes then measure.

If/when the loop needs real episodes *inside* training (PRD's canary
tasks), the pattern is env-as-a-service (verifiers/Atropos style): fan out
all candidates of a step as concurrent harness pods, train one step
off-policy while the next step's pods run. The `EpisodeResult` seam is
where that plugs in; the prototype deliberately does not build it.

## 3. Model choice

Requirement: ≤9B open weights, the smaller the better; explore Qwen3.5
(0.8B/2B/4B/9B) but stay parameterized (`RT_MODEL` / `MODEL_LADDER`).

**Default: `Qwen/Qwen3-1.7B` (text-only), LoRA r=16 on attention
projections, bf16 on Ampere / fp16 on Turing.** Two findings force the
text-only default for now:

- Every Qwen3.5 checkpoint — including 0.8B — is natively multimodal
  (`Qwen3_5ForConditionalGeneration`, hybrid GatedDeltaNet attention).
  TRL GRPO fails on the arch today (weights nested under
  `language_model.*`: trl#5269 open; no vLLM text-only path: vllm#39993),
  and 2026 RL recipes still default to Qwen3 text bases. The ladder keeps
  the Qwen3.5 rungs (`*-qwen35`) for the moment upstream closes those.
- Sub-1B models are fine for format acquisition but weak at the actual
  skill (reading code and reasoning about data sizes); 1.7B–4B is the
  reported sweet spot for cheap-but-capable GRPO. The smoke profile can
  still run `Qwen/Qwen3-0.6B` (`RT_MODEL`) for pipeline plumbing tests.

QLoRA (nf4 double-quant) is wired for ladder rungs that don't fit a small
GPU in 16-bit; below ~2B it costs quality for no needed VRAM.

**GPU: smallest that trains.** Default `T4:1` (g4dn.xlarge pool on demo —
plentiful, and 16GB fits 1.7B fp16 + LoRA with 128-token completions).
T4 is Turing: no bf16, so dtype follows the device. `RT_TRAIN_GPU=A10G:1`
for bf16 or bigger rungs. This also keeps the tuner factory off the single
spare A10G that basic-model-factory's training contends for.

## 4. Reward curriculum

Stage A (`success`) — prove the fundamental loop: reward = format(0.1) +
success(1.0), binary, unhackable. The go/no-go signal is mean reward
climbing over ~10–150 steps. Nothing else is trained until this moves.

Stage B (`composite`) — the PRD formula, with the constraints research
says keep it stable:

- waste penalty ≤ 0.5, computed **only on successful episodes** (waste on
  a failed run would reward failing cheaply), mean of memory + CPU
  overprovision fractions;
- OOM penalty asymmetric: worst-case success (1.1 − 0.55 = 0.55) still
  beats any OOM (0.1 − 0.5 = −0.4);
- throttle is a nudge (−0.05), not a failure — CPU is compressible;
- every component logged separately, so curves show which term moved.

The waste penalty doubles as the fix for GRPO's all-pass degenerate
groups: once most proposals fit, waste ranks them within the group and
restores advantage variance. Trainer knobs follow the 2026 small-model
consensus: `beta=0`, `loss_type="dapo"`, `scale_rewards="batch"`,
`num_generations` 8–16, temperature 1.0, lr 5e-6–1e-5.

Rollouts are **unconstrained** (format failures must be *seen* to be
trained away); grammar-constrained decoding belongs on the serving path
only (PRD: proposals schema-validated regardless).

## 5. Evaluation and the promotion bar

Every checkpoint is scored on the held-out corpus split against the
**rule-based baseline** (per-family median footprint + 25% margin,
bucketed — the honest cold-start version of Autopilot/VPA-style
percentile+margin). The PRD's rule: a learned model that can't beat the
percentile baseline on both OOM-risk and waste has no reason to exist.

Report metrics (tuner-eval-report artifact): schema validity, success
rate, median overprovision % (policy vs baseline), and the real-episode
sim-vs-real table. Auto-gate: validity ≥ 95% AND success ≥ baseline AND
waste ≤ baseline.

## 6. Metrics plugin integration

`flyteplugins-union @ niels/get-metrics` supplies
`Metrics.get_for_action(run_name, action_name)` → per-pod Prometheus
series (`used_memory_bytes_avg`, `used_cpu_avg`, `request_*`, GPU set).
Facts that shaped the design:

- attempts under ~30s answer OutOfRange → every corpus workload holds its
  footprint ≥60s (`_hold_loop`);
- pod "used memory" includes page cache → harness rusage is the
  working-set number; pod series are the OOM-relevant cross-check;
- `used_cpu_avg` is a 5m irate → corpus CPU demand is sustained, not bursty;
- the repo is **private** → optional dependency (`--extra metrics`,
  needs GitHub credentials), lazy-imported, and everything degrades to
  rusage-only when absent. Task images install it only when
  `RT_WITH_METRICS=1` (+ `RT_GH_TOKEN`) at deploy time.

## 7. Roadmap (post-prototype)

1. Corpus realism: GPU task families; footprint constants refitted from
   harness measurements (closing the sim-to-real loop the same way the
   PRD's production loop closes on fleet telemetry).
2. OnArtifact trigger wiring + HITL promotion gate (mechanical port from
   basic-model-factory).
3. RFT/DPO on episode logs (zero extra live runs), then canary on-policy
   episodes inside training via the env-as-a-service seam.
4. Serve the promoted tuner behind the PRD's tune service; distillation
   ladder 4B → 1.7B → 0.6B.
5. Swap the ladder to Qwen3.5 when trl#5269 / vllm#39993 close.
