# Research notes: poolside "Model Factory" blog series

Source posts (fetched 2026-08-12):

- https://poolside.ai/blog/introducing-the-model-factory
- https://poolside.ai/blog/gathering-and-processing-raw-materials-for-the-model-factory
- https://poolside.ai/blog/titan-the-model-factory-s-furnace
- https://poolside.ai/blog/designing-a-world-class-code-execution-environment
- https://poolside.ai/blog/the-carrier-and-the-beacon
- https://poolside.ai/blog/post-training-in-the-model-factory

## Post 1: "Introducing the Model Factory"

**Component and role.** The framing post for the six-part series. The Model
Factory is poolside's end-to-end internal platform for producing foundation
models, structured as a closed loop: data ingestion → preprocessing → training
→ evaluation → RL → synthetic data generation → data mixing, and back around.
The central thesis is that model production should be a *factory* — codified,
reproducible, largely automated — rather than a series of artisanal one-off
runs.

**Architecture details.**

- Runs on a ~10,000-GPU NVIDIA H200 cluster orchestrated by Kubernetes; Helm
  charts for deployment, Terraform for cluster management; heterogeneous node
  pools (GPU + CPU); elastic task scaling to keep GPUs from idling.
- Data layer: Apache Iceberg data lake (versioning + lineage) with Dagster
  orchestrating both data workflows and training pipelines; Spark for scalable
  data experiments. A key decision: data is **streamed** into training rather
  than pre-materialized as fixed datasets, so data mixes can be adjusted
  flexibly per run.
- Training: **Titan**, a PyTorch-based library with custom CUDA/Triton kernels
  plus torch.compile; metrics stream to Neptune for cross-run comparison;
  Grafana for system health.
- Inference: **Atlas**, an inference wrapper spanning NVIDIA, AMD, and AWS
  Trainium accelerators, serving evals, production inference, and pre-training
  experiments on dedicated or elastic schedules.
- Code execution platform: **~1 million GitHub repositories containerized with
  executable test suites** (OCI containers built via heuristic rules, later
  augmented by an agent-based builder), providing the secure isolated
  environments that underpin RL from code execution.

**Humans vs. automation.** Automation wins cited: deduplication went from 1+
week of effort to one click; experiment scheduling from weeks to under 10
minutes; evaluations auto-trigger every 100–1,000 training steps; conditional
asset validation automatically excludes low-quality datasets. Humans remain
for company-wide "vibe checking" of models (engineers through go-to-market),
manual model querying/comparison via the internal **Podium** tool, and a QA
verification platform that deploys fresh checkpoints for stress testing.

**Reward/synthetic data.** Introduces RL-from-code-execution at a high level:
a modular RL system with a **Task Engine orchestrating "millions of
complicated agentic workflows,"** using test-suite execution inside
containerized repos as the feedback mechanism. Synthetic data: the inference
engine is integrated into the Factory and has generated **hundreds of billions
of synthetic tokens**, scheduled through the same orchestrator as everything
else.

## Post 2: "Gathering and Processing Raw Materials"

**Component and role.** The data pipelines — the "raw materials" stage:
ingestion, synthetic dataset generation, quality filtering, deduplication,
code dependency ordering, document packing, and dataset blending for training
consumption.

**Architecture details.**

- **Iceberg tables are the universal dataset representation**; every asset is
  immutable with full lineage/provenance, so researchers can trace how any
  processing decision affected results.
- **Dagster** orchestrates via declarative Python asset definitions; **Apache
  Spark** provides distributed compute with **~20 trillion tokens/day
  ingestion capacity**.
- **Blender**: a custom gRPC-based data *streaming* service that handles
  dataset configuration and fetching — the delivery mechanism between the data
  lake and training (and later, between RL actors and trainers).
- Algorithmic choices: **weighted MinHash** for dedup (chosen over SimHash for
  better precision across similarity thresholds); **best-fit bin packing** for
  document packing (~100% context-window utilization vs. 85% with next-fit);
  **topological sort via DFS** to order code files by dependency; **two-stage
  k-means clustering** (15K coarse clusters, then 3M fine-grained) for
  document categorization.
- Filtering results: ~5% average improvement on pre-training benchmarks (up to
  150% on individual metrics); successive filtering passes removed 24%, 19%,
  and 14% of data; some final filtered datasets are single-digit percentages
  of the original size.

**Humans vs. automation.** Automated: synthetic generation (fully automatic
once parameterized), heuristic/metadata filtering, clustering, dedup,
dependency sorting, packing. Human: the data team visually inspects datasets
during ingestion, iterates on filtering steps, reviews synthetic data quality,
and decides which filters apply to which datasets.

**Synthetic data / batch inference.** One concrete recipe: execute predefined
tests against repository inputs and **record the outputs as training data
teaching models to reason about code execution**. Generation workers run as
load-balanced, preemptible services. Crucially, synthetic and real datasets
flow through *identical* downstream pipeline stages.

## Post 3: "Titan: The Model Factory's Furnace"

**Component and role.** Titan is the distributed training codebase —
pre-training, mid-training, SFT, and the training half of RL.

**Architecture details.**

- Built on **TorchTitan** (the PyTorch team's open-source project), with
  selectively migrated components from their predecessor codebase "Monster"
  and a custom **mixture-of-experts stack**. PyTorch-native distributed
  training with multiple parallelism strategies; scales from one machine to
  the 10K-H200 cluster; Kubernetes + torchrun for node scheduling.
- Deep Dagster integration: training jobs launch from the Dagster UI or CI;
  every run produces **versioned Dagster assets** giving historical lineage of
  experiments.
- Experimentation workflow: scaling-law studies with hyperparameter-transfer
  techniques (**mu-P family**); thousands of small configs generated
  programmatically, each a versioned asset; checkpoint evals auto-trigger
  (sometimes every few hundred steps); metrics stream to Neptune (loss curves,
  iteration timing, hardware utilization). A batch-size sweep is "define a
  parameter dict, click a button, launch hundreds of parallel experiments."
- Reliability at production scale: pre-flight node stress tests catch
  hardware/network issues before launch; logs to Sentry; PyTorch
  **FlightRecorder** for retrospective debugging; Grafana metrics.
  **Automated recovery** handles faulty nodes and stalls without humans —
  auto-restart after a configurable window (~10 min); unrecoverable errors
  escalate via **incident.io** (soft alert at 30 min, engineer page at ~60
  min, auto-created Slack incident channels).

**Humans vs. automation.** Automated: experiment launching, eval scheduling,
checkpointing, standard failure recovery, metrics. Human: config/architecture
design, rare failure investigation, on-call. Their first production model
(Malibu v1) required "weeks of near-continuous babysitting"; the automation
stack exists to eliminate that.

**Design lesson.** They explicitly traded hyper-optimization for flexibility:
Monster's early micro-optimizations made the code rigid, and (Amdahl's law)
microbenchmark wins didn't translate to end-to-end gains.

## Post 4: "Designing a World-Class Code Execution Environment"

**Component and role.** The infrastructure enabling **RLCEF (Reinforcement
Learning via Code Execution Feedback)** — poolside's core training innovation.
It supplies reproducible execution environments for **800K+ repositories** and
turns real code execution into reward signal.

**Architecture details.**

- **Saucer** (repository serving): gRPC service providing efficient file
  access at *any* repository revision. Ingestion is triggered via **Redpanda
  (Kafka-compatible) topics** for auditability and deterministic replay;
  two-stage downloads (parallel runners with single-worker fallback); git
  packfiles + index files in a read-optimized layout.
- **Image building pipeline**: a heuristic-rules layer detects build systems
  (Cargo, pyproject.toml, CMake/Makefiles, etc.) and produces OCI images; an
  **AI-agent layer** handles repos the heuristics can't — reading READMEs,
  iterating on build failures, resolving gnarly C++ dependencies. Notably,
  **that build agent was itself trained in the Factory** — the factory
  improving its own tooling. Images land in a **custom OCI registry backed by
  S3**. Build pipeline also captures test-suite execution and coverage
  baselines.
- **Revision handling via layers**: diffs between commits become separate OCI
  image layers, composed at runtime with **Linux OverlayFS** — one layered
  image per repo instead of a full image per revision.
- **Execution service**: gRPC endpoints for low-level ops (filesystem
  mutation, arbitrary commands) plus high-level abstractions via the **Task
  Engine** (run tests, lint, report coverage).

**Reward signals (the most concrete in the series).** Execution feedback
comprises **test pass/fail results, test coverage metrics, compilation/build
error messages, and linter output** — the raw material for code reward
functions.

**Humans vs. automation.** Almost fully automated (ingestion, build detection,
test execution, registry ops); Saucer ingestions are deliberately
human-triggered as a safety choice; the AI agent covers the long tail of hard
builds.

## Post 5: "The Carrier and the Beacon" (Atlas + Evaluation)

**Component and role.** **Atlas** ("the carrier") is the inference codebase;
the evaluation platform ("the beacon") tells them whether training is on
course.

**Architecture details.**

- Atlas is a **composition-based library with swappable components**, wrapping
  open-source engines — **notably vLLM** — alongside custom implementations
  behind generic interfaces. Supports **CUDA, ROCm, and AWS Trainium**;
  per-platform custom kernels; whole-system profiling down to NUMA affinity
  and GC tuning.
- Scheduling: **Kubernetes + Volcano** (batch scheduler) — priority queues,
  backfill, evictable/preemptible jobs so urgent work can displace batch
  inference, automatic TTL updates. Dynamic GPU allocation coexists with
  static reservations. Dagster-integrated.
- Checkpoints reach inference three ways: **direct checkpoint identifiers,
  Dagster asset materialization, or direct GPU-to-GPU weight transfer** (the
  last is the RL fast path).
- Evaluation: **evals are native Factory (Dagster) assets**, independently
  schedulable and parallelized across checkpoints. Checkpoint notifications
  from training **automatically trigger dependent eval jobs**; results push to
  Neptune. Early-training evals target foundational capabilities (math,
  language understanding) as leading indicators for downstream
  code-engineering skill.

**Humans vs. automation.** Manual GPU allocation and checkpoint tracking are
eliminated; human involvement in the eval loop reduces to error recovery plus
qualitative vibe-checking.

## Post 6: "Post-Training in the Model Factory"

**Component and role.** SFT + RL: turning base models into usable assistants;
the stage where the whole factory's components compose into what "almost
operates as its own sub-factory."

**Architecture details.**

- **SFT** is pure component reuse: Blender streams datasets, Titan trains from
  pre-trained checkpoints, the eval platform benchmarks, checkpoints are
  stored as assets.
- **RLCEF architecture — the off-policy actor/trainer split**: **actors**
  solve tasks (agentic rollouts against the code execution environment) and
  **stream session transcripts to Blender**; **training nodes** consume those
  transcripts and update weights. Acting and training scale **independently**
  — poolside explicitly chose asynchronous off-policy RL "for scalability
  reasons," rejecting co-located training+inference because it bottlenecks
  both and prevents independent scaling.
- **Weight sync**: direct **GPU-to-GPU weight transfer** (no storage hop) on
  **P5e nodes (8× H200, ~3200 Gbps via EFAv3)**, so actors get fresh policies
  quickly while training continues asynchronously.
- Scale: the **Task Engine handles tens of millions of concurrent code
  execution tasks**; algorithms like **GRPO, PPO, and actor-critic loops**
  slot modularly into the orchestration. Full reproducibility via pinned
  Docker images + config management.
- **Synthetic SFT data generation**: models generate function arguments, the
  **code execution service computes the ground-truth outputs**, and Blender
  streams the resulting dataset to training — execution as the oracle, not a
  judge model.

**Humans vs. automation.** Researchers spend significant deliberate time in
**Podium** (internal Iceberg-table dataset viewer) manually inspecting
fine-tuning data — catching malformed samples, iterating on generation recipes
*before* scaling them, and tracking model evolution across checkpoints. Human
feedback is essentially curation and vibe-checking; the reward comes from
execution, not annotators.

---

# Consolidated reference architecture: the poolside loop

**The loop:** raw + synthetic data → pipelines (filter/dedup/pack/blend) →
streamed into training → checkpoints → inference deployments → (a) automated
evals, (b) RL actor rollouts against real repos, (c) synthetic data generation
→ execution feedback as reward / new tokens → back into data and training.

**Named components and their generic roles:**

1. **Orchestration spine — Dagster + Kubernetes.** Everything — datasets,
   training runs, checkpoints, evals — is a *versioned, declarative asset*
   with lineage. Asset dependencies drive automation: a materialized
   checkpoint automatically triggers downstream evals. Load-bearing
   requirements when swapping orchestrators: typed/versioned artifacts,
   lineage, event-driven downstream triggering, config-as-code
   reproducibility, conditional validation gates that quarantine bad assets.
2. **Data lake — Apache Iceberg** as the universal dataset format; immutable,
   provenance-complete. **Spark** for heavy transforms.
3. **Data delivery — Blender**, a custom gRPC streaming service. Training
   consumes streams, not materialized files; the same channel carries RL
   transcripts from actors to trainers.
4. **Training — Titan** (TorchTitan-based PyTorch, mu-P scaling transfer),
   with pre-flight node checks, auto-recovery, and
   incident.io/Sentry/Grafana/Neptune observability.
5. **Inference — Atlas**, a composition layer over vLLM + custom backends,
   scheduled with priority/preemption/backfill so batch inference (synthetic
   data, evals) soaks up idle GPUs and yields to urgent jobs.
6. **Code execution environment — Saucer + image pipeline + Task Engine.**
   Real repos as OCI images, per-revision OverlayFS layers, gRPC execution
   service. Simultaneously the **reward function** (test pass/fail, coverage,
   compile errors, lint), the **synthetic data oracle** (execute code, record
   outputs), and the **agent playground**.
7. **RL — RLCEF with off-policy actor/trainer split.** Actors roll out agentic
   coding tasks, stream transcripts; trainers (GRPO/PPO) consume
   asynchronously; fresh weights flow back via GPU-to-GPU sync.
8. **Evaluation — evals as first-class assets**, auto-triggered per
   checkpoint, reusing inference + execution infra.
9. **Human layer — Podium** (dataset/model inspection UI) plus company-wide
   qualitative testing. Humans design configs, curate data recipes, review
   synthetic data before scaling, and handle rare incidents.

**Signature ideas worth carrying to a re-implementation:**
everything-as-a-versioned-asset on one orchestrator; streaming data delivery;
real repositories with executable tests as the reward source; off-policy
actor/trainer decoupling with weight sync; batch inference as a preemptible
filler workload; humans positioned at data curation and recipe design rather
than in the execution path.
