"""RightSize Bench: a measured, head-to-head benchmark for resource right-sizing.

Every workload in the bench is real, executable code; every label is the peak
memory and CPU the code actually used when it ran in a pod. Arms -- simple
rules, classical ML, autoregressive LLMs and a Jev-style decision model -- see
only what a scheduler could see at submit time (the task's source, its input
metadata, and the task's own run history) and are scored by what it would
cost to get every run to SUCCEEDED, OOM retries included.

Modules:
    grid        the request grid, prices, and the oracle label
    workloads   workload families: code + inputs + the knobs that drive memory
    measure     runs a workload in a subprocess and measures it
    features    scheduler-visible features (the information boundary)
    prompting   the one prompt every LLM-backed arm reads
    arms        static / history / size-scaling / GBT arms
    llm_arms    zero-shot LLM, SFT generative LLM, Jev-style decision head
    scoring     cost model with retries, regret vs oracle, calibration
    stations    the Flyte tasks
    factory     the bench as a Union factory
"""
