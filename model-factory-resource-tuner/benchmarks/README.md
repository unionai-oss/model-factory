# Benchmarks

Self-contained HTML results pages for the resource-tuner's RightSize benches.
Each bench has a long-form report (datasets, cost model, how every approach
works, full leaderboard) and a one-page abstract. Open a file in a browser;
nothing is fetched at view time beyond Google Fonts.

The runs, findings and corrections behind every number live in
[`model-factory-resource-tuner/research_log/`](../research_log/README.md).

| bench | release | measured runs | report | abstract | log entry |
|---|---|---|---|---|---|
| RightSizeMem | v2 | 14,234 | [rightsize-mem-bench.html](rightsize-mem-bench.html) | [abstract](rightsize-mem-bench-abstract.html) | [round 17](../research_log/2026-10-15-round-17-rightsize-bench.md) |
| RightSizeCPU | small-1 | 220 | [rightsize-cpu-bench.html](rightsize-cpu-bench.html) | [abstract](rightsize-cpu-bench-abstract.html) | [round 18](../research_log/2026-10-23-round-18-rightsize-cpu-gpu.md) |
| RightSizeGPU | small-1 | 128 | [rightsize-gpu-bench.html](rightsize-gpu-bench.html) | [abstract](rightsize-gpu-bench-abstract.html) | [round 18](../research_log/2026-10-23-round-18-rightsize-cpu-gpu.md) |
| RightSizeCPU | big-1 | 9,900 | [rightsize-cpu-bench-big.html](rightsize-cpu-bench-big.html) | [abstract](rightsize-cpu-bench-big-abstract.html) | [round 20](../research_log/2026-10-24-round-20-rightsize-big-and-combined.md) |
| RightSizeGPU | big-1 | 10,000 | [rightsize-gpu-bench-big.html](rightsize-gpu-bench-big.html) | [abstract](rightsize-gpu-bench-big-abstract.html) | [round 20](../research_log/2026-10-24-round-20-rightsize-big-and-combined.md) |
| Combined (memory + CPU + GPU, with RL) | big-1 | 33,700 | [rightsize-bench-combined.html](rightsize-bench-combined.html) | [abstract](rightsize-bench-combined-abstract.html) | [round 20](../research_log/2026-10-24-round-20-rightsize-big-and-combined.md) |
| RightSizeCPU, with trees x LLM hybrids | big-1 (round 21) | 9,900 | [rightsize-cpu-bench-big-hybrids.html](rightsize-cpu-bench-big-hybrids.html) | [abstract](rightsize-cpu-bench-big-hybrids-abstract.html) | [round 21](../research_log/2026-10-26-round-21-gpu-big-2-hybrids.md) |
| RightSizeGPU, whole machine ladder, with hybrids | big-2 | 10,000 | [rightsize-gpu-bench-big-2.html](rightsize-gpu-bench-big-2.html) | [abstract](rightsize-gpu-bench-big-2-abstract.html) | [round 21](../research_log/2026-10-26-round-21-gpu-big-2-hybrids.md) |
| Combined (memory + CPU + GPU, with RL and hybrids) | big-2 | 32,082 | [rightsize-bench-combined-big-2.html](rightsize-bench-combined-big-2.html) | [abstract](rightsize-bench-combined-big-2-abstract.html) | [round 21](../research_log/2026-10-26-round-21-gpu-big-2-hybrids.md) |

The round-19 full releases (CPU full-1, GPU full-1) have no pages of their
own; their numbers appear as the comparison columns in the big-1 reports.
