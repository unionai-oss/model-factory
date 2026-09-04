# 2026-09-04 — round 4: archetype-scale corpus (100k pipelines)

## Context

Target: a synthetic corpus of **100,000 pipelines** with real variety and
complexity. Teacher-per-task doesn't scale — one 27B llama.cpp replica ≈
a minute per generation → 100k ≈ two months of teacher time — so the work
factors into `archetypes × instantiation`:

    teacher writes ~150 PARAMETERIZED archetypes (PARAMS dict, declared
    numeric ranges, the param that dominates memory)
      → AST safety screen
      → oracle CALIBRATION: harness pods measure each archetype at 3
        log-spaced parameter points (real peak RSS + avg CPU)
      → per-archetype linear fit labels sampled instantiations
      → ~100k records; label_source = measured (calibration rows) |
        fitted (interpolations)

Labels stay measurement-anchored; the teacher's own resource guess is
never used. Variants of one archetype share code shape but differ in
PARAMS (memory param sampled log-uniformly), so footprints span orders of
magnitude per archetype.

Reporting reworked per review: `archetype_data_release` shows summary
stats, a rejection-reason histogram, and head/tail of the archetype table
(throttled to one flush per 3s); `publish_synthetic_corpus` reports
dataset summary statistics (rows, archetypes, families, peak-MiB
percentiles, cpu/duration/code-length stats, label sources).

## Run ledger

| run | what | outcome |
|---|---|---|
| [urjghvd6flrj2m29tsjq](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/urjghvd6flrj2m29tsjq) | first 100k launch | ❌ NameError: `run_generated` was imported inside the *other* release task's body — never in scope here (import moved to module top) |
| [up7jxnw2ldj5wsfq7rhg](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/up7jxnw2ldj5wsfq7rhg) | 100k relaunch (150 archetypes, K=3, teacher qwen38-27b) | (in flight — expected several hours: ~150 serialized teacher generations + ~450 calibration pods at concurrency 24 + instantiation) |

Dark chain: the merged corpus publish should fire train (smoke profile)
→ checkpoint → dark eval.

## Also fixed this round

- **Reproducibility bug** exposed by an intermittent test failure:
  `generate_task` seeded its RNG with `hash(family)` — Python string
  hashes are **salted per process**, so "the same seed" produced a
  different template corpus in every process/run. Now `zlib.crc32`.
- Driver memory 4Gi → 8Gi (10⁵-row pandas materialization).
- `_find_params_span` lru-cached (the instantiation hot path parses each
  archetype's AST once, not 10⁵ times).

## Status

- (pending) yield, corpus stats, and dark-chain results on completion.
