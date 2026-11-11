# Round 23: review, the twin in real pods, and the k8s-2 release

## Context

Round 22 produced a paper draft (`benchmarks/rightsize-bench-k8s.html`).
Before it goes out, this round reviews it as a referee would. An
independent reviewer agent read the paper and the code, and every claim it
made was checked against the code and the data. The round then fixes what
the review found, measuring where measuring was possible, and builds a final
release, **k8s-2**, from the corrected code. The paper is rewritten to put
Kubernetes first: the workloads are Kubernetes pods; a raw Job and a Flyte
task are two ways to submit them, and Flyte is the interface that adds
typed I/O, caching, retries and durable execution on top.

The questions:

1. Do the round-22 numbers hold under a task-clustered bootstrap, and is
   the pair metric measuring what the paper says?
2. Do the twin's labels hold in **real pods**: Job-shaped pods with an init
   container, a real `emptyDir`, and per-container cgroups? (rt-k8sval-2 ran
   everything as sibling processes in one Flyte pod.)
3. Is the Flyte runtime's 180 MiB real?
4. Which approach results change once the review's implementation defects
   are fixed? The defects: zero-shot parsing, the Flyte-only ablation tuned
   on Kubernetes rows, the kNN scale, the adapter's CPU, the baselines, and
   the cache keys.

## Run ledger

| run | what | outcome |
|---|---|---|
| [rt-paper-r2-rows](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-r2-rows), [-picks](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-r2-picks), [-prompts](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-r2-prompts) | `paper.paper_export` on rsk-k8s-1-gate. Exports every evaluated row's scoring columns, every arm's request per run, and prompt lengths, inline | succeeded. Re-scoring offline reproduces every scorecard exactly (Findings 1-3) |
| [rt-k8spod-pilot-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8spod-pilot-1) | `k8s_probe.validate_pods` pilot (25 runs): a real Flyte task pod plus a Job-shaped pod per run | **every job pod rejected**: the namespace's LimitRange gives containers without a CPU limit a CPU limit of 0. Fix: explicit CPU limits |
| [rt-k8spod-pilot-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8spod-pilot-2) | the same, fixed | succeeded, and found that **k8s-1's init-tmpfs manifests do not run** (Finding 5). Also three harness bugs: `job.py`'s own `/data/out/result.json` overwrote the probe's; a process pool cannot pickle a function from an `exec`'d module; spot scheduling failures. Fixed |
| [rt-k8spod-pilot-3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8spod-pilot-3) | the same, fixed, plus init-tmpfs at a sized init limit and at k8s-1's 256Mi, and the job at its label without the staged bytes | succeeded (25 runs) |
| [rt-k8spod-1](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-k8spod-1) | **the twin in real pods**: 100 runs (48 RightSizeMem, 44 RightSizeCPU, 8 init-tmpfs with inputs over 300 MiB). Each run as a real Flyte task, and as its Job pod at a generous limit, at its label cell and one cell below. init-tmpfs also with k8s-1's 256Mi init container, and at its label without the staged bytes. About 480 pods | succeeded: Findings 4-6 |
| [rsk-k8s-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-2) | **k8s-2**: the corrected corpus (measured Flyte runtime, sized init containers) and every arm, seed 1, with the new arms (rules-sized, single-platform trees, jev-pointer-k8s, zeroshot-27b with room to answer, zeroshot-27b-json) and the counterfactual station | succeeded, about 15 h (the counterfactual station alone took about 3 h): Findings 7-12 |
| [rsk-k8s-2-s2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-2-s2), [rsk-k8s-2-s3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-2-s3) | the same corpus, retrained at seeds 2 and 3 (release `k8s-2-s<k>`, `stations.seed_of`), every arm but zero-shot: three trainings of k8s-2 | both succeeded: Finding 12 |

| [rt-paper-k2s1-rows](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s1-rows), [-picks](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s1-picks), [-prompts](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s1-prompts), [rt-paper-k2s1g-picks](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s1g-picks), [rt-paper-k2s2g-picks](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s2g-picks), [rt-paper-k2s3g-picks](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s3g-picks) | `paper_export` on the three k8s-2 runs (the `g` exports include the gate's arm) | succeeded |
| [rt-paper-k2s1-cf](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s1-cf), [rt-paper-k2s2-cf](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s2-cf), [rt-paper-k2s3-cf](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rt-paper-k2s3-cf) | `paper_file` on each counterfactual station's output | succeeded |

## Findings

### 1. The round-22 numbers under a task-clustered bootstrap (replicate 2)

Intervals are 95%, Poisson bootstrap over tasks; a run, its siblings and
its twin resample together.

- The decision model's K8s-minus-Flyte gap on test is +1.1 [−0.0, +2.3]
  points. RL's is −0.4 [−1.1, +0.4]. The trees' is −3.0 [−3.9, −2.1] and
  the rules' +3.0 [+1.3, +4.5]: both nonzero, both within 3. jev-pointer
  argmax is +6.9 [5.1, 8.7]. The Flyte-only model's is +2.9 [1.4, 4.4],
  and the Flyte-only adapted one's +5.4 [4.1, 6.8].
- The top of the test leaderboard overlaps. On K8s test rows:
  - pool × decision: 7.0 [5.9, 7.9]
  - RL argmax: 7.1 [5.7, 8.4]
  - pool × RL argmax: 7.3 [6.0, 8.5]
  - gate: 7.8 [6.6, 9.1]

  The gate is not best on K8s test rows.
- Zero-shot 27B's intervals span about 40 points (95.5% [76.4, 115.6] on K8s test).

### 2. The pair metric was measuring the runtime, not the manifest

Of the 3,950 test CPU-node run–twin pairs, the label moves in 69% of them,
and 88% of those moves are **down**: the Job does not carry the Flyte
runtime. Only 326 moves are up, from staged bytes on tmpfs. An arm that
gives the run and its twin the same request scores 31.0% (the unmoved
share). On the pairs whose label moved:

| arm | moved, direction right | moved up (tmpfs) | unmoved, kept |
|---|---|---|---|
| jev-pointer | 82.3% | 79.8% | 64.9% |
| gbt-dist | 82.3% | 75.8% | 65.1% |
| pool × RL argmax | 86.4% | 85.0% | 71.3% |
| rules | 25.7% | 32.2% | 83.0% |
| jev-pointer-flyte | 28.4% | 46.0% | 58.7% |

So round 22's "reads the one manifest line" was mostly "knows a Job has no
Flyte runtime". k8s-2 adds a controlled test (`counterfactual`): the same
run rendered under all four stagings, with the platform held fixed.

### 3. Truncation never happens; zero-shot never finished

- **Truncation.** No evaluated prompt reaches 3,072 tokens. The longest is
  about 2.5k (Qwen3-1.7B tokenizer, exact on 1,020 sampled prompts,
  estimated for all 28,058). K8s median 1,394-1,814, Flyte 994-1,425. So
  `import boto3` and `medium: Memory` are never cut.
- **Zero-shot.** It failed to parse 444 of 560 answers. The cause: qwen38-27b
  reasons in plain text even with thinking off, and stopped mid-reasoning
  at `max_tokens=1024`. The stored tails end mid-sentence. Its round-22 row
  measures the 4Gi / 1-core fallback.
- **Parser.** `parse_answer` also read `"cpu": "500m"` as 8 cores, and
  `"2 GiB"` and `"2GB"` as unparseable.

### 4. The Flyte runtime is 55-70 MiB, not 180 (rt-k8spod-1)

Each run was measured as a real Flyte task: the rendered task module, its
`File`s downloaded by Flyte, in a fresh task pod, with the container
cgroup's anon + shmem sampled from task start.

- **At rest.** The container holds 55.0 MiB before the workload's module is
  imported (p10-p90 54.9-55.0, n=100).
- **Effective.** The container peak exceeds the same run's Job workload
  peak by a median 70 MiB (p10 53, p90 242). Running the workload inside
  the runtime's process costs a little more than the runtime at rest.
- **The old constant.** The 180 MiB added since round 17 has no probe
  behind it: `measure_runtime_overhead`, which its comment cites, never
  existed.
- **Labels.** Against the real Flyte container's cell, labels with
  R = 180 are in the same cell for 23% of runs and one over for 70%. With
  R = 70: 58% same, 33% over, 9% under. The unders are process pools (the
  container counts every worker) and DuckDB and polars runs.
- `k8s.FLYTE_RUNTIME_MIB = 70` for k8s-2. It moves 61% of the Flyte
  CPU-node labels down one cell.

### 5. The twin in real pods (rt-k8spod-1)

- **Where tmpfs pages are charged.** A tmpfs page is charged to the
  container that writes it.
  - boto3-tmpfs: the job container's shmem rises by the staged bytes ×
    1.0001 (n=23).
  - init-tmpfs: the **init** container's shmem is the staged bytes ×
    1.0000, and the job container's is 0 (n=31).
- **k8s-1's init-tmpfs manifests fail.** With the manifest's 256Mi init
  container, every staging up to 162 MiB ran. None of 254 MiB or more did:
  10 were confirmed OOMKilled in `stage-inputs`, and 6 more failed with
  truncated errors. In k8s-1, 982 of the 2,756 evaluated init-tmpfs twins
  stage more than 200 MiB: 7% of all K8s rows describe Jobs that could not
  have run.
  - The k8s-2 manifest sizes the init container's limit at 256 MiB plus
    the staged bytes (`k8s.init_limit_mib`).
- **Pages still count against the pod.** The pages outlive the init
  container (they are reparented to the pod), and still count:
  - at its label minus the staged bytes, the job was OOMKilled in 9 of 31
    init-tmpfs pods;
  - at its full label, 30 of 31 ran.

  So the label's "+ staged bytes" is right, by way of the pod's limit, not
  the container's.
- **Labels as limits.** With the job container's limit at its label cell,
  98 of 100 runs succeeded. The two OOMKilled are both `process_pool`: the
  label is `ru_maxrss` of the largest process, while the container counts
  all of them. One cell below the label, 25 of the 97 runs that
  ran were OOMKilled.
- **Cells against the container cgroup.** Twin label from the corpus vs
  the job container's own anon + shmem peak (× 1.05): same cell for 52-68%
  of runs by staging, within one cell for 91-100%, one or more cells over
  for 26-48%, under for 0-6%.
  - The over-cells are the `ru_maxrss` floor (Finding 2 of round 22) and
    file-mapped pages: `numpy_linalg` and `image_batch` mmap their inputs,
    which RSS counts and the kernel reclaims.
  - rt-k8sval-2's 90.3% same-cell used the same RSS instrument for both
    sides, so it could not see this.
  - The K8s labels are conservative by about one cell for a third of runs.
    The oracle inherits that.

### 6. Findings from the review, checked

| review claim | verdict | what changed |
|---|---|---|
| The Flyte-only ablation is calibrated and pooled on K8s val rows | **true** (`_policy_predictions`, `pool_all`) | single-platform arms tune on their own platform's val rows and pool with single-platform trees (`gbt-dist-flyte`, `gbt-dist-k8s`) |
| The kNN gate's AUROC 0.9999 is a scale artifact | the scaling bug was real (constant-on-train features got sd = 1e-9), but it was **not** the cause: fixed, the AUROC is still 0.999 (Finding 11) | constant features keep their raw unit; the old scaling stays as `knn-raw` for comparison |
| The adapter still moves CPU | **true** (`adapted` passed its own distribution through `adapt_probs`) | memory only, as the paper said |
| The cost model is not a raw Job's (`backoffLimit: 0`) | true as stated | OOM-penalty sweep offline (1, 5, 20 oracle costs per retry) |
| The rules baseline is weak for K8s | fair | `rules-sized`: history scaled by input size, cold starts from input bytes, staging read off the manifest |
| The family name is in every prompt | **true** (`TaskEnvironment(name=family)`, manifest names) | k8s-2's counterfactual station re-scores with neutral names, no retraining |
| No K8s-only ablation | true | `jev-pointer-k8s` |
| Gate members chosen after seeing unseen results | **true** for the members and signal | k8s-2 is the pre-registered confirmation: members, signal and q fixed before it ran, at a new seed |
| "Within 3 points" unsupported without CIs | partly: holds for decoded arms, with CIs (Finding 1) | CIs everywhere in the paper |
| Dollar regret ranks differently on unseen | true (already §9.6) | both metrics in the abstract |

### 7. k8s-2 leaderboard (seed 1, Job pods, 95% task-clustered intervals)

Each split has 5,095 Job pods for test and 6,416 for unseen.

| approach | test | unseen | Flyte test / unseen | Job $ regret, test | 3 seeds, test / unseen |
|---|---|---|---|---|---|
| static | 89.3 [84.7, 93.7] | 115.1 | 91.5 / 112.8 | 183.0 | fixed |
| rules | 42.4 [37.9, 46.7] | 49.8 | 41.3 / 48.0 | 56.9 | fixed |
| rules-sized (new) | 37.7 [33.8, 41.6] | **39.2** [37.3, 41.2] | 37.3 / 41.1 | 55.9 | fixed |
| zero-shot 27B, 8,192 tokens | 48.8 [33.1, 67.2] | 42.8 [29.6, 57.2] | 35.7 / 52.0 | 86.0 | one training |
| trees (gbt-dist) | 11.5 [10.2, 12.8] | 69.1 | 14.6 / 69.0 | 30.6 | fixed |
| decision (jev-pointer) | 12.6 [11.3, 14.0] | 47.8 | 11.3 / 43.0 | 39.6 | 12.6-15.0 / 44.2-56.6 |
| RL (rl-pointer) | 14.6 [13.2, 16.1] | 39.3 [37.3, 41.4] | 14.3 / 40.2 | 50.6 | 14.6-17.4 / 35.5-39.3 |
| pool x decision | **6.6** [5.6, 7.6] | 56.8 | 5.5 / 51.4 | 27.6 | 6.6-6.9 / 52.1-58.3 |
| pool x RL, argmax | 7.2 [6.0, 8.4] | 79.4 | 7.1 / 78.1 | **15.7** | 6.8-11.8 / 78.6-110.9 |
| gate (kNN, q 0.95, pre-registered) | 8.1 [6.9, 9.4] | 39.3 [37.3, 41.4] | 7.6 / 40.2 | 19.5 | 8.1-12.7 / 35.5-39.3 |

- Familiar code: the pools lead, and pool x decision and pool x RL argmax
  are tied.
- New code: RL, the gate and `rules-sized` tie at about 39%. The pools
  fall to 50-79%.
- The pre-registered gate is confirmed in the sense that matters. It is
  near the top on both splits, which no other trained arm is. But it does
  not beat `rules-sized` on new code.

### 8. Training data: Jobs only, Flyte only, both

Regret on Job pods, test:

| arm | trained on both | Jobs only | Flyte only |
|---|---|---|---|
| decision | 12.6 | 13.2 | 37.3 |
| pool x decision | 6.6 | 7.7 | 24.8 |
| trees | 11.5 | 11.8 | 18.3 |

- Flyte rows do not help with Jobs. decision minus decision-k8s is
  -0.6 [-1.8, +0.7] points on test.
- A Flyte-only model does not transfer to Jobs.
- The Flyte-to-Job adapter now helps on every seed:
  - seed 1: 37.3 -> 28.6
  - seed 2: 33.8 -> 22.1
  - seed 3: 37.5 -> 28.6

  Round 22 found that it hurt. That was the wrong 180 MiB constant:
  the adapter subtracted a runtime that was not there.

### 9. The counterfactual: models trained on Jobs read the manifest

The setup: 1,189 first runs (test and unseen), each rendered as a Job
under all four stagings, 4,756 pods in all. Only the manifest and the
fetch code change, and with no history the manifest is the only
evidence of the staging. On test, switching disk to tmpfs moves the
label in 48.6% of pairs, and init to boto3 in 20.4%.

Where the label moved under the medium contrast, the request moved the
right way:

| arm | test | unseen |
|---|---|---|
| decision, trees, RL, pools | 72-78% | 64-78% |
| decision, Flyte-only | 21% | 32% |
| static, rules | 0% | 0% |

Static and rules have nothing but defaults on a first run, so they
never move.

On unseen pairs whose label did not move, the trained arms still moved
their request in 57-80% of them. They overreact to `medium: Memory`
when the staged bytes are small.

The real run/Job pairs were also re-measured. With R = 70, only 29.9%
of pairs move: 638 up and 543 down. On moved pairs, the trained arms
move the right way in 56-67%, and the rules in 25%.

### 10. The task's name barely matters

The counterfactual's first runs and their real twins were re-scored
with neutral names, with no retraining.

- Test: every trained arm except Flyte-only changes by less than a
  point. Flyte-only changes by +1.8.
- Unseen: arms change by +0.5 to +3.6.

### 11. The gate's kNN fix changed nothing

With constant-on-train features left in raw units, the kNN distance
still separates unseen from validation with AUROC 0.999. It sends 100%
of unseen pods to RL, and its requests are identical to `knn-raw`.

So the 1e-9 scaling was not what made the detector perfect. The unseen
families really are far from the training code: median distance 7.6,
against 1.0 for validation.

Without code tokens the AUROC is 0.67 and unseen regret is 62%.

### 12. Three trainings, retry cost, dollars

**Seeds.**
- The top of the familiar-code board is stable:
  - pool x decision 6.6-6.9
  - RL argmax 8.2-8.6
- New-code numbers move by up to 12 points for the decision model and
  32 for pool x RL argmax. RL varies least (35.5-39.3).

**Retry cost.** At an OOM penalty of 5 or 20 oracle costs (a raw Job
with `backoffLimit: 0`, resubmitted by hand), `rules-sized` is the best
arm on unseen code:
- penalty 5: 112.5% vs RL/gate 135.7%
- penalty 20: 387.2% vs 497.1%

pool x decision stays best on familiar code at every penalty from 1 to
20.

**Dollars.** On unseen code the trees are cheapest ($959 per 1,000 Job
runs). The gate and RL cost $1,177, `rules-sized` $1,126, and the
oracle $517.

**Zero-shot.** With 8,192 tokens it answers, and it is about as good as
the rules: 48.8% test and 42.8% unseen, with wide intervals. The JSON
schema variant is 48.8% / 44.4%.

## Code changes

- `bench_all/paper.py` (new): `paper_export`, which exports a finished
  release for offline analysis.
- `bench_all/pod_probe.py` (new), plus `k8s_probe.job_template`,
  `upload_inputs` and `validate_pods`: the twin in real pods, and the Flyte
  task measured as a Flyte task.
- `bench_all/k8s.py`:
  - `FLYTE_RUNTIME_MIB = 70` (measured) and `flyte_need_mib`;
  - `init_limit_mib`, so the manifest's init container holds its staged
    bytes;
  - `twin(staging=)` and `task_twins` for counterfactual renditions.
- `bench_all/dataset.py`: both corpora's Flyte needs from the measured
  peak and runtime.
- `bench_all/prompting.py`: history in the measured runtime's units;
  `_mem` accepts `GiB` and `GB`, and `_cpu` accepts millicores.
- `bench_all/counterfactual.py` (new) and the `staging_counterfactual`
  station: the four-staging test and the neutral-name test.
- `bench_all/stations.py`:
  - zero-shot as `zeroshot-27b` (8,192 tokens) and `zeroshot-27b-json`
    (decoding constrained to the answer schema), with
    `at_token_limit` / `p95_completion_tokens` stats;
  - `jev-pointer-k8s`;
  - `gbt-dist-flyte` / `-k8s`;
  - platform-restricted tuning;
  - seeds;
  - counterfactual and neutral names on the leaderboard.
- `bench_all/arms.py`: `rules-sized`.
- `bench_all/gating.py`: the kNN scale fix and `knn-raw`; the adapter is
  memory only.
- `bench_all/scoring.py`: `paired` reports `label_moved`,
  `moved_direction` and `unmoved_kept`.
- `bench/versioning.py`: `all_train_cache` keys the trained arms on their
  own code (`_hash_defs` hashes named top-level definitions of
  `stations.py`). A post-hoc station can no longer retrain the suite. Test:
  `test_trained_arms_do_not_key_on_post_hoc_code`.
- `shared/llm_client.chat(extra=)`, used for structured output.
- `bench_all/factory.py`: the `counterfactual` node; `K8S_ARM_KNOBS`
  seed 1; `K8S_ARMS`.
- `paper/` (new): the LaTeX paper (`make`, `make arxiv`), with:
  - `scripts/export.py`, `export_cf.py` and `analyze.py`: exports,
    bootstrap and sensitivity;
  - `scripts/tables.py`: the generated tables;
  - `scripts/to_html.py`: the HTML report, rendered from the same LaTeX
    by pandoc;
  - `scripts/abstract_html.py`: the one-page abstract.
- `bench_all/paper.py`: `paper_file`, a station's JSON output returned
  inline (for the counterfactual).
- `benchmarks/rightsize-bench-k8s{,-abstract}.html` and
  `rightsize-bench-k8s.pdf`: rewritten Kubernetes-first from k8s-2.

