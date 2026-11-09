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
| [rsk-k8s-2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-2) | **k8s-2**: the corrected corpus (measured Flyte runtime, sized init containers) and every arm, seed 1, with the new arms (rules-sized, single-platform trees, jev-pointer-k8s, zeroshot-27b with room to answer, zeroshot-27b-json) and the counterfactual station | running |
| [rsk-k8s-2-s2](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-2-s2), [rsk-k8s-2-s3](https://demo.hosted.unionai.cloud/v2/domain/development/project/resource-tuner-model-factory/runs/rsk-k8s-2-s3) | the same corpus, retrained at seeds 2 and 3 (release `k8s-2-s<k>`, `stations.seed_of`), every arm but zero-shot: three trainings of k8s-2 | running |

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
| The kNN gate's AUROC 0.9999 is a scale artifact | **true**: constant-on-train features got sd = 1e-9 | constant features keep their raw unit; the old scaling stays as `knn-raw` for comparison |
| The adapter still moves CPU | **true** (`adapted` passed its own distribution through `adapt_probs`) | memory only, as the paper said |
| The cost model is not a raw Job's (`backoffLimit: 0`) | true as stated | OOM-penalty sweep offline (1, 5, 20 oracle costs per retry) |
| The rules baseline is weak for K8s | fair | `rules-sized`: history scaled by input size, cold starts from input bytes, staging read off the manifest |
| The family name is in every prompt | **true** (`TaskEnvironment(name=family)`, manifest names) | k8s-2's counterfactual station re-scores with neutral names, no retraining |
| No K8s-only ablation | true | `jev-pointer-k8s` |
| Gate members chosen after seeing unseen results | **true** for the members and signal | k8s-2 is the pre-registered confirmation: members, signal and q fixed before it ran, at a new seed |
| "Within 3 points" unsupported without CIs | partly: holds for decoded arms, with CIs (Finding 1) | CIs everywhere in the paper |
| Dollar regret ranks differently on unseen | true (already §9.6) | both metrics in the abstract |

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
