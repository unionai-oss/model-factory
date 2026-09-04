"""The TRL reward bridge: completions + truth columns → scalars."""

from resource_tuner.rewards.rewards import MAX_REWARD
from resource_tuner.training.grpo import make_reward_fn


def test_reward_fn_scores_a_mixed_batch():
    fn = make_reward_fn("success")
    completions = [
        [{"role": "assistant", "content": '{"cpu": 1, "memory": "2Gi"}'}],  # fits
        [{"role": "assistant", "content": '{"cpu": 1, "memory": "256Mi"}'}],  # ooms
        [{"role": "assistant", "content": "I think 4 CPUs should do"}],  # invalid
    ]
    rewards = fn(
        completions,
        true_peak_memory_mib=[800.0, 800.0, 800.0],
        true_cpu_cores=[1.0, 1.0, 1.0],
        duration_s=[60, 60, 60],
    )
    assert rewards[0] == MAX_REWARD
    assert rewards[2] < rewards[1] < rewards[0]


def test_reward_fn_accepts_plain_string_completions():
    fn = make_reward_fn("composite")
    (r,) = fn(
        ['{"cpu": 1, "memory": "1Gi"}'],
        true_peak_memory_mib=[800.0],
        true_cpu_cores=[1.0],
        duration_s=[60],
    )
    assert r > 0


def test_shaped_stage_uses_gpu_truth_and_tiebreak():
    """The stage-C path: GPU column flows through, and within a group the
    cheapest successful proposal earns the tie-break bonus."""
    fn = make_reward_fn("c-bucket", num_generations=2, max_steps=10)
    lean = '{"cpu": 1, "memory": "1Gi"}'
    padded = '{"cpu": 4, "memory": "16Gi"}'
    needs_gpu = '{"cpu": 2, "memory": "4Gi", "gpu": "T4:1"}'
    no_gpu = '{"cpu": 2, "memory": "4Gi"}'
    rewards = fn(
        [lean, padded, needs_gpu, no_gpu],
        true_peak_memory_mib=[700.0, 700.0, 700.0, 700.0],
        true_cpu_cores=[1.0, 1.0, 1.0, 1.0],
        duration_s=[60, 60, 60, 60],
        true_gpu_mem_mib=[0.0, 0.0, 8000.0, 8000.0],
        baseline_cost_per_hr=[0.2, 0.2, 0.5, 0.5],
    )
    # group 1 (CPU task): lean beats padded, and lean got the tie-break
    assert rewards[0] > rewards[1]
    # group 2 (GPU task): proposing the T4 succeeds; omitting it fails
    assert rewards[2] > rewards[3]


def test_shaped_stage_anneal_increases_waste_pressure():
    fn = make_reward_fn("c-log", max_steps=2)
    padded = ['{"cpu": 4, "memory": "16Gi"}']
    kwargs = dict(
        true_peak_memory_mib=[700.0], true_cpu_cores=[1.0], duration_s=[60]
    )
    (early,) = fn(padded, **kwargs)  # call 1 → step_frac 0.5
    (late,) = fn(padded, **kwargs)  # call 2 → step_frac 1.0
    assert late < early  # same proposal, more waste pressure later
