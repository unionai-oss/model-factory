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
