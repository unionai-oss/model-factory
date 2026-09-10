"""Round-14 throughput fixes, from the post-mortem of the 1M release.

Run upnz22h5cdt6gdgrtshv asked for 1,000,000 rows and shipped 31,704. Three
mechanisms failed together, and each one gets tests here:

- the wave loop only checked its deadline BETWEEN waves, so a first wave
  that overran (19.5h against a 15h budget) could not be stopped and there
  was never a second wave  → `plan_wave`
- work kept being routed to teacher endpoints that had been returning 504
  for hours                                                → `TeacherPool`
- the newly-admitted libraries generated code that raised at runtime, and
  each such archetype burned all 7 calibration pods to find out
                                          → `STACK_PITFALLS` + the probe
"""

from __future__ import annotations

import inspect
import random

from resource_tuner.shared.llm_client import (
    TeacherError,
    TeacherPool,
    chat,
    wait_until_ready,
)
from resource_tuner.taskgen import synthetic as syn
from resource_tuner.taskgen.synthetic import curate_measurement
from resource_tuner.training.stations import MIN_WAVE_SIZE, plan_wave, probe_is_fatal


# ── wave sizing ─────────────────────────────────────────────────────────
def test_first_wave_is_small_regardless_of_target():
    """Wave 1 measures; it does not try to finish the release. Round 13 ran
    the caller's full archetype count here and never got a wave 2."""
    size, _ = plan_wave(
        wave=0, first_wave=150, remaining=1250, attempted=0, kept=0,
        max_wave_size=600, time_left_s=16 * 3600, secs_per_attempt=0.0,
    )
    assert size == 150


def test_later_waves_size_from_observed_keep_rate():
    # 30 kept of 150 attempted = 20%; covering 200 more needs ~1000 attempts,
    # capped by max_wave_size.
    size, note = plan_wave(
        wave=1, first_wave=150, remaining=200, attempted=150, kept=30,
        max_wave_size=600, time_left_s=16 * 3600, secs_per_attempt=10.0,
    )
    assert size == 600
    assert "20%" in note

    # A healthier keep rate asks for proportionally fewer attempts.
    size, _ = plan_wave(
        wave=1, first_wave=150, remaining=200, attempted=150, kept=120,
        max_wave_size=600, time_left_s=16 * 3600, secs_per_attempt=10.0,
    )
    assert 240 < size < 270  # 200 / 0.8 + 8


def test_wave_is_trimmed_to_what_the_clock_allows():
    """The round-13 failure, in one assertion: a wave is sized by what fits
    in the remaining budget, not only by what the target wants."""
    size, note = plan_wave(
        wave=1, first_wave=150, remaining=2400, attempted=600, kept=78,
        max_wave_size=600, time_left_s=3600,      # 1 hour left
        secs_per_attempt=117.0,                   # observed in run upnz22h5
    )
    assert size == int(3600 / 117)  # ~30, not 600
    assert "trimming" in note


def test_no_time_left_yields_a_wave_too_small_to_run():
    size, _ = plan_wave(
        wave=2, first_wave=150, remaining=2000, attempted=800, kept=100,
        max_wave_size=600, time_left_s=30.0, secs_per_attempt=117.0,
    )
    assert size < MIN_WAVE_SIZE  # the caller stops instead of starting it


def test_unmeasured_cost_does_not_trim():
    """Before the first wave completes there is no cost observation, and a
    zero must not be read as 'infinitely affordable' OR as 'stop'."""
    size, _ = plan_wave(
        wave=0, first_wave=150, remaining=1250, attempted=0, kept=0,
        max_wave_size=600, time_left_s=60.0, secs_per_attempt=0.0,
    )
    assert size == 150


# ── teacher circuit breaker ─────────────────────────────────────────────
def test_pool_round_robins_while_healthy():
    pool = TeacherPool(["a", "b", "c"])
    assert [pool.pick(i) for i in range(4)] == [0, 1, 2, 0]


def test_endpoint_trips_after_consecutive_failures_and_leaves_rotation():
    pool = TeacherPool(["a", "b", "c"], trip_after=3)
    assert pool.note_fail(1) is False
    assert pool.note_fail(1) is False
    assert pool.note_fail(1) is True  # tripped
    assert pool.live() == [0, 2]
    # 'b' no longer receives any work.
    assert {pool.pick(i) for i in range(10)} == {0, 2}


def test_a_success_resets_the_streak():
    """Only CONSECUTIVE failures trip: an endpoint that hiccups and recovers
    is healthy, and retrying it is the right call."""
    pool = TeacherPool(["a", "b"], trip_after=3)
    pool.note_fail(0)
    pool.note_fail(0)
    pool.note_ok(0)
    assert pool.note_fail(0) is False
    assert pool.live() == [0, 1]


def test_the_last_live_endpoint_never_trips():
    """Degraded beats nowhere to send work."""
    pool = TeacherPool(["a", "b"], trip_after=2)
    pool.note_fail(0)
    assert pool.note_fail(0) is True
    for _ in range(10):
        pool.note_fail(1)
    assert pool.live() == [1]
    assert pool.pick(0) == 1


def test_revive_is_half_open_not_a_clean_slate():
    """Tripped endpoints get another chance each wave (they scale from
    zero and do come back), but a still-dead one must re-trip fast rather
    than absorb another full trip_after of archetypes."""
    pool = TeacherPool(["a", "b"], trip_after=6)
    for _ in range(6):
        pool.note_fail(0)
    assert pool.live() == [1]

    assert pool.revive() == ["a"]
    assert pool.live() == [0, 1]
    # Two strikes, not six.
    assert pool.note_fail(0) is False
    assert pool.note_fail(0) is True


def test_all_endpoints_down_is_reported_not_guessed():
    pool = TeacherPool([], trip_after=3)
    assert pool.pick(0) is None


def test_a_loading_model_is_not_a_dead_endpoint():
    """/health returns 200 once the listener is up but BEFORE the weights
    are in, so a teacher can pass the wake check and still 503 every
    completion for minutes (qwen35-397b, run ukh2f7p6jhzbp8bm247x). That
    gets a patient retry budget of its own, and must not count against the
    breaker — dropping an endpoint for becoming useful is backwards."""
    sig = inspect.signature(chat).parameters
    assert sig["loading_retries"].default > sig["retries"].default * 4

    err = TeacherError("teacher HTTP 503: Loading model", status=503, loading=True)
    assert err.loading
    plain = TeacherError("teacher HTTP 504: gateway", status=504)
    assert not plain.loading


def test_health_probe_outlasts_a_frontier_model_load():
    """llama.cpp BLOCKS its HTTP listener while loading weights, so a big
    model does not answer 503-while-loading — it answers nothing at all.
    At 20s the probe gave up before the server could ever reply and run
    us86v7zcfphfz76gjrdw lost qwen35-397b and minimax-m3 for 30 minutes
    while both were healthy-but-loading. A slow answer is not a dead
    endpoint."""
    default = inspect.signature(wait_until_ready).parameters["probe_timeout_s"].default
    assert default >= 60


# ── the viability probe ─────────────────────────────────────────────────
def test_probe_kills_the_archetype_only_when_the_code_is_broken():
    """The 462-pod bucket: code that ran and raised. One pod is enough to
    know, and the other six can only rediscover it."""
    assert probe_is_fatal("execution failed: ValueError: n_splits=5 > n_samples=3")
    assert probe_is_fatal("pod failed: OOMKilled")


def test_probe_tolerates_a_point_that_is_merely_out_of_bounds():
    """A range low under the memory floor says nothing about the code — the
    remaining points still calibrate it. Rejecting here would have the probe
    throw away archetypes it was never meant to judge."""
    assert not probe_is_fatal("peak 88MiB out of bounds")
    assert not probe_is_fatal("duration 12s out of bounds")
    assert not probe_is_fatal("gpu vram 0MiB out of bounds")
    assert not probe_is_fatal(None)


def test_probe_classification_matches_what_curation_actually_emits():
    """Guard the coupling: `probe_is_fatal` matches on the prefixes
    `curate_measurement` produces, so a reworded reason must not silently
    turn a fatal failure into a tolerated one."""
    broke = curate_measurement({"ok": False, "error": "boom"})
    assert probe_is_fatal(broke)

    small = curate_measurement(
        {"ok": True, "peak_rss_mib": 10.0, "duration_s": 60.0}
    )
    assert small and not probe_is_fatal(small)

    fine = curate_measurement(
        {"ok": True, "peak_rss_mib": 512.0, "duration_s": 60.0}
    )
    assert fine is None and not probe_is_fatal(fine)


# ── stack API contracts ─────────────────────────────────────────────────
def test_every_offered_stack_has_an_api_contract_or_is_stdlib():
    """A library admitted to the corpus without concrete API guidance is how
    round 13 got 462 `execution failed` pods. Stacks with no pitfall note
    must be ones the teacher already writes correctly."""
    known_safe = {"numpy", "pandas", "sklearn", "torch", "scipy", "pyarrow", "stdlib"}
    offered = {
        name
        for stacks in syn.STACKS_BY_FAMILY.values()
        for name, _ in stacks
    }
    missing = offered - known_safe - set(syn.STACK_PITFALLS)
    assert not missing, f"stacks offered with no API contract: {sorted(missing)}"


def test_contracts_name_the_specific_calls_that_crashed():
    """Spot-check that the guidance is concrete (a named symbol), not just
    another sentence about the library."""
    assert "group_by" in syn.STACK_PITFALLS["polars"]
    assert "verbose_eval" in syn.STACK_PITFALLS["lightgbm"]  # removed in 4.x
    assert "from_numpy_array" in syn.STACK_PITFALLS["networkx"]
    assert "float32" in syn.STACK_PITFALLS["faiss"]
    assert "recursion_limit" in syn.STACK_PITFALLS["langgraph"]
    assert "divisible" in syn.STACK_PITFALLS["transformers"]
    # lightning's defaults write to the working directory and fail there.
    assert "logger=False" in syn.STACK_PITFALLS["lightning"]


def test_the_contract_reaches_the_prompt():
    rng = random.Random(0)
    seen = set()
    for _ in range(200):
        scenario = syn.build_scenario(rng, family="ml_training")
        for name, contract in syn.STACK_PITFALLS.items():
            if f"API contract for that stack — {contract}" in scenario:
                seen.add(name)
    # Every ml_training stack that HAS a contract should surface one; plain
    # torch deliberately has none (it is in the known-safe set).
    assert seen == {
        n for n, _ in syn.STACKS_BY_FAMILY["ml_training"] if n in syn.STACK_PITFALLS
    }
    assert "torch" not in seen


def test_scenario_without_a_family_has_no_contract():
    assert "API contract" not in syn.build_scenario(random.Random(0))
