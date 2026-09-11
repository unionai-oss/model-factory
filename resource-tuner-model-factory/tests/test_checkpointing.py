"""Training checkpointing: intra-task resume, artifact checkpoints,
warm-start parameterization."""

import json
import pathlib

import pytest

from resource_tuner.config import AMBITIOUS, PROFILES, SMOKE, SMOKE_CKPT
from resource_tuner.contracts import ARTIFACT_TUNER_CHECKPOINT_INTERMEDIATE
from resource_tuner.training.grpo import _resolve_resume, find_trl_checkpoint


def _mk_ckpt(root: pathlib.Path, name: str) -> pathlib.Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "trainer_state.json").write_text(json.dumps({"global_step": 1}))
    return d


def test_find_trl_checkpoint_handles_both_restore_layouts(tmp_path):
    # layout A: the checkpoint dir's CONTENTS at the restore root
    root_a = tmp_path / "a"
    root_a.mkdir()
    (root_a / "trainer_state.json").write_text("{}")
    assert find_trl_checkpoint(str(root_a)) == str(root_a)
    # layout B: checkpoint-<N> subdirs — highest step wins
    root_b = tmp_path / "b"
    _mk_ckpt(root_b, "checkpoint-5")
    best = _mk_ckpt(root_b, "checkpoint-40")
    _mk_ckpt(root_b, "checkpoint-15")
    assert find_trl_checkpoint(str(root_b)) == str(best)
    # junk-named dirs are skipped, empty root is None
    root_c = tmp_path / "c"
    (root_c / "checkpoint-final").mkdir(parents=True)
    assert find_trl_checkpoint(str(root_c)) is None
    assert find_trl_checkpoint(str(tmp_path / "missing")) is None


def test_incomplete_checkpoint_dirs_are_ignored(tmp_path):
    # a dir without trainer_state.json (crash mid-save) must not be picked
    (tmp_path / "checkpoint-99").mkdir()
    good = _mk_ckpt(tmp_path, "checkpoint-10")
    assert find_trl_checkpoint(str(tmp_path)) == str(good)


def test_resolve_resume_prefers_explicit_dir():
    import asyncio

    class FakeDir:
        path = "s3://bucket/ckpt"

    d, source = asyncio.run(_resolve_resume(FakeDir(), "ignored-artifact"))
    assert d is not None and source.startswith("dir:")
    d, source = asyncio.run(_resolve_resume(None, ""))
    assert d is None and source == ""


def test_resolve_resume_artifact_fails_loudly_when_unversioned(monkeypatch):
    import asyncio

    from resource_tuner.shared import assets

    async def none_version(name):
        return None

    monkeypatch.setattr(assets, "latest_version", none_version)
    with pytest.raises(RuntimeError, match="no\\s+blob-resolvable"):
        asyncio.run(_resolve_resume(None, ARTIFACT_TUNER_CHECKPOINT_INTERMEDIATE))


def test_checkpointing_profiles():
    # defaults stay off — checkpointing must be opt-in per profile
    assert SMOKE.save_steps == 0 and SMOKE.artifact_checkpoint_every == 0
    # the smoke rung saves within its 10 steps and publishes at step 5
    assert 0 < SMOKE_CKPT.save_steps < SMOKE_CKPT.max_steps
    assert 0 < SMOKE_CKPT.artifact_checkpoint_every < SMOKE_CKPT.max_steps
    # the ambitious rung: 4B-class QLoRA on one T4, checkpointed
    assert "4B" in AMBITIOUS.base_model and AMBITIOUS.use_qlora
    assert AMBITIOUS.save_steps and AMBITIOUS.artifact_checkpoint_every
    assert "ambitious" in PROFILES and "smoke-ckpt" in PROFILES


# ── prompt budget (round 15) ────────────────────────────────────────────
def test_clip_prompt_keeps_both_ends_and_cuts_the_middle():
    """Current TRL removed max_prompt_length from GRPOConfig, so nothing
    bounds a prompt unless we do — and WHERE the cut falls decides whether
    the sample is still usable. The system instructions open the prompt and
    the answer cue closes it; both must survive, so the trim comes out of
    the middle of the code."""
    from resource_tuner.training.grpo import clip_prompt

    class FakeTok:
        """1 token per 4 chars, which is all clip_prompt needs."""

        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return [0] * (sum(len(m["content"]) for m in messages) // 4)

    tok = FakeTok()
    msgs = [
        {"role": "system", "content": "SYSTEM-RULES-" + "s" * 100},
        {"role": "user", "content": "HEAD-OF-CODE-" + "x" * 4000 + "-ANSWER-CUE"},
    ]
    out, clipped = clip_prompt(msgs, tok, max_tokens=300)
    assert clipped
    assert len(tok.apply_chat_template(out)) <= 300
    # System message untouched, and both ends of the user message survive.
    assert out[0]["content"] == msgs[0]["content"]
    assert out[1]["content"].startswith("HEAD-OF-CODE-")
    assert out[1]["content"].endswith("-ANSWER-CUE")
    assert "truncated" in out[1]["content"]


def test_clip_prompt_leaves_short_prompts_alone():
    from resource_tuner.training.grpo import clip_prompt

    class FakeTok:
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
            return [0] * (sum(len(m["content"]) for m in messages) // 4)

    msgs = [{"role": "system", "content": "short"}, {"role": "user", "content": "also short"}]
    out, clipped = clip_prompt(msgs, FakeTok(), max_tokens=500)
    assert out == msgs and not clipped


def test_vram_estimate_scales_linearly_in_batch_and_sequence():
    """The arithmetic that decides GPU fit. Both terms are linear, which is
    why halving the group size is a real lever."""
    import dataclasses as dc

    from resource_tuner.config import AMBITIOUS
    from resource_tuner.training.grpo import vram_estimate_gib

    base = vram_estimate_gib(AMBITIOUS, 248_000)
    twice_batch = vram_estimate_gib(dc.replace(AMBITIOUS, per_device_batch=AMBITIOUS.per_device_batch * 2), 248_000)
    assert abs(twice_batch["needed_gib"] - 2 * base["needed_gib"]) < 1e-6
    longer = vram_estimate_gib(dc.replace(AMBITIOUS, max_prompt_length=AMBITIOUS.max_prompt_length * 2), 248_000)
    assert longer["needed_gib"] > base["needed_gib"]
