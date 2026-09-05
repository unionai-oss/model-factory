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
