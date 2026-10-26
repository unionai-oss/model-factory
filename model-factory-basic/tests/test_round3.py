"""Round 3: checkpoint shapes, the served teacher, profile wiring."""

import json

import pytest

from model_factory.config import get_profile
from model_factory.data_engineering.synthetic import _strip_reasoning, parse_generation, teacher_url
from model_factory.shared.checkpoints import read_manifest, weights_source


def _dir(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return str(tmp_path)


def test_round2_adapter_manifests_read_as_lora(tmp_path):
    d = _dir(tmp_path, {"base_model": "B", "adapter": True, "generation": 1})
    m = read_manifest(d)
    assert m["weights"] == "lora"
    assert weights_source(d, m) == ("B", d)


def test_round2_base_champion_reads_as_base(tmp_path):
    d = _dir(tmp_path, {"base_model": "B", "adapter": False})
    m = read_manifest(d)
    assert (m["weights"], m["generation"]) == ("base", 0)
    assert weights_source(d, m) == ("B", None)


def test_full_weights_load_from_the_directory_itself(tmp_path):
    d = _dir(tmp_path, {"base_model": "B", "weights": "full", "generation": 3})
    assert weights_source(d, read_manifest(d)) == (d, None)


def test_unknown_weights_kind_is_an_error(tmp_path):
    d = _dir(tmp_path, {"base_model": "B", "weights": "gguf"})
    with pytest.raises(ValueError):
        weights_source(d, read_manifest(d))


def test_teacher_url_is_the_in_cluster_service_name():
    assert teacher_url("qwen38-27b-vllm@llm-service", "development") == (
        "http://qwen38-27b-vllm.llm-service-development.svc.cluster.local"
    )
    with pytest.raises(ValueError):
        teacher_url("no-project")


def test_reasoning_before_the_answer_is_stripped():
    # What the 27B app actually returned in probe run bmf-r3-teacher-probe.
    text = "We need answer user... Provide code.\n</think>\n\nQUESTION:\nAdd two numbers.\n\nSOLUTION:\n```python\ndef add(a, b):\n    return a + b\n```\n\nTESTS:\n```python\ndef test_add():\n    assert add(1, 2) == 3\n```"
    parsed = parse_generation(_strip_reasoning(text))
    assert parsed is not None and parsed["question"] == "Add two numbers."
    assert _strip_reasoning("no reasoning here") == "no reasoning here"


def test_daily_uses_vllm_a_7b_teacher_and_the_big_heldout_split():
    p = get_profile("daily-r3")
    assert p.use_vllm and p.teacher_app is None
    assert "7B" in p.synthetic_model
    assert p.eval_tasks >= 500
    assert get_profile("daily").eval_tasks == 200  # round 2's name keeps round 2's meaning


def test_the_factory_picks_the_curriculum_task_from_the_profile(monkeypatch):
    from dataclasses import replace

    from model_factory import config
    from model_factory import contracts as c
    from model_factory.factory import build_graph

    local = build_graph("daily-r3")._nodes[c.ARTIFACT_SYNTHETIC].producer.task.name
    assert local == "de-gpu.generate_synthetic_tasks"

    served_profile = replace(config.DAILY, name="served", teacher_app="qwen38-27b-vllm@llm-service")
    monkeypatch.setitem(config.PROFILES, "served", served_profile)
    served = build_graph("served")._nodes[c.ARTIFACT_SYNTHETIC].producer.task.name
    assert served == "de-cpu.generate_synthetic_tasks_served"
