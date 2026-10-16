"""Pin the cross-module call shapes locally, end to end, with a tiny model.

Helper calls inside async flyte tasks only fail on the cluster — after the
episodes, and maybe a 20-minute fine-tune, have already run. So this test
drives the SAME functions the tasks call, in the same order, with the same
argument shapes, using a tiny random Qwen2 and a BPE tokenizer trained in
memory (no downloads):

    data.build_rows -> training.training_inputs -> heads.modeling.train
    -> training.write_manifest -> evaluation.build_scorecard (+ render)
    -> evaluation.select.choose (+ render) -> serving.decide_with

The numbers are meaningless; the shapes, keys and JSON-serializability are
what is being checked.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from decision_factory.config import get_profile  # noqa: E402
from decision_factory.contracts import CHAMPION_MANIFEST_KEYS, MODEL_MANIFEST_KEYS, SCORECARD_KEYS  # noqa: E402
from decision_factory.data.tasks import build_rows  # noqa: E402
from decision_factory.evaluation.select import choose, render_selection  # noqa: E402
from decision_factory.evaluation.tasks import build_scorecard, render_scorecard  # noqa: E402
from decision_factory.harness.policy import DECISIONS, Observation  # noqa: E402
from decision_factory.harness.prompt import build_chat  # noqa: E402
from decision_factory.heads import modeling  # noqa: E402
from decision_factory.heads.encoding import encode_pointer, render_prompt  # noqa: E402
from decision_factory.training.tasks import training_inputs, write_manifest  # noqa: E402

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    import pandas as pd
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    rows = build_rows(48, 16, 24, seed=1)
    texts = [render_prompt_text(Observation.from_row(r)) for r in rows] + ['{"decision": "ask_user"}']
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=2000, special_tokens=["<pad>", "</s>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="</s>", pad_token="<pad>")
    tok.chat_template = CHAT_TEMPLATE
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                      tie_word_embeddings=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    base_dir = str(tmp_path_factory.mktemp("tiny-base"))
    Qwen2ForCausalLM(cfg).save_pretrained(base_dir)
    tok.save_pretrained(base_dir)
    return base_dir, pd.DataFrame(rows)


def render_prompt_text(obs):
    return "\n".join(m["content"] for m in build_chat(obs))


@pytest.fixture(scope="module")
def trained(tiny_base, tmp_path_factory):
    base_dir, df = tiny_base
    profile = get_profile("smoke")
    rows = df[df["split"] == "train"].to_dict("records")
    chats, options, labels = training_inputs(rows)
    dirs = {}
    for head in ("pointer", "sft"):
        out = str(tmp_path_factory.mktemp(f"tiny-{head}"))
        info = modeling.train(head, base_dir, chats, options, labels, out, epochs=1.0, batch_size=8,
                              pointer_rank=16, max_len=8192)
        manifest = write_manifest(out, info, slug=f"tiny-{head}", torso="tiny", head=head, base_model=base_dir,
                                  params_m=0, profile=profile, train_n=len(rows))
        assert set(MODEL_MANIFEST_KEYS) <= set(manifest)
        dirs[f"tiny-{head}"] = out
    return base_dir, df, dirs


def test_pointer_positions_land_on_the_options(tiny_base):
    base_dir, df = tiny_base
    tok = modeling.load_tokenizer(base_dir)
    chat = build_chat(Observation.from_row(df.iloc[0].to_dict()), ["deny", "allow", "ask_user"])
    ex = encode_pointer(tok, chat, ["deny", "allow", "ask_user"], "allow")
    assert ex.label == 1
    assert ex.answer_pos == len(ex.input_ids) - 1
    text = render_prompt(tok, chat)
    for pos, opt in zip(ex.option_pos, ["deny", "allow", "ask_user"]):
        # the token at each pointer position decodes to the tail of its option
        assert opt.endswith(tok.decode([ex.input_ids[pos]]).strip()) or tok.decode([ex.input_ids[pos]]).strip() in opt
    assert text.endswith("<|assistant|>\n")


def test_scorecard_selection_and_serving_shapes(trained, tmp_path):
    base_dir, df, dirs = trained
    cards = {}
    for slug, d in dirs.items():
        head = slug.split("-", 1)[1]
        card = build_scorecard(df, d, slug=slug, torso="tiny", head=head, base_model=base_dir, params_m=0,
                               profile_name="smoke", latency_samples=2)
        assert set(SCORECARD_KEYS) <= set(card)
        assert card["primary"] in card["readouts"]
        assert set(card["readouts"]) == ({"pointer"} if head == "pointer" else {"json", "logprob"})
        for r in card["readouts"].values():
            assert {"accuracy", "unsafe_allow_rate", "brier", "ece", "gate", "frontier", "latency"} <= set(r)
        json.dumps(card)  # the scorecard File must serialize
        assert "<svg" in render_scorecard(card)
        cards[slug] = json.loads(json.dumps(card))  # what the selector reads back

    out = str(tmp_path / "champion")
    manifest, ranked = choose(cards, dirs, 0.0, out)
    assert set(CHAMPION_MANIFEST_KEYS) <= set(manifest)
    assert "<svg" in render_selection(ranked, manifest["leaderboard"], ranked[0])
    with open(os.path.join(out, "manifest.json")) as f:
        assert json.load(f)["model_slug"] == ranked[0]["model_slug"]

    from decision_factory.serving.service import decide_with

    dm = modeling.DecisionModel(out)
    obs = Observation.from_row(df.iloc[0].to_dict())
    res = decide_with(dm, obs, manifest["gate_threshold"])
    assert res["decision"] in DECISIONS and res["gated"]["decision"] in DECISIONS
    assert set(res["probabilities"]) == set(DECISIONS)
    assert abs(sum(res["probabilities"].values()) - 1) < 1e-3


def test_floor_rejects_a_bad_suite(trained, tmp_path):
    base_dir, df, dirs = trained
    slug = "tiny-pointer"
    card = build_scorecard(df, dirs[slug], slug=slug, torso="tiny", head="pointer", base_model=base_dir,
                           params_m=0, profile_name="smoke", latency_samples=1)
    with pytest.raises(ValueError, match="floor"):
        choose({slug: card}, {slug: dirs[slug]}, 1.01, str(tmp_path / "c"))
