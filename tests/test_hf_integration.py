"""Hugging Face integration checks for the remote-code (Hub) usage path.

These tests use tiny random models, so they run on CPU without mamba_ssm.
They cover what a Hub user touches: AutoConfig/AutoModelForCausalLM with
trust_remote_code, safetensors save/load, dtype handling, padded batches,
generate() and pipeline().
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, pipeline

from barbet import BarbetConfig, BarbetForCausalLM

ROOT = Path(__file__).resolve().parents[1]
REMOTE_CODE_FILES = ("configuration_barbet.py", "modeling_barbet.py", "megatron_long_context.py")


def tiny_config(**overrides) -> BarbetConfig:
    values = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        sliding_window_size=6,
        global_attention_layers=[0, 1, 4],
        mamba_layers=[3],
        mtp_enabled=False,
        pad_token_id=3,
        bos_token_id=1,
        eos_token_id=2,
        unk_token_id=0,
    )
    values.update(overrides)
    return BarbetConfig(**values)


def make_hub_folder(tmp_path: Path, config: BarbetConfig) -> Path:
    """Write a folder laid out like the Hub repo: config, weights, remote code."""
    torch.manual_seed(0)
    model = BarbetForCausalLM(config).eval()
    model.config.auto_map = {
        "AutoConfig": "configuration_barbet.BarbetConfig",
        "AutoModel": "modeling_barbet.BarbetModel",
        "AutoModelForCausalLM": "modeling_barbet.BarbetForCausalLM",
    }
    model.save_pretrained(tmp_path, safe_serialization=True)
    for name in REMOTE_CODE_FILES:
        shutil.copy(ROOT / name, tmp_path / name)
    return tmp_path


def test_root_remote_code_matches_package() -> None:
    for name in REMOTE_CODE_FILES:
        root_copy = (ROOT / name).read_text()
        package_copy = (ROOT / "src" / "barbet" / name).read_text()
        assert root_copy == package_copy, f"{name} differs between repo root and src/barbet"


def test_release_config_matches_factory() -> None:
    release = json.loads((ROOT / "configs" / "barbet_1b_base" / "config.json").read_text())
    factory = BarbetConfig.barbet_1b_base()
    for key in (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "global_attention_layers",
        "mamba_layers",
        "max_position_embeddings",
        "rope_theta",
        "vocab_size",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
    ):
        assert getattr(factory, key) == release[key], key
    assert release["rope_scaling"] is None
    assert [factory.layer_type(i) for i in range(29)].count("mamba") == 7
    assert "sliding_attention" not in {factory.layer_type(i) for i in range(29)}


def test_auto_classes_with_trust_remote_code(tmp_path) -> None:
    folder = make_hub_folder(tmp_path, tiny_config())
    config = AutoConfig.from_pretrained(folder, trust_remote_code=True)
    assert config.model_type == "barbet"
    model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True).eval()
    assert type(model).__name__ == "BarbetForCausalLM"
    assert model.lm_head.weight is model.model.embed_tokens.weight
    reference = BarbetForCausalLM.from_pretrained(folder).eval()
    ids = torch.randint(4, 128, (2, 9))
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, reference(input_ids=ids).logits, atol=1e-6)


def test_bf16_loading_keeps_mamba_state_parameters_in_fp32(tmp_path) -> None:
    folder = make_hub_folder(tmp_path, tiny_config())
    model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True, dtype=torch.bfloat16)
    mixer = model.model.layers[3].mixer
    assert mixer.A_log.dtype == torch.float32
    assert mixer.D.dtype == torch.float32
    assert model.model.layers[0].mixer.q_proj.weight.dtype == torch.bfloat16


def test_padded_batch_matches_unpadded_rows() -> None:
    """Left padding must not change real tokens, including Mamba layers."""
    torch.manual_seed(0)
    model = BarbetForCausalLM(tiny_config()).eval()
    short = torch.randint(4, 128, (1, 5))
    long = torch.randint(4, 128, (1, 9))
    pad = model.config.pad_token_id
    batch = torch.cat([torch.cat([torch.full((1, 4), pad), short], dim=1), long], dim=0)
    mask = torch.ones_like(batch)
    mask[0, :4] = 0
    position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
    with torch.no_grad():
        batched = model(input_ids=batch, attention_mask=mask, position_ids=position_ids, use_cache=False).logits
        alone_short = model(input_ids=short, use_cache=False).logits
        alone_long = model(input_ids=long, use_cache=False).logits
    assert torch.isfinite(batched).all()
    assert torch.allclose(batched[0, 4:], alone_short[0], atol=1e-5)
    assert torch.allclose(batched[1], alone_long[0], atol=1e-5)


def test_padded_generate_matches_single_prompt_generate() -> None:
    torch.manual_seed(0)
    model = BarbetForCausalLM(tiny_config()).eval()
    short = torch.randint(4, 128, (1, 4))
    long = torch.randint(4, 128, (1, 10))
    pad = model.config.pad_token_id
    batch = torch.cat([torch.cat([torch.full((1, 6), pad), short], dim=1), long], dim=0)
    mask = torch.ones_like(batch)
    mask[0, :6] = 0
    kwargs = dict(max_new_tokens=8, do_sample=False, min_new_tokens=8)
    with torch.no_grad():
        batched = model.generate(batch, attention_mask=mask, **kwargs)[:, 10:]
        alone_short = model.generate(short, attention_mask=torch.ones_like(short), **kwargs)[:, 4:]
        alone_long = model.generate(long, attention_mask=torch.ones_like(long), **kwargs)[:, 10:]
    assert torch.equal(batched[0], alone_short[0])
    assert torch.equal(batched[1], alone_long[0])


def test_padded_training_batch_has_finite_loss() -> None:
    model = BarbetForCausalLM(tiny_config())
    ids = torch.randint(4, 128, (2, 12))
    mask = torch.ones_like(ids)
    mask[1, 8:] = 0  # right padding, as in training collators
    labels = ids.masked_fill(mask == 0, -100)
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_text_generation_pipeline(tmp_path) -> None:
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(128)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="t0", bos_token="t1", eos_token="t2", pad_token="t3"
    )
    folder = make_hub_folder(tmp_path, tiny_config())
    model = AutoModelForCausalLM.from_pretrained(folder, trust_remote_code=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    text = generator("t10 t11 t12", max_new_tokens=4, do_sample=False)[0]["generated_text"]
    assert text.startswith("t10 t11 t12")
