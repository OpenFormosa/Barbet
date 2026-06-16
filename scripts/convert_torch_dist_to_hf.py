#!/usr/bin/env python3
"""Convert a Megatron torch_dist Barbet-1B checkpoint to HF safetensors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


ROOT = Path(__file__).resolve().parents[1]
REMOTE_CODE_FILES = ("configuration_barbet.py", "modeling_barbet.py")


def resolve_iter_dir(checkpoint: Path) -> Path:
    if checkpoint.name.startswith("iter_"):
        return checkpoint
    latest = checkpoint / "latest_checkpointed_iteration.txt"
    if not latest.exists():
        raise FileNotFoundError(f"Cannot resolve iteration: {latest} does not exist")
    iteration = int(latest.read_text().strip())
    iter_dir = checkpoint / f"iter_{iteration:07d}"
    if not iter_dir.exists():
        raise FileNotFoundError(f"Resolved iteration directory does not exist: {iter_dir}")
    return iter_dir


def tensor_metadata(iter_dir: Path) -> dict[str, TensorStorageMetadata]:
    metadata = FileSystemReader(str(iter_dir)).read_metadata()
    return {
        key: value
        for key, value in metadata.state_dict_metadata.items()
        if isinstance(value, TensorStorageMetadata)
    }


def load_tensors(iter_dir: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    metadata = tensor_metadata(iter_dir)
    missing = [key for key in keys if key not in metadata]
    if missing:
        raise KeyError("Missing Megatron tensor keys:\n" + "\n".join(missing))
    state = {
        key: torch.empty(tuple(metadata[key].size), dtype=metadata[key].properties.dtype)
        for key in keys
    }
    dcp.load_state_dict(state, storage_reader=FileSystemReader(str(iter_dir)), no_dist=True)
    return state


def split_grouped_qkv(
    weight: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads_per_group = num_attention_heads // num_key_value_heads
    group_width = (heads_per_group + 2) * head_dim
    hidden_size = weight.shape[1]
    grouped = weight.view(num_key_value_heads, group_width, hidden_size)
    query = grouped[:, : heads_per_group * head_dim, :].reshape(num_attention_heads * head_dim, hidden_size)
    key_start = heads_per_group * head_dim
    key = grouped[:, key_start : key_start + head_dim, :].reshape(num_key_value_heads * head_dim, hidden_size)
    value = grouped[:, key_start + head_dim :, :].reshape(num_key_value_heads * head_dim, hidden_size)
    return query.contiguous(), key.contiguous(), value.contiguous()


def build_required_keys(config: dict) -> list[str]:
    keys = ["embedding.word_embeddings.weight", "decoder.final_norm.weight"]
    mamba_layers = set(config["mamba_layers"])
    for layer_idx in range(config["num_hidden_layers"]):
        mixer_idx = 2 * layer_idx
        mlp_idx = mixer_idx + 1
        if layer_idx in mamba_layers:
            prefix = f"decoder.layers.{mixer_idx}.mixer"
            keys.extend(
                [
                    f"{prefix}.in_proj.layer_norm_weight",
                    f"{prefix}.in_proj.weight.z",
                    f"{prefix}.in_proj.weight.x",
                    f"{prefix}.in_proj.weight.B",
                    f"{prefix}.in_proj.weight.C",
                    f"{prefix}.in_proj.weight.dt",
                    f"{prefix}.conv1d.weight.x",
                    f"{prefix}.conv1d.weight.B",
                    f"{prefix}.conv1d.weight.C",
                    f"{prefix}.conv1d.bias.x",
                    f"{prefix}.conv1d.bias.B",
                    f"{prefix}.conv1d.bias.C",
                    f"{prefix}.dt_bias",
                    f"{prefix}.A_log",
                    f"{prefix}.D",
                    f"{prefix}.norm.weight",
                    f"{prefix}.out_proj.weight",
                ]
            )
        else:
            prefix = f"decoder.layers.{mixer_idx}.self_attention"
            keys.extend(
                [
                    f"{prefix}.linear_qkv.layer_norm_weight",
                    f"{prefix}.linear_qkv.weight",
                    f"{prefix}.q_layernorm.weight",
                    f"{prefix}.k_layernorm.weight",
                    f"{prefix}.linear_proj.weight",
                ]
            )
        mlp_prefix = f"decoder.layers.{mlp_idx}.mlp"
        keys.extend(
            [
                f"{mlp_prefix}.linear_fc1.layer_norm_weight",
                f"{mlp_prefix}.linear_fc1.weight",
                f"{mlp_prefix}.linear_fc2.weight",
            ]
        )
    return keys


def convert_state(megatron: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    hf: dict[str, torch.Tensor] = {}
    hf["model.embed_tokens.weight"] = megatron["embedding.word_embeddings.weight"].contiguous()
    hf["model.norm.weight"] = megatron["decoder.final_norm.weight"].contiguous()

    mamba_layers = set(config["mamba_layers"])
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    for layer_idx in range(config["num_hidden_layers"]):
        mixer_idx = 2 * layer_idx
        mlp_idx = mixer_idx + 1
        hf_prefix = f"model.layers.{layer_idx}"
        if layer_idx in mamba_layers:
            prefix = f"decoder.layers.{mixer_idx}.mixer"
            hf[f"{hf_prefix}.input_layernorm.weight"] = megatron[
                f"{prefix}.in_proj.layer_norm_weight"
            ].contiguous()
            mixer = f"{hf_prefix}.mixer"
            rename = {
                f"{prefix}.in_proj.weight.z": f"{mixer}.in_proj_z.weight",
                f"{prefix}.in_proj.weight.x": f"{mixer}.in_proj_x.weight",
                f"{prefix}.in_proj.weight.B": f"{mixer}.in_proj_b.weight",
                f"{prefix}.in_proj.weight.C": f"{mixer}.in_proj_c.weight",
                f"{prefix}.in_proj.weight.dt": f"{mixer}.in_proj_dt.weight",
                f"{prefix}.conv1d.weight.x": f"{mixer}.conv_x.weight",
                f"{prefix}.conv1d.weight.B": f"{mixer}.conv_b.weight",
                f"{prefix}.conv1d.weight.C": f"{mixer}.conv_c.weight",
                f"{prefix}.conv1d.bias.x": f"{mixer}.conv_x.bias",
                f"{prefix}.conv1d.bias.B": f"{mixer}.conv_b.bias",
                f"{prefix}.conv1d.bias.C": f"{mixer}.conv_c.bias",
                f"{prefix}.dt_bias": f"{mixer}.dt_bias",
                f"{prefix}.A_log": f"{mixer}.A_log",
                f"{prefix}.D": f"{mixer}.D",
                f"{prefix}.norm.weight": f"{mixer}.norm.weight",
                f"{prefix}.out_proj.weight": f"{mixer}.out_proj.weight",
            }
            for source, target in rename.items():
                hf[target] = megatron[source].contiguous()
        else:
            prefix = f"decoder.layers.{mixer_idx}.self_attention"
            hf[f"{hf_prefix}.input_layernorm.weight"] = megatron[
                f"{prefix}.linear_qkv.layer_norm_weight"
            ].contiguous()
            q, k, v = split_grouped_qkv(
                megatron[f"{prefix}.linear_qkv.weight"],
                num_attention_heads=config["num_attention_heads"],
                num_key_value_heads=config["num_key_value_heads"],
                head_dim=config["head_dim"],
            )
            hf[f"{hf_prefix}.mixer.q_proj.weight"] = q
            hf[f"{hf_prefix}.mixer.k_proj.weight"] = k
            hf[f"{hf_prefix}.mixer.v_proj.weight"] = v
            hf[f"{hf_prefix}.mixer.q_norm.weight"] = megatron[f"{prefix}.q_layernorm.weight"].contiguous()
            hf[f"{hf_prefix}.mixer.k_norm.weight"] = megatron[f"{prefix}.k_layernorm.weight"].contiguous()
            hf[f"{hf_prefix}.mixer.o_proj.weight"] = megatron[f"{prefix}.linear_proj.weight"].contiguous()

        mlp_prefix = f"decoder.layers.{mlp_idx}.mlp"
        hf[f"{hf_prefix}.post_attention_layernorm.weight"] = megatron[
            f"{mlp_prefix}.linear_fc1.layer_norm_weight"
        ].contiguous()
        fc1 = megatron[f"{mlp_prefix}.linear_fc1.weight"]
        if fc1.shape != (2 * intermediate, hidden):
            raise ValueError(f"Unexpected SwiGLU fc1 shape for layer {layer_idx}: {tuple(fc1.shape)}")
        gate, up = torch.split(fc1, intermediate, dim=0)
        hf[f"{hf_prefix}.mlp.gate_proj.weight"] = gate.contiguous()
        hf[f"{hf_prefix}.mlp.up_proj.weight"] = up.contiguous()
        hf[f"{hf_prefix}.mlp.down_proj.weight"] = megatron[f"{mlp_prefix}.linear_fc2.weight"].contiguous()

    return hf


def write_configs(output_dir: Path) -> None:
    native = json.loads((ROOT / "configs" / "barbet_1b" / "config.json").read_text())
    native["mtp_enabled"] = False
    native["use_cache"] = True
    native.pop("rope_scaling", None)
    (output_dir / "config.json").write_text(json.dumps(native, indent=2, sort_keys=True) + "\n")

    extension = json.loads((ROOT / "configs" / "barbet_1b_1m" / "config.json").read_text())
    extension["mtp_enabled"] = False
    extension["use_cache"] = True
    (output_dir / "config_1m_extension.json").write_text(
        json.dumps(extension, indent=2, sort_keys=True) + "\n"
    )


def copy_remote_code(output_dir: Path) -> None:
    for filename in REMOTE_CODE_FILES:
        shutil.copy2(ROOT / filename, output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path, help="Megatron checkpoint dir or iter_* dir")
    parser.add_argument("--output-dir", required=True, type=Path, help="HF output directory")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory")
    args = parser.parse_args()

    iter_dir = resolve_iter_dir(args.checkpoint)
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} is not empty; pass --force to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((ROOT / "configs" / "barbet_1b" / "config.json").read_text())
    required_keys = build_required_keys(config)
    megatron = load_tensors(iter_dir, required_keys)
    hf_state = convert_state(megatron, config)

    save_file(
        hf_state,
        output_dir / "model.safetensors",
        metadata={
            "format": "pt",
            "source_checkpoint": str(args.checkpoint),
            "source_iteration_dir": str(iter_dir),
            "mtp_exported": "false",
        },
    )
    write_configs(output_dir)
    copy_remote_code(output_dir)
    report = {
        "source_checkpoint": str(args.checkpoint),
        "source_iteration_dir": str(iter_dir),
        "hf_tensor_count": len(hf_state),
        "megatron_tensor_count_loaded": len(megatron),
        "safetensors": str(output_dir / "model.safetensors"),
        "default_config_context": 262144,
        "extension_config": "config_1m_extension.json",
        "mtp_exported": False,
    }
    (output_dir / "conversion_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
