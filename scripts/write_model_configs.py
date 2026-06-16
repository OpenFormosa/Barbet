#!/usr/bin/env python3
"""Regenerate Barbet config.json files."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_MODULE = ROOT / "src" / "barbet" / "configuration_barbet.py"
REMOTE_CODE_AUTO_MAP = {
    "AutoConfig": "configuration_barbet.BarbetConfig",
    "AutoModel": "modeling_barbet.BarbetModel",
    "AutoModelForCausalLM": "modeling_barbet.BarbetForCausalLM",
}


def load_config_class():
    spec = importlib.util.spec_from_file_location("barbet_configuration_for_config_writer", CONFIG_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CONFIG_MODULE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BarbetConfig


BarbetConfig = load_config_class()


def write_config(config: BarbetConfig, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config.architectures = ["BarbetForCausalLM"]
    config.auto_map = dict(REMOTE_CODE_AUTO_MAP)
    config.torch_dtype = "bfloat16"
    config.save_pretrained(path)
    postprocess_config_json(config, path)


def postprocess_config_json(config: BarbetConfig, path: Path) -> None:
    config_path = path / "config.json"
    data = json.loads(config_path.read_text())
    data["auto_map"] = dict(REMOTE_CODE_AUTO_MAP)
    data["torch_dtype"] = "bfloat16"
    data["transformers_version"] = "4.45.0"
    data.pop("dtype", None)
    if getattr(config, "rope_scaling", None):
        rope_scaling = dict(config.rope_scaling)
        rope_scaling.pop("rope_type", None)
        data["rope_scaling"] = rope_scaling
        data.pop("rope_parameters", None)
    else:
        data.pop("rope_scaling", None)
        data.pop("rope_parameters", None)
    config_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    # save_pretrained copies custom config code when register_for_auto_class()
    # is active. The config folders intentionally stay metadata-only; Hub repos
    # receive the remote-code files at the model repository root.
    generated_code = path / "configuration_barbet.py"
    generated_code.unlink(missing_ok=True)


def main() -> None:
    write_config(BarbetConfig.barbet_300m(), ROOT / "configs" / "barbet_300m")
    write_config(BarbetConfig.barbet_1b(), ROOT / "configs" / "barbet_1b")
    write_config(BarbetConfig.barbet_1b_1m_extension(), ROOT / "configs" / "barbet_1b_1m")


if __name__ == "__main__":
    main()
