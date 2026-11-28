#!/usr/bin/env python3
"""Check that a Barbet checkpoint works through the standard transformers API.

Usage:
    python scripts/verify_hf_checkpoint.py OpenFormosa/barbet-1b-base
    python scripts/verify_hf_checkpoint.py /path/to/local/folder --device cuda

The script loads the tokenizer and model with the Auto classes
(trust_remote_code=True), then checks precision, weight tying, cached versus
uncached generation, left-padded batch generation and the text-generation
pipeline. It exits non-zero on the first failed check.
"""

from __future__ import annotations

import argparse
import collections
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="Hub repo id or local folder")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, dtype="auto"
    ).to(args.device).eval()

    census = collections.Counter(str(p.dtype) for p in model.parameters())
    check("parameters loaded", sum(p.numel() for p in model.parameters()) > 0, str(dict(census)))
    check("lm_head tied to embeddings", model.lm_head.weight is model.model.embed_tokens.weight)
    mamba = [layer.mixer for layer in model.model.layers if layer.layer_type == "mamba"]
    if mamba:
        check(
            "Mamba A_log and D kept in FP32",
            all(m.A_log.dtype == torch.float32 and m.D.dtype == torch.float32 for m in mamba),
        )

    gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False, min_new_tokens=args.max_new_tokens)
    prompts = ["台灣的首都是", "今天天氣很好，我們一起去"]
    singles = []
    with torch.inference_mode():
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt").to(args.device)
            cached = model.generate(**enc, **gen)
            uncached = model.generate(**enc, use_cache=False, **gen)
            check(f"cache == no cache ({prompt})", torch.equal(cached, uncached))
            singles.append(cached[0, enc["input_ids"].shape[1]:])
            print("    ", repr(tokenizer.decode(singles[-1], skip_special_tokens=True)))

        tokenizer.padding_side = "left"
        batch = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
        batched = model.generate(**batch, **gen)[:, batch["input_ids"].shape[1]:]
    for i, prompt in enumerate(prompts):
        check(f"left-padded batch == single ({prompt})", torch.equal(batched[i], singles[i]))

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    text = generator(prompts[0], max_new_tokens=4, do_sample=False)[0]["generated_text"]
    check("text-generation pipeline", text.startswith(prompts[0]), repr(text))
    print("All checks passed.")


if __name__ == "__main__":
    main()
