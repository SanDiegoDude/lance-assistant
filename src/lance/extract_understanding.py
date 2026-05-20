"""Extract the Lance "understanding-only" weights into a vanilla
Qwen2.5-VL-3B checkpoint loadable by HF transformers.

Strategy (see notes/ARCHITECTURE.md):
    * keep `language_model.model.<X>`  -> rename to `model.language_model.<X>`
    * keep `language_model.lm_head.weight` -> `lm_head.weight`
    * keep ViT (either from Lance_3B_Video bundle's `vit_model.*` or from the
      standalone `Qwen2.5-VL-ViT/vit.safetensors`) -> prefix with `model.visual.`
    * drop every `*_moe_gen*` weight (those are the generation expert)
    * drop `latent_pos_embed`, `vae2llm`, `llm2vae`, `time_embedder` (those
      are the generation pathway and have no slot in vanilla Qwen2.5-VL)

We also rewrite `llm_config.json` into a compatible `config.json` so the
resulting directory can be loaded directly via
`Qwen2_5_VLForConditionalGeneration.from_pretrained(target_dir)`.

Usage:
    python -m lance extract_understanding \\
        --lance weights/Lance_hf/Lance_3B/model.safetensors \\
        --vit   weights/Lance_hf/Qwen2.5-VL-ViT/vit.safetensors \\
        --config weights/Lance_hf/Lance_3B/llm_config.json \\
        --tokenizer-dir weights/Lance_hf/Lance_3B \\
        --target weights/lance_3b_understand \\
        --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


DROP_PREFIXES = (
    "latent_pos_embed.",
    "vae2llm.",
    "llm2vae.",
    "time_embedder.",
)


def is_moe_gen(name: str) -> bool:
    """True if this tensor is part of the generation expert path."""
    return "_moe_gen" in name


def remap_lance_to_hf(name: str) -> str | None:
    """Return the HF Qwen2.5-VL state-dict key for a Lance key, or None to drop."""
    if any(name.startswith(p) for p in DROP_PREFIXES):
        return None
    if is_moe_gen(name):
        return None
    if name.startswith("vit_model."):
        return "model.visual." + name[len("vit_model.") :]
    if name == "language_model.lm_head.weight":
        return "lm_head.weight"
    if name.startswith("language_model.model."):
        return "model.language_model." + name[len("language_model.model.") :]
    if name.startswith("language_model."):
        # safety net for anything stray we missed
        return "model.language_model." + name[len("language_model.") :]
    return None


def remap_vit_standalone(name: str) -> str:
    """Standalone Qwen2.5-VL-ViT keys -> HF model.visual.*."""
    return "model.visual." + name


def cast_dtype(t: torch.Tensor, target: str) -> torch.Tensor:
    if target == "fp32":
        return t.to(torch.float32)
    if target == "bf16":
        return t.to(torch.bfloat16)
    if target == "fp16":
        return t.to(torch.float16)
    raise ValueError(f"unknown dtype: {target}")


def rewrite_config(src: Path, out: Path, target_dtype: str) -> None:
    """Reshape llm_config.json into a standalone config.json HF can load."""
    cfg = json.loads(src.read_text())
    cfg["torch_dtype"] = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[target_dtype]
    out.write_text(json.dumps(cfg, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lance extract_understanding")
    p.add_argument("--lance", type=Path, required=True, help="Lance_3B/model.safetensors")
    p.add_argument(
        "--vit",
        type=Path,
        default=None,
        help="Qwen2.5-VL-ViT/vit.safetensors (if the Lance ckpt does NOT bundle vit_model.*)",
    )
    p.add_argument("--config", type=Path, required=True, help="Lance_3B/llm_config.json")
    p.add_argument("--tokenizer-dir", type=Path, required=True, help="dir containing tokenizer.json etc.")
    p.add_argument("--target", type=Path, required=True, help="output directory")
    p.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    target: Path = args.target
    target.mkdir(parents=True, exist_ok=True)

    print(f"[extract] loading lance ckpt: {args.lance}")
    out_state: dict[str, torch.Tensor] = {}
    dropped_moe_gen = 0
    dropped_genpath = 0
    bundled_vit = False
    with safe_open(str(args.lance), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            target_key = remap_lance_to_hf(k)
            if target_key is None:
                if is_moe_gen(k):
                    dropped_moe_gen += 1
                else:
                    dropped_genpath += 1
                continue
            if target_key.startswith("model.visual."):
                bundled_vit = True
            t = f.get_tensor(k)
            out_state[target_key] = cast_dtype(t, args.dtype)
        print(
            f"[extract] from lance: kept={len(out_state)} "
            f"drop_moe_gen={dropped_moe_gen} drop_genpath={dropped_genpath}"
        )

    if bundled_vit:
        print("[extract] Lance ckpt has bundled vit_model.* -> not loading standalone ViT")
    elif args.vit is not None:
        print(f"[extract] loading standalone ViT: {args.vit}")
        with safe_open(str(args.vit), framework="pt") as f:
            for k in f.keys():
                target_key = remap_vit_standalone(k)
                t = f.get_tensor(k)
                out_state[target_key] = cast_dtype(t, args.dtype)
        print(f"[extract] total after ViT splice: {len(out_state)}")
    else:
        raise SystemExit(
            "no ViT source: Lance ckpt doesn't bundle vit_model.*, and --vit was not provided"
        )

    if args.dry_run:
        print("[extract] dry-run: would write the following key groups:")
        prefixes: dict[str, int] = {}
        for k in out_state:
            head = ".".join(k.split(".")[:3])
            prefixes[head] = prefixes.get(head, 0) + 1
        for head, n in sorted(prefixes.items()):
            print(f"  {head}: {n}")
        return 0

    print(f"[extract] writing safetensors -> {target / 'model.safetensors'}")
    save_file(out_state, str(target / "model.safetensors"), metadata={"format": "pt"})

    print(f"[extract] writing config.json")
    rewrite_config(args.config, target / "config.json", args.dtype)

    print(f"[extract] copying generation_config + tokenizer files from {args.tokenizer_dir}")
    for fname in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        src = args.tokenizer_dir / fname
        if src.exists():
            shutil.copy2(src, target / fname)
            print(f"  copied {fname}")

    print(f"[extract] done. try:")
    print(
        f"  from transformers import Qwen2_5_VLForConditionalGeneration\n"
        f"  m = Qwen2_5_VLForConditionalGeneration.from_pretrained('{target}', torch_dtype='auto', device_map='cuda')"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
