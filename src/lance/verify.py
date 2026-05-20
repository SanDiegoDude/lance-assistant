"""Structural verification of the Lance safetensors against expected schema.

Run after downloading the big shards. Confirms every transformer layer has
both the understanding and the generation expert weights, the projector head
shapes match the Wan 2.2 VAE (48 channels), and the MaPE bank has the
expected row count.

Usage:
    python -m lance verify --target weights/Lance_hf
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

EXPECTED_LM_LAYERS = 36
EXPECTED_HIDDEN = 2048
EXPECTED_KV_HIDDEN = 256                # num_kv_heads(2) * head_dim(128)
EXPECTED_HEAD_DIM = 128
EXPECTED_INTERMEDIATE = 11008
EXPECTED_VAE_LATENT_DIM = 48
EXPECTED_VOCAB = 151936

LAYER_KEY_SUFFIXES = (
    ("input_layernorm.weight",                   [EXPECTED_HIDDEN]),
    ("input_layernorm_moe_gen.weight",           [EXPECTED_HIDDEN]),
    ("post_attention_layernorm.weight",          [EXPECTED_HIDDEN]),
    ("post_attention_layernorm_moe_gen.weight",  [EXPECTED_HIDDEN]),
    ("self_attn.q_proj.weight",                  [EXPECTED_HIDDEN, EXPECTED_HIDDEN]),
    ("self_attn.q_proj.bias",                    [EXPECTED_HIDDEN]),
    ("self_attn.k_proj.weight",                  [EXPECTED_KV_HIDDEN, EXPECTED_HIDDEN]),
    ("self_attn.k_proj.bias",                    [EXPECTED_KV_HIDDEN]),
    ("self_attn.v_proj.weight",                  [EXPECTED_KV_HIDDEN, EXPECTED_HIDDEN]),
    ("self_attn.v_proj.bias",                    [EXPECTED_KV_HIDDEN]),
    ("self_attn.o_proj.weight",                  [EXPECTED_HIDDEN, EXPECTED_HIDDEN]),
    ("self_attn.q_norm.weight",                  [EXPECTED_HEAD_DIM]),
    ("self_attn.k_norm.weight",                  [EXPECTED_HEAD_DIM]),
    ("self_attn.q_proj_moe_gen.weight",          [EXPECTED_HIDDEN, EXPECTED_HIDDEN]),
    ("self_attn.q_proj_moe_gen.bias",            [EXPECTED_HIDDEN]),
    ("self_attn.k_proj_moe_gen.weight",          [EXPECTED_KV_HIDDEN, EXPECTED_HIDDEN]),
    ("self_attn.k_proj_moe_gen.bias",            [EXPECTED_KV_HIDDEN]),
    ("self_attn.v_proj_moe_gen.weight",          [EXPECTED_KV_HIDDEN, EXPECTED_HIDDEN]),
    ("self_attn.v_proj_moe_gen.bias",            [EXPECTED_KV_HIDDEN]),
    ("self_attn.o_proj_moe_gen.weight",          [EXPECTED_HIDDEN, EXPECTED_HIDDEN]),
    ("self_attn.q_norm_moe_gen.weight",          [EXPECTED_HEAD_DIM]),
    ("self_attn.k_norm_moe_gen.weight",          [EXPECTED_HEAD_DIM]),
    ("mlp.gate_proj.weight",                     [EXPECTED_INTERMEDIATE, EXPECTED_HIDDEN]),
    ("mlp.up_proj.weight",                       [EXPECTED_INTERMEDIATE, EXPECTED_HIDDEN]),
    ("mlp.down_proj.weight",                     [EXPECTED_HIDDEN, EXPECTED_INTERMEDIATE]),
    ("mlp_moe_gen.gate_proj.weight",             [EXPECTED_INTERMEDIATE, EXPECTED_HIDDEN]),
    ("mlp_moe_gen.up_proj.weight",               [EXPECTED_INTERMEDIATE, EXPECTED_HIDDEN]),
    ("mlp_moe_gen.down_proj.weight",             [EXPECTED_HIDDEN, EXPECTED_INTERMEDIATE]),
)

TOP_KEYS = {
    "language_model.model.embed_tokens.weight":    [EXPECTED_VOCAB, EXPECTED_HIDDEN],
    "language_model.lm_head.weight":               [EXPECTED_VOCAB, EXPECTED_HIDDEN],
    "language_model.model.norm.weight":            [EXPECTED_HIDDEN],
    "language_model.model.norm_moe_gen.weight":    [EXPECTED_HIDDEN],
    "vae2llm.weight":                              [EXPECTED_HIDDEN, EXPECTED_VAE_LATENT_DIM],
    "vae2llm.bias":                                [EXPECTED_HIDDEN],
    "llm2vae.weight":                              [EXPECTED_VAE_LATENT_DIM, EXPECTED_HIDDEN],
    "llm2vae.bias":                                [EXPECTED_VAE_LATENT_DIM],
    "time_embedder.mlp.0.weight":                  [EXPECTED_HIDDEN, 256],
    "time_embedder.mlp.0.bias":                    [EXPECTED_HIDDEN],
    "time_embedder.mlp.2.weight":                  [EXPECTED_HIDDEN, EXPECTED_HIDDEN],
    "time_embedder.mlp.2.bias":                    [EXPECTED_HIDDEN],
}

MAPE_EXPECTED_ROWS_IMAGE = 4_096
MAPE_EXPECTED_ROWS_VIDEO = 126_976


def safetensors_header(path: Path) -> dict:
    """Read the JSON header of a safetensors file from disk."""
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def check_one(name: str, path: Path, expect_video: bool) -> list[str]:
    errors: list[str] = []
    print(f"\n=== verifying {name} ({path}) ===")
    if not path.exists():
        return [f"{name}: file missing"]
    header = safetensors_header(path)
    header.pop("__metadata__", None)
    shapes = {k: v["shape"] for k, v in header.items()}

    for L in range(EXPECTED_LM_LAYERS):
        prefix = f"language_model.model.layers.{L}."
        for suf, want in LAYER_KEY_SUFFIXES:
            key = prefix + suf
            if key not in shapes:
                errors.append(f"  missing: {key}")
                continue
            got = shapes[key]
            if got != want:
                errors.append(f"  shape mismatch {key}: got {got} want {want}")

    for key, want in TOP_KEYS.items():
        if key not in shapes:
            errors.append(f"  missing: {key}")
            continue
        got = shapes[key]
        if got != want:
            errors.append(f"  shape mismatch {key}: got {got} want {want}")

    mape_key = "latent_pos_embed.pos_embed"
    want_rows = MAPE_EXPECTED_ROWS_VIDEO if expect_video else MAPE_EXPECTED_ROWS_IMAGE
    if mape_key not in shapes:
        errors.append(f"  missing: {mape_key}")
    else:
        got = shapes[mape_key]
        if got != [want_rows, EXPECTED_HIDDEN]:
            errors.append(f"  shape mismatch {mape_key}: got {got} want [{want_rows}, {EXPECTED_HIDDEN}]")

    if expect_video:
        if not any(k.startswith("vit_model.") for k in shapes):
            errors.append("  video checkpoint missing bundled vit_model.* keys")

    total = sum(int(s["shape"][0]) * (1 if len(s["shape"]) == 1 else int(s["shape"][1])) for s in header.values() if len(s["shape"]) <= 2)
    print(f"  tensors: {len(shapes)}  errors: {len(errors)}")
    for e in errors[:8]:
        print(e)
    if len(errors) > 8:
        print(f"  ... and {len(errors) - 8} more")
    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lance verify")
    p.add_argument(
        "--target",
        type=Path,
        default=Path("weights/Lance_hf"),
        help="local mirror directory (default: weights/Lance_hf)",
    )
    p.add_argument(
        "--ckpt",
        choices=("image", "video", "both"),
        default="both",
    )
    args = p.parse_args(argv)

    all_errors: list[str] = []
    if args.ckpt in ("image", "both"):
        all_errors += check_one("Lance_3B", args.target / "Lance_3B" / "model.safetensors", expect_video=False)
    if args.ckpt in ("video", "both"):
        all_errors += check_one("Lance_3B_Video", args.target / "Lance_3B_Video" / "model.safetensors", expect_video=True)

    print(f"\n=== verify complete: {len(all_errors)} error(s) ===")
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
