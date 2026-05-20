"""End-to-end text-to-image generation for Lance.

Loads:
  - LanceForGeneration weights from weights/Lance_hf/Lance_3B/model.safetensors
  - Wan 2.2 VAE   from weights/Lance_hf/Wan2.2_VAE.pth
  - Qwen tokenizer from weights/Lance_hf/Lance_3B (or the extracted ckpt dir)

Runs a flow-matching denoise loop and saves a PNG.

Example:
    python scripts/t2i.py --prompt "a photo of a corgi on the moon, 4k" \\
        --steps 20 --width 512 --height 512 --out out.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lance.flow_match import LanceGenerator, T2IRequest
from lance.lance_model import LanceConfig, LanceForGeneration
from lance.wan_vae import Wan2_2_VAE


CKPT_DIR = Path("weights/Lance_hf/Lance_3B")
VAE_PATH = Path("weights/Lance_hf/Wan2.2_VAE.pth")
# Lance_3B/ is missing tokenizer_config.json (only Lance_3B_Video has it).
# Our extracted understanding ckpt already has a complete tokenizer dir.
TOKENIZER_DIR = Path("weights/lance_3b_understand")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="low quality, blurry, distorted")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("out_t2i.png"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--vae-device", default=None, help="defaults to --device")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae_device = torch.device(args.vae_device or args.device)

    print(f"[t2i] loading config from {CKPT_DIR / 'llm_config.json'}")
    config = LanceConfig.from_llm_config(CKPT_DIR / "llm_config.json", video=False)
    print(f"[t2i] config: {config}")

    print(f"[t2i] building LanceForGeneration ...")
    model = LanceForGeneration(config)
    print(f"[t2i]   model params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"[t2i] loading weights from {CKPT_DIR / 'model.safetensors'} (dtype={args.dtype})")
    t0 = time.time()
    counts = model.load_lance_safetensors(CKPT_DIR / "model.safetensors", dtype=dtype)
    print(f"[t2i]   load took {time.time() - t0:.1f}s  kept={counts['kept']} missing={counts['missing']}")
    if counts["missing"]:
        raise SystemExit(f"missing weight assignments: {counts['missing']}")

    # Sanity: ensure all model params got initialized (not still on meta or zeros).
    # We rely on the fact that load above asserts shape match and assigns.
    model = model.to(device=device, dtype=dtype)
    print(f"[t2i]   moved to {device} as {dtype}")

    print(f"[t2i] loading Wan 2.2 VAE from {VAE_PATH}")
    vae = Wan2_2_VAE(
        z_dim=48,
        c_dim=160,
        vae_pth=str(VAE_PATH),
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        dtype=torch.float32,
        device=str(vae_device),
    )
    print(f"[t2i]   VAE loaded")

    print(f"[t2i] loading tokenizer from {TOKENIZER_DIR}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    gen = LanceGenerator(model=model, tokenizer=tok, device=device, dtype=dtype)
    req = T2IRequest(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        timestep_shift=args.shift,
        cfg_scale=args.cfg,
        negative_prompt=args.negative,
        seed=args.seed,
        num_frames=1,
    )
    print(f"[t2i] request: {req}")

    t0 = time.time()
    latent = gen.generate(req)  # [48, T_lat=1, H_lat, W_lat]
    print(f"[t2i] denoise done in {time.time() - t0:.1f}s, latent shape={tuple(latent.shape)}")

    # Decode with VAE.
    # Wan2_2_VAE.decode expects a list of latents shaped [C, T, H, W]
    print(f"[t2i] VAE decode ...")
    t0 = time.time()
    latent_for_vae = latent.to(device=vae_device, dtype=torch.float32)
    out = vae.decode([latent_for_vae])  # list of [3, T, H, W] in [-1, 1]
    img = out[0]  # [3, T=1, H, W]
    print(f"[t2i] VAE decode done in {time.time() - t0:.1f}s, shape={tuple(img.shape)}")

    arr = img[:, 0].clamp(-1, 1).float().cpu().numpy()  # [3, H, W]
    arr = ((arr + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    Image.fromarray(arr).save(args.out)
    print(f"[t2i] wrote {args.out} ({arr.shape[1]}x{arr.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
