"""End-to-end text-to-video generation for Lance — SCRATCH IMPL, DO NOT USE.

This is the pre-official-release reverse-engineered T2V loop. It is broken
in several ways that we only learned after ByteDance published their code:

  - Uses Qwen2 1D RoPE; the real model needs Qwen 2.5-VL mrope
    (`mrope_section=(16, 24, 24)` splitting head_dim over (t, h, w)).
  - Latents pinned at position 0; real model uses `pos_shift ~ 1000`.
  - Hand-rolled "3D causal" attention; real model uses standard
    causal-text + bidirectional-noise-block via flex_attention's
    `create_sparse_mask`.

Use `refs/lance_official/inference_lance.sh` with `TASK_NAME=t2v` instead.
Kept in-tree as a learning artifact.

Number of frames must satisfy `T = 1 + 4 * k` due to Wan 2.2 VAE temporal
compression (4x with a single-frame head). Typical values: 5, 9, 13, ..., 121.
At 16 fps the maximum 121 frames is ~7.5 seconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lance.flow_match import LanceGenerator, T2IRequest
from lance.lance_model import LanceConfig, LanceForGeneration
from lance.wan_vae import Wan2_2_VAE


CKPT_DIR = Path("weights/Lance_hf/Lance_3B_Video")
VAE_PATH = Path("weights/Lance_hf/Wan2.2_VAE.pth")
TOKENIZER_DIR = Path("weights/Lance_hf/Lance_3B_Video")  # has tokenizer_config.json


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative", default="low quality, blurry, distorted, static")
    p.add_argument("--frames", type=int, default=49,
                   help="number of output frames (must be 1 or 1+4k); 5..121")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--cfg-interval", type=float, nargs=2, default=(0.0, 1.0),
                   metavar=("LO", "HI"),
                   help="CFG only applied when t in [LO, HI]; default (0.0, 1.0)")
    p.add_argument("--cfg-renorm-min", type=float, default=0.5,
                   help="BAGEL global CFG renorm clamp min; 0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--out", type=Path, default=Path("out_t2v.mp4"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    args = p.parse_args()

    if args.frames != 1 and (args.frames - 1) % 4 != 0:
        raise SystemExit(f"--frames must be 1 or 1+4k; got {args.frames}")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"[t2v] loading video config from {CKPT_DIR / 'llm_config.json'}")
    config = LanceConfig.from_llm_config(CKPT_DIR / "llm_config.json", video=True)
    print(f"[t2v] config: max_mape={config.max_latent_tokens}")

    print(f"[t2v] building LanceForGeneration (video MaPE bank: 126976 rows = 31 frames x 64x64)")
    model = LanceForGeneration(config)
    print(f"[t2v]   model params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"[t2v] loading weights from {CKPT_DIR / 'model.safetensors'} (dtype={args.dtype})")
    t0 = time.time()
    counts = model.load_lance_safetensors(CKPT_DIR / "model.safetensors", dtype=dtype)
    print(f"[t2v]   load took {time.time() - t0:.1f}s  kept={counts['kept']} missing={counts['missing']} skipped_vit={counts['skipped_vit']}")
    if counts["missing"]:
        raise SystemExit(f"missing weight assignments: {counts['missing']}")

    model = model.to(device=device, dtype=dtype)
    print(f"[t2v]   moved to {device} as {dtype}")

    print(f"[t2v] loading Wan 2.2 VAE from {VAE_PATH}")
    vae = Wan2_2_VAE(
        z_dim=48,
        c_dim=160,
        vae_pth=str(VAE_PATH),
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        dtype=torch.float32,
        device=str(device),
    )

    print(f"[t2v] loading tokenizer from {TOKENIZER_DIR}")
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
        num_frames=args.frames,
        cfg_interval=tuple(args.cfg_interval),
        cfg_renorm_min=args.cfg_renorm_min,
    )
    print(f"[t2v] request: {req}")

    t0 = time.time()
    latent = gen.generate(req)  # [48, T_lat, H_lat, W_lat]
    print(f"[t2v] denoise done in {time.time() - t0:.1f}s, latent shape={tuple(latent.shape)}")

    print(f"[t2v] VAE decode ...")
    t0 = time.time()
    latent_for_vae = latent.to(device=device, dtype=torch.float32)
    out = vae.decode([latent_for_vae])  # list of [3, T, H, W] in [-1, 1]
    video = out[0]  # [3, T, H, W]
    print(f"[t2v] decoded shape={tuple(video.shape)} in {time.time() - t0:.1f}s")

    # to uint8 frames [T, H, W, 3]
    arr = video.clamp(-1, 1).float().cpu().numpy()
    arr = ((arr + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    frames = np.transpose(arr, (1, 2, 3, 0))  # [T, H, W, 3]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, frames, fps=args.fps, codec="libx264", pixelformat="yuv420p")
    print(f"[t2v] wrote {args.out} ({frames.shape[0]} frames @ {args.fps}fps, {frames.shape[2]}x{frames.shape[1]})")
    # also dump first/middle/last as png for quick preview
    preview_dir = args.out.with_suffix("")
    preview_dir.mkdir(exist_ok=True)
    indices = [0, len(frames) // 2, len(frames) - 1]
    for idx in indices:
        from PIL import Image as _Img
        _Img.fromarray(frames[idx]).save(preview_dir / f"frame_{idx:03d}.png")
    print(f"[t2v] wrote preview PNGs to {preview_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
