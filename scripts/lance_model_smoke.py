"""Smoke-test the LanceForGeneration model: load weights + tiny forward.

Verifies that every Lance ckpt key (excluding the bundled ViT for the video
checkpoint) finds a matching parameter in our model.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lance.lance_model import LanceConfig, LanceForGeneration


def main() -> int:
    ckpt_dir = Path("weights/Lance_hf/Lance_3B")
    config = LanceConfig.from_llm_config(ckpt_dir / "llm_config.json", video=False)
    print(f"[smoke] config: hidden={config.hidden_size} layers={config.num_hidden_layers} max_mape={config.max_latent_tokens}")

    print(f"[smoke] instantiating LanceForGeneration on meta ...")
    with torch.device("meta"):
        m = LanceForGeneration(config)
    print(f"[smoke]   params: {sum(p.numel() for p in m.parameters()):,}")

    print(f"[smoke] moving to CPU and zero-initialising ...")
    m = m.to_empty(device="cpu")
    for p in m.parameters():
        p.data.zero_()

    print(f"[smoke] loading Lance_3B/model.safetensors (bf16) ...")
    t0 = time.time()
    counts = m.load_lance_safetensors(ckpt_dir / "model.safetensors", dtype=torch.bfloat16)
    print(f"[smoke]   load took {time.time() - t0:.1f}s  kept={counts['kept']} missing={counts['missing']} skipped_vit={counts['skipped_vit']}")
    if counts["missing"]:
        raise SystemExit(f"missing keys: {counts['missing']}")

    # also count how many of OUR parameters were not assigned
    not_loaded = []
    for n, p in m.named_parameters():
        if torch.equal(p, torch.zeros_like(p)):
            not_loaded.append(n)
    if not_loaded:
        print(f"[smoke] WARNING {len(not_loaded)} parameters left at zero:")
        for n in not_loaded[:10]:
            print(f"  - {n}")

    print(f"[smoke] running a tiny forward pass on CPU ...")
    # A 32-token text prompt + 16 latent positions = 48 token sequence
    text_ids = torch.tensor([151644, 8948, 198, 9836, 13, 151645], dtype=torch.long)
    n_lat = 16
    full_ids = torch.cat([text_ids, torch.tensor([151652], dtype=torch.long),
                          torch.full((n_lat,), 151655, dtype=torch.long),
                          torch.tensor([151653], dtype=torch.long)])
    print(f"[smoke]   seq_len = {full_ids.shape[0]}, n_lat = {n_lat}")

    latent_embeds = torch.randn(n_lat, config.hidden_size, dtype=torch.bfloat16) * 0.01
    latent_positions = torch.nonzero(full_ids == 151655, as_tuple=False).flatten()
    position_ids = torch.arange(full_ids.shape[0], dtype=torch.long)
    position_ids[latent_positions] = int(latent_positions[0].item())

    m = m.to(torch.bfloat16)
    with torch.inference_mode():
        h = m.forward_packed(
            token_ids=full_ids,
            latent_embeds=latent_embeds,
            latent_positions=latent_positions,
            position_ids=position_ids,
            is_causal=False,
        )
    print(f"[smoke]   output shape={tuple(h.shape)}  dtype={h.dtype}")
    print(f"[smoke]   norm at first text pos = {h[0].norm().item():.3f}")
    print(f"[smoke]   norm at first latent  = {h[latent_positions[0]].norm().item():.3f}")
    print(f"[smoke]   any NaN? {bool(torch.isnan(h).any().item())}")

    # Test the velocity prediction head
    v = m.llm2vae(h[latent_positions])
    print(f"[smoke]   v shape={tuple(v.shape)}  dtype={v.dtype}  norm={v.norm().item():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
