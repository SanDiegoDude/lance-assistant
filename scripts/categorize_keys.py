"""Bucket the 1021 tensor names from Lance_3B into logical pathways.

Reads `notes/Lance_3B__model.safetensors.keys.txt` (and the video one if
present) and reports a parameter count per bucket. Helpful for designing the
state-dict mapping into:

  - "understanding LM"  : every layer's non-_moe_gen weights -> vanilla Qwen2.5-VL
  - "generation LM"     : every layer's _moe_gen weights
  - "input projector"   : vae2llm
  - "output head"       : llm2vae
  - "MaPE"              : latent_pos_embed
  - "time embed"        : time_embedder.mlp
  - "top-level shared"  : embed_tokens, lm_head, final norms
  - "bundled ViT"       : vit_model.* (video ckpt only)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

NOTES = Path(__file__).resolve().parent.parent / "notes"

BUCKETS = [
    ("vit_bundled",       re.compile(r"^vit_model\.")),
    ("mape",              re.compile(r"^latent_pos_embed\.")),
    ("vae2llm",           re.compile(r"^vae2llm\.")),
    ("llm2vae",           re.compile(r"^llm2vae\.")),
    ("time_embed",        re.compile(r"^time_embedder\.")),
    ("embed_tokens",      re.compile(r"^language_model\.model\.embed_tokens\.")),
    ("lm_head",           re.compile(r"^language_model\.lm_head\.")),
    ("final_norm_u",      re.compile(r"^language_model\.model\.norm\.weight$")),
    ("final_norm_g",      re.compile(r"^language_model\.model\.norm_moe_gen\.weight$")),
    ("layer_gen",         re.compile(r"^language_model\.model\.layers\.\d+\.(.*_moe_gen.*)$")),
    ("layer_understand",  re.compile(r"^language_model\.model\.layers\.\d+\.")),
]


def numel_from_shape_str(s: str) -> int:
    """Parse 'shape=[a, b, c]' into a*b*c."""
    inside = s.split("shape=")[1].split("]")[0].lstrip("[")
    if not inside.strip():
        return 1
    out = 1
    for part in inside.split(","):
        out *= int(part.strip())
    return out


def categorize(path: Path) -> None:
    print(f"\n=== {path.name} ===")
    counts: dict[str, int] = {b: 0 for b, _ in BUCKETS}
    nums: dict[str, int] = {b: 0 for b, _ in BUCKETS}
    unknown: list[str] = []

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        key = parts[0]
        n = numel_from_shape_str(parts[1])
        for bucket, pat in BUCKETS:
            if pat.search(key):
                counts[bucket] += 1
                nums[bucket] += n
                break
        else:
            unknown.append(key)

    total_n = sum(nums.values())
    total_t = sum(counts.values())
    width = max(len(b) for b in counts)
    print(f"  {'bucket':<{width}}  tensors    params       %")
    for b, _ in BUCKETS:
        if counts[b] == 0:
            continue
        pct = 100.0 * nums[b] / total_n if total_n else 0.0
        print(f"  {b:<{width}}   {counts[b]:5d}   {nums[b]:>12,}   {pct:5.1f}%")
    print(f"  {'TOTAL':<{width}}   {total_t:5d}   {total_n:>12,}   100.0%")
    if unknown:
        print(f"  WARN unknown keys: {len(unknown)}")
        for k in unknown[:10]:
            print(f"    {k}")


def main() -> int:
    files = sorted(NOTES.glob("*.keys.txt"))
    if not files:
        print(f"no .keys.txt files in {NOTES}", file=sys.stderr)
        return 1
    for f in files:
        categorize(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
