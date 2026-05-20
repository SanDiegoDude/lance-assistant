"""Greedy text chat against the extracted Lance understanding checkpoint."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lance.qknorm_patch import patch_qwen2_5_vl_qknorm

patch_qwen2_5_vl_qknorm()
from transformers import (  # noqa: E402
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

CKPT = Path("weights/lance_3b_understand")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"[chat] device={args.device} ckpt={CKPT}")
    tok = AutoTokenizer.from_pretrained(CKPT)
    print(f"[chat] loading model in bf16 onto {args.device} ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    print(f"[chat] loaded in {time.time() - t0:.1f}s, params={sum(p.numel() for p in model.parameters()):,}")

    text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    print(f"[chat] prompt:\n{text}")

    ids = tok(text, return_tensors="pt").input_ids.to(args.device)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")

    out_ids = []
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids=ids,
            max_new_tokens=args.max_new,
            do_sample=False,
            eos_token_id=eos_id,
        )
    new_tokens = out[0, ids.shape[1] :].tolist()
    print(f"[chat] generated {len(new_tokens)} tokens in {time.time() - t0:.1f}s")
    print(f"\n=== response ===\n{tok.decode(new_tokens, skip_special_tokens=True)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
