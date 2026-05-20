"""Load the extracted understanding-only checkpoint with HF transformers.

Reports any missing / unexpected keys and tries a tiny forward pass on CPU
(text only). If this works, the understanding path is wired correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lance.qknorm_patch import patch_qwen2_5_vl_qknorm

patch_qwen2_5_vl_qknorm()
from transformers import (  # noqa: E402  (after patch)
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

CKPT = Path("weights/lance_3b_understand")


def main() -> int:
    print(f"[load_test] QK-norm patch applied")
    print(f"[load_test] loading from {CKPT}")
    tok = AutoTokenizer.from_pretrained(CKPT)
    print(f"  tokenizer: vocab={tok.vocab_size} model_max_length={tok.model_max_length}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[load_test] loaded {sum(p.numel() for p in model.parameters()):,} params  device={next(model.parameters()).device}")

    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello! What is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text, return_tensors="pt").input_ids
    print(f"[load_test] running tiny text forward (CPU, seq_len={ids.shape[1]}) ...")
    with torch.inference_mode():
        out = model(input_ids=ids)
    print(f"  logits shape={tuple(out.logits.shape)}  dtype={out.logits.dtype}")
    next_id = int(out.logits[0, -1].argmax())
    print(f"  greedy next token id={next_id}  text={tok.decode([next_id])!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
