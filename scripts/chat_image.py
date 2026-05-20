"""Image VQA / captioning against the extracted Lance understanding ckpt.

Uses the standard Qwen2.5-VL processor for image preprocessing; the ViT
weights spliced into our checkpoint are stock Qwen2.5-VL ViT so this Just
Works once the QK-norm patch is applied to the language model.

Usage:
    python scripts/chat_image.py --image refs/bagel/test_images/meme.jpg \\
        --prompt "Describe this image in detail."
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from lance.qknorm_patch import patch_qwen2_5_vl_qknorm

patch_qwen2_5_vl_qknorm()
from transformers import (  # noqa: E402
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

CKPT = Path("weights/lance_3b_understand")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--prompt", default="Describe this image in one sentence.")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--processor",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF id for the image processor (we don't ship one in the extracted ckpt)",
    )
    args = p.parse_args()

    print(f"[vqa] image={args.image} device={args.device}")
    processor = AutoProcessor.from_pretrained(args.processor)
    print(f"[vqa] processor loaded: {type(processor).__name__}")

    print(f"[vqa] loading model in bf16 onto {args.device} ...")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        CKPT,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()
    print(f"[vqa] model loaded in {time.time() - t0:.1f}s")

    image = Image.open(args.image).convert("RGB")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(args.device)
    print(f"[vqa] inputs: input_ids={tuple(inputs['input_ids'].shape)}  pixel_values={tuple(inputs['pixel_values'].shape)}")

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new,
            do_sample=False,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :].tolist()
    print(f"[vqa] generated {len(new_tokens)} tokens in {time.time() - t0:.1f}s")
    print(f"\n=== prompt ===\n{args.prompt}\n=== response ===\n{processor.batch_decode([new_tokens], skip_special_tokens=True)[0]}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
