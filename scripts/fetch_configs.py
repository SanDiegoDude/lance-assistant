"""Download just the small config / tokenizer files from bytedance-research/Lance.

This lets us understand the architecture without pulling the multi-GB
safetensors shards.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "bytedance-research/Lance"
TARGET_DIR = Path(__file__).resolve().parent.parent / "weights" / "Lance_hf"
TARGET_DIR.mkdir(parents=True, exist_ok=True)

SMALL_FILES = [
    "README.md",
    "Lance_3B/generation_config.json",
    "Lance_3B/llm_config.json",
    "Lance_3B/merges.txt",
    "Lance_3B/tokenizer.json",
    "Lance_3B/vocab.json",
    "Lance_3B_Video/generation_config.json",
    "Lance_3B_Video/llm_config.json",
    "Lance_3B_Video/merges.txt",
    "Lance_3B_Video/tokenizer.json",
    "Lance_3B_Video/tokenizer_config.json",
    "Lance_3B_Video/vocab.json",
    "Qwen2.5-VL-ViT/config.json",
]


def main() -> None:
    for fname in SMALL_FILES:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=fname,
            local_dir=str(TARGET_DIR),
        )
        print(f"  ok -> {path}")


if __name__ == "__main__":
    main()
