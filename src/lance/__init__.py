"""Lance tinker package: utilities for working with bytedance-research/Lance."""

__version__ = "0.0.1"

REPO_ID = "bytedance-research/Lance"

ALL_FILES = (
    "README.md",
    "Lance_3B/generation_config.json",
    "Lance_3B/llm_config.json",
    "Lance_3B/merges.txt",
    "Lance_3B/model.safetensors",
    "Lance_3B/tokenizer.json",
    "Lance_3B/vocab.json",
    "Lance_3B_Video/generation_config.json",
    "Lance_3B_Video/llm_config.json",
    "Lance_3B_Video/merges.txt",
    "Lance_3B_Video/model.safetensors",
    "Lance_3B_Video/tokenizer.json",
    "Lance_3B_Video/tokenizer_config.json",
    "Lance_3B_Video/vocab.json",
    "Qwen2.5-VL-ViT/config.json",
    "Qwen2.5-VL-ViT/vit.safetensors",
    "Wan2.2_VAE.pth",
)

SMALL_FILES = tuple(f for f in ALL_FILES if not f.endswith((".safetensors", ".pth")))
LARGE_FILES = tuple(f for f in ALL_FILES if f.endswith((".safetensors", ".pth")))
