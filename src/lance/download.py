"""Download the Lance checkpoint shards from Hugging Face with resume.

Uses `huggingface_hub.hf_hub_download`, which already handles resume + sha
verification. We expose group selectors so the user can fetch just what they
need (e.g. only the image checkpoint, or only the configs).

CLI:
    python -m lance download --target weights/Lance_hf
    python -m lance download --group image
    python -m lance download --group video
    python -m lance download --group small      # configs + tokenizers only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from . import ALL_FILES, LARGE_FILES, REPO_ID, SMALL_FILES

GROUPS: dict[str, tuple[str, ...]] = {
    "all": ALL_FILES,
    "small": SMALL_FILES,
    "large": LARGE_FILES,
    "image": tuple(f for f in ALL_FILES if f.startswith("Lance_3B/") or f.startswith("Qwen2.5-VL-ViT/") or f == "Wan2.2_VAE.pth" or f == "README.md"),
    "video": tuple(f for f in ALL_FILES if f.startswith("Lance_3B_Video/") or f.startswith("Qwen2.5-VL-ViT/") or f == "Wan2.2_VAE.pth" or f == "README.md"),
    "vit": ("Qwen2.5-VL-ViT/config.json", "Qwen2.5-VL-ViT/vit.safetensors"),
    "vae": ("Wan2.2_VAE.pth",),
}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:6.1f}{unit}"
        n //= 1024
    return f"{n}PB"


def download_files(target: Path, files: tuple[str, ...], dry_run: bool = False) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for rel in files:
        local = target / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"  would fetch -> {rel}")
            continue
        print(f"  fetching {rel} ...", flush=True)
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=rel,
            local_dir=str(target),
            resume_download=True,
        )
        size = os.path.getsize(path)
        print(f"    ok {human(size):>10}  {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m lance download")
    p.add_argument(
        "--target",
        type=Path,
        default=Path("weights/Lance_hf"),
        help="local directory to mirror the HF tree into (default: weights/Lance_hf)",
    )
    p.add_argument(
        "--group",
        choices=sorted(GROUPS.keys()),
        default="small",
        help="which subset of files to fetch (default: small, ~1MB total)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    files = GROUPS[args.group]
    print(f"[lance.download] repo={REPO_ID} group={args.group} files={len(files)} target={args.target}")
    download_files(args.target, files, dry_run=args.dry_run)
    print("[lance.download] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
