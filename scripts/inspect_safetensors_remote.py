"""Read safetensors headers remotely via HTTP range requests.

safetensors layout:
    [8 bytes little-endian uint64 = header_size]
    [header_size bytes of JSON]
    [raw tensor bytes ...]

We only need to fetch the first ~few MB to map every weight key + shape, no
matter how big the file. Saves us downloading 50+ GB just to read the schema.
"""

from __future__ import annotations

import json
import struct
import sys
import urllib.request
from pathlib import Path

REPO_ID = "bytedance-research/Lance"
REVISION = "main"

TARGETS = [
    "Lance_3B/model.safetensors",
    "Lance_3B_Video/model.safetensors",
    "Qwen2.5-VL-ViT/vit.safetensors",
]

OUT_DIR = Path(__file__).resolve().parent.parent / "notes"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def safetensors_header(url: str) -> dict:
    head = fetch_range(url, 0, 7)
    (header_len,) = struct.unpack("<Q", head)
    if header_len > 200_000_000:
        raise RuntimeError(f"absurd header_len {header_len} from {url}")
    payload = fetch_range(url, 8, 8 + header_len - 1)
    return json.loads(payload)


def main() -> None:
    summary: list[str] = []
    for rel in TARGETS:
        url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{rel}"
        print(f"=== {rel} ===", flush=True)
        try:
            header = safetensors_header(url)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue
        meta = header.pop("__metadata__", {})
        keys = sorted(header.keys())
        print(f"  __metadata__: {meta}")
        print(f"  num_tensors: {len(keys)}")
        total_numel = 0
        for k in keys:
            info = header[k]
            shape = info.get("shape", [])
            dtype = info.get("dtype", "?")
            n = 1
            for d in shape:
                n *= int(d)
            total_numel += n
            print(f"    {k}  shape={shape} dtype={dtype}")
        print(f"  approx_param_count: {total_numel:,}")
        summary.append(f"{rel}\n  tensors: {len(keys)}  params: {total_numel:,}\n")
        out_path = OUT_DIR / (rel.replace("/", "__") + ".keys.txt")
        out_path.write_text(
            "\n".join(f"{k}\tshape={header[k].get('shape')}\tdtype={header[k].get('dtype')}" for k in keys)
        )
        print(f"  saved -> {out_path}")

    print("\n=== SUMMARY ===\n" + "".join(summary))


if __name__ == "__main__":
    main()
