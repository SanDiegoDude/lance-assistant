#!/bin/bash
# First-time setup for Lance Assistant.
#
# Run this once after `git clone`:
#   ./scripts/setup.sh
#
# What it does:
#   1. Clones ByteDance/Lance into refs/lance_official/ (the inference code
#      that we drive — gitignored because it's a separate upstream repo).
#   2. Applies our small `decord` soft-import patch so the code works on
#      Python 3.12 (where decord has no wheel).
#   3. Optionally creates a Python virtualenv and installs dependencies.
#
# It does NOT download model weights — that happens automatically the first
# time you launch the web UI. (~32 GB pulled from Hugging Face, resumable.)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OFFICIAL_DIR="$REPO_ROOT/refs/lance_official"
SKIP_VENV="${SKIP_VENV:-0}"
VENV_DIR="${VENV_DIR:-.venv}"
PY="${PYTHON:-python3}"

echo "================================================================"
echo "  Lance Assistant — setup"
echo "================================================================"

# ───────────────────────────────────────────────────────── 1. official code

if [ -d "$OFFICIAL_DIR/.git" ]; then
    echo "[1/3] refs/lance_official/ already cloned; pulling latest"
    git -C "$OFFICIAL_DIR" fetch --depth 1 origin
    git -C "$OFFICIAL_DIR" reset --hard origin/HEAD
else
    echo "[1/3] cloning bytedance/Lance into refs/lance_official/"
    mkdir -p "$REPO_ROOT/refs"
    git clone --depth 1 https://github.com/bytedance/Lance.git "$OFFICIAL_DIR"
fi

# ───────────────────────────────────────────────────────── 2. apply patch

VAL_DS="$OFFICIAL_DIR/data/datasets_custom/validation_dataset.py"
if [ -f "$VAL_DS" ] && grep -q "^import decord$" "$VAL_DS"; then
    echo "[2/3] applying decord soft-import patch to $VAL_DS"
    python3 - <<PY
import io, re, sys
from pathlib import Path
p = Path("$VAL_DS")
src = p.read_text()
patched = re.sub(
    r"^import decord\nfrom decord import VideoReader\n",
    "try:\n    import decord\n    from decord import VideoReader\nexcept ImportError:\n    decord = None\n    VideoReader = None\n",
    src,
    count=1,
    flags=re.MULTILINE,
)
if patched != src:
    p.write_text(patched)
    print("  patched")
else:
    print("  no change needed")
PY
else
    echo "[2/3] decord patch not needed"
fi

# ───────────────────────────────────────────────────────── 3. venv + deps

if [ "$SKIP_VENV" = "1" ]; then
    echo "[3/3] SKIP_VENV=1 → skipping virtualenv + pip install"
    echo "      Make sure your environment satisfies pyproject.toml deps."
else
    if [ ! -d "$VENV_DIR" ]; then
        echo "[3/3] creating venv at $VENV_DIR"
        "$PY" -m venv "$VENV_DIR"
    else
        echo "[3/3] using existing venv at $VENV_DIR"
    fi
    # Upgrade pip first, then install
    "$VENV_DIR/bin/pip" install --upgrade pip
    # Core deps. For GPU-specific deps (torch, flash-attn) you may want a
    # CUDA-matched build — pyproject pins minimums but you can override.
    "$VENV_DIR/bin/pip" install -e .
    echo ""
    echo "Installed packages:"
    "$VENV_DIR/bin/pip" list | grep -iE "^(torch|transformers|fastapi|uvicorn|safetensors|huggingface-hub|httpx|python-dotenv|accelerate|einops|imageio|flash-attn)" || true
fi

echo ""
echo "================================================================"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    1. cp .env.example .env       # configure orchestrator (optional)"
echo "    2. ./scripts/run_webui.sh     # launches the UI on http://localhost:7861"
echo ""
echo "  First launch downloads ~32 GB of Lance weights from Hugging Face"
echo "  (resumable). After that, startup is fast."
echo "================================================================"
