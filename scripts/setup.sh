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
#   3. Creates a venv and installs deps in the right order to avoid
#      flash-attn's notorious build pitfalls:
#        - pip + wheel + packaging + ninja first (so flash-attn's setup.py works)
#        - pinned torch from the PyTorch index that has prebuilt flash-attn wheels
#        - project deps from pyproject.toml
#        - flash-attn with --no-build-isolation (uses the installed torch)
#
# It does NOT download model weights — that happens automatically the first
# time you launch the web UI. (~32 GB pulled from Hugging Face, resumable.)
#
# Environment knobs:
#   SKIP_VENV=1               don't create/use a venv; install into current env
#   VENV_DIR=.venv            where the venv lives (default .venv)
#   PYTHON=python3            interpreter used to create the venv
#   GPU_DEPS=auto|cu124|cu128|cu121|cpu|skip
#                             which torch + flash-attn pin to install:
#                               auto    — detect from nvidia-smi (default)
#                               cu124   — torch 2.5.1+cu124 / flash-attn 2.7.4
#                                         (Ampere sm_80/86, Ada sm_89, Hopper sm_90)
#                               cu128   — torch 2.9.0+cu128 / NO flash-attn
#                                         (Blackwell sm_120/121, e.g. GB10)
#                               cu121   — torch 2.5.1+cu121 / flash-attn 2.6.3
#                                         (official Lance pin)
#                               cpu     — torch CPU-only, no flash-attn
#                               skip    — don't touch torch/flash-attn (BYO)
#   FLASH_ATTN_SKIP=1         install torch but skip flash-attn (e.g. you'll
#                             rely on SDPA + flex_attention only)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OFFICIAL_DIR="$REPO_ROOT/refs/lance_official"
SKIP_VENV="${SKIP_VENV:-0}"
VENV_DIR="${VENV_DIR:-.venv}"
PY="${PYTHON:-python3}"
GPU_DEPS="${GPU_DEPS:-auto}"
FLASH_ATTN_SKIP="${FLASH_ATTN_SKIP:-0}"

echo "================================================================"
echo "  Lance Assistant — setup"
echo "================================================================"

# ───────────────────────────────────────────────────────── 1. official code

if [ -d "$OFFICIAL_DIR/.git" ]; then
    echo "[1/4] refs/lance_official/ already cloned; pulling latest"
    git -C "$OFFICIAL_DIR" fetch --depth 1 origin
    git -C "$OFFICIAL_DIR" reset --hard origin/HEAD
else
    echo "[1/4] cloning bytedance/Lance into refs/lance_official/"
    mkdir -p "$REPO_ROOT/refs"
    git clone --depth 1 https://github.com/bytedance/Lance.git "$OFFICIAL_DIR"
fi

# ───────────────────────────────────────────────────────── 2. apply patch

VAL_DS="$OFFICIAL_DIR/data/datasets_custom/validation_dataset.py"
if [ -f "$VAL_DS" ] && grep -q "^import decord$" "$VAL_DS"; then
    echo "[2/4] applying decord soft-import patch to $VAL_DS"
    python3 - <<PY
import re
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
    echo "[2/4] decord patch not needed"
fi

# ───────────────────────────────────────────────────────── 3. venv

if [ "$SKIP_VENV" = "1" ]; then
    echo "[3/4] SKIP_VENV=1 → using current environment"
    PIP="pip"
    PYTHON_VENV="${PYTHON:-python3}"
else
    if [ ! -d "$VENV_DIR" ]; then
        echo "[3/4] creating venv at $VENV_DIR"
        "$PY" -m venv "$VENV_DIR"
    else
        echo "[3/4] using existing venv at $VENV_DIR"
    fi
    PIP="$VENV_DIR/bin/pip"
    PYTHON_VENV="$VENV_DIR/bin/python"
fi

# ───────────────────────────────────────────────────────── 4. install deps
echo "[4/4] installing dependencies"

# 4a. Build prerequisites. These MUST come before torch/flash-attn because
#     flash-attn's setup.py imports `packaging` and needs `ninja` to build
#     (even when it ends up grabbing a prebuilt wheel, the build script
#     still tries to import them).
echo "  - upgrading pip + build tools"
"$PIP" install --upgrade pip wheel setuptools packaging ninja >/dev/null

# 4b. Decide which torch + flash-attn stack to install.
detect_gpu_arch() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "none"
        return
    fi
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ' || true)
    if [ -z "$cc" ]; then
        echo "unknown"
        return
    fi
    case "$cc" in
        80|86|89|90) echo "ampere_ada_hopper" ;;
        120|121)     echo "blackwell" ;;
        *)           echo "other_$cc" ;;
    esac
}

TORCH_PIN=""
FLASH_ATTN_PIN=""
TORCH_INDEX=""

resolve_gpu_deps() {
    local kind="$1"
    case "$kind" in
        cu124)
            TORCH_PIN="torch==2.5.1"
            FLASH_ATTN_PIN="flash-attn==2.7.4.post1"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            ;;
        cu121)
            TORCH_PIN="torch==2.5.1"
            FLASH_ATTN_PIN="flash-attn==2.6.3"
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            ;;
        cu128)
            # Blackwell — cu128 needed for sm_120/121. flash-attn's prebuilt
            # wheels don't yet cover this combo and source build often fails
            # on these GPUs, so we skip it and rely on SDPA + flex_attention
            # (which is what webui/server.py forces anyway on Blackwell).
            TORCH_PIN="torch==2.9.0"
            FLASH_ATTN_PIN=""        # skip
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
            ;;
        cpu)
            TORCH_PIN="torch==2.5.1+cpu"
            FLASH_ATTN_PIN=""
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            ;;
        skip)
            TORCH_PIN=""
            FLASH_ATTN_PIN=""
            TORCH_INDEX=""
            ;;
    esac
}

if [ "$GPU_DEPS" = "auto" ]; then
    arch="$(detect_gpu_arch)"
    case "$arch" in
        ampere_ada_hopper)
            echo "  - detected Ampere/Ada/Hopper GPU → using cu124 stack"
            resolve_gpu_deps cu124
            ;;
        blackwell)
            echo "  - detected Blackwell GPU (sm_120/121) → using cu128 stack (no flash-attn; SDPA only)"
            resolve_gpu_deps cu128
            ;;
        none)
            echo "  - no nvidia-smi found; installing CPU-only torch"
            resolve_gpu_deps cpu
            ;;
        *)
            echo "  - unknown GPU arch ($arch); defaulting to cu124 stack — override with GPU_DEPS=..."
            resolve_gpu_deps cu124
            ;;
    esac
else
    echo "  - GPU_DEPS=$GPU_DEPS (explicit override)"
    resolve_gpu_deps "$GPU_DEPS"
fi

# 4c. Install torch FIRST (and pin it), before anything else. This makes
#     sure flash-attn's setup.py sees the exact torch it wants.
if [ -n "$TORCH_PIN" ]; then
    echo "  - installing $TORCH_PIN from $TORCH_INDEX"
    "$PIP" install "$TORCH_PIN" torchvision torchaudio --index-url "$TORCH_INDEX"
fi

# 4d. Project deps from pyproject.toml. --upgrade-strategy=only-if-needed
#     means torch won't get bumped if pyproject's spec is satisfied.
echo "  - installing project deps (pyproject.toml)"
"$PIP" install -e . --upgrade-strategy only-if-needed

# 4e. flash-attn — opt-in, --no-build-isolation always so it sees torch.
if [ "$FLASH_ATTN_SKIP" = "1" ] || [ -z "$FLASH_ATTN_PIN" ]; then
    if [ "$FLASH_ATTN_SKIP" = "1" ]; then
        echo "  - skipping flash-attn (FLASH_ATTN_SKIP=1)"
    else
        echo "  - skipping flash-attn (not needed / not supported for this GPU)"
    fi
else
    echo "  - installing $FLASH_ATTN_PIN (with --no-build-isolation)"
    if ! "$PIP" install "$FLASH_ATTN_PIN" --no-build-isolation; then
        echo ""
        echo "  flash-attn install failed."
        echo "  Common causes:"
        echo "    1. No prebuilt wheel for your torch / cuda / python combo, and"
        echo "       no nvcc available to compile from source."
        echo "    2. Python version too new or too old for the pinned flash-attn."
        echo ""
        echo "  Try one of:"
        echo "    - ./scripts/setup.sh again with GPU_DEPS=cu121 (older but more wheels)"
        echo "    - FLASH_ATTN_SKIP=1 ./scripts/setup.sh  (rely on SDPA only)"
        echo "    - manually: $PIP install flash-attn==<version> --no-build-isolation"
        echo "  See README → Troubleshooting for the full matrix."
        echo ""
        exit 1
    fi
fi

echo ""
echo "Installed packages:"
"$PIP" list 2>/dev/null | grep -iE "^(torch|transformers|fastapi|uvicorn|safetensors|huggingface-hub|httpx|python-dotenv|accelerate|einops|imageio|flash-attn) " || true

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
