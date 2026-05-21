#!/bin/bash
# Launch the Lance multimodal chat web UI.
#
# This is a FastAPI server (webui/server.py) that holds the official Lance
# pipeline (`refs/lance_official/`) in memory and dispatches across all six
# multimodal tasks (t2i, t2v, image_edit, video_edit, x2t_image, x2t_video)
# plus pure text chat (using the extracted understanding checkpoint at
# weights/lance_3b_understand). One model, one process, chat-style UI.
#
# Usage:
#   ./scripts/run_webui.sh                  # binds 0.0.0.0:7861
#   ./scripts/run_webui.sh 8080             # custom port
#   ./scripts/run_webui.sh --lowvram        # one variant on GPU at a time
#   ./scripts/run_webui.sh --nopersist      # don't write conversations to disk
#   ./scripts/run_webui.sh --lowvram --nopersist 8080
#   PORT=8080 HOST=127.0.0.1 ./scripts/run_webui.sh
#
# Flags:
#   --lowvram    enable hot-swapping between image / video Lance variants
#                (only one resides on the GPU at a time; the other is
#                parked in system RAM). Sets LANCE_LOWVRAM=1. Recommended
#                for 24 GB cards (4090); leave off on big-VRAM hardware
#                so both variants stay GPU-resident and swap is free.
#   --nopersist  do NOT save conversations to disk. The default is to
#                store every chat under webui/tmp/state/ so it survives
#                a server restart. With --nopersist nothing is written
#                and any state from previous runs is ignored; on
#                shutdown all conversations are gone. Use this if you
#                don't want chat history to outlive the process.
#                Sets LANCE_NOPERSIST=1.
#   --debug      verbose tracing through the orchestrator + agentic-turn
#                path: every request payload (truncated), every SSE
#                chunk received, tool-call decisions, job submissions.
#                Output also mirrored to webui/tmp/logs/debug.log
#                (truncated on each start). Sets LANCE_DEBUG=1.
#
# Environment overrides:
#   PYTHON          path to python (defaults to .venv/bin/python, then `python3`)
#   HOST            bind address (default 0.0.0.0)
#   PORT            bind port    (default 7861)
#   LANCE_LOWVRAM   same as --lowvram (1 / true / yes / on enables)
#   LANCE_NOPERSIST same as --nopersist
#   LANCE_DEBUG     same as --debug
#   LANCE_MODEL_VARIANT  image | video | auto (default auto — both available)
#   LANCE_DTYPE     bfloat16 (default) | float16 | float32

set -euo pipefail

# Parse our own flags before falling through to positional args.
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --lowvram)
            export LANCE_LOWVRAM=1
            ;;
        --nopersist)
            export LANCE_NOPERSIST=1
            ;;
        --debug)
            export LANCE_DEBUG=1
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done
set -- "${ARGS[@]}"

PORT="${1:-${PORT:-7861}}"
HOST="${HOST:-0.0.0.0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_DIR="$REPO_ROOT/refs/lance_official"

# ── Pick a Python interpreter ────────────────────────────────────────────
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
else
    PY="$(command -v python3 || true)"
fi

if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "ERROR: no Python interpreter found." >&2
    echo "       Run ./scripts/setup.sh first, or set PYTHON=/path/to/python" >&2
    exit 1
fi

# ── Make sure the upstream Lance code is present ────────────────────────
if [ ! -d "$OFFICIAL_DIR/.git" ]; then
    echo ""
    echo "First-time setup needed: refs/lance_official/ is missing."
    echo "Running ./scripts/setup.sh for you..."
    echo ""
    SKIP_VENV=1 "$REPO_ROOT/scripts/setup.sh"
fi

cd "$REPO_ROOT"

# expandable_segments dramatically reduces allocator fragmentation at the
# cost of a tiny bit of bookkeeping — useful when a long-running process
# alternates between large and small allocations (which is exactly what
# the edit + decode pipeline does). The OOM error message itself recommends
# this. Only set if the user hasn't picked their own value.
if [ -z "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi

echo "================================================================"
echo "  Lance Assistant"
echo "  Open  http://${HOST}:${PORT}  (or http://localhost:${PORT})"
echo ""
if [ "${LANCE_LOWVRAM:-0}" = "1" ] || [ "${LANCE_LOWVRAM:-0}" = "true" ]; then
    echo "  Mode: --lowvram (one variant on GPU at a time, hot-swap on demand)"
fi
if [ "${LANCE_NOPERSIST:-0}" = "1" ] || [ "${LANCE_NOPERSIST:-0}" = "true" ]; then
    echo "  Mode: --nopersist (conversations live in RAM only — no disk writes)"
fi
if [ "${LANCE_DEBUG:-0}" = "1" ] || [ "${LANCE_DEBUG:-0}" = "true" ]; then
    echo "  Mode: --debug (verbose trace — tail webui/tmp/logs/debug.log)"
fi
echo "  Note: first launch will download ~32 GB of Lance weights from"
echo "  Hugging Face into weights/Lance_hf/ (resumable, only once)."
echo "  Under LANCE_MODEL_VARIANT=auto (default), the second variant is"
echo "  downloaded lazily on the first task that needs it (~16 GB more)."
echo "================================================================"
echo ""

exec "$PY" -m uvicorn webui.server:app \
    --host "$HOST" --port "$PORT" \
    --log-level info
