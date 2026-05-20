#!/bin/bash
# Thin wrapper around refs/lance_official/inference_lance.sh that:
#  - sets up the downloads/ symlinks if needed
#  - lets you pick task from argv: t2i | t2v | image_edit | video_edit |
#    x2t_image | x2t_video
#  - uses .venv/ (or $PYTHON) instead of system python
#
# Usage:
#   ./scripts/run_official.sh t2v
#   ./scripts/run_official.sh t2i 768 30 4.0       # task, res, steps, cfg
#
# Outputs go to refs/lance_official/results/<TASK>_sample_*/000000.{png,mp4}

set -euo pipefail

TASK="${1:-t2i}"
RES="${2:-}"                  # 768 for image, 480 for video; auto if blank
STEPS="${3:-30}"
CFG="${4:-4.0}"
FRAMES="${5:-81}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFICIAL_DIR="$REPO_ROOT/refs/lance_official"
WEIGHTS_DIR="$REPO_ROOT/weights/Lance_hf"

# Pick interpreter / accelerate launcher
if [ -n "${PYTHON:-}" ]; then
    VENV_PY="$PYTHON"
    VENV_ACCELERATE="$(dirname "$PYTHON")/accelerate"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    VENV_PY="$REPO_ROOT/.venv/bin/python"
    VENV_ACCELERATE="$REPO_ROOT/.venv/bin/accelerate"
else
    VENV_PY="$(command -v python3 || true)"
    VENV_ACCELERATE="$(command -v accelerate || true)"
fi
if [ -z "$VENV_PY" ] || [ ! -x "$VENV_PY" ]; then
    echo "ERROR: no Python interpreter found. Run ./scripts/setup.sh first." >&2
    exit 1
fi

if [ ! -d "$OFFICIAL_DIR/.git" ]; then
    echo "First-time setup needed (cloning refs/lance_official)..."
    SKIP_VENV=1 "$REPO_ROOT/scripts/setup.sh"
fi

WHICH_GROUP="all"
case "$TASK" in
    t2v|video_edit|x2t_video) WHICH_GROUP="video" ;;
    t2i|image_edit|x2t_image) WHICH_GROUP="image" ;;
esac
if [ ! -d "$WEIGHTS_DIR" ] || \
   { [ "$WHICH_GROUP" = "image" ] && [ ! -d "$WEIGHTS_DIR/Lance_3B" ]; } || \
   { [ "$WHICH_GROUP" = "video" ] && [ ! -d "$WEIGHTS_DIR/Lance_3B_Video" ]; }; then
    echo "Weights missing — downloading group=$WHICH_GROUP from Hugging Face..."
    "$VENV_PY" -m lance download --group "$WHICH_GROUP" --target "$WEIGHTS_DIR"
fi

mkdir -p "$OFFICIAL_DIR/downloads"
ln -sfn "../../../weights/Lance_hf/Lance_3B"        "$OFFICIAL_DIR/downloads/lance_3b"
ln -sfn "../../../weights/Lance_hf/Lance_3B_Video"  "$OFFICIAL_DIR/downloads/lance_3b_video"
ln -sfn "../../../weights/Lance_hf/Qwen2.5-VL-ViT"  "$OFFICIAL_DIR/downloads/Qwen2.5-VL-ViT"
ln -sfn "../../../weights/Lance_hf/Wan2.2_VAE.pth"  "$OFFICIAL_DIR/downloads/Wan2.2_VAE.pth"

case "$TASK" in
    t2i|image_edit|x2t_image)
        MODEL_PATH="downloads/lance_3b"
        RESOLUTION="image_768res"
        VH=${RES:-768}; VW=${RES:-768}
        NUM_FRAMES=1
        ;;
    t2v|video_edit|x2t_video)
        MODEL_PATH="downloads/lance_3b_video"
        RESOLUTION="video_480p"
        VH=480; VW=832
        NUM_FRAMES="$FRAMES"
        ;;
    *)
        echo "unknown task: $TASK"
        echo "valid: t2i | t2v | image_edit | video_edit | x2t_image | x2t_video"
        exit 2
        ;;
esac

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAVE_PATH="results/${TASK}_sample_ts${STEPS}_cfg${CFG}_${TIMESTAMP}"

cd "$OFFICIAL_DIR"
# shellcheck source=/dev/null
source benchmarks/sample_env.sh
lance_setup_common_env
lance_setup_distributed_env 1
lance_setup_shard_env 1

"$VENV_ACCELERATE" launch \
    --num_machines $NUM_MACHINES \
    --num_processes $TOTAL_RANK \
    --machine_rank $MACHINE_RANK \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MAIN_PROCESS_PORT \
    --mixed_precision bf16 \
    inference_lance.py \
    --model_path            "$MODEL_PATH" \
    --vit_type              qwen_2_5_vl_original \
    --llm_qk_norm           true \
    --llm_qk_norm_und       true \
    --llm_qk_norm_gen       true \
    --tie_word_embeddings   false \
    --validation_num_timesteps "$STEPS" \
    --validation_timestep_shift 3.5 \
    --copy_init_moe         true \
    --max_num_frames        121 \
    --max_latent_size       64 \
    --latent_patch_size     1 1 1 \
    --visual_und            true \
    --visual_gen            true \
    --vae_model_type        wan \
    --apply_qwen_2_5_vl_pos_emb true \
    --apply_chat_template   false \
    --cfg_type              0 \
    --validation_data_seed  42 \
    --video_height          "$VH" \
    --video_width           "$VW" \
    --num_frames            "$NUM_FRAMES" \
    --task                  "$TASK" \
    --save_path_gen         "$SAVE_PATH" \
    --resolution            "$RESOLUTION" \
    --text_template         true \
    --cfg_text_scale        "$CFG" \
    --use_KVcache           true

echo ""
echo "================================================"
echo "Outputs: $OFFICIAL_DIR/$SAVE_PATH"
echo "================================================"
