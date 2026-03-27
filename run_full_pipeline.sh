#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <image_folder_or_video> [run_dir] [fps_for_video]"
  exit 1
fi

INPUT_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
RUN_DIR="${2:-$PROJECT_ROOT/runs/$(date +%Y%m%d_%H%M%S)}"
VIDEO_FPS="${3:-8}"
USE_CPU="${USE_CPU:-1}"
MAX_LOOPS="${MAX_LOOPS:-0}"
EXTRACT_MAX_SECONDS="${EXTRACT_MAX_SECONDS:-}"
EXTRACT_START_SECONDS="${EXTRACT_START_SECONDS:-}"
VIDEO_START_SECONDS="${EXTRACT_START_SECONDS:-0}"
VIDEO_DURATION_SECONDS="${EXTRACT_MAX_SECONDS:-0}"
STAGE2_MAX_FRAMES="${STAGE2_MAX_FRAMES:-0}"
BACKEND="${BACKEND:-vggt}"
LOGER_ENV="${LOGER_ENV:-loger_a10}"
LOGER_CONF_THRES="${LOGER_CONF_THRES:-0.05}"
LOGER_WINDOW_SIZE="${LOGER_WINDOW_SIZE:-32}"
LOGER_OVERLAP_SIZE="${LOGER_OVERLAP_SIZE:-3}"
LOGER_MODEL_NAME="${LOGER_MODEL_NAME:-ckpts/LoGeR_star/latest.pt}"
LOGER_CONFIG_PATH="${LOGER_CONFIG_PATH:-ckpts/LoGeR_star/original_config.yaml}"
RUN_OS="${RUN_OS:-0}"
OS_QUERIES="${OS_QUERIES:-door,sign}"
OS_SHOW_MASKS="${OS_SHOW_MASKS:-0}"
SAM3_CKPT_PATH="${SAM3_CKPT_PATH:-}"
SAM3_LOAD_FROM_HF="${SAM3_LOAD_FROM_HF:-1}"
DETECTOR_BACKEND="${DETECTOR_BACKEND:-sam3}"
SAM3_QUERY_FILTER="${SAM3_QUERY_FILTER:-sign,door}"
SAM3_MIN_SAM_SCORE="${SAM3_MIN_SAM_SCORE:-0.55}"
SAM3_MIN_SEMANTIC_SCORE="${SAM3_MIN_SEMANTIC_SCORE:--1.0}"
SAM3_FALLBACK_TO_DETR="${SAM3_FALLBACK_TO_DETR:-1}"
DET_THRESHOLD="${DET_THRESHOLD:-0.75}"
OCR_BACKEND="${OCR_BACKEND:-auto}"
OCR_TESS_LANG="${OCR_TESS_LANG:-eng}"
OCR_PADDLE_LANG="${OCR_PADDLE_LANG:-korean}"
OCR_DISABLE_ENHANCE="${OCR_DISABLE_ENHANCE:-0}"
OCR_NUMERIC_MODE="${OCR_NUMERIC_MODE:-1}"
OCR_NUMERIC_ONLY="${OCR_NUMERIC_ONLY:-1}"
OCR_NUMERIC_MIN_DIGITS="${OCR_NUMERIC_MIN_DIGITS:-2}"
OCR_NUMERIC_CHARS="${OCR_NUMERIC_CHARS:-B0123456789 }"
OCR_TEXT_PATTERN="${OCR_TEXT_PATTERN:-[^B0-9\\s]+}"
OCR_NUMERIC_PSM_LIST="${OCR_NUMERIC_PSM_LIST:-7,6,8,13}"
OCR_NUMERIC_USE_PADDLE="${OCR_NUMERIC_USE_PADDLE:-1}"
OCR_NUMERIC_TOKEN_MIN_LEN="${OCR_NUMERIC_TOKEN_MIN_LEN:-2}"
OCR_NUMERIC_TOKEN_MAX_LEN="${OCR_NUMERIC_TOKEN_MAX_LEN:-0}"
OCR_NUMERIC_MIN_TOKENS="${OCR_NUMERIC_MIN_TOKENS:-1}"
OCR_NUMERIC_MAX_TOKENS="${OCR_NUMERIC_MAX_TOKENS:-0}"
OCR_NUMERIC_PREFER_MULTI_TOKEN="${OCR_NUMERIC_PREFER_MULTI_TOKEN:-1}"
OCR_NUMERIC_PREFER_THREE_DIGIT="${OCR_NUMERIC_PREFER_THREE_DIGIT:-0}"
OCR_NUMERIC_ROI_STRICT="${OCR_NUMERIC_ROI_STRICT:-0}"
OCR_PRECHECK_BEFORE_UPSCALE="${OCR_PRECHECK_BEFORE_UPSCALE:-1}"
OCR_PRECHECK_MIN_SCORE="${OCR_PRECHECK_MIN_SCORE:-95}"
OCR_PRECHECK_MIN_DIGITS="${OCR_PRECHECK_MIN_DIGITS:-3}"
OCR_UPSCALE_ACCEPT_MARGIN="${OCR_UPSCALE_ACCEPT_MARGIN:-10}"
OCR_UPSCALE_MAX_EXTRA_DIGITS="${OCR_UPSCALE_MAX_EXTRA_DIGITS:-0}"
OCR_UPSCALE_EXTRA_DIGIT_PENALTY="${OCR_UPSCALE_EXTRA_DIGIT_PENALTY:-28}"
OCR_TROCR_MODEL="${OCR_TROCR_MODEL:-microsoft/trocr-small-printed}"
SUPERRES_BACKEND="${SUPERRES_BACKEND:-auto}"
SUPERRES_SCALE="${SUPERRES_SCALE:-4.0}"
REALESRGAN_MODEL_VARIANT="${REALESRGAN_MODEL_VARIANT:-general-x4v3}"
REALESRGAN_MODEL_PATH="${REALESRGAN_MODEL_PATH:-}"
DIFFUSION_MODEL_ID="${DIFFUSION_MODEL_ID:-stabilityai/stable-diffusion-x4-upscaler}"
DIFFUSION_PROMPT="${DIFFUSION_PROMPT:-clean sharp readable sign text}"
DIFFUSION_NEGATIVE_PROMPT="${DIFFUSION_NEGATIVE_PROMPT:-blur, noise, artifact}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-20}"
DIFFUSION_GUIDANCE_SCALE="${DIFFUSION_GUIDANCE_SCALE:-7.0}"
DIFFUSION_MAX_INPUT_SIDE="${DIFFUSION_MAX_INPUT_SIDE:-256}"
DIFFUSION_SELECTIVE_ENABLED="${DIFFUSION_SELECTIVE_ENABLED:-0}"
DIFFUSION_SELECTIVE_MIN_AREA_RATIO="${DIFFUSION_SELECTIVE_MIN_AREA_RATIO:-0.0012}"
DIFFUSION_SELECTIVE_MIN_FRONTALNESS="${DIFFUSION_SELECTIVE_MIN_FRONTALNESS:-0.55}"
DIFFUSION_SELECTIVE_MAX_CROPS="${DIFFUSION_SELECTIVE_MAX_CROPS:-2}"
DIFFUSION_UNCERTAIN_MIN_DIGITS="${DIFFUSION_UNCERTAIN_MIN_DIGITS:-4}"
DIFFUSION_UNCERTAIN_MIN_TEXT_SCORE="${DIFFUSION_UNCERTAIN_MIN_TEXT_SCORE:-20}"
OCR_SUPERRES_SCALES="${OCR_SUPERRES_SCALES:-2,4}"
OCR_UPSCALE="${OCR_UPSCALE:-2.5}"
OCR_MIN_CROP_SIDE="${OCR_MIN_CROP_SIDE:-120}"
OCR_MAX_CANDIDATES="${OCR_MAX_CANDIDATES:-10}"
CROP_EXPAND_RATIO="${CROP_EXPAND_RATIO:-0.08}"
TRACKER="${TRACKER:-bytetrack}"
TRACKER_IOU_THRES="${TRACKER_IOU_THRES:-0.35}"
TRACKER_LOST_FRAMES="${TRACKER_LOST_FRAMES:-10}"
TRACKER_MIN_HITS="${TRACKER_MIN_HITS:-2}"
VOTE_MIN_COUNT="${VOTE_MIN_COUNT:-2}"
FINAL_LABEL_VOCAB="${FINAL_LABEL_VOCAB:-}"
FINAL_LABEL_VOCAB_ENABLE="${FINAL_LABEL_VOCAB_ENABLE:-0}"
FINAL_LABEL_VOCAB_ENFORCE="${FINAL_LABEL_VOCAB_ENFORCE:-0}"
FINAL_LABEL_VOCAB_MIN_SIM="${FINAL_LABEL_VOCAB_MIN_SIM:-0.72}"
FINAL_LABEL_VOCAB_VOTE_BONUS="${FINAL_LABEL_VOCAB_VOTE_BONUS:-8.0}"
FINAL_LABEL_VOCAB_OOV_PENALTY="${FINAL_LABEL_VOCAB_OOV_PENALTY:-0.25}"
REPR_USE_FRONTAL="${REPR_USE_FRONTAL:-1}"
REPR_FRONTAL_WEIGHT="${REPR_FRONTAL_WEIGHT:-18.0}"
TRACK_MERGE_ENABLED="${TRACK_MERGE_ENABLED:-1}"
TRACK_MERGE_3D_DIST="${TRACK_MERGE_3D_DIST:-1.25}"
MIN_BBOX_WIDTH="${MIN_BBOX_WIDTH:-18}"
MIN_BBOX_HEIGHT="${MIN_BBOX_HEIGHT:-18}"
MIN_BBOX_AREA_RATIO="${MIN_BBOX_AREA_RATIO:-0.00035}"
WAIT_GPU_IDLE="${WAIT_GPU_IDLE:-0}"
GPU_IDLE_POLL_SECONDS="${GPU_IDLE_POLL_SECONDS:-5}"

VGGT_REPO="${VGGT_REPO:-$PROJECT_ROOT/VGGT-SLAM}"
VGGT_PY_REPO="${VGGT_PY_REPO:-$PROJECT_ROOT/vggt}"
LOGER_REPO="${LOGER_REPO:-$PROJECT_ROOT/LoGeR}"
PI3_REPO="${PI3_REPO:-$WORKSPACE_ROOT/Pi3}"
DETR_REPO="${DETR_REPO:-$WORKSPACE_ROOT/detr}"
PIPELINE_PY="${PIPELINE_PY:-$PROJECT_ROOT/pipeline.py}"
PI3_LOG_PY="${PI3_LOG_PY:-$PROJECT_ROOT/generate_pi3_logs.py}"
LOGER_LOG_PY="${LOGER_LOG_PY:-$PROJECT_ROOT/generate_loger_logs.py}"
DETR_WEIGHTS="${DETR_WEIGHTS:-$DETR_REPO/outputs/sign_finetune/sign_model_final.pth}"

if [[ "$USE_CPU" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export VGGT_DISABLE_IMAGE_RETRIEVAL="${VGGT_DISABLE_IMAGE_RETRIEVAL:-1}"
  echo "[mode] CPU mode enabled (CUDA disabled)."
fi

wait_gpu_idle() {
  if [[ "$USE_CPU" == "1" ]]; then
    return 0
  fi
  if [[ "$WAIT_GPU_IDLE" != "1" ]]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  echo "[gpu] Waiting until GPU utilization becomes 0%..."
  while true; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{sum+=$1} END {print sum+0}')
    if [[ "$util" == "0" ]]; then
      echo "[gpu] idle."
      break
    fi
    sleep "$GPU_IDLE_POLL_SECONDS"
  done
}

mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/vggt_poses.txt"
LOG_DIR="${LOG_PATH%.txt}_logs"
OS_SAVE_PATH="${OS_SAVE_PATH:-$RUN_DIR/open_set_3d.json}"

if [[ -d "$INPUT_PATH" ]]; then
  IMAGE_FOLDER="$INPUT_PATH"
elif [[ -f "$INPUT_PATH" ]]; then
  FRAME_DIR="$RUN_DIR/frames"
  mkdir -p "$FRAME_DIR"
  echo "[0/2] Extracting frames from video: $INPUT_PATH (fps=$VIDEO_FPS, start=${EXTRACT_START_SECONDS:-0}, dur=${EXTRACT_MAX_SECONDS:-all})"
  if [[ -n "$EXTRACT_MAX_SECONDS" && -n "$EXTRACT_START_SECONDS" ]]; then
    ffmpeg -y -ss "$EXTRACT_START_SECONDS" -t "$EXTRACT_MAX_SECONDS" -i "$INPUT_PATH" -vf "fps=${VIDEO_FPS}" "$FRAME_DIR/frame_%06d.jpg" >/dev/null 2>&1
  elif [[ -n "$EXTRACT_START_SECONDS" ]]; then
    ffmpeg -y -ss "$EXTRACT_START_SECONDS" -i "$INPUT_PATH" -vf "fps=${VIDEO_FPS}" "$FRAME_DIR/frame_%06d.jpg" >/dev/null 2>&1
  elif [[ -n "$EXTRACT_MAX_SECONDS" ]]; then
    ffmpeg -y -t "$EXTRACT_MAX_SECONDS" -i "$INPUT_PATH" -vf "fps=${VIDEO_FPS}" "$FRAME_DIR/frame_%06d.jpg" >/dev/null 2>&1
  else
    ffmpeg -y -i "$INPUT_PATH" -vf "fps=${VIDEO_FPS}" "$FRAME_DIR/frame_%06d.jpg" >/dev/null 2>&1
  fi
  IMAGE_FOLDER="$FRAME_DIR"
else
  echo "Input not found: $INPUT_PATH"
  exit 1
fi

if [[ "$BACKEND" == "pi3" ]]; then
  wait_gpu_idle
  echo "[1/2] Running PI3 to produce framewise pointcloud logs..."
  PYTHONPATH="$PI3_REPO:${PYTHONPATH:-}" conda run -n sign-map-ros2 python "$PI3_LOG_PY" \
    --image_folder "$IMAGE_FOLDER" \
    --log_path "$LOG_PATH" \
    --pi3_repo "$PI3_REPO"
elif [[ "$BACKEND" == "loger" ]]; then
  wait_gpu_idle
  echo "[1/2] Running LoGeR to produce framewise pointcloud logs..."
  if [[ ! -d "$LOGER_REPO" ]]; then
    echo "LoGeR repo not found: $LOGER_REPO"
    exit 1
  fi
  conda run -n "$LOGER_ENV" python "$LOGER_LOG_PY" \
    --image_folder "$IMAGE_FOLDER" \
    --log_path "$LOG_PATH" \
    --loger_repo "$LOGER_REPO" \
    --model_name "$LOGER_MODEL_NAME" \
    --config_path "$LOGER_CONFIG_PATH" \
    --window_size "$LOGER_WINDOW_SIZE" \
    --overlap_size "$LOGER_OVERLAP_SIZE" \
    --conf_thres "$LOGER_CONF_THRES"
else
  wait_gpu_idle
  echo "[1/2] Running VGGT-SLAM 2.0 to produce framewise pointcloud logs..."
  if [[ ! -d "$VGGT_REPO" ]]; then
    echo "VGGT-SLAM repo not found: $VGGT_REPO"
    exit 1
  fi
  if [[ ! -d "$VGGT_PY_REPO" ]]; then
    echo "VGGT python repo not found: $VGGT_PY_REPO"
    exit 1
  fi
  VGGT_EXTRA_ARGS=()
  if [[ "$RUN_OS" == "1" ]]; then
    VGGT_EXTRA_ARGS+=(--run_os)
    VGGT_EXTRA_ARGS+=(--sam3_load_from_hf "$SAM3_LOAD_FROM_HF")
    VGGT_EXTRA_ARGS+=(--os_save_path "$OS_SAVE_PATH")
    if [[ -n "$SAM3_CKPT_PATH" ]]; then
      VGGT_EXTRA_ARGS+=(--sam3_checkpoint_path "$SAM3_CKPT_PATH")
    fi
    if [[ -n "$OS_QUERIES" ]]; then
      VGGT_EXTRA_ARGS+=(--os_queries "$OS_QUERIES")
    fi
    if [[ "$OS_SHOW_MASKS" == "1" ]]; then
      VGGT_EXTRA_ARGS+=(--os_show_masks 1)
    fi
  fi
  PYTHONPATH="$VGGT_PY_REPO:${PYTHONPATH:-}" conda run -n vggt-slam python "$VGGT_REPO/main.py" \
    --image_folder "$IMAGE_FOLDER" \
    --max_loops "$MAX_LOOPS" \
    --log_results \
    --log_path "$LOG_PATH" \
    "${VGGT_EXTRA_ARGS[@]}"
fi

echo "[2/2] Running DETR sign detection + OCR + 3D mapping + web view..."
wait_gpu_idle
OCR_EXTRA_ARGS=()
if [[ "$OCR_DISABLE_ENHANCE" == "1" ]]; then
  OCR_EXTRA_ARGS+=(--ocr_disable_enhance)
fi
if [[ "$OCR_NUMERIC_MODE" == "1" ]]; then
  OCR_EXTRA_ARGS+=(--ocr_numeric_mode --ocr_numeric_min_digits "$OCR_NUMERIC_MIN_DIGITS" --ocr_numeric_chars "$OCR_NUMERIC_CHARS")
  OCR_EXTRA_ARGS+=(--ocr_numeric_psm_list "$OCR_NUMERIC_PSM_LIST")
  OCR_EXTRA_ARGS+=(--ocr_numeric_use_paddle "$OCR_NUMERIC_USE_PADDLE")
  OCR_EXTRA_ARGS+=(--ocr_numeric_token_min_len "$OCR_NUMERIC_TOKEN_MIN_LEN")
  OCR_EXTRA_ARGS+=(--ocr_numeric_token_max_len "$OCR_NUMERIC_TOKEN_MAX_LEN")
  OCR_EXTRA_ARGS+=(--ocr_numeric_min_tokens "$OCR_NUMERIC_MIN_TOKENS")
  OCR_EXTRA_ARGS+=(--ocr_numeric_max_tokens "$OCR_NUMERIC_MAX_TOKENS")
  OCR_EXTRA_ARGS+=(--ocr_numeric_prefer_multi_token "$OCR_NUMERIC_PREFER_MULTI_TOKEN")
  OCR_EXTRA_ARGS+=(--ocr_numeric_prefer_three_digit "$OCR_NUMERIC_PREFER_THREE_DIGIT")
fi
if [[ "$OCR_NUMERIC_ONLY" == "1" ]]; then
  OCR_EXTRA_ARGS+=(--ocr_numeric_only)
fi
LABEL_VOCAB_ARGS=()
if [[ "$FINAL_LABEL_VOCAB_ENABLE" == "1" ]]; then
  LABEL_VOCAB_ARGS+=(--final_label_vocab "$FINAL_LABEL_VOCAB")
  LABEL_VOCAB_ARGS+=(--final_label_vocab_enforce "$FINAL_LABEL_VOCAB_ENFORCE")
  LABEL_VOCAB_ARGS+=(--final_label_vocab_min_sim "$FINAL_LABEL_VOCAB_MIN_SIM")
  LABEL_VOCAB_ARGS+=(--final_label_vocab_vote_bonus "$FINAL_LABEL_VOCAB_VOTE_BONUS")
  LABEL_VOCAB_ARGS+=(--final_label_vocab_oov_penalty "$FINAL_LABEL_VOCAB_OOV_PENALTY")
fi
PIPELINE_EXTRA_ARGS=()
if [[ "$RUN_OS" == "1" ]]; then
  PIPELINE_EXTRA_ARGS+=(--open_set_json "$OS_SAVE_PATH")
fi
conda run -n sign-map-ros2 python "$PIPELINE_PY" \
  --image_folder "$IMAGE_FOLDER" \
  --pointcloud_log_dir "$LOG_DIR" \
  --output_dir "$RUN_DIR" \
  --detr_repo "$DETR_REPO" \
  --detr_weights "$DETR_WEIGHTS" \
  --detector_backend "$DETECTOR_BACKEND" \
  --sam3_query_filter "$SAM3_QUERY_FILTER" \
  --sam3_min_sam_score "$SAM3_MIN_SAM_SCORE" \
  --sam3_min_semantic_score "$SAM3_MIN_SEMANTIC_SCORE" \
  --sam3_fallback_to_detr_when_empty "$SAM3_FALLBACK_TO_DETR" \
  --sign_label_id 1 \
  --det_threshold "$DET_THRESHOLD" \
  --min_bbox_width "$MIN_BBOX_WIDTH" \
  --min_bbox_height "$MIN_BBOX_HEIGHT" \
  --min_bbox_area_ratio "$MIN_BBOX_AREA_RATIO" \
  --ocr_backend "$OCR_BACKEND" \
  --trocr_model_name "$OCR_TROCR_MODEL" \
  --ocr_tesseract_lang "$OCR_TESS_LANG" \
  --ocr_paddle_lang "$OCR_PADDLE_LANG" \
  --ocr_text_pattern "$OCR_TEXT_PATTERN" \
  --ocr_superres_scales "$OCR_SUPERRES_SCALES" \
  --ocr_upscale "$OCR_UPSCALE" \
  --ocr_min_crop_side "$OCR_MIN_CROP_SIDE" \
  --ocr_max_candidates "$OCR_MAX_CANDIDATES" \
  --ocr_precheck_before_upscale "$OCR_PRECHECK_BEFORE_UPSCALE" \
  --ocr_precheck_min_score "$OCR_PRECHECK_MIN_SCORE" \
  --ocr_precheck_min_digits "$OCR_PRECHECK_MIN_DIGITS" \
  --ocr_upscale_accept_margin "$OCR_UPSCALE_ACCEPT_MARGIN" \
  --ocr_upscale_max_extra_digits "$OCR_UPSCALE_MAX_EXTRA_DIGITS" \
  --ocr_upscale_extra_digit_penalty "$OCR_UPSCALE_EXTRA_DIGIT_PENALTY" \
  --crop_expand_ratio "$CROP_EXPAND_RATIO" \
  --superres_backend "$SUPERRES_BACKEND" \
  --superres_scale "$SUPERRES_SCALE" \
  --realesrgan_model_variant "$REALESRGAN_MODEL_VARIANT" \
  --realesrgan_model_path "$REALESRGAN_MODEL_PATH" \
  --diffusion_model_id "$DIFFUSION_MODEL_ID" \
  --diffusion_prompt "$DIFFUSION_PROMPT" \
  --diffusion_negative_prompt "$DIFFUSION_NEGATIVE_PROMPT" \
  --diffusion_steps "$DIFFUSION_STEPS" \
  --diffusion_guidance_scale "$DIFFUSION_GUIDANCE_SCALE" \
  --diffusion_max_input_side "$DIFFUSION_MAX_INPUT_SIDE" \
  --diffusion_selective_enabled "$DIFFUSION_SELECTIVE_ENABLED" \
  --diffusion_selective_min_area_ratio "$DIFFUSION_SELECTIVE_MIN_AREA_RATIO" \
  --diffusion_selective_min_frontalness "$DIFFUSION_SELECTIVE_MIN_FRONTALNESS" \
  --diffusion_selective_max_crops "$DIFFUSION_SELECTIVE_MAX_CROPS" \
  --diffusion_uncertain_min_digits "$DIFFUSION_UNCERTAIN_MIN_DIGITS" \
  --diffusion_uncertain_min_text_score "$DIFFUSION_UNCERTAIN_MIN_TEXT_SCORE" \
  --tracker "$TRACKER" \
  --tracker_iou_thres "$TRACKER_IOU_THRES" \
  --tracker_lost_frames "$TRACKER_LOST_FRAMES" \
  --tracker_min_track_hits "$TRACKER_MIN_HITS" \
  --vote_min_count "$VOTE_MIN_COUNT" \
  "${LABEL_VOCAB_ARGS[@]}" \
  --repr_use_frontal "$REPR_USE_FRONTAL" \
  --repr_frontal_weight "$REPR_FRONTAL_WEIGHT" \
  --track_merge_enabled "$TRACK_MERGE_ENABLED" \
  --track_merge_3d_dist "$TRACK_MERGE_3D_DIST" \
  --max_frames "$STAGE2_MAX_FRAMES" \
  --pose_log_path "$LOG_PATH" \
  --video_start_seconds "$VIDEO_START_SECONDS" \
  --video_duration_seconds "$VIDEO_DURATION_SECONDS" \
  --video_fps "$VIDEO_FPS" \
  --ocr_numeric_roi_strict "$OCR_NUMERIC_ROI_STRICT" \
  "${OCR_EXTRA_ARGS[@]}" \
  "${PIPELINE_EXTRA_ARGS[@]}"

echo
echo "Done."
echo "Input frames: $IMAGE_FOLDER"
echo "Viewer: $RUN_DIR/viewer.html"
echo "Detections JSON: $RUN_DIR/sign_3d_detections.json"
echo "Track Labels JSON: $RUN_DIR/sign_track_labels.json"
