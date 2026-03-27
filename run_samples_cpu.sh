#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
SAMPLES_DIR="${SAMPLES_DIR:-$PROJECT_ROOT/samples}"
RUNS_ROOT="${RUNS_ROOT:-$PROJECT_ROOT/runs}"

mkdir -p "$RUNS_ROOT"

run_one() {
  local video_path="$1"
  local stem
  stem="$(basename "${video_path%.*}")"
  local run_dir="$RUNS_ROOT/sample_cpu_${stem}_$(date +%Y%m%d_%H%M%S)"

  echo "[sample] running: $video_path"
  USE_CPU=1 \
  BACKEND="${BACKEND:-vggt}" \
  VIDEO_FPS="${VIDEO_FPS:-2}" \
  DETECTOR_BACKEND="${DETECTOR_BACKEND:-hybrid}" \
  OCR_NUMERIC_MODE=1 \
  OCR_NUMERIC_ONLY=1 \
  OCR_NUMERIC_CHARS='B0123456789 ' \
  OCR_TEXT_PATTERN='[^B0-9\s]+' \
  OCR_NUMERIC_MIN_DIGITS="${OCR_NUMERIC_MIN_DIGITS:-2}" \
  OCR_PRECHECK_BEFORE_UPSCALE=1 \
  OCR_PRECHECK_MIN_SCORE="${OCR_PRECHECK_MIN_SCORE:-92}" \
  OCR_PRECHECK_MIN_DIGITS="${OCR_PRECHECK_MIN_DIGITS:-2}" \
  OCR_UPSCALE_ACCEPT_MARGIN="${OCR_UPSCALE_ACCEPT_MARGIN:-10}" \
  OCR_UPSCALE_MAX_EXTRA_DIGITS="${OCR_UPSCALE_MAX_EXTRA_DIGITS:-0}" \
  OCR_UPSCALE_EXTRA_DIGIT_PENALTY="${OCR_UPSCALE_EXTRA_DIGIT_PENALTY:-28}" \
  SUPERRES_BACKEND="${SUPERRES_BACKEND:-realesrgan}" \
  SUPERRES_SCALE="${SUPERRES_SCALE:-4.0}" \
  REALESRGAN_MODEL_VARIANT="${REALESRGAN_MODEL_VARIANT:-general-x4v3}" \
  "$PROJECT_ROOT/run_full_pipeline.sh" "$video_path" "$run_dir" "${VIDEO_FPS:-2}"

  echo "[sample] viewer: $run_dir/viewer.html"
}

for video in "$SAMPLES_DIR"/*.mp4; do
  [ -f "$video" ] || continue
  run_one "$video"
done
