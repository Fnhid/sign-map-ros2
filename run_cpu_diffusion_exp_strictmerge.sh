#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export FLAGS_use_cuda=0

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate sign-map-ros2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$SCRIPT_DIR}"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

python "$PROJECT_ROOT/pipeline.py" \
  --image_folder "$PROJECT_ROOT/runs/208_1_1m15_1m35/frames" \
  --pointcloud_log_dir "$PROJECT_ROOT/runs/208_1_1m15_1m35/vggt_poses_logs" \
  --output_dir "$PROJECT_ROOT/runs/exp_cpu_diffusion_full_1m15_1m35_strictmerge" \
  --detr_repo "$WORKSPACE_ROOT/detr" \
  --detr_weights "$WORKSPACE_ROOT/detr/outputs/sign_finetune/sign_model_final.pth" \
  --sign_label_id 1 \
  --det_threshold 0.60 \
  --max_detections_per_frame 1 \
  --ocr_backend paddleocr \
  --ocr_tesseract_lang kor+eng \
  --ocr_paddle_lang korean \
  --ocr_numeric_mode \
  --ocr_numeric_only \
  --ocr_numeric_min_digits 2 \
  --ocr_numeric_chars 'B0123456789 ' \
  --ocr_text_pattern '[^B0-9\\s]+' \
  --ocr_numeric_use_paddle 1 \
  --ocr_precheck_before_upscale 1 \
  --ocr_precheck_min_score 92 \
  --ocr_precheck_min_digits 3 \
  --ocr_upscale_accept_margin 10 \
  --ocr_upscale_max_extra_digits 0 \
  --ocr_upscale_extra_digit_penalty 18 \
  --ocr_roi_hint_weight 18 \
  --ocr_superres_scales 4 \
  --ocr_upscale 2.0 \
  --ocr_min_crop_side 120 \
  --ocr_max_candidates 3 \
  --crop_expand_ratio 0.12 \
  --superres_backend realesrgan \
  --superres_scale 4.0 \
  --realesrgan_model_variant general-x4v3 \
  --diffusion_selective_enabled 1 \
  --diffusion_model_id stabilityai/stable-diffusion-x4-upscaler \
  --diffusion_steps 20 \
  --diffusion_guidance_scale 7.0 \
  --diffusion_max_input_side 256 \
  --diffusion_selective_min_area_ratio 0.001 \
  --diffusion_selective_min_frontalness 0.68 \
  --diffusion_selective_max_crops 1 \
  --diffusion_uncertain_min_digits 4 \
  --tracker simple \
  --tracker_iou_thres 0.35 \
  --tracker_lost_frames 8 \
  --tracker_min_track_hits 2 \
  --vote_min_count 2 \
  --repr_use_frontal 1 \
  --repr_frontal_weight 18 \
  --final_label_unify_enabled 0 \
  --track_merge_enabled 1 \
  --track_merge_3d_dist 0.06 \
  --track_merge_xy_dist 0.022 \
  --track_merge_z_dist 0.012 \
  --track_merge_min_score 0.64 \
  --track_merge_cooccur_block 0 \
  --track_merge_ocr_conflict_block 0 \
  --final_label_repr_bonus 0.95
