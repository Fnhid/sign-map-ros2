#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from html import escape
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import numpy as np
import plotly.graph_objects as go
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DETR_REPO = Path(os.environ.get("DETR_REPO", str((WORKSPACE_ROOT / "detr").resolve())))
DEFAULT_DETR_WEIGHTS = Path(
    os.environ.get(
        "DETR_WEIGHTS",
        str((DEFAULT_DETR_REPO / "outputs" / "sign_finetune" / "sign_model_final.pth").resolve()),
    )
)
DEFAULT_REALESRGAN_CACHE_DIR = Path(
    os.environ.get("REALESRGAN_CACHE_DIR", str((PROJECT_ROOT / "models" / "realesrgan").resolve()))
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match DETR sign detections + OCR text to VGGT-SLAM pointcloud logs and build web viewer."
    )
    parser.add_argument("--image_folder", type=Path, required=True, help="Folder of image sequence.")
    parser.add_argument("--pointcloud_log_dir", type=Path, required=True, help="Folder with framewise *.npz logs from VGGT-SLAM.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output folder for html/json.")
    parser.add_argument("--detr_repo", type=Path, default=DEFAULT_DETR_REPO)
    parser.add_argument(
        "--detr_weights",
        type=Path,
        default=DEFAULT_DETR_WEIGHTS,
    )
    parser.add_argument(
        "--detector_backend",
        type=str,
        default="hybrid",
        choices=["detr", "sam3", "hybrid"],
        help="Detection source: DETR only, SAM3(open-set box_xyxy) only, or hybrid merge.",
    )
    parser.add_argument("--sign_label_id", type=int, default=1, help="Class id used for sign in trained DETR.")
    parser.add_argument("--det_threshold", type=float, default=0.75)
    parser.add_argument(
        "--det_merge_iou_thres",
        type=float,
        default=0.65,
        help="IoU threshold for NMS when merging detections from multiple backends.",
    )
    parser.add_argument(
        "--sam3_query_filter",
        type=str,
        default="sign",
        help="Comma-separated query names from open_set_json to use as sign detections.",
    )
    parser.add_argument("--sam3_min_sam_score", type=float, default=0.55)
    parser.add_argument("--sam3_min_semantic_score", type=float, default=-1.0)
    parser.add_argument("--sam3_max_detections_per_frame", type=int, default=5)
    parser.add_argument(
        "--sam3_frame_id_tolerance",
        type=float,
        default=0.25,
        help="Allowed frame_id tolerance when matching open-set frame ids to image frame ids.",
    )
    parser.add_argument(
        "--sam3_fallback_to_detr_when_empty",
        type=int,
        default=1,
        help="When detector_backend=sam3, fallback to DETR if no SAM3 box exists for a frame (1/0).",
    )
    parser.add_argument("--min_bbox_width", type=int, default=18)
    parser.add_argument("--min_bbox_height", type=int, default=18)
    parser.add_argument(
        "--min_bbox_area_ratio",
        type=float,
        default=0.00035,
        help="Minimum bbox area ratio (bbox area / frame area) to keep detection.",
    )
    parser.add_argument("--max_detections_per_frame", type=int, default=3)
    parser.add_argument("--max_frames", type=int, default=0, help="0 means all frames.")
    parser.add_argument(
        "--ocr_backend",
        type=str,
        default="auto",
        choices=["auto", "trocr", "paddleocr", "easyocr", "tesseract"],
    )
    parser.add_argument("--trocr_model_name", type=str, default="microsoft/trocr-small-printed")
    parser.add_argument("--trocr_max_new_tokens", type=int, default=24)
    parser.add_argument("--ocr_psm", type=int, default=6, help="Tesseract page segmentation mode.")
    parser.add_argument(
        "--ocr_tesseract_lang",
        type=str,
        default="kor+eng",
        help="Preferred tesseract language(s), e.g., kor+eng",
    )
    parser.add_argument(
        "--ocr_easyocr_langs",
        type=str,
        default="ko,en",
        help="Comma-separated easyocr languages, e.g., ko,en",
    )
    parser.add_argument("--ocr_easyocr_min_conf", type=float, default=0.20)
    parser.add_argument("--ocr_paddle_lang", type=str, default="korean")
    parser.add_argument("--ocr_paddle_min_conf", type=float, default=0.30)
    parser.add_argument("--ocr_paddle_numeric_min_conf", type=float, default=0.10)
    parser.add_argument("--ocr_trocr_min_conf", type=float, default=0.0)
    parser.add_argument("--ocr_upscale", type=float, default=2.0, help="Upscale factor for OCR crops.")
    parser.add_argument(
        "--ocr_superres_scales",
        type=str,
        default="2,4",
        help="Comma-separated OCR super-res scales (e.g., 2,4 or 2,4,6).",
    )
    parser.add_argument(
        "--ocr_min_crop_side",
        type=int,
        default=80,
        help="Ensure OCR crop short side is at least this many pixels.",
    )
    parser.add_argument("--ocr_max_candidates", type=int, default=10)
    parser.add_argument(
        "--crop_expand_ratio",
        type=float,
        default=0.08,
        help="Expand DETR bbox by this ratio before OCR crop.",
    )
    parser.add_argument("--ocr_disable_enhance", action="store_true", help="Disable OCR crop enhancement.")
    parser.add_argument(
        "--ocr_numeric_mode",
        action="store_true",
        help="Prioritize numeric extraction from OCR outputs and return numeric text only.",
    )
    parser.add_argument(
        "--ocr_numeric_only",
        action="store_true",
        help="Run numeric-only OCR path (digit whitelist) and ignore non-numeric texts.",
    )
    parser.add_argument(
        "--ocr_numeric_min_digits",
        type=int,
        default=2,
        help="Minimum digit count for accepting numeric OCR text in numeric mode.",
    )
    parser.add_argument(
        "--ocr_numeric_chars",
        type=str,
        default="B0123456789 ",
        help="Allowed characters for sign OCR output in numeric mode (e.g., B + digits).",
    )
    parser.add_argument(
        "--ocr_numeric_psm_list",
        type=str,
        default="7,6,8,13",
        help="Comma-separated tesseract PSM list for numeric OCR path.",
    )
    parser.add_argument(
        "--ocr_numeric_use_paddle",
        type=int,
        default=1,
        help="Use PaddleOCR as additional numeric candidate generator (1/0).",
    )
    parser.add_argument("--ocr_numeric_token_min_len", type=int, default=2)
    parser.add_argument(
        "--ocr_numeric_token_max_len",
        type=int,
        default=0,
        help="Maximum token length for numeric OCR tokens (0 means unlimited).",
    )
    parser.add_argument("--ocr_numeric_min_tokens", type=int, default=1)
    parser.add_argument("--ocr_numeric_max_tokens", type=int, default=0)
    parser.add_argument("--ocr_numeric_prefer_multi_token", type=int, default=1)
    parser.add_argument(
        "--ocr_numeric_prefer_three_digit",
        type=int,
        default=0,
        help="Prefer 3-digit tokens in numeric ranking (1/0).",
    )
    parser.add_argument(
        "--ocr_numeric_specialist",
        type=str,
        default="easyocr_digits",
        choices=["off", "easyocr_digits"],
        help="Additional numeric-specialized OCR engine.",
    )
    parser.add_argument(
        "--ocr_numeric_specialist_replace",
        type=int,
        default=0,
        help="Use numeric specialist only (1) or ensemble with existing numeric OCR (0).",
    )
    parser.add_argument(
        "--ocr_numeric_specialist_min_conf",
        type=float,
        default=0.20,
        help="Minimum confidence for numeric specialist OCR candidates.",
    )
    parser.add_argument(
        "--ocr_numeric_roi_enabled",
        type=int,
        default=1,
        help="Detect numeric text regions inside sign crop and OCR that ROI first (1/0).",
    )
    parser.add_argument(
        "--ocr_numeric_roi_min_conf",
        type=float,
        default=0.05,
        help="Minimum PaddleOCR confidence for numeric ROI candidates.",
    )
    parser.add_argument(
        "--ocr_numeric_roi_expand_ratio",
        type=float,
        default=0.10,
        help="Expand numeric ROI bbox by this ratio within sign crop.",
    )
    parser.add_argument(
        "--ocr_numeric_roi_min_area_ratio",
        type=float,
        default=0.01,
        help="Minimum numeric ROI area ratio within sign crop.",
    )
    parser.add_argument(
        "--ocr_numeric_roi_max_count",
        type=int,
        default=4,
        help="Maximum number of numeric ROI candidates to evaluate per sign.",
    )
    parser.add_argument(
        "--ocr_numeric_roi_allow_nondigit",
        type=int,
        default=0,
        help="Allow non-digit Paddle text boxes as fallback numeric ROI candidates (1/0).",
    )
    parser.add_argument(
        "--ocr_numeric_roi_strict",
        type=int,
        default=0,
        help="If 1, no numeric ROI means empty OCR. If 0, fallback to full-crop OCR.",
    )
    parser.add_argument(
        "--ocr_precheck_before_upscale",
        type=int,
        default=1,
        help="Run OCR on raw ROI before super-resolution and skip SR when raw OCR is already confident (1/0).",
    )
    parser.add_argument(
        "--ocr_precheck_min_score",
        type=float,
        default=90.0,
        help="Minimum OCR score to accept raw pre-upscale OCR.",
    )
    parser.add_argument(
        "--ocr_precheck_min_digits",
        type=int,
        default=3,
        help="Minimum digit count to accept raw pre-upscale OCR in numeric mode.",
    )
    parser.add_argument(
        "--ocr_upscale_accept_margin",
        type=float,
        default=10.0,
        help="Required score margin for SR/diffusion OCR result to replace raw OCR.",
    )
    parser.add_argument(
        "--ocr_upscale_max_extra_digits",
        type=int,
        default=0,
        help="Maximum allowed extra digit count introduced by SR OCR compared to raw OCR.",
    )
    parser.add_argument(
        "--ocr_upscale_extra_digit_penalty",
        type=float,
        default=28.0,
        help="Penalty per extra digit when SR OCR inflates digit count.",
    )
    parser.add_argument(
        "--ocr_roi_hint_weight",
        type=float,
        default=14.0,
        help="Weight for similarity between OCR text and Paddle ROI text hint.",
    )
    parser.add_argument(
        "--ocr_text_pattern",
        type=str,
        default=r"[^0-9A-Za-z가-힣\s]+",
        help="Regex pattern to remove from OCR output.",
    )
    parser.add_argument(
        "--superres_backend",
        type=str,
        default="auto",
        choices=["auto", "realesrgan", "bicubic", "diffusion", "none"],
    )
    parser.add_argument("--superres_scale", type=float, default=2.5)
    parser.add_argument(
        "--realesrgan_model_variant",
        type=str,
        default="general-x4v3",
        choices=["general-x4v3", "x4plus"],
    )
    parser.add_argument(
        "--realesrgan_model_path",
        type=str,
        default="",
    )
    parser.add_argument("--realesrgan_tile", type=int, default=0)
    parser.add_argument("--realesrgan_half", action="store_true")
    parser.add_argument(
        "--diffusion_model_id",
        type=str,
        default="stabilityai/stable-diffusion-x4-upscaler",
    )
    parser.add_argument("--diffusion_prompt", type=str, default="clean sharp readable sign text")
    parser.add_argument("--diffusion_negative_prompt", type=str, default="blur, noise, artifact")
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_guidance_scale", type=float, default=7.0)
    parser.add_argument("--diffusion_max_input_side", type=int, default=256)
    parser.add_argument("--diffusion_selective_enabled", type=int, default=0)
    parser.add_argument("--diffusion_selective_min_area_ratio", type=float, default=0.0012)
    parser.add_argument("--diffusion_selective_min_frontalness", type=float, default=0.55)
    parser.add_argument("--diffusion_selective_max_crops", type=int, default=2)
    parser.add_argument("--diffusion_uncertain_min_digits", type=int, default=4)
    parser.add_argument("--diffusion_uncertain_min_text_score", type=int, default=20)
    parser.add_argument("--tracker", type=str, default="bytetrack", choices=["bytetrack", "simple", "off"])
    parser.add_argument("--tracker_iou_thres", type=float, default=0.35)
    parser.add_argument("--tracker_lost_frames", type=int, default=10)
    parser.add_argument("--tracker_min_track_hits", type=int, default=2)
    parser.add_argument("--tracker_frame_rate", type=float, default=8.0)
    parser.add_argument("--vote_min_count", type=int, default=2)
    parser.add_argument("--vote_min_text_len", type=int, default=1)
    parser.add_argument(
        "--final_label_vocab",
        type=str,
        default="",
        help="Optional comma-separated allowed final labels (e.g., 425,426).",
    )
    parser.add_argument(
        "--final_label_vocab_enforce",
        type=int,
        default=0,
        help="If 1, only labels matched to final_label_vocab are allowed.",
    )
    parser.add_argument(
        "--final_label_vocab_min_sim",
        type=float,
        default=0.72,
        help="Minimum similarity to snap OCR text to final_label_vocab.",
    )
    parser.add_argument(
        "--final_label_vocab_vote_bonus",
        type=float,
        default=8.0,
        help="Additive vote bonus when OCR text matches allowed vocab.",
    )
    parser.add_argument(
        "--final_label_vocab_oov_penalty",
        type=float,
        default=0.25,
        help="Weight multiplier for OCR votes that do not match allowed vocab.",
    )
    parser.add_argument(
        "--final_label_drop_weak",
        type=int,
        default=1,
        help="If 1, drop merged final labels whose vote_count is below vote_min_count.",
    )
    parser.add_argument("--repr_use_frontal", type=int, default=1)
    parser.add_argument("--repr_frontal_weight", type=float, default=18.0)
    parser.add_argument("--track_merge_enabled", type=int, default=1)
    parser.add_argument(
        "--track_merge_3d_dist",
        type=float,
        default=1.25,
        help="Merge final sign tracks whose 3D centroids are within this distance.",
    )
    parser.add_argument(
        "--track_merge_xy_dist",
        type=float,
        default=0.03,
        help="Anisotropic merge gate on XY distance (meters).",
    )
    parser.add_argument(
        "--track_merge_z_dist",
        type=float,
        default=0.015,
        help="Anisotropic merge gate on Z distance (meters).",
    )
    parser.add_argument(
        "--track_merge_min_score",
        type=float,
        default=0.62,
        help="Minimum pairwise merge score to link two tracks.",
    )
    parser.add_argument("--track_merge_w_3d", type=float, default=0.55)
    parser.add_argument("--track_merge_w_ocr", type=float, default=0.30)
    parser.add_argument("--track_merge_w_view", type=float, default=0.15)
    parser.add_argument(
        "--track_merge_cooccur_block",
        type=int,
        default=0,
        help="Block merge when two tracks appear in the same frame (1/0).",
    )
    parser.add_argument(
        "--track_merge_ocr_conflict_block",
        type=int,
        default=0,
        help="Block merge when confident OCR labels conflict (1/0).",
    )
    parser.add_argument(
        "--track_merge_ocr_conflict_min_vote_weight",
        type=float,
        default=0.50,
        help="Minimum vote_weight for OCR conflict blocking.",
    )
    parser.add_argument(
        "--track_merge_ocr_conflict_min_support",
        type=int,
        default=1,
        help="Minimum vote_count for OCR conflict blocking.",
    )
    parser.add_argument(
        "--final_label_repr_bonus",
        type=float,
        default=0.85,
        help="Bonus score when final label candidate matches representative-frame OCR text.",
    )
    parser.add_argument(
        "--final_label_unify_enabled",
        type=int,
        default=1,
        help="Unify final label text among nearby merged signs (1/0).",
    )
    parser.add_argument(
        "--final_label_unify_dist",
        type=float,
        default=1.60,
        help="3D distance threshold for final label unification among nearby signs.",
    )
    parser.add_argument(
        "--final_label_unify_conflict_block",
        type=int,
        default=1,
        help="Block final label unification when nearby signs already have conflicting non-empty labels (1/0).",
    )
    parser.add_argument(
        "--open_set_json",
        type=Path,
        default=Path(""),
        help="Optional open-set 3D detection JSON (from VGGT-SLAM --run_os) to overlay in viewer.",
    )
    parser.add_argument(
        "--viewer_show_open_set",
        type=int,
        default=1,
        help="Show open-set 3D OBB overlays in viewer when open_set_json is provided (1/0).",
    )
    parser.add_argument(
        "--viewer_open_set_max_objects",
        type=int,
        default=200,
        help="Maximum number of open-set objects to render in viewer (0 means all).",
    )
    parser.add_argument(
        "--viewer_open_set_line_width",
        type=float,
        default=6.0,
        help="Line width for open-set OBB wireframes in viewer.",
    )
    parser.add_argument("--viewer_max_points", type=int, default=220000)
    parser.add_argument("--viewer_marker_size", type=float, default=1.3)
    parser.add_argument(
        "--viewer_hide_empty_ocr",
        type=int,
        default=1,
        help="Hide detections with empty OCR text in 3D viewer (1/0).",
    )
    parser.add_argument(
        "--pose_log_path",
        type=Path,
        default=Path(""),
        help="Optional pose log text file (e.g., vggt_poses.txt) to display trajectory/pose info.",
    )
    parser.add_argument("--video_start_seconds", type=float, default=0.0, help="Video clip start time in seconds.")
    parser.add_argument(
        "--video_duration_seconds",
        type=float,
        default=-1.0,
        help="Video clip duration in seconds. <=0 means unknown.",
    )
    parser.add_argument("--video_fps", type=float, default=0.0, help="Frame extraction FPS used for time conversion.")
    parser.add_argument(
        "--viewer_side_max_frames",
        type=int,
        default=180,
        help="Maximum number of side-panel keyframes to render.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_frame_id(name: str):
    m = re.search(r"\d+(?:\.\d+)?", name)
    if not m:
        return None
    return float(m.group())


def load_pose_records(pose_log_path: Path):
    if pose_log_path is None:
        return {}, []
    path_str = str(pose_log_path).strip()
    if not path_str:
        return {}, []
    p = Path(path_str)
    if not p.exists():
        return {}, []

    pose_by_frame = {}
    pose_rows = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        try:
            fid = float(parts[0])
            tx, ty, tz, qx, qy, qz, qw = [float(v) for v in parts[1:8]]
        except Exception:
            continue
        rec = {
            "frame_id": fid,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "qx": qx,
            "qy": qy,
            "qz": qz,
            "qw": qw,
        }
        pose_rows.append(rec)
        if fid not in pose_by_frame:
            pose_by_frame[fid] = rec
    return pose_by_frame, pose_rows


def format_seconds_hms(seconds: float):
    try:
        v = float(seconds)
    except Exception:
        return "-"
    if not np.isfinite(v):
        return "-"
    sign = "-" if v < 0 else ""
    v = abs(v)
    h = int(v // 3600)
    m = int((v % 3600) // 60)
    s = v - h * 3600 - m * 60
    if h > 0:
        return f"{sign}{h:02d}:{m:02d}:{s:05.2f}"
    return f"{sign}{m:02d}:{s:05.2f}"


def load_detr_model(detr_repo: Path, detr_weights: Path, device: str):
    model = torch.hub.load(str(detr_repo), "detr_resnet50", pretrained=False, source="local")
    ckpt = torch.load(detr_weights, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def run_detr_inference(model, transform, image_pil: Image.Image, sign_label_id: int, det_threshold: float):
    w, h = image_pil.size
    device = next(model.parameters()).device
    tensor = transform(image_pil).unsqueeze(0).to(device)
    out = model(tensor)
    probs = out["pred_logits"].softmax(-1)[0, :, :-1]
    scores, labels = probs.max(-1)
    keep = (labels == sign_label_id) & (scores >= det_threshold)
    if keep.sum().item() == 0:
        return []

    boxes = out["pred_boxes"][0, keep]
    kept_scores = scores[keep].detach().cpu().numpy()
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    scale = torch.tensor([w, h, w, h], device=boxes_xyxy.device)
    boxes_xyxy = (boxes_xyxy * scale).detach().cpu().numpy()
    results = []
    for score, box in zip(kept_scores, boxes_xyxy):
        x1, y1, x2, y2 = box.tolist()
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(1, min(int(round(x2)), w))
        y2 = max(1, min(int(round(y2)), h))
        if x2 <= x1 or y2 <= y1:
            continue
        results.append({"bbox_xyxy": [x1, y1, x2, y2], "score": float(score), "source": "detr"})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def filter_detections_by_bbox(dets, img_w, img_h, args):
    kept = []
    frame_area = float(max(1, img_w * img_h))
    for d in dets:
        x1, y1, x2, y2 = d["bbox_xyxy"]
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        area_ratio = (bw * bh) / frame_area
        if bw < args.min_bbox_width or bh < args.min_bbox_height:
            continue
        if area_ratio < args.min_bbox_area_ratio:
            continue
        d["bbox_w"] = int(bw)
        d["bbox_h"] = int(bh)
        d["bbox_area_ratio"] = float(area_ratio)
        kept.append(d)
    return kept


def expand_bbox_xyxy(x1, y1, x2, y2, img_w, img_h, ratio):
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    pad_x = bw * float(ratio)
    pad_y = bh * float(ratio)
    nx1 = int(max(0, math.floor(x1 - pad_x)))
    ny1 = int(max(0, math.floor(y1 - pad_y)))
    nx2 = int(min(img_w, math.ceil(x2 + pad_x)))
    ny2 = int(min(img_h, math.ceil(y2 + pad_y)))
    if nx2 <= nx1:
        nx2 = min(img_w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(img_h, ny1 + 1)
    return nx1, ny1, nx2, ny2


def init_superres(args, device: str, backend_override=None):
    backend = str(backend_override) if backend_override else args.superres_backend
    if backend == "none":
        return {"backend": "none", "upsampler": None}

    def resolve_model_path():
        if args.realesrgan_model_path:
            src = str(args.realesrgan_model_path)
        elif args.realesrgan_model_variant == "general-x4v3":
            src = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
        else:
            src = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

        if src.startswith("http://") or src.startswith("https://"):
            cache_dir = DEFAULT_REALESRGAN_CACHE_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            filename = Path(urlparse(src).path).name or "realesrgan_model.pth"
            local_path = cache_dir / filename
            if not local_path.exists():
                urlretrieve(src, str(local_path))
            return str(local_path)
        return src

    if backend == "diffusion":
        from diffusers import StableDiffusionUpscalePipeline  # type: ignore

        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            args.diffusion_model_id,
            torch_dtype=dtype,
        )
        if device == "cuda":
            pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        return {"backend": "diffusion", "pipe": pipe}

    if backend in ("auto", "realesrgan"):
        try:
            if "torchvision.transforms.functional_tensor" not in sys.modules:
                import torchvision.transforms.functional as _tvf  # type: ignore

                shim = types.ModuleType("torchvision.transforms.functional_tensor")
                shim.rgb_to_grayscale = _tvf.rgb_to_grayscale  # type: ignore[attr-defined]
                sys.modules["torchvision.transforms.functional_tensor"] = shim
            from realesrgan import RealESRGANer  # type: ignore

            if args.realesrgan_model_variant == "general-x4v3":
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact  # type: ignore

                model = SRVGGNetCompact(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_conv=32,
                    upscale=4,
                    act_type="prelu",
                )
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore

                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
            upsampler = RealESRGANer(
                scale=4,
                model_path=resolve_model_path(),
                model=model,
                tile=max(0, args.realesrgan_tile),
                tile_pad=10,
                pre_pad=0,
                half=bool(args.realesrgan_half and device == "cuda"),
                gpu_id=0 if device == "cuda" else None,
            )
            return {"backend": "realesrgan", "upsampler": upsampler}
        except Exception:
            if backend == "realesrgan":
                raise
    return {"backend": "bicubic", "upsampler": None}


def upscale_crop_for_ocr(image_crop: Image.Image, superres_cfg, args, outscale=None):
    if superres_cfg is None:
        return image_crop
    backend = superres_cfg.get("backend", "bicubic")
    if backend == "none":
        return image_crop
    target_scale = float(outscale if outscale is not None else args.superres_scale)
    target_scale = max(1.0, target_scale)
    if backend == "realesrgan" and superres_cfg.get("upsampler") is not None:
        try:
            arr = np.array(image_crop.convert("RGB"))
            out, _ = superres_cfg["upsampler"].enhance(arr, outscale=target_scale)
            return Image.fromarray(out.astype(np.uint8)).convert("RGB")
        except Exception:
            pass
    if backend == "diffusion" and superres_cfg.get("pipe") is not None:
        try:
            pipe = superres_cfg["pipe"]
            low = image_crop.convert("RGB")
            max_side = int(max(64, args.diffusion_max_input_side))
            lw, lh = low.size
            if max(lw, lh) > max_side:
                s = float(max_side) / float(max(1, max(lw, lh)))
                low = low.resize(
                    (max(1, int(round(lw * s))), max(1, int(round(lh * s)))),
                    Image.Resampling.LANCZOS,
                )
            out = pipe(
                prompt=str(args.diffusion_prompt),
                image=low,
                num_inference_steps=int(args.diffusion_steps),
                guidance_scale=float(args.diffusion_guidance_scale),
                negative_prompt=str(args.diffusion_negative_prompt) if args.diffusion_negative_prompt else None,
            ).images[0].convert("RGB")
            bw, bh = image_crop.size
            tw = max(1, int(round(bw * target_scale)))
            th = max(1, int(round(bh * target_scale)))
            if out.size != (tw, th):
                out = out.resize((tw, th), Image.Resampling.LANCZOS)
            return out
        except Exception:
            pass
    if backend in ("auto", "bicubic", "realesrgan", "diffusion"):
        w, h = image_crop.size
        return image_crop.resize(
            (max(1, int(round(w * target_scale))), max(1, int(round(h * target_scale)))),
            Image.Resampling.LANCZOS,
        )
    return image_crop


def build_superres_inputs(image_crop: Image.Image, superres_cfg, args):
    imgs = [image_crop.convert("RGB")]
    scales = []
    for tok in str(args.ocr_superres_scales).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            s = float(tok)
        except Exception:
            continue
        if s > 1.0:
            scales.append(s)
    if not scales:
        scales = [max(1.0, float(args.superres_scale))]

    for s in scales:
        imgs.append(upscale_crop_for_ocr(image_crop, superres_cfg, args, outscale=s))

    # Deduplicate by size to avoid repeated expensive OCR calls.
    uniq = []
    seen = set()
    for im in imgs:
        key = (int(im.size[0]), int(im.size[1]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(im)
    return uniq


def get_max_superres_scale(args):
    vals = []
    for tok in str(args.ocr_superres_scales).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except Exception:
            continue
        if v > 1.0:
            vals.append(v)
    if vals:
        return max(vals)
    return max(1.0, float(args.superres_scale))


def run_tesseract_ocr(image_crop: Image.Image, lang: str = "eng", psm: int = 6, whitelist: str = ""):
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td) / "crop.png"
        image_crop.save(tmp_path)
        cmd = [
            "tesseract",
            str(tmp_path),
            "stdout",
            "--psm",
            str(psm),
            "-l",
            lang,
        ]
        if whitelist:
            cmd.extend(["-c", f"tessedit_char_whitelist={whitelist}"])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return ""
        text = proc.stdout.strip()
        text = re.sub(r"\s+", " ", text).strip()
        return text


def parse_psm_list(psm_list: str):
    out = []
    for tok in str(psm_list).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except Exception:
            continue
        if v < 0 or v > 13:
            continue
        out.append(v)
    return out or [7, 6]


def list_tesseract_languages():
    proc = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True)
    if proc.returncode != 0:
        return set()
    langs = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("List of available languages"):
            continue
        langs.append(line)
    return set(langs)


def resolve_tesseract_lang(requested: str, available_langs: set):
    if not available_langs:
        return "eng"
    req = [x.strip() for x in requested.split("+") if x.strip()]
    kept = [x for x in req if x in available_langs]
    if kept:
        return "+".join(kept)
    if "eng" in available_langs:
        return "eng"
    return sorted(available_langs)[0]


def normalize_ocr_text(text: str, text_pattern: str):
    if not text:
        return ""
    text = re.sub(text_pattern, " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_ocr_variants(image_crop: Image.Image, args):
    variants = [image_crop.convert("RGB")]
    if args.ocr_disable_enhance:
        return variants

    w, h = image_crop.size
    short_side = max(1, min(w, h))
    scale = max(float(args.ocr_upscale), float(args.ocr_min_crop_side) / float(short_side))
    if scale <= 1.01:
        return variants

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    up = image_crop.convert("L").resize((new_w, new_h), Image.Resampling.LANCZOS)
    up = ImageOps.autocontrast(up)
    up = up.filter(ImageFilter.MedianFilter(size=3))
    sharp = up.filter(ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=2)).convert("RGB")
    bw = up.point(lambda x: 255 if x >= 150 else 0).convert("RGB")
    variants.append(sharp)
    variants.append(bw)
    return variants


def ocr_text_score(text: str):
    if not text:
        return -1
    useful = len(re.findall(r"[0-9A-Za-z가-힣]", text))
    return useful * 10 + min(len(text), 60)


def extract_numeric_candidates(text: str, args):
    if not text:
        return []
    allowed_set = {ch.upper() for ch in str(args.ocr_numeric_chars)}
    allowed_set.add(" ")
    up_text = str(text).upper()
    matches = re.findall(r"[B0-9][B0-9\s\-\./]*", up_text)
    candidates = []
    for m in matches:
        cleaned = "".join(ch if (ch in allowed_set and (ch.isdigit() or ch == " " or ch == "B")) else " " for ch in m)
        tokens = re.findall(r"B?\d+", cleaned)
        if args.ocr_numeric_token_min_len > 0:
            tokens = [t for t in tokens if len(re.sub(r"\D", "", t)) >= int(args.ocr_numeric_token_min_len)]
        if args.ocr_numeric_token_max_len > 0:
            tokens = [t for t in tokens if len(re.sub(r"\D", "", t)) <= int(args.ocr_numeric_token_max_len)]
        if int(args.ocr_numeric_max_tokens) > 0 and len(tokens) > int(args.ocr_numeric_max_tokens):
            tokens = tokens[: int(args.ocr_numeric_max_tokens)]
        if len(tokens) < int(args.ocr_numeric_min_tokens):
            continue
        cleaned = " ".join(tokens).strip()
        digit_count = sum(ch.isdigit() for ch in cleaned)
        if digit_count < int(args.ocr_numeric_min_digits):
            continue
        candidates.append(cleaned)
    # preserve order, remove duplicates
    uniq = []
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def numeric_text_score(text: str, args):
    if not text:
        return -1
    tokens = re.findall(r"\d+", text)
    if not tokens:
        return -1
    digit_count = sum(len(t) for t in tokens)
    score = digit_count * 20 + len(tokens) * 6
    if int(args.ocr_numeric_prefer_multi_token) == 1 and len(tokens) >= 2:
        score += 18
    if int(args.ocr_numeric_prefer_three_digit) == 1:
        score += sum(6 for t in tokens if len(t) == 3)
    return score


def extract_text_items_from_paddle_result(result, min_conf: float):
    if not result:
        return []
    lines = result[0] if isinstance(result, list) else result
    if lines is None:
        return []
    texts = []
    for item in lines:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        rec = item[1]
        if not isinstance(rec, (list, tuple)) or len(rec) == 0:
            continue
        txt = str(rec[0]).strip()
        conf = float(rec[1]) if len(rec) > 1 else 1.0
        if not txt or conf < min_conf:
            continue
        texts.append(txt)
    return texts


def extract_text_from_paddle_result(result, min_conf: float):
    texts = extract_text_items_from_paddle_result(result, min_conf)
    return " ".join(texts).strip()


def init_trocr_reader(args, device: str):
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore

    processor = TrOCRProcessor.from_pretrained(args.trocr_model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model_name)
    model.to(device)
    model.eval()
    return {"processor": processor, "model": model, "device": device}


def create_ocr_reader(args, device: str):
    backend = args.ocr_backend
    tesseract_available = list_tesseract_languages()
    tesseract_lang = resolve_tesseract_lang(args.ocr_tesseract_lang, tesseract_available)

    engines = []

    if backend in ("auto", "trocr"):
        try:
            reader = init_trocr_reader(args, device=device)
            engines.append({"name": "trocr", "reader": reader})
        except Exception:
            if backend == "trocr":
                raise RuntimeError(
                    "trocr backend requested but TrOCR model load failed. "
                    "Check transformers/torch install and network/model cache."
                )

    if backend in ("auto", "paddleocr"):
        try:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR  # type: ignore

            use_gpu = device == "cuda"
            try:
                reader = PaddleOCR(
                    use_angle_cls=False,
                    lang=args.ocr_paddle_lang,
                    use_gpu=use_gpu,
                    show_log=False,
                )
            except Exception:
                # Fallback to CPU if GPU runtime is not available in paddle.
                reader = PaddleOCR(
                    use_angle_cls=False,
                    lang=args.ocr_paddle_lang,
                    use_gpu=False,
                    show_log=False,
                )
            engines.append({"name": "paddleocr", "reader": reader})
        except Exception:
            if backend == "paddleocr":
                raise RuntimeError(
                    "paddleocr backend requested but paddleocr is not available. "
                    "Install it in sign-map-ros2 env or use --ocr_backend easyocr/tesseract."
                )

    if backend in ("auto", "easyocr"):
        try:
            import easyocr  # type: ignore

            easyocr_langs = [x.strip() for x in args.ocr_easyocr_langs.split(",") if x.strip()]
            reader = easyocr.Reader(easyocr_langs, gpu=device == "cuda")
            engines.append({"name": "easyocr", "reader": reader})
        except Exception:
            if backend == "easyocr":
                raise RuntimeError(
                    "easyocr backend requested but easyocr is not available. "
                    "Install it in sign-map-ros2 env or use --ocr_backend tesseract."
                )

    if backend in ("auto", "tesseract", "paddleocr"):
        engines.append({"name": "tesseract", "reader": None})

    if backend != "auto" and not engines:
        raise RuntimeError(f"OCR backend '{backend}' is not available.")

    numeric_specialist = None
    if str(args.ocr_numeric_specialist) == "easyocr_digits":
        easy_reader = None
        for eng in engines:
            if eng.get("name") == "easyocr" and eng.get("reader") is not None:
                easy_reader = eng.get("reader")
                break
        if easy_reader is None:
            try:
                import easyocr  # type: ignore

                easy_reader = easyocr.Reader(["en"], gpu=device == "cuda")
            except Exception:
                easy_reader = None
        if easy_reader is not None:
            numeric_specialist = {"name": "easyocr_digits", "reader": easy_reader}

    if backend == "auto":
        # auto mode: run all available engines and keep the best scoring text.
        return {
            "backend": "ensemble",
            "engines": engines,
            "tesseract_lang": tesseract_lang,
            "numeric_specialist": numeric_specialist,
        }

    return {
        "backend": backend,
        "engines": engines,
        "tesseract_lang": tesseract_lang,
        "numeric_specialist": numeric_specialist,
    }


def run_easyocr_ocr(image_crop: Image.Image, easyocr_reader, min_conf: float):
    arr = np.array(image_crop)
    out = easyocr_reader.readtext(arr, detail=1, paragraph=False)
    texts = []
    for item in out:
        if len(item) != 3:
            continue
        _, txt, conf = item
        if conf is None or conf < min_conf:
            continue
        txt = str(txt).strip()
        if txt:
            texts.append(txt)
    return " ".join(texts).strip()


def run_paddleocr_ocr(image_crop: Image.Image, paddle_reader, min_conf: float):
    arr = np.array(image_crop.convert("RGB"))
    out = paddle_reader.ocr(arr, cls=False)
    return extract_text_from_paddle_result(out, min_conf=min_conf)


def run_trocr_ocr(image_crop: Image.Image, trocr_reader, args):
    processor = trocr_reader["processor"]
    model = trocr_reader["model"]
    device = trocr_reader["device"]
    with torch.no_grad():
        pixel_values = processor(images=image_crop.convert("RGB"), return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values, max_new_tokens=int(args.trocr_max_new_tokens))
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return str(text).strip()


def run_ocr_single_backend(image_crop: Image.Image, backend_name: str, reader, tesseract_lang: str, args):
    if backend_name == "trocr":
        return run_trocr_ocr(image_crop, reader, args)
    if backend_name == "easyocr":
        return run_easyocr_ocr(image_crop, reader, args.ocr_easyocr_min_conf)
    if backend_name == "paddleocr":
        return run_paddleocr_ocr(image_crop, reader, args.ocr_paddle_min_conf)
    return run_tesseract_ocr(image_crop, lang=tesseract_lang, psm=args.ocr_psm)


def run_numeric_tesseract_candidates(image_crop: Image.Image, args):
    numeric_texts = []
    for psm in parse_psm_list(args.ocr_numeric_psm_list):
        raw = run_tesseract_ocr(
            image_crop,
            lang="eng",
            psm=psm,
            whitelist=args.ocr_numeric_chars,
        )
        raw = normalize_ocr_text(raw, r"[^B0-9\s]+")
        for cand in extract_numeric_candidates(raw, args):
            numeric_texts.append(cand)
    return numeric_texts


def run_numeric_paddle_candidates(image_crop: Image.Image, engines, args):
    if int(args.ocr_numeric_use_paddle) == 0:
        return []
    out = []
    for eng in engines:
        if eng.get("name") != "paddleocr":
            continue
        reader = eng.get("reader")
        if reader is None:
            continue
        try:
            arr = np.array(image_crop.convert("RGB"))
            result = reader.ocr(arr, cls=False)
        except Exception:
            continue
        for seg_text in extract_text_items_from_paddle_result(result, min_conf=args.ocr_paddle_numeric_min_conf):
            seg_text = normalize_ocr_text(seg_text, r"[^B0-9\s]+")
            for cand in extract_numeric_candidates(seg_text, args):
                out.append(cand)
    return out


def run_numeric_specialist_candidates(image_crop: Image.Image, specialist, args):
    if not specialist or specialist.get("name") != "easyocr_digits":
        return []
    reader = specialist.get("reader")
    if reader is None:
        return []
    arr = np.array(image_crop.convert("RGB"))
    try:
        out = reader.readtext(
            arr,
            detail=1,
            paragraph=False,
            allowlist=args.ocr_numeric_chars,
        )
    except Exception:
        try:
            out = reader.readtext(arr, detail=1, paragraph=False)
        except Exception:
            return []

    candidates = []
    for item in out:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        _, txt, conf = item
        if conf is None or float(conf) < float(args.ocr_numeric_specialist_min_conf):
            continue
        txt = normalize_ocr_text(str(txt).strip(), r"[^B0-9\s]+")
        for cand in extract_numeric_candidates(txt, args):
            candidates.append(cand)
    return candidates


def _clip_local_bbox_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(math.floor(x1)), w - 1))
    y1 = max(0, min(int(math.floor(y1)), h - 1))
    x2 = max(x1 + 1, min(int(math.ceil(x2)), w))
    y2 = max(y1 + 1, min(int(math.ceil(y2)), h))
    return x1, y1, x2, y2


def _quad_to_xyxy(quad, w, h):
    try:
        pts = np.array(quad, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 4:
            return None
    except Exception:
        return None
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return _clip_local_bbox_xyxy(x1, y1, x2, y2, w, h)


def _expand_local_bbox(x1, y1, x2, y2, w, h, ratio):
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    ex = bw * float(ratio)
    ey = bh * float(ratio)
    return _clip_local_bbox_xyxy(x1 - ex, y1 - ey, x2 + ex, y2 + ey, w, h)


def _bbox_iou_xyxy_local(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(1e-6, area_a + area_b - inter))


def detect_numeric_rois_with_paddle(image_crop: Image.Image, engines, args):
    paddle_reader = None
    for eng in engines:
        if eng.get("name") == "paddleocr" and eng.get("reader") is not None:
            paddle_reader = eng.get("reader")
            break
    if paddle_reader is None:
        return []

    arr = np.array(image_crop.convert("RGB"))
    try:
        result = paddle_reader.ocr(arr, cls=False)
    except Exception:
        return []
    lines = result[0] if isinstance(result, list) else result
    if lines is None:
        return []

    w, h = image_crop.size
    img_area = float(max(1, w * h))
    candidates = []
    for item in lines:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        quad = item[0]
        rec = item[1]
        if not isinstance(rec, (list, tuple)) or len(rec) == 0:
            continue
        txt = str(rec[0]).strip()
        conf = float(rec[1]) if len(rec) > 1 else 1.0
        if conf < float(args.ocr_numeric_roi_min_conf):
            continue
        bbox = _quad_to_xyxy(quad, w, h)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratio = float((bw * bh) / img_area)
        if area_ratio < float(args.ocr_numeric_roi_min_area_ratio):
            continue
        digit_count = len(re.findall(r"\d", txt))
        if digit_count == 0 and int(args.ocr_numeric_roi_allow_nondigit) != 1:
            continue
        priority = digit_count * 40.0 + float(conf) * 10.0 + area_ratio * 100.0
        if digit_count > 0:
            priority += 20.0
        candidates.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "text_hint": txt,
                "conf": float(conf),
                "digit_count_hint": int(digit_count),
                "priority": float(priority),
            }
        )

    if not candidates:
        return []
    candidates.sort(key=lambda x: x["priority"], reverse=True)

    out = []
    for c in candidates:
        x1, y1, x2, y2 = c["bbox_xyxy"]
        x1, y1, x2, y2 = _expand_local_bbox(
            x1,
            y1,
            x2,
            y2,
            w,
            h,
            ratio=float(args.ocr_numeric_roi_expand_ratio),
        )
        c["bbox_xyxy"] = [x1, y1, x2, y2]
        keep = True
        for k in out:
            if _bbox_iou_xyxy_local(c["bbox_xyxy"], k["bbox_xyxy"]) > 0.65:
                keep = False
                break
        if keep:
            out.append(c)
        if len(out) >= int(args.ocr_numeric_roi_max_count):
            break
    return out


def run_ocr_with_numeric_roi(
    image_crop: Image.Image,
    det_meta,
    ocr,
    args,
    superres_cfg=None,
    diffusion_cfg=None,
    selective_diffusion_used=0,
):
    engines = ocr.get("engines", [])
    rois = detect_numeric_rois_with_paddle(image_crop, engines, args)
    if not rois:
        if int(args.ocr_numeric_roi_strict) == 1:
            return {"ocr_text": "", "ocr_raw_text": "", "ocr_numeric_text": ""}, None, False, selective_diffusion_used, None
        # Robust fallback: when ROI detection fails, still use the same
        # precheck -> selective SR/diffusion policy on the full sign crop.
        ocr_out, _, crop_for_ocr, _, used_diffusion, selective_diffusion_used, _, _ = run_ocr_with_precheck(
            image_crop,
            det_meta,
            ocr,
            args,
            superres_cfg=superres_cfg,
            diffusion_cfg=diffusion_cfg,
            selective_diffusion_used=selective_diffusion_used,
            roi_text_hint="",
        )
        return ocr_out, None, used_diffusion, selective_diffusion_used, crop_for_ocr

    best = None
    for roi in rois:
        x1, y1, x2, y2 = roi["bbox_xyxy"]
        roi_crop = image_crop.crop((x1, y1, x2, y2))
        chosen_ocr_out, chosen_score, roi_for_ocr, used_backend, _, _, precheck_locked, raw_base_numeric_text = run_ocr_with_precheck(
            roi_crop,
            None,
            ocr,
            args,
            superres_cfg=superres_cfg,
            diffusion_cfg=None,
            selective_diffusion_used=0,
            roi_text_hint=roi.get("text_hint", ""),
        )

        score = chosen_score
        score += float(roi.get("priority", 0.0)) * 0.25
        item = {
            "roi": roi,
            "ocr_out": chosen_ocr_out,
            "score": float(score),
            "roi_for_ocr": roi_for_ocr,
            "roi_backend": used_backend,
            "precheck_locked": bool(precheck_locked),
            "raw_base_numeric_text": str(raw_base_numeric_text),
        }
        if best is None or item["score"] > best["score"]:
            best = item

    if best is None:
        return run_ocr_text(image_crop, ocr, args, superres_cfg=None), None, False, selective_diffusion_used, None

    used_diffusion = False
    best_ocr_out = best["ocr_out"]
    best_roi_for_ocr = best["roi_for_ocr"]
    best_score = score_numeric_ocr_candidate(best_ocr_out, args, roi_text_hint=best["roi"].get("text_hint", ""))
    max_sr_scale = get_max_superres_scale(args)
    if (not bool(best.get("precheck_locked", False))) and diffusion_cfg is not None and should_try_selective_diffusion(
        det_meta, best_ocr_out, args, selective_diffusion_used
    ):
        x1, y1, x2, y2 = best["roi"]["bbox_xyxy"]
        roi_crop = image_crop.crop((x1, y1, x2, y2))
        roi_diff_for_ocr = upscale_crop_for_ocr(roi_crop, diffusion_cfg, args, outscale=max_sr_scale)
        ocr_diff = run_ocr_text(roi_diff_for_ocr, ocr, args, superres_cfg=None)
        diff_score = score_numeric_ocr_candidate(
            ocr_diff,
            args,
            roi_text_hint=best["roi"].get("text_hint", ""),
            base_numeric_text=str(best.get("raw_base_numeric_text", "")),
        )
        if diff_score >= best_score + float(args.ocr_upscale_accept_margin):
            best_ocr_out = ocr_diff
            best_roi_for_ocr = roi_diff_for_ocr
            used_diffusion = True
            selective_diffusion_used += 1

    return best_ocr_out, best["roi"], used_diffusion, selective_diffusion_used, best_roi_for_ocr


def run_ocr_text(image_crop: Image.Image, ocr, args, superres_cfg=None):
    best_text = ""
    best_score = -1
    best_numeric = ""
    best_numeric_score = -1
    engines = ocr.get("engines", [])
    numeric_specialist = ocr.get("numeric_specialist")
    if not engines:
        engines = [{"name": "tesseract", "reader": None}]
    candidate_images = []
    if superres_cfg is None:
        candidate_images.extend(prepare_ocr_variants(image_crop, args))
    else:
        for sr_img in build_superres_inputs(image_crop, superres_cfg, args):
            candidate_images.extend(prepare_ocr_variants(sr_img, args))
    if args.ocr_max_candidates > 0 and len(candidate_images) > args.ocr_max_candidates:
        candidate_images = candidate_images[: args.ocr_max_candidates]

    # Numeric-only mode: skip general OCR engines and use digit-whitelist OCR path.
    if args.ocr_numeric_mode and args.ocr_numeric_only:
        best_numeric = ""
        best_numeric_score = -1
        for variant in candidate_images:
            if int(args.ocr_numeric_specialist_replace) != 1:
                for cand in run_numeric_tesseract_candidates(variant, args):
                    nscore = numeric_text_score(cand, args)
                    if nscore > best_numeric_score:
                        best_numeric_score = nscore
                        best_numeric = cand
                for cand in run_numeric_paddle_candidates(variant, engines, args):
                    nscore = numeric_text_score(cand, args)
                    if nscore > best_numeric_score:
                        best_numeric_score = nscore
                        best_numeric = cand
            for cand in run_numeric_specialist_candidates(variant, numeric_specialist, args):
                nscore = numeric_text_score(cand, args)
                if nscore > best_numeric_score:
                    best_numeric_score = nscore
                    best_numeric = cand
        return {
            "ocr_text": best_numeric if best_numeric_score >= 0 else "",
            "ocr_raw_text": best_numeric if best_numeric_score >= 0 else "",
            "ocr_numeric_text": best_numeric if best_numeric_score >= 0 else "",
        }

    for variant in candidate_images:
        if args.ocr_numeric_mode:
            if int(args.ocr_numeric_specialist_replace) != 1:
                for cand in run_numeric_tesseract_candidates(variant, args):
                    nscore = numeric_text_score(cand, args)
                    if nscore > best_numeric_score:
                        best_numeric_score = nscore
                        best_numeric = cand
                for cand in run_numeric_paddle_candidates(variant, engines, args):
                    nscore = numeric_text_score(cand, args)
                    if nscore > best_numeric_score:
                        best_numeric_score = nscore
                        best_numeric = cand
            for cand in run_numeric_specialist_candidates(variant, numeric_specialist, args):
                nscore = numeric_text_score(cand, args)
                if nscore > best_numeric_score:
                    best_numeric_score = nscore
                    best_numeric = cand
        for eng in engines:
            try:
                text = run_ocr_single_backend(
                    variant,
                    eng["name"],
                    eng.get("reader"),
                    ocr["tesseract_lang"],
                    args,
                )
            except Exception:
                continue
            text = normalize_ocr_text(text, args.ocr_text_pattern)
            score = ocr_text_score(text)
            if score > best_score:
                best_score = score
                best_text = text
            if args.ocr_numeric_mode:
                for cand in extract_numeric_candidates(text, args):
                    nscore = numeric_text_score(cand, args)
                    if nscore > best_numeric_score:
                        best_numeric_score = nscore
                        best_numeric = cand

    if args.ocr_numeric_mode:
        # Numeric-priority mode: prefer numeric text, but fall back to regular OCR
        # so English/Korean strings can still be returned when numbers are absent.
        final_text = best_numeric if best_numeric_score >= 0 else best_text
        return {
            "ocr_text": final_text,
            "ocr_raw_text": best_text,
            "ocr_numeric_text": best_numeric if best_numeric_score >= 0 else "",
        }

    return {
        "ocr_text": best_text,
        "ocr_raw_text": best_text,
        "ocr_numeric_text": "",
    }


def bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = max(1e-6, area_a + area_b - inter)
    return float(inter / denom)


class SimpleIoUTracker:
    def __init__(self, iou_thres=0.35, lost_frames=10):
        self.iou_thres = float(iou_thres)
        self.lost_frames = int(lost_frames)
        self.next_id = 1
        self.tracks = {}

    def update(self, frame_idx, boxes):
        assigned = [-1] * len(boxes)
        active_ids = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr["last_frame"] <= self.lost_frames:
                active_ids.append(tid)

        used_tracks = set()
        for i, box in enumerate(boxes):
            best_tid = None
            best_iou = 0.0
            for tid in active_ids:
                if tid in used_tracks:
                    continue
                tr_box = self.tracks[tid]["bbox"]
                iou = bbox_iou_xyxy(box, tr_box)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid
            if best_tid is not None and best_iou >= self.iou_thres:
                assigned[i] = best_tid
                used_tracks.add(best_tid)
                self.tracks[best_tid]["bbox"] = box
                self.tracks[best_tid]["last_frame"] = frame_idx
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": box, "last_frame": frame_idx}
                assigned[i] = tid
                used_tracks.add(tid)
        return assigned


def assign_track_ids(detections_out, args):
    frame_groups = defaultdict(list)
    for idx, d in enumerate(detections_out):
        frame_groups[int(d["frame_idx"])].append((idx, d))

    frame_indices = sorted(frame_groups.keys())
    if not frame_indices:
        return

    if args.tracker == "off":
        for i, d in enumerate(detections_out, start=1):
            d["track_id"] = i
        return

    if args.tracker == "bytetrack":
        try:
            import supervision as sv  # type: ignore

            tracker = sv.ByteTrack(
                frame_rate=max(1.0, float(args.tracker_frame_rate)),
                minimum_matching_threshold=float(args.tracker_iou_thres),
                lost_track_buffer=int(args.tracker_lost_frames),
            )
            for fidx in frame_indices:
                items = frame_groups[fidx]
                boxes = np.array([it[1]["bbox_xyxy"] for it in items], dtype=np.float32)
                conf = np.array([it[1]["score"] for it in items], dtype=np.float32)
                class_id = np.zeros((len(items),), dtype=np.int32)
                det = sv.Detections(xyxy=boxes, confidence=conf, class_id=class_id)
                tracked = tracker.update_with_detections(det)
                t_boxes = tracked.xyxy if tracked.xyxy is not None else np.zeros((0, 4), dtype=np.float32)
                t_ids = tracked.tracker_id if tracked.tracker_id is not None else np.zeros((0,), dtype=np.int32)

                used = set()
                for tb, tid in zip(t_boxes, t_ids):
                    best_j = None
                    best_iou = 0.0
                    for j, (_, rec) in enumerate(items):
                        if j in used:
                            continue
                        iou = bbox_iou_xyxy(tb, rec["bbox_xyxy"])
                        if iou > best_iou:
                            best_iou = iou
                            best_j = j
                    if best_j is not None and best_iou > 0.1:
                        used.add(best_j)
                        rec = items[best_j][1]
                        rec["track_id"] = int(tid)

                # unmatched detections become short-lived unique tracks
                next_temp = 1_000_000 + fidx * 1000
                for j, (_, rec) in enumerate(items):
                    if "track_id" not in rec:
                        rec["track_id"] = next_temp
                        next_temp += 1
            hist = Counter(int(d.get("track_id", -1)) for d in detections_out)
            repeated_non_temp = sum(1 for tid, c in hist.items() if tid >= 0 and tid < 1_000_000 and c >= 2)
            if repeated_non_temp >= 1:
                return
            # If ByteTrack does not produce stable tracks on this sequence,
            # clear and fallback to simple IoU tracking.
            for d in detections_out:
                if "track_id" in d:
                    del d["track_id"]
        except Exception:
            pass

    # fallback tracker
    tracker = SimpleIoUTracker(iou_thres=args.tracker_iou_thres, lost_frames=args.tracker_lost_frames)
    for fidx in frame_indices:
        items = frame_groups[fidx]
        boxes = [it[1]["bbox_xyxy"] for it in items]
        tids = tracker.update(fidx, boxes)
        for (_, rec), tid in zip(items, tids):
            rec["track_id"] = int(tid)


def finalize_track_labels(detections_out, args):
    tracks = defaultdict(list)
    for d in detections_out:
        tracks[int(d.get("track_id", -1))].append(d)
    label_vocab = parse_final_label_vocab(args)

    def detection_vote_weight(d, canon_text):
        w = float(d.get("score", 0.0)) + min(len(canon_text), 12) * 0.05
        if d.get("number_roi_bbox_crop") is None:
            w *= 0.75
        hint = (d.get("number_roi_text_hint") or "").strip()
        hint_num = ""
        if hint:
            hc = extract_numeric_candidates(hint, args)
            hint_num = hc[0] if hc else ""
        if bool(d.get("ocr_used_diffusion")):
            if hint_num:
                if canon_text.replace(" ", "") == hint_num.replace(" ", ""):
                    w *= 1.05
                else:
                    w *= 0.50
            else:
                w *= 0.70
        return float(w)

    finalized = []
    for tid, items in sorted(tracks.items(), key=lambda kv: kv[0]):
        if tid < 0:
            continue
        if len(items) < int(args.tracker_min_track_hits):
            continue

        votes = Counter()
        support = Counter()
        for d in items:
            t = (d.get("ocr_text") or "").strip()
            if len(t) < int(args.vote_min_text_len):
                continue
            canon = re.sub(r"\s+", " ", t).strip().upper()
            if not canon:
                continue
            canon, sim, in_vocab = snap_text_to_label_vocab(canon, label_vocab, args)
            if not canon:
                continue
            w = detection_vote_weight(d, canon)
            if label_vocab:
                if in_vocab:
                    w += float(args.final_label_vocab_vote_bonus) * float(sim)
                else:
                    w *= float(args.final_label_vocab_oov_penalty)
            votes[canon] += w
            support[canon] += 1
        if not votes:
            continue
        label, weight = votes.most_common(1)[0]
        vote_count = int(support.get(label, 0))
        weak_label = False
        if vote_count < int(args.vote_min_count):
            # fallback: still output the top weighted label for this track
            # so downstream point-cloud labeling is not empty.
            weak_label = True

        centroids = [d["centroid_xyz"] for d in items if d.get("centroid_xyz") is not None]
        centroid = None
        if centroids:
            c = np.array(centroids, dtype=np.float32)
            centroid = np.median(c, axis=0).tolist()

        finalized.append(
            {
                "track_id": int(tid),
                "final_text": label,
                "vote_count": int(vote_count),
                "vote_weight": float(weight),
                "detections": len(items),
                "centroid_xyz": centroid,
                "weak_label": weak_label,
            }
        )

    final_by_tid = {int(t["track_id"]): t["final_text"] for t in finalized}
    for d in detections_out:
        d["track_text_final"] = final_by_tid.get(int(d.get("track_id", -1)), "")

    return finalized


def estimate_frontalness_score(d):
    bw = max(1.0, float(d.get("bbox_w", 1.0)))
    bh = max(1.0, float(d.get("bbox_h", 1.0)))
    area_ratio = float(d.get("bbox_area_ratio", 0.0))
    center = d.get("bbox_center_norm") or [0.5, 0.5]
    cx = float(center[0]) if len(center) >= 1 else 0.5
    cy = float(center[1]) if len(center) >= 2 else 0.5

    # 1) bigger bbox is usually closer/more frontal
    size_score = min(1.0, max(0.0, area_ratio / 0.02))

    # 2) closer to image center is usually better observed by a forward camera
    dist = math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / 0.70710678
    center_score = max(0.0, 1.0 - dist)

    # 3) oblique views tend to produce very elongated boxes
    aspect = bw / bh
    if aspect < 1.0:
        aspect = 1.0 / aspect
    shape_score = max(0.0, 1.0 - min(1.0, abs(math.log(aspect)) / 1.2))

    return float(size_score * 0.55 + center_score * 0.30 + shape_score * 0.15)


def detection_quality_score(d, args):
    text = (d.get("ocr_text") or "").strip()
    numeric = (d.get("ocr_numeric_text") or "").strip()
    useful = len(re.findall(r"[0-9A-Za-z가-힣]", text))
    digits = len(re.findall(r"\d", numeric if numeric else text))
    area_ratio = float(d.get("bbox_area_ratio", 0.0))
    point_count = int(d.get("point_count", 0))
    det_score = float(d.get("score", 0.0))
    score = useful * 2.0 + digits * 8.0 + det_score * 4.0
    score += min(40.0, area_ratio * 4500.0)
    score += min(10.0, point_count / 25.0)
    if args.ocr_numeric_mode and numeric:
        score += 15.0
    if int(args.repr_use_frontal) == 1:
        score += float(args.repr_frontal_weight) * estimate_frontalness_score(d)
    return float(score)


def ocr_output_score(ocr_out, args):
    text = (ocr_out.get("ocr_text") or "").strip()
    raw = (ocr_out.get("ocr_raw_text") or "").strip()
    numeric = (ocr_out.get("ocr_numeric_text") or "").strip()
    score = float(ocr_text_score(text))
    score = max(score, float(ocr_text_score(raw)))
    if args.ocr_numeric_mode:
        score = max(score, float(numeric_text_score(numeric, args)) + 30.0)
    return score


def is_ocr_uncertain(ocr_out, args):
    if args.ocr_numeric_mode:
        numeric = (ocr_out.get("ocr_numeric_text") or "").strip()
        digit_count = len(re.findall(r"\d", numeric))
        return digit_count < int(args.diffusion_uncertain_min_digits)
    txt = (ocr_out.get("ocr_text") or "").strip()
    return ocr_text_score(txt) < int(args.diffusion_uncertain_min_text_score)


def _extract_num_for_compare(ocr_out):
    num = (ocr_out.get("ocr_numeric_text") or "").strip()
    if not num:
        num = (ocr_out.get("ocr_text") or "").strip()
    return num


def _digit_count(text):
    return len(re.findall(r"\d", text or ""))


def _compact_num_text(text):
    return re.sub(r"\s+", "", str(text or "").strip())


def parse_final_label_vocab(args):
    raw = str(getattr(args, "final_label_vocab", "") or "")
    out = []
    seen = set()
    for tok in raw.split(","):
        t = re.sub(r"\s+", "", str(tok).strip().upper())
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def snap_text_to_label_vocab(text, vocab, args):
    canon = re.sub(r"\s+", "", str(text or "").strip().upper())
    if not canon:
        return "", 0.0, False
    if not vocab:
        return canon, 1.0, False
    best = ""
    best_sim = -1.0
    for v in vocab:
        sim = float(SequenceMatcher(None, canon, v).ratio())
        if sim > best_sim:
            best_sim = sim
            best = v
    if best_sim < float(args.final_label_vocab_min_sim):
        if int(args.final_label_vocab_enforce) == 1:
            return "", best_sim, False
        return canon, best_sim, False
    return best, best_sim, True


def score_numeric_ocr_candidate(ocr_out, args, roi_text_hint="", base_numeric_text=""):
    score = float(ocr_output_score(ocr_out, args))
    cand_num = _compact_num_text(_extract_num_for_compare(ocr_out))

    hint_compact = ""
    if roi_text_hint:
        hint_cands = extract_numeric_candidates(str(roi_text_hint), args)
        if hint_cands:
            hint_compact = _compact_num_text(hint_cands[0])
    if hint_compact and cand_num:
        sim = float(SequenceMatcher(None, cand_num, hint_compact).ratio())
        score += sim * float(args.ocr_roi_hint_weight)

    if base_numeric_text:
        base_cnt = _digit_count(base_numeric_text)
        cand_cnt = _digit_count(cand_num)
        extra = max(0, cand_cnt - base_cnt - int(args.ocr_upscale_max_extra_digits))
        if extra > 0:
            score -= float(args.ocr_upscale_extra_digit_penalty) * float(extra)
    return float(score)


def is_raw_precheck_confident(ocr_out, args):
    if int(args.ocr_precheck_before_upscale) != 1:
        return False
    if float(ocr_output_score(ocr_out, args)) < float(args.ocr_precheck_min_score):
        return False
    if args.ocr_numeric_mode:
        if _digit_count(_extract_num_for_compare(ocr_out)) < int(args.ocr_precheck_min_digits):
            return False
    return True


def run_ocr_with_precheck(
    image_crop: Image.Image,
    det_meta,
    ocr,
    args,
    superres_cfg=None,
    diffusion_cfg=None,
    selective_diffusion_used=0,
    roi_text_hint="",
):
    max_sr_scale = get_max_superres_scale(args)
    raw_ocr_out = run_ocr_text(image_crop, ocr, args, superres_cfg=None)
    raw_score = score_numeric_ocr_candidate(raw_ocr_out, args, roi_text_hint=roi_text_hint)
    raw_base_numeric_text = _extract_num_for_compare(raw_ocr_out)

    chosen_ocr_out = raw_ocr_out
    chosen_score = raw_score
    crop_for_ocr = image_crop
    used_backend = "none"
    used_diffusion = False
    precheck_locked = is_raw_precheck_confident(raw_ocr_out, args)

    if (not precheck_locked) and superres_cfg is not None:
        sr_crop_for_ocr = upscale_crop_for_ocr(image_crop, superres_cfg, args, outscale=max_sr_scale)
        sr_ocr_out = run_ocr_text(sr_crop_for_ocr, ocr, args, superres_cfg=None)
        sr_score = score_numeric_ocr_candidate(
            sr_ocr_out,
            args,
            roi_text_hint=roi_text_hint,
            base_numeric_text=raw_base_numeric_text,
        )
        if sr_score >= chosen_score + float(args.ocr_upscale_accept_margin):
            chosen_ocr_out = sr_ocr_out
            chosen_score = sr_score
            crop_for_ocr = sr_crop_for_ocr
            used_backend = str(superres_cfg.get("backend", "none"))

    if (
        (not precheck_locked)
        and diffusion_cfg is not None
        and det_meta is not None
        and should_try_selective_diffusion(det_meta, chosen_ocr_out, args, selective_diffusion_used)
    ):
        diff_crop_for_ocr = upscale_crop_for_ocr(image_crop, diffusion_cfg, args, outscale=max_sr_scale)
        ocr_diff = run_ocr_text(diff_crop_for_ocr, ocr, args, superres_cfg=None)
        diff_score = score_numeric_ocr_candidate(
            ocr_diff,
            args,
            roi_text_hint=roi_text_hint,
            base_numeric_text=raw_base_numeric_text,
        )
        if diff_score >= chosen_score + float(args.ocr_upscale_accept_margin):
            chosen_ocr_out = ocr_diff
            chosen_score = diff_score
            crop_for_ocr = diff_crop_for_ocr
            used_backend = str(diffusion_cfg.get("backend", "diffusion"))
            used_diffusion = True
            selective_diffusion_used += 1

    return (
        chosen_ocr_out,
        float(chosen_score),
        crop_for_ocr,
        used_backend,
        bool(used_diffusion),
        int(selective_diffusion_used),
        bool(precheck_locked),
        str(raw_base_numeric_text),
    )


def should_try_selective_diffusion(d, ocr_out, args, used_count):
    if int(args.diffusion_selective_enabled) != 1:
        return False
    if used_count >= int(args.diffusion_selective_max_crops):
        return False
    area_ratio = float(d.get("bbox_area_ratio", 0.0))
    if area_ratio < float(args.diffusion_selective_min_area_ratio):
        return False
    frontal = estimate_frontalness_score(d)
    if frontal < float(args.diffusion_selective_min_frontalness):
        return False
    if not is_ocr_uncertain(ocr_out, args):
        return False
    return True


def merge_track_labels(finalized_tracks, detections_out, args):
    label_vocab = parse_final_label_vocab(args)

    def detection_vote_weight(d, canon_text):
        w = float(d.get("score", 0.0)) + min(len(canon_text), 12) * 0.05
        if d.get("number_roi_bbox_crop") is None:
            w *= 0.75
        hint = (d.get("number_roi_text_hint") or "").strip()
        hint_num = ""
        if hint:
            hc = extract_numeric_candidates(hint, args)
            hint_num = hc[0] if hc else ""
        if bool(d.get("ocr_used_diffusion")):
            if hint_num:
                if canon_text.replace(" ", "") == hint_num.replace(" ", ""):
                    w *= 1.05
                else:
                    w *= 0.50
            else:
                w *= 0.70
        return float(w)

    if int(args.track_merge_enabled) != 1:
        out = []
        for i, t in enumerate(finalized_tracks, start=1):
            rec = dict(t)
            rec["sign_id"] = i
            rec["merged_track_ids"] = [int(t["track_id"])]
            out.append(rec)
        return out

    with_centroid = [t for t in finalized_tracks if t.get("centroid_xyz") is not None]
    no_centroid = [t for t in finalized_tracks if t.get("centroid_xyz") is None]
    n = len(with_centroid)
    if n == 0:
        out = []
        for i, t in enumerate(no_centroid, start=1):
            rec = dict(t)
            rec["sign_id"] = i
            rec["merged_track_ids"] = [int(t["track_id"])]
            out.append(rec)
        return out

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    merge_dist = float(args.track_merge_3d_dist)
    xy_th = max(1e-9, float(args.track_merge_xy_dist))
    z_th = max(1e-9, float(args.track_merge_z_dist))
    min_score = float(args.track_merge_min_score)
    w_3d = float(args.track_merge_w_3d)
    w_ocr = float(args.track_merge_w_ocr)
    w_view = float(args.track_merge_w_view)
    w_sum = max(1e-9, w_3d + w_ocr + w_view)
    w_3d, w_ocr, w_view = w_3d / w_sum, w_ocr / w_sum, w_view / w_sum

    by_tid = defaultdict(list)
    for d in detections_out:
        by_tid[int(d.get("track_id", -1))].append(d)

    def canon_track_text(t):
        txt = (t.get("final_text") or "").strip().upper()
        txt = re.sub(r"\s+", "", txt)
        return txt

    def track_has_confident_text(t):
        if not canon_track_text(t):
            return False
        if int(t.get("vote_count", 0)) < int(args.track_merge_ocr_conflict_min_support):
            return False
        if float(t.get("vote_weight", 0.0)) < float(args.track_merge_ocr_conflict_min_vote_weight):
            return False
        return True

    def best_detection_for_track(track_id):
        det_items = by_tid.get(int(track_id), [])
        best_det = None
        best_q = -1e9
        for d in det_items:
            q = detection_quality_score(d, args)
            if q > best_q:
                best_q = q
                best_det = d
        return best_det

    track_meta = {}
    for t in with_centroid:
        tid = int(t["track_id"])
        det_items = by_tid.get(tid, [])
        frames = set()
        for d in det_items:
            fn = d.get("frame_name")
            if fn:
                frames.add(str(fn))
        rep = best_detection_for_track(tid)
        frontal = float(estimate_frontalness_score(rep)) if rep is not None else 0.0
        area_ratio = float(rep.get("bbox_area_ratio", 0.0)) if rep is not None else 0.0
        track_meta[tid] = {
            "frames": frames,
            "rep": rep,
            "frontal": frontal,
            "area_ratio": area_ratio,
            "canon_text": canon_track_text(t),
            "text_confident": track_has_confident_text(t),
        }

    centers = np.array([t["centroid_xyz"] for t in with_centroid], dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            ti = with_centroid[i]
            tj = with_centroid[j]
            tid_i = int(ti["track_id"])
            tid_j = int(tj["track_id"])

            ci = centers[i]
            cj = centers[j]
            delta = ci - cj
            xy_dist = float(np.linalg.norm(delta[:2]))
            z_dist = float(abs(delta[2]))
            dist = float(np.linalg.norm(delta))

            if merge_dist > 0 and dist > merge_dist:
                continue
            if xy_dist > xy_th or z_dist > z_th:
                continue

            meta_i = track_meta.get(tid_i, {})
            meta_j = track_meta.get(tid_j, {})

            if int(args.track_merge_cooccur_block) == 1:
                if (meta_i.get("frames") or set()) & (meta_j.get("frames") or set()):
                    continue

            if int(args.track_merge_ocr_conflict_block) == 1:
                txt_i = str(meta_i.get("canon_text") or "")
                txt_j = str(meta_j.get("canon_text") or "")
                if txt_i and txt_j and txt_i != txt_j:
                    if bool(meta_i.get("text_confident")) and bool(meta_j.get("text_confident")):
                        continue

            s_xy = max(0.0, 1.0 - (xy_dist / xy_th))
            s_z = max(0.0, 1.0 - (z_dist / z_th))
            s_3d = 0.5 * (s_xy + s_z)

            txt_i = str(meta_i.get("canon_text") or "")
            txt_j = str(meta_j.get("canon_text") or "")
            if txt_i and txt_j:
                s_ocr = float(SequenceMatcher(None, txt_i, txt_j).ratio())
            else:
                s_ocr = 0.40

            front_i = float(meta_i.get("frontal", 0.0))
            front_j = float(meta_j.get("frontal", 0.0))
            s_front = max(0.0, 1.0 - abs(front_i - front_j))
            area_i = float(meta_i.get("area_ratio", 0.0))
            area_j = float(meta_j.get("area_ratio", 0.0))
            if area_i > 0.0 and area_j > 0.0:
                ratio_log = abs(math.log((area_i + 1e-9) / (area_j + 1e-9)))
                s_area = max(0.0, 1.0 - min(1.0, ratio_log / 1.4))
            else:
                s_area = 0.5
            s_view = 0.5 * (s_front + s_area)

            merge_score = w_3d * s_3d + w_ocr * s_ocr + w_view * s_view
            if merge_score >= min_score:
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(with_centroid[i])

    merged = []
    next_sign_id = 1
    for _, tracks in groups.items():
        track_ids = sorted(int(t["track_id"]) for t in tracks)
        track_id_set = set(track_ids)
        det_items = [d for d in detections_out if int(d.get("track_id", -1)) in track_id_set]
        votes = Counter()
        support = Counter()
        for d in det_items:
            txt = re.sub(r"\s+", " ", (d.get("ocr_text") or "").strip()).upper()
            if not txt:
                continue
            txt, sim, in_vocab = snap_text_to_label_vocab(txt, label_vocab, args)
            if not txt:
                continue
            w = detection_vote_weight(d, txt)
            if label_vocab:
                if in_vocab:
                    w += float(args.final_label_vocab_vote_bonus) * float(sim)
                else:
                    w *= float(args.final_label_vocab_oov_penalty)
            votes[txt] += float(w)
            support[txt] += 1
        best_det = None
        best_q = -1e9
        for d in det_items:
            q = detection_quality_score(d, args)
            if q > best_q:
                best_q = q
                best_det = d

        rep_txt = re.sub(r"\s+", " ", (best_det.get("ocr_text") or "").strip()).upper() if best_det else ""
        if rep_txt:
            rep_txt, _, _ = snap_text_to_label_vocab(rep_txt, label_vocab, args)
        if votes:
            candidate_score = {}
            for txt in votes.keys():
                s = float(votes[txt]) + float(support[txt]) * 0.35
                if rep_txt and txt == rep_txt:
                    s += float(args.final_label_repr_bonus)
                candidate_score[txt] = s
            final_text = sorted(
                votes.keys(),
                key=lambda t: (candidate_score[t], support[t], votes[t]),
                reverse=True,
            )[0]
            chosen_vote_count = int(support[final_text])
            chosen_vote_weight = float(votes[final_text])
        else:
            final_text = rep_txt
            chosen_vote_count = 0
            chosen_vote_weight = 0.0

        if final_text:
            snapped, _, _ = snap_text_to_label_vocab(final_text, label_vocab, args)
            final_text = snapped
        weak_label = int(chosen_vote_count) < int(args.vote_min_count)
        if weak_label and int(args.final_label_drop_weak) == 1:
            final_text = ""

        c = np.array([t["centroid_xyz"] for t in tracks], dtype=np.float32)
        centroid = np.median(c, axis=0).tolist()

        merged.append(
            {
                "sign_id": int(next_sign_id),
                "final_text": final_text,
                "merged_track_ids": track_ids,
                "tracks": len(track_ids),
                "detections": int(sum(int(t.get("detections", 0)) for t in tracks)),
                "vote_count": chosen_vote_count,
                "vote_weight": chosen_vote_weight,
                "weak_label": bool(weak_label),
                "centroid_xyz": centroid,
                "representative_frame": best_det.get("frame_name") if best_det else "",
                "representative_track_id": int(best_det.get("track_id", -1)) if best_det else -1,
                "representative_ocr_text": best_det.get("ocr_text", "") if best_det else "",
                "representative_crop_path": best_det.get("crop_path", "") if best_det else "",
                "representative_upscaled_crop_path": best_det.get("upscaled_crop_path", "") if best_det else "",
                "representative_quality": float(best_q if best_det else 0.0),
                "representative_frontalness": float(estimate_frontalness_score(best_det) if best_det else 0.0),
            }
        )
        next_sign_id += 1

    for t in no_centroid:
        track_id = int(t["track_id"])
        det_items = [d for d in detections_out if int(d.get("track_id", -1)) == track_id]
        best_det = None
        best_q = -1e9
        for d in det_items:
            q = detection_quality_score(d, args)
            if q > best_q:
                best_q = q
                best_det = d
        merged.append(
            {
                "sign_id": int(next_sign_id),
                "final_text": ("" if (int(args.final_label_drop_weak) == 1 and int(t.get("vote_count", 0)) < int(args.vote_min_count)) else (t.get("final_text") or "")),
                "merged_track_ids": [track_id],
                "tracks": 1,
                "detections": int(t.get("detections", len(det_items))),
                "vote_count": int(t.get("vote_count", 0)),
                "vote_weight": float(t.get("vote_weight", 0.0)),
                "weak_label": bool(int(t.get("vote_count", 0)) < int(args.vote_min_count)),
                "centroid_xyz": t.get("centroid_xyz"),
                "representative_frame": best_det.get("frame_name") if best_det else "",
                "representative_track_id": int(track_id),
                "representative_ocr_text": best_det.get("ocr_text", "") if best_det else "",
                "representative_crop_path": best_det.get("crop_path", "") if best_det else "",
                "representative_upscaled_crop_path": best_det.get("upscaled_crop_path", "") if best_det else "",
                "representative_quality": float(best_q if best_det else 0.0),
                "representative_frontalness": float(estimate_frontalness_score(best_det) if best_det else 0.0),
            }
        )
        next_sign_id += 1

    merged.sort(key=lambda x: int(x["sign_id"]))
    return merged


def unify_nearby_final_labels(final_track_labels, args):
    if int(args.final_label_unify_enabled) != 1:
        return final_track_labels
    if not final_track_labels:
        return final_track_labels

    def has_reliable_label(s):
        txt = re.sub(r"\s+", "", str(s.get("final_text", "")).strip().upper())
        if not txt:
            return False
        if int(s.get("vote_count", 0)) < int(args.vote_min_count):
            return False
        return True

    # Only unify already-reliable labels.
    # This prevents weak/empty labels from acting as bridges that can
    # accidentally overwrite stronger neighboring labels.
    idxs = [
        i
        for i, s in enumerate(final_track_labels)
        if s.get("centroid_xyz") is not None and has_reliable_label(s)
    ]
    if len(idxs) < 2:
        return final_track_labels

    parent = {i: i for i in idxs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    centers = {i: np.array(final_track_labels[i]["centroid_xyz"], dtype=np.float32) for i in idxs}
    dist_th = float(args.final_label_unify_dist)
    for i_pos in range(len(idxs)):
        i = idxs[i_pos]
        ci = centers[i]
        for j_pos in range(i_pos + 1, len(idxs)):
            j = idxs[j_pos]
            cj = centers[j]
            dist = float(np.linalg.norm(ci - cj))
            if dist <= dist_th:
                if int(args.final_label_unify_conflict_block) == 1:
                    ti = re.sub(r"\s+", "", str(final_track_labels[i].get("final_text", "")).strip().upper())
                    tj = re.sub(r"\s+", "", str(final_track_labels[j].get("final_text", "")).strip().upper())
                    if ti and tj and ti != tj:
                        continue
                union(i, j)

    groups = defaultdict(list)
    for i in idxs:
        groups[find(i)].append(i)

    for gid, members in groups.items():
        votes = Counter()
        for i in members:
            s = final_track_labels[i]
            txt = (s.get("final_text") or "").strip()
            if not txt:
                continue
            w = float(s.get("vote_weight", 0.0))
            w += float(s.get("detections", 0)) * 0.5
            votes[txt] += w
        if not votes:
            continue
        chosen = votes.most_common(1)[0][0]
        for i in members:
            final_track_labels[i]["final_text"] = chosen
            final_track_labels[i]["label_unify_group"] = int(gid)
    return final_track_labels


def map_bbox_to_points(pointcloud, conf_mask, img_w, img_h, bbox_xyxy):
    ph, pw = pointcloud.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    sx = pw / float(img_w)
    sy = ph / float(img_h)
    px1 = max(0, min(int(math.floor(x1 * sx)), pw - 1))
    py1 = max(0, min(int(math.floor(y1 * sy)), ph - 1))
    px2 = max(px1 + 1, min(int(math.ceil(x2 * sx)), pw))
    py2 = max(py1 + 1, min(int(math.ceil(y2 * sy)), ph))

    pts = pointcloud[py1:py2, px1:px2, :]
    m = conf_mask[py1:py2, px1:px2]
    pts = pts[m]
    if pts.shape[0] == 0:
        return None

    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return None
    return pts


def build_pointcloud_view(points_list, max_points, rng):
    if not points_list:
        return np.zeros((0, 3), dtype=np.float32)
    merged = []
    for pts in points_list:
        if pts.shape[0] == 0:
            continue
        if pts.shape[0] > 2000:
            idx = rng.choice(pts.shape[0], size=2000, replace=False)
            pts = pts[idx]
        merged.append(pts.astype(np.float32))
    if not merged:
        return np.zeros((0, 3), dtype=np.float32)
    merged = np.vstack(merged)
    if merged.shape[0] > max_points:
        idx = rng.choice(merged.shape[0], size=max_points, replace=False)
        merged = merged[idx]
    return merged


def load_open_set_objects(open_set_json: Path):
    if open_set_json is None:
        return []
    path_str = str(open_set_json).strip()
    if not path_str:
        return []
    p = Path(path_str)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    if isinstance(payload, dict):
        objs = payload.get("objects", [])
    elif isinstance(payload, list):
        objs = payload
    else:
        objs = []
    if not isinstance(objs, list):
        return []
    return objs


def _bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(float(ax1), float(bx1))
    iy1 = max(float(ay1), float(by1))
    ix2 = min(float(ax2), float(bx2))
    iy2 = min(float(ay2), float(by2))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (float(ax2) - float(ax1)) * (float(ay2) - float(ay1)))
    area_b = max(0.0, (float(bx2) - float(bx1)) * (float(by2) - float(by1)))
    return float(inter / max(1e-6, area_a + area_b - inter))


def nms_xyxy(dets, iou_thres=0.65):
    if not dets:
        return []
    src = sorted(dets, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    out = []
    for d in src:
        cls_d = str(d.get("query", d.get("object_class", ""))).strip().lower()
        keep = True
        for k in out:
            cls_k = str(k.get("query", k.get("object_class", ""))).strip().lower()
            if cls_d and cls_k and cls_d != cls_k:
                # Do not suppress different semantic classes (e.g., sign vs door).
                continue
            if _bbox_iou_xyxy(d["bbox_xyxy"], k["bbox_xyxy"]) > float(iou_thres):
                keep = False
                break
        if keep:
            out.append(d)
    return out


def build_open_set_index_by_frame(open_set_objects):
    idx = defaultdict(list)
    for o in open_set_objects:
        try:
            fid = float(o.get("frame_id"))
        except Exception:
            continue
        idx[fid].append(o)
    return idx


def _fetch_open_set_for_frame(frame_id, os_idx_by_frame, tol=0.25):
    if not os_idx_by_frame:
        return []
    if frame_id in os_idx_by_frame:
        return list(os_idx_by_frame[frame_id])
    out = []
    ftol = max(0.0, float(tol))
    if ftol <= 0:
        return out
    for fid, items in os_idx_by_frame.items():
        if abs(float(fid) - float(frame_id)) <= ftol:
            out.extend(items)
    return out


def run_sam3_box_inference(frame_id, img_w, img_h, os_idx_by_frame, args):
    objs = _fetch_open_set_for_frame(frame_id, os_idx_by_frame, tol=float(args.sam3_frame_id_tolerance))
    if not objs:
        return []
    qset = {q.strip().lower() for q in str(args.sam3_query_filter).split(",") if q.strip()}
    out = []
    for o in objs:
        query = str(o.get("query", "")).strip().lower()
        if qset and query not in qset:
            continue
        box = o.get("box_xyxy")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
            sam_score = float(o.get("sam_score", 0.0))
            sem_score = float(o.get("semantic_score", 0.0))
            point_count = float(o.get("point_count", 0.0))
        except Exception:
            continue
        if sam_score < float(args.sam3_min_sam_score):
            continue
        if sem_score < float(args.sam3_min_semantic_score):
            continue
        x1 = max(0, min(int(round(x1)), img_w - 1))
        y1 = max(0, min(int(round(y1)), img_h - 1))
        x2 = max(1, min(int(round(x2)), img_w))
        y2 = max(1, min(int(round(y2)), img_h))
        if x2 <= x1 or y2 <= y1:
            continue
        score = 0.75 * sam_score + 0.25 * max(0.0, sem_score)
        if point_count > 0:
            score += min(0.08, math.log10(point_count + 1.0) * 0.02)
        out.append(
            {
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": float(max(0.0, min(1.0, score))),
                "source": "sam3",
                "query": query,
                "sam_score": float(sam_score),
                "semantic_score": float(sem_score),
                "point_count": int(max(0.0, point_count)),
            }
        )
    out.sort(key=lambda d: d["score"], reverse=True)
    out = nms_xyxy(out, iou_thres=float(args.det_merge_iou_thres))
    if int(args.sam3_max_detections_per_frame) > 0:
        out = out[: int(args.sam3_max_detections_per_frame)]
    return out


def _obb_corners_world(center, extent, rotation):
    c = np.asarray(center, dtype=np.float64).reshape(-1)
    e = np.asarray(extent, dtype=np.float64).reshape(-1)
    r = np.asarray(rotation, dtype=np.float64)
    if c.shape[0] != 3 or e.shape[0] != 3 or r.shape != (3, 3):
        return None
    if np.any(~np.isfinite(c)) or np.any(~np.isfinite(e)) or np.any(~np.isfinite(r)):
        return None
    dx, dy, dz = e / 2.0
    corners_local = np.array(
        [
            [-dx, -dy, -dz],
            [dx, -dy, -dz],
            [dx, dy, -dz],
            [-dx, dy, -dz],
            [-dx, -dy, dz],
            [dx, -dy, dz],
            [dx, dy, dz],
            [-dx, dy, dz],
        ],
        dtype=np.float64,
    )
    corners_world = (r @ corners_local.T).T + c
    return corners_world


def make_plotly_view(
    pointcloud_xyz,
    detections,
    final_track_labels,
    open_set_objects,
    out_html: Path,
    marker_size: float,
    hide_empty_ocr: bool = True,
    show_open_set: bool = True,
    open_set_max_objects: int = 0,
    open_set_line_width: float = 6.0,
    pose_rows=None,
    side_frames=None,
    video_meta=None,
    side_max_frames: int = 180,
):
    pose_rows = pose_rows or []
    side_frames = side_frames or []
    video_meta = video_meta or {}
    fig = go.Figure()

    if pointcloud_xyz.shape[0] > 0:
        fig.add_trace(
            go.Scatter3d(
                x=pointcloud_xyz[:, 0],
                y=pointcloud_xyz[:, 1],
                z=pointcloud_xyz[:, 2],
                mode="markers",
                marker=dict(size=marker_size, color="rgba(130,130,130,0.30)"),
                name="map_points",
                hoverinfo="skip",
            )
        )

    mapped_all = [d for d in detections if d.get("centroid_xyz") is not None]
    mapped_sign = [d for d in mapped_all if str(d.get("object_class", "sign")).strip().lower() == "sign"]
    mapped_door = [d for d in mapped_all if str(d.get("object_class", "")).strip().lower() == "door"]
    if hide_empty_ocr:
        mapped_sign = [d for d in mapped_sign if (d.get("ocr_text") or "").strip()]
    if mapped_sign:
        det_xyz = np.array([d["centroid_xyz"] for d in mapped_sign], dtype=np.float32)
        hover_text = [
            (
                f"text: {d['ocr_text'] or '(empty)'}<br>"
                f"class: {d.get('object_class', 'sign')}<br>"
                f"track: {d.get('track_id', -1)}<br>"
                f"sign: {d.get('sign_id', -1)}<br>"
                f"final: {d.get('sign_text_final', d.get('track_text_final', '')) or '(empty)'}<br>"
                f"det: {d.get('detector_source', '-') }<br>"
                f"score: {d['score']:.3f}<br>"
                f"frame: {d['frame_name']}<br>"
                f"points: {d['point_count']}"
            )
            for d in mapped_sign
        ]
        fig.add_trace(
            go.Scatter3d(
                x=det_xyz[:, 0],
                y=det_xyz[:, 1],
                z=det_xyz[:, 2],
                mode="markers",
                marker=dict(size=6.5, color="rgb(214,39,40)", symbol="diamond"),
                name="sign_text_points",
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
    if mapped_door:
        dxyz = np.array([d["centroid_xyz"] for d in mapped_door], dtype=np.float32)
        d_hover = [
            (
                f"class: {d.get('object_class', 'door')}<br>"
                f"det: {d.get('detector_source', '-') }<br>"
                f"score: {d['score']:.3f}<br>"
                f"frame: {d['frame_name']}<br>"
                f"points: {d['point_count']}"
            )
            for d in mapped_door
        ]
        fig.add_trace(
            go.Scatter3d(
                x=dxyz[:, 0],
                y=dxyz[:, 1],
                z=dxyz[:, 2],
                mode="markers",
                marker=dict(size=6.8, color="rgb(46,204,113)", symbol="square"),
                name="door_points",
                hovertext=d_hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    track_labeled = [t for t in final_track_labels if t.get("centroid_xyz") is not None and t.get("final_text")]
    if track_labeled:
        xyz = np.array([t["centroid_xyz"] for t in track_labeled], dtype=np.float32)
        hover = [
            (
                f"sign: {t.get('sign_id', t.get('track_id', -1))}<br>"
                f"label: {t['final_text']}<br>"
                f"votes: {t.get('vote_count', 0)}/{t.get('detections', 0)}<br>"
                f"rep_frame: {t.get('representative_frame', '')}<br>"
                f"frontal: {float(t.get('representative_frontalness', 0.0)):.3f}"
            )
            for t in track_labeled
        ]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers+text",
                marker=dict(size=8.2, color="rgb(31,119,180)", symbol="circle"),
                name="final_labels",
                text=[t["final_text"] for t in track_labeled],
                textposition="top center",
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    centers = []
    center_hover = []
    center_labels = []
    if show_open_set and open_set_objects:
        objs = open_set_objects
        if int(open_set_max_objects) > 0 and len(objs) > int(open_set_max_objects):
            objs = objs[: int(open_set_max_objects)]

        line_x = []
        line_y = []
        line_z = []
        line_hover = []
        edges_idx = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]

        for o in objs:
            c = o.get("center_xyz")
            e = o.get("extent_xyz")
            r = o.get("rotation")
            corners = _obb_corners_world(c, e, r)
            if corners is None:
                continue
            center = np.asarray(c, dtype=np.float64).reshape(3)
            query = str(o.get("query", "object"))
            sem = o.get("semantic_score", None)
            sam = o.get("sam_score", None)
            frame_id = o.get("frame_id", None)
            point_count = o.get("point_count", None)
            hover = (
                f"query: {query}<br>"
                f"semantic: {sem if sem is not None else '-'}<br>"
                f"sam: {sam if sam is not None else '-'}<br>"
                f"frame_id: {frame_id if frame_id is not None else '-'}<br>"
                f"pts: {point_count if point_count is not None else '-'}"
            )

            centers.append(center)
            center_hover.append(hover)
            center_labels.append(query)

            for i, j in edges_idx:
                line_x.extend([corners[i, 0], corners[j, 0], None])
                line_y.extend([corners[i, 1], corners[j, 1], None])
                line_z.extend([corners[i, 2], corners[j, 2], None])
                line_hover.extend([hover, hover, None])

        if line_x:
            fig.add_trace(
                go.Scatter3d(
                    x=line_x,
                    y=line_y,
                    z=line_z,
                    mode="lines",
                    line=dict(color="rgb(255,127,14)", width=float(open_set_line_width)),
                    name="open_set_obb",
                    hovertext=line_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )

    if pose_rows:
        traj_xyz = np.array([[p["tx"], p["ty"], p["tz"]] for p in pose_rows], dtype=np.float32)
        if traj_xyz.shape[0] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=traj_xyz[:, 0],
                    y=traj_xyz[:, 1],
                    z=traj_xyz[:, 2],
                    mode="lines+markers",
                    line=dict(color="rgb(44,160,44)", width=4),
                    marker=dict(size=3.6, color="rgb(44,160,44)"),
                    name="pose_trajectory",
                    hovertemplate="tx:%{x:.3f}<br>ty:%{y:.3f}<br>tz:%{z:.3f}<extra></extra>",
                )
            )

        if centers:
            cxyz = np.asarray(centers, dtype=np.float32)
            fig.add_trace(
                go.Scatter3d(
                    x=cxyz[:, 0],
                    y=cxyz[:, 1],
                    z=cxyz[:, 2],
                    mode="markers+text",
                    marker=dict(size=7.0, color="rgb(255,127,14)", symbol="x"),
                    name="open_set_centers",
                    text=center_labels,
                    textposition="top center",
                    hovertext=center_hover,
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )

    fig.update_layout(
        title="VGGT-SLAM point cloud + SAM3(sign/door) + sign OCR mapping",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(orientation="h", x=0.01, y=0.99),
    )
    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False, config={"responsive": True, "displaylogo": False})

    start_sec = float(video_meta.get("start_sec", 0.0))
    end_sec = video_meta.get("end_sec", None)
    if end_sec is not None and np.isfinite(float(end_sec)):
        range_text = f"{format_seconds_hms(start_sec)} ~ {format_seconds_hms(float(end_sec))}"
    else:
        range_text = f"{format_seconds_hms(start_sec)} ~ -"

    side_items = side_frames
    if int(side_max_frames) > 0 and len(side_items) > int(side_max_frames):
        side_items = side_items[: int(side_max_frames)]
    cards = []
    for it in side_items:
        fid = it.get("frame_id", "-")
        t_hms = it.get("time_hms", "-")
        img_rel = str(it.get("image_rel", "")).strip()
        pose_txt = str(it.get("pose_text", "-")).strip()
        if img_rel:
            img_html = (
                f'<a href="{escape(img_rel)}" target="_blank">'
                f'<img loading="lazy" src="{escape(img_rel)}" alt="frame {escape(str(fid))}"></a>'
            )
        else:
            img_html = '<div class="img-missing">image missing</div>'
        cards.append(
            f"""
            <div class="frame-card">
              <div class="frame-meta">frame {escape(str(fid))} | {escape(t_hms)}</div>
              {img_html}
              <div class="pose-meta">{escape(pose_txt)}</div>
            </div>
            """
        )
    if not cards:
        cards.append('<div class="frame-card"><div class="img-missing">No keyframes to show</div></div>')

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sign Map Viewer</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      min-height: 100%;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background: #f4f6f8;
      color: #111827;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 10px;
      height: 100dvh;
      min-height: 100dvh;
      padding: 10px;
      box-sizing: border-box;
    }}
    .main {{
      background: #ffffff;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid #e5e7eb;
      display: flex;
      min-height: 0;
    }}
    .main > div {{
      flex: 1 1 auto;
      min-height: 0;
    }}
    .main .js-plotly-plot,
    .main .plotly,
    .main .plotly-graph-div {{
      height: 100% !important;
      min-height: 100% !important;
    }}
    .sidebar {{
      background: #ffffff;
      border-radius: 10px;
      border: 1px solid #e5e7eb;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      min-height: 0;
    }}
    .summary {{
      padding: 10px 12px;
      border-bottom: 1px solid #e5e7eb;
      font-size: 13px;
      line-height: 1.5;
      background: #f9fafb;
    }}
    .summary b {{
      color: #111827;
    }}
    .frames {{
      padding: 8px;
      overflow: auto;
      display: grid;
      gap: 8px;
      align-content: start;
    }}
    .frame-card {{
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 6px;
      background: #ffffff;
    }}
    .frame-meta {{
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 6px;
      color: #1f2937;
    }}
    .pose-meta {{
      margin-top: 6px;
      font-size: 11px;
      color: #4b5563;
      line-height: 1.35;
      word-break: break-word;
    }}
    .frame-card img {{
      width: 100%;
      height: auto;
      border-radius: 6px;
      display: block;
      border: 1px solid #e5e7eb;
    }}
    .img-missing {{
      font-size: 12px;
      color: #6b7280;
      padding: 10px 8px;
      background: #f9fafb;
      border: 1px dashed #d1d5db;
      border-radius: 6px;
      text-align: center;
    }}
    @media (max-width: 960px) {{
      .layout {{
        grid-template-columns: 1fr;
        grid-template-rows: minmax(540px, 72dvh) minmax(0, 1fr);
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <div class="main">{fig_html}</div>
    <aside class="sidebar">
      <div class="summary">
        <div><b>Video Range</b>: {escape(range_text)}</div>
        <div><b>Keyframes Used</b>: {len(side_frames)}</div>
        <div><b>Pose Rows</b>: {len(pose_rows)}</div>
        <div><b>Detections</b>: {len(detections)} | <b>Sign</b>: {sum(1 for d in detections if str(d.get('object_class', 'sign')).strip().lower() == 'sign')} | <b>Door</b>: {sum(1 for d in detections if str(d.get('object_class', '')).strip().lower() == 'door')}</div>
        <div><b>Final Sign Labels</b>: {len(final_track_labels)}</div>
        <div><b>Open-set Objects</b>: {len(open_set_objects)}</div>
      </div>
      <div class="frames">
        {''.join(cards)}
      </div>
    </aside>
  </div>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def save_annotated_frame(image: Image.Image, frame_detections, out_path: Path):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for det in frame_detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        obj_cls = str(det.get("object_class", "sign")).strip().lower()
        color = (46, 204, 113) if obj_cls == "door" else (255, 60, 60)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if obj_cls == "door":
            label = "door"
        else:
            label = det["ocr_text"][:24] if det["ocr_text"] else "(empty)"
        tag = f"#{det['det_in_frame']} [{obj_cls}] {label}"
        text_y = max(0, y1 - 16)
        draw.rectangle([x1, text_y, min(x1 + 320, x2 + 220), text_y + 14], fill=color)
        draw.text((x1 + 2, text_y), tag, fill=(255, 255, 255))
    canvas.save(out_path)


def make_detection_report(detections, out_html: Path):
    def img_cell(path, width, alt="-"):
        if not path:
            return alt
        return f"<a href=\"{escape(path)}\"><img src=\"{escape(path)}\" width=\"{int(width)}\"></a>"

    rows = []
    for d in detections:
        crop_size = d.get("crop_size") or [0, 0]
        up_size = d.get("upscaled_crop_size") or [0, 0]
        up_img = d.get("upscaled_crop_path") or ""
        num_roi_img = d.get("number_roi_path") or ""
        num_roi_up_img = d.get("number_roi_upscaled_path") or ""
        crop_size_txt = f"{int(crop_size[0])}x{int(crop_size[1])}" if len(crop_size) >= 2 else "-"
        up_size_txt = "-"
        if len(up_size) >= 2 and int(up_size[0]) > 0 and int(up_size[1]) > 0:
            up_size_txt = f"{int(up_size[0])}x{int(up_size[1])}"
        rows.append(
            "<tr>"
            f"<td>{escape(str(d['frame_name']))}</td>"
            f"<td>{d['det_in_frame']}</td>"
            f"<td>{escape(str(d.get('object_class', 'sign')))}</td>"
            f"<td>{int(d.get('track_id', -1))}</td>"
            f"<td>{int(d.get('sign_id', -1))}</td>"
            f"<td>{d['score']:.3f}</td>"
            f"<td>{'yes' if d['mapped_3d'] else 'no'}</td>"
            f"<td>{d['point_count']}</td>"
            f"<td>{escape(d['ocr_text'] or '')}</td>"
            f"<td>{escape(d.get('ocr_raw_text', '') or '')}</td>"
            f"<td>{escape(d.get('sign_text_final', d.get('track_text_final', '')) or '')}</td>"
            f"<td>{'yes' if d.get('ocr_used_diffusion') else 'no'}</td>"
            f"<td>{crop_size_txt}</td>"
            f"<td>{img_cell(d['crop_path'], 220)}</td>"
            f"<td>{escape(str(d.get('number_roi_bbox_crop') or ''))}</td>"
            f"<td>{escape(str(d.get('number_roi_bbox_frame') or ''))}</td>"
            f"<td>{img_cell(num_roi_img, 200, '-')}</td>"
            f"<td>{up_size_txt}</td>"
            f"<td>{img_cell(up_img, 260, '-')}</td>"
            f"<td>{img_cell(num_roi_up_img, 240, '-')}</td>"
            f"<td>{img_cell(d['annotated_frame_path'], 300)}</td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else "<tr><td colspan='21'>No detections</td></tr>"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>SAM3/DETR Crop + OCR Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 12px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f4f4f4; }}
    img {{ border: 1px solid #ccc; }}
    .meta {{ margin-bottom: 8px; }}
  </style>
</head>
<body>
  <div class="meta">
    <h3>SAM3/DETR Detections + OCR</h3>
    <p>Total detections: {len(detections)}</p>
  </div>
  <table>
    <thead>
      <tr>
        <th>Frame</th>
        <th>Det#</th>
        <th>Class</th>
        <th>Track</th>
        <th>Sign</th>
        <th>Score</th>
        <th>Mapped3D</th>
        <th>Pts</th>
        <th>OCR Text</th>
        <th>Raw OCR</th>
        <th>Final Label</th>
        <th>Diffusion</th>
        <th>Crop Size</th>
        <th>Crop</th>
        <th>NumROI (Crop XYXY)</th>
        <th>NumROI (Frame XYXY)</th>
        <th>Number ROI</th>
        <th>Upscaled Size</th>
        <th>Upscaled Crop</th>
        <th>Upscaled NumROI</th>
        <th>Frame+Box</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    detector_backend = str(args.detector_backend).strip().lower()
    if detector_backend not in {"detr", "sam3", "hybrid"}:
        detector_backend = "hybrid"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform = load_detr_model(args.detr_repo, args.detr_weights, device)
    print(f"Detector backend: {detector_backend}")
    ocr = create_ocr_reader(args, device)
    print(f"OCR backend: {ocr['backend']}")
    engine_names = [e["name"] for e in ocr.get("engines", [])]
    if engine_names:
        print(f"OCR engines: {', '.join(engine_names)}")
    if "tesseract" in engine_names:
        print(f"Tesseract lang: {ocr['tesseract_lang']}")
    if ocr.get("numeric_specialist") is not None:
        print(f"Numeric specialist: {ocr['numeric_specialist'].get('name')}")

    image_paths = [p for p in sorted(args.image_folder.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_folder}")

    npz_files = sorted(args.pointcloud_log_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz pointcloud logs found in {args.pointcloud_log_dir}")

    npz_by_id = {}
    for p in npz_files:
        fid = parse_frame_id(p.stem)
        if fid is not None:
            npz_by_id[fid] = p
    pose_by_frame, pose_rows = load_pose_records(args.pose_log_path)
    open_set_objects = load_open_set_objects(args.open_set_json)
    os_idx_by_frame = build_open_set_index_by_frame(open_set_objects)
    if detector_backend in {"sam3", "hybrid"}:
        print(f"SAM3 open-set objects available: {len(open_set_objects)}")
    frame_path_by_id = {}
    for p in image_paths:
        fid = parse_frame_id(p.stem)
        if fid is not None and fid not in frame_path_by_id:
            frame_path_by_id[fid] = p

    global_points_for_view = []
    detections_out = []
    crop_dir = args.output_dir / "crops"
    num_roi_dir = args.output_dir / "number_rois"
    up_crop_dir = args.output_dir / "crops_upscaled"
    num_roi_up_dir = args.output_dir / "number_rois_upscaled"
    ann_dir = args.output_dir / "annotated_frames"
    crop_dir.mkdir(parents=True, exist_ok=True)
    num_roi_dir.mkdir(parents=True, exist_ok=True)
    up_crop_dir.mkdir(parents=True, exist_ok=True)
    num_roi_up_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    superres_cfg = init_superres(args, device)
    diffusion_cfg = None
    if int(args.diffusion_selective_enabled) == 1:
        try:
            diffusion_cfg = init_superres(args, device, backend_override="diffusion")
        except Exception as e:
            diffusion_cfg = None
            print(f"Selective diffusion disabled (init failed): {e}")
    max_sr_scale = get_max_superres_scale(args)
    print(f"Super-res backend: {superres_cfg['backend']}")
    if diffusion_cfg is not None:
        print(f"Selective diffusion backend: {diffusion_cfg['backend']}")

    processed = 0
    selective_diffusion_used = 0
    for frame_idx, img_path in enumerate(image_paths):
        frame_id = parse_frame_id(img_path.stem)
        if frame_id is None or frame_id not in npz_by_id:
            continue

        npz = np.load(npz_by_id[frame_id], allow_pickle=True)
        if "pointcloud" not in npz or "mask" not in npz:
            continue
        pointcloud = npz["pointcloud"]
        conf_mask = npz["mask"].astype(bool)

        # for global display cloud
        pflat = pointcloud[conf_mask]
        pflat = pflat[np.isfinite(pflat).all(axis=1)]
        if pflat.shape[0] > 0:
            global_points_for_view.append(pflat)

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        dets_detr = []
        dets_sam3 = []
        need_detr = detector_backend in {"detr", "hybrid"} or (
            detector_backend == "sam3" and int(args.sam3_fallback_to_detr_when_empty) == 1
        )
        if need_detr:
            dets_detr = run_detr_inference(
                model,
                transform,
                image,
                sign_label_id=args.sign_label_id,
                det_threshold=args.det_threshold,
            )
        if detector_backend in {"sam3", "hybrid"}:
            dets_sam3 = run_sam3_box_inference(
                frame_id=frame_id,
                img_w=img_w,
                img_h=img_h,
                os_idx_by_frame=os_idx_by_frame,
                args=args,
            )

        if detector_backend == "detr":
            dets = dets_detr
        elif detector_backend == "sam3":
            dets = dets_sam3
            if (not dets) and int(args.sam3_fallback_to_detr_when_empty) == 1:
                dets = dets_detr
        else:
            dets = nms_xyxy(dets_sam3 + dets_detr, iou_thres=float(args.det_merge_iou_thres))

        dets = filter_detections_by_bbox(dets, img_w, img_h, args)
        if args.max_detections_per_frame > 0:
            dets = dets[: args.max_detections_per_frame]

        frame_detections = []
        for det_in_frame, d in enumerate(dets, start=1):
            x1, y1, x2, y2 = d["bbox_xyxy"]
            raw_query = str(d.get("query", "")).strip().lower()
            object_class = raw_query if raw_query else "sign"
            if object_class not in {"sign", "door"}:
                object_class = "sign"
            cx1, cy1, cx2, cy2 = expand_bbox_xyxy(x1, y1, x2, y2, img_w, img_h, ratio=args.crop_expand_ratio)
            crop = image.crop((cx1, cy1, cx2, cy2))
            crop_name = f"{img_path.stem}_det{det_in_frame:02d}.png"
            crop_path = crop_dir / crop_name
            crop.save(crop_path)
            num_roi = None
            roi_for_ocr = None
            if object_class == "sign":
                if int(args.ocr_numeric_roi_enabled) == 1 and args.ocr_numeric_mode:
                    ocr_out, num_roi, used_diffusion, selective_diffusion_used, roi_for_ocr = run_ocr_with_numeric_roi(
                        crop,
                        d,
                        ocr,
                        args,
                        superres_cfg=superres_cfg,
                        diffusion_cfg=diffusion_cfg,
                        selective_diffusion_used=selective_diffusion_used,
                    )
                else:
                    ocr_out, _, _, _, used_diffusion, selective_diffusion_used, _, _ = run_ocr_with_precheck(
                        crop,
                        d,
                        ocr,
                        args,
                        superres_cfg=superres_cfg,
                        diffusion_cfg=diffusion_cfg,
                        selective_diffusion_used=selective_diffusion_used,
                        roi_text_hint="",
                    )
            else:
                # Door detections are geometry/semantic targets; OCR is sign-specific.
                ocr_out = {"ocr_text": "", "ocr_raw_text": "", "ocr_numeric_text": ""}
                used_diffusion = False

            has_numeric_text = bool((ocr_out.get("ocr_numeric_text") or "").strip())
            upscaled_preview = None
            up_crop_path_rel = ""
            upscaled_crop_size = [0, 0]
            upscaled_preview_enabled = (
                object_class == "sign"
                and
                has_numeric_text
                and float(max_sr_scale) > 1.0
                and superres_cfg is not None
                and str(superres_cfg.get("backend", "bicubic")) != "none"
            )
            if upscaled_preview_enabled:
                up_crop_name = f"{img_path.stem}_det{det_in_frame:02d}_x{max_sr_scale:g}.png"
                up_crop_path = up_crop_dir / up_crop_name
                upscaled_preview = upscale_crop_for_ocr(crop, superres_cfg, args, outscale=max_sr_scale)
                if upscaled_preview.size != crop.size:
                    upscaled_preview.save(up_crop_path)
                    up_crop_path_rel = f"crops_upscaled/{up_crop_name}"
                    upscaled_crop_size = [int(upscaled_preview.width), int(upscaled_preview.height)]

            number_roi_bbox_crop = None
            number_roi_bbox_frame = None
            number_roi_path_rel = ""
            number_roi_up_path_rel = ""
            number_roi_size = [0, 0]
            number_roi_upscaled_size = [0, 0]
            if num_roi is not None:
                rx1, ry1, rx2, ry2 = [int(v) for v in num_roi["bbox_xyxy"]]
                roi_crop = crop.crop((rx1, ry1, rx2, ry2))
                number_roi_bbox_crop = [rx1, ry1, rx2, ry2]
                number_roi_bbox_frame = [int(cx1 + rx1), int(cy1 + ry1), int(cx1 + rx2), int(cy1 + ry2)]
                roi_name = f"{img_path.stem}_det{det_in_frame:02d}_numroi.png"
                roi_path = num_roi_dir / roi_name
                roi_crop.save(roi_path)
                number_roi_path_rel = f"number_rois/{roi_name}"
                number_roi_size = [int(roi_crop.width), int(roi_crop.height)]
                if upscaled_preview_enabled:
                    roi_up_name = f"{img_path.stem}_det{det_in_frame:02d}_numroi_x{max_sr_scale:g}.png"
                    roi_up_path = num_roi_up_dir / roi_up_name
                    roi_up = roi_for_ocr if roi_for_ocr is not None else upscale_crop_for_ocr(
                        roi_crop, superres_cfg, args, outscale=max_sr_scale
                    )
                    if roi_up.size != roi_crop.size:
                        roi_up.save(roi_up_path)
                        number_roi_up_path_rel = f"number_rois_upscaled/{roi_up_name}"
                        number_roi_upscaled_size = [int(roi_up.width), int(roi_up.height)]

            pts = map_bbox_to_points(pointcloud, conf_mask, img_w, img_h, d["bbox_xyxy"])
            mapped_3d = pts is not None and pts.shape[0] >= 15
            centroid = np.median(pts, axis=0) if mapped_3d else None
            point_count = int(pts.shape[0]) if pts is not None else 0

            rec = {
                "frame_name": img_path.name,
                "frame_idx": int(frame_idx),
                "frame_id": frame_id,
                "det_in_frame": det_in_frame,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "bbox_w": int(d.get("bbox_w", x2 - x1)),
                "bbox_h": int(d.get("bbox_h", y2 - y1)),
                "bbox_area_ratio": float(d.get("bbox_area_ratio", ((x2 - x1) * (y2 - y1)) / float(img_w * img_h))),
                "bbox_center_norm": [
                    float(((x1 + x2) * 0.5) / float(max(1, img_w))),
                    float(((y1 + y2) * 0.5) / float(max(1, img_h))),
                ],
                "object_class": object_class,
                "score": float(d["score"]),
                "detector_source": str(d.get("source", detector_backend)),
                "query": raw_query,
                "sam_score": float(d.get("sam_score", 0.0)),
                "semantic_score": float(d.get("semantic_score", 0.0)),
                "ocr_text": ocr_out["ocr_text"],
                "ocr_raw_text": ocr_out["ocr_raw_text"],
                "ocr_numeric_text": ocr_out["ocr_numeric_text"],
                "mapped_3d": mapped_3d,
                "centroid_xyz": [float(centroid[0]), float(centroid[1]), float(centroid[2])] if mapped_3d else None,
                "point_count": point_count,
                "crop_size": [int(crop.width), int(crop.height)],
                "upscaled_crop_size": upscaled_crop_size,
                "crop_path": f"crops/{crop_name}",
                "upscaled_crop_path": up_crop_path_rel,
                "number_roi_bbox_crop": number_roi_bbox_crop,
                "number_roi_bbox_frame": number_roi_bbox_frame,
                "number_roi_text_hint": (num_roi.get("text_hint", "") if num_roi is not None else ""),
                "number_roi_size": number_roi_size,
                "number_roi_upscaled_size": number_roi_upscaled_size,
                "number_roi_path": number_roi_path_rel,
                "number_roi_upscaled_path": number_roi_up_path_rel,
                "ocr_used_diffusion": bool(used_diffusion),
                "annotated_frame_path": "",
            }
            detections_out.append(rec)
            frame_detections.append(rec)

        if frame_detections:
            ann_name = f"{img_path.stem}_annotated.png"
            ann_path = ann_dir / ann_name
            save_annotated_frame(image, frame_detections, ann_path)
            for rec in frame_detections:
                rec["annotated_frame_path"] = f"annotated_frames/{ann_name}"

        processed += 1
        if processed % 20 == 0:
            print(f"processed {processed} frames")

    # Sign-only tracking/labeling. Door detections stay untracked and unlabeled for OCR.
    for d in detections_out:
        d["track_id"] = -1
        d["sign_id"] = -1
        d["track_text_final"] = ""
        d["sign_text_final"] = ""

    sign_detections = [d for d in detections_out if str(d.get("object_class", "sign")).strip().lower() == "sign"]
    if sign_detections:
        assign_track_ids(sign_detections, args)
        raw_track_labels = finalize_track_labels(sign_detections, args)
        final_track_labels = merge_track_labels(raw_track_labels, sign_detections, args)
        final_track_labels = unify_nearby_final_labels(final_track_labels, args)
    else:
        raw_track_labels = []
        final_track_labels = []
    tid_to_sign = {}
    sign_text_by_id = {}
    for s in final_track_labels:
        sid = int(s.get("sign_id", -1))
        txt = (s.get("final_text") or "").strip()
        sign_text_by_id[sid] = txt
        for tid in s.get("merged_track_ids", []):
            tid_to_sign[int(tid)] = sid
    for d in detections_out:
        tid = int(d.get("track_id", -1))
        sid = tid_to_sign.get(tid, -1)
        d["sign_id"] = int(sid)
        d["sign_text_final"] = sign_text_by_id.get(sid, "")
        if not d.get("track_text_final"):
            d["track_text_final"] = d["sign_text_final"]

    view_points = build_pointcloud_view(global_points_for_view, args.viewer_max_points, rng)
    used_frame_ids = sorted(npz_by_id.keys())
    clip_start_sec = float(args.video_start_seconds)
    clip_fps = float(args.video_fps)
    clip_duration_sec = float(args.video_duration_seconds)
    side_frames = []
    for fid in used_frame_ids:
        ipath = frame_path_by_id.get(fid, None)
        rel_path = ""
        if ipath is not None:
            try:
                rel_path = os.path.relpath(ipath, args.output_dir).replace("\\", "/")
            except Exception:
                rel_path = str(ipath)
        ts_sec = None
        ts_hms = "-"
        if clip_fps > 0:
            ts_sec = clip_start_sec + max(0.0, float(fid) - 1.0) / clip_fps
            ts_hms = format_seconds_hms(ts_sec)
        pose = pose_by_frame.get(fid, None)
        if pose is not None:
            pose_txt = (
                f"t=({pose['tx']:.3f}, {pose['ty']:.3f}, {pose['tz']:.3f}) "
                f"q=({pose['qx']:.3f}, {pose['qy']:.3f}, {pose['qz']:.3f}, {pose['qw']:.3f})"
            )
        else:
            pose_txt = "-"
        side_frames.append(
            {
                "frame_id": fid,
                "frame_name": (ipath.name if ipath is not None else f"frame_{int(fid):06d}.jpg"),
                "image_rel": rel_path,
                "time_sec": ts_sec,
                "time_hms": ts_hms,
                "pose_text": pose_txt,
            }
        )
    if clip_duration_sec > 0:
        clip_end_sec = clip_start_sec + clip_duration_sec
    elif clip_fps > 0 and used_frame_ids:
        clip_end_sec = clip_start_sec + max(0.0, float(max(used_frame_ids)) - 1.0) / clip_fps
    else:
        clip_end_sec = None
    video_meta = {"start_sec": clip_start_sec, "end_sec": clip_end_sec}

    out_json = args.output_dir / "sign_3d_detections.json"
    out_track_json = args.output_dir / "sign_track_labels.json"
    out_track_raw_json = args.output_dir / "sign_track_labels_raw.json"
    out_html = args.output_dir / "viewer.html"
    out_report = args.output_dir / "ocr_report.html"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(detections_out, f, indent=2)
    with out_track_json.open("w", encoding="utf-8") as f:
        json.dump(final_track_labels, f, indent=2)
    with out_track_raw_json.open("w", encoding="utf-8") as f:
        json.dump(raw_track_labels, f, indent=2)

    make_plotly_view(
        view_points,
        detections_out,
        final_track_labels,
        open_set_objects,
        out_html,
        marker_size=args.viewer_marker_size,
        hide_empty_ocr=bool(int(args.viewer_hide_empty_ocr)),
        show_open_set=bool(int(args.viewer_show_open_set)),
        open_set_max_objects=int(args.viewer_open_set_max_objects),
        open_set_line_width=float(args.viewer_open_set_line_width),
        pose_rows=pose_rows,
        side_frames=side_frames,
        video_meta=video_meta,
        side_max_frames=int(args.viewer_side_max_frames),
    )
    make_detection_report(detections_out, out_report)

    print(f"frames scanned: {processed}")
    mapped_count = sum(1 for d in detections_out if d["mapped_3d"])
    sign_count = sum(1 for d in detections_out if str(d.get("object_class", "sign")).strip().lower() == "sign")
    door_count = sum(1 for d in detections_out if str(d.get("object_class", "")).strip().lower() == "door")
    numeric_non_empty = sum(1 for d in detections_out if (d.get("ocr_numeric_text") or "").strip())
    final_labels_count = len(final_track_labels)
    print(f"detections total: {len(detections_out)}")
    print(f"detections sign: {sign_count}")
    print(f"detections door: {door_count}")
    print(f"detections mapped: {mapped_count}")
    print(f"numeric non-empty: {numeric_non_empty}")
    if int(args.diffusion_selective_enabled) == 1:
        print(f"selective diffusion used: {selective_diffusion_used}")
    print(f"finalized track labels: {final_labels_count}")
    print(f"open-set objects loaded: {len(open_set_objects)}")
    print(f"saved json: {out_json}")
    print(f"saved track labels: {out_track_json}")
    print(f"saved raw track labels: {out_track_raw_json}")
    print(f"saved html: {out_html}")
    print(f"saved report: {out_report}")


if __name__ == "__main__":
    main()
    def detection_vote_weight(d, canon_text):
        w = float(d.get("score", 0.0)) + min(len(canon_text), 12) * 0.05
        if d.get("number_roi_bbox_crop") is None:
            w *= 0.75
        hint = (d.get("number_roi_text_hint") or "").strip()
        hint_num = ""
        if hint:
            hc = extract_numeric_candidates(hint, args)
            hint_num = hc[0] if hc else ""
        if bool(d.get("ocr_used_diffusion")):
            if hint_num:
                if canon_text.replace(" ", "") == hint_num.replace(" ", ""):
                    w *= 1.05
                else:
                    w *= 0.50
            else:
                w *= 0.70
        return float(w)
