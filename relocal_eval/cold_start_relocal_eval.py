#!/usr/bin/env python3
"""
Cold-start relocalization evaluation on TUM RGB-D:

VPR retrieval + LoGeR geometric verification + pose evaluation.

This script is intentionally modular so each stage can be replaced independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TUM_ROOT = Path(
    os.environ.get("TUM_ROOT", "/workspace/VGGT-SLAM-mixed-20260306_133724/datasets/tum")
)


def _ensure_loger_importable(loger_repo: Path) -> None:
    import sys

    repo = str(loger_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


@dataclass
class TumFrame:
    seq_name: str
    frame_idx: int
    timestamp: float
    rgb_path: str
    pose_w_c: List[List[float]]


@dataclass
class EvalRecord:
    sequence: str
    query_idx: int
    query_timestamp: float
    query_rgb: str
    status: str
    best_candidate_idx: Optional[int]
    best_candidate_rgb: Optional[str]
    score: Optional[float]
    match_count: Optional[int]
    translation_error_m: Optional[float]
    rotation_error_deg: Optional[float]
    success: bool
    map_size: Optional[int] = None
    map_exclude_start: Optional[int] = None
    map_exclude_end: Optional[int] = None
    topk_candidates: Optional[List[int]] = None
    query_jump_translation_m: Optional[float] = None
    query_jump_rotation_deg: Optional[float] = None
    baseline_vpr_translation_error_m: Optional[float] = None
    baseline_vpr_rotation_error_deg: Optional[float] = None
    baseline_random_translation_error_m: Optional[float] = None
    baseline_random_rotation_error_deg: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold-start relocalization evaluation (VPR + LoGeR)")
    parser.add_argument(
        "--tum_root",
        type=Path,
        default=DEFAULT_TUM_ROOT,
        help="Root directory that contains rgbd_dataset_freiburg*/ folders.",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        default="freiburg1_room,freiburg2_desk,freiburg2_360_kidnap",
        help="Comma-separated sequence short names.",
    )
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_radius", type=int, default=2)
    parser.add_argument(
        "--map_exclude_radius",
        type=int,
        default=0,
        help="Exclude query +/- this many frames from the retrieval map (kidnap-like hold-out).",
    )
    parser.add_argument(
        "--query_mode",
        type=str,
        default="random",
        choices=["random", "kidnap_jump"],
        help="Query sampling strategy: random or motion-jump-heavy (kidnap-like).",
    )
    parser.add_argument(
        "--kidnap_min_translation_m",
        type=float,
        default=0.20,
        help="In kidnap_jump mode, prefer frames where jump translation from previous frame exceeds this.",
    )
    parser.add_argument(
        "--kidnap_min_rotation_deg",
        type=float,
        default=12.0,
        help="In kidnap_jump mode, prefer frames where jump rotation from previous frame exceeds this.",
    )
    parser.add_argument("--assoc_tolerance_sec", type=float, default=0.03)
    parser.add_argument(
        "--descriptor_model",
        type=str,
        default="resnet18",
        choices=["resnet18", "dinov2_vits14", "salad"],
    )
    parser.add_argument("--descriptor_batch_size", type=int, default=32)
    parser.add_argument(
        "--salad_root",
        type=Path,
        default=Path(
            os.environ.get(
                "SALAD_ROOT",
                "/workspace/Depth-Anything-3-ori/da3_streaming/loop_utils/salad",
            )
        ),
        help="Local SALAD source root (contains models/).",
    )
    parser.add_argument(
        "--salad_ckpt_path",
        type=Path,
        default=Path(
            os.environ.get(
                "SALAD_CKPT_PATH",
                "/workspace/Depth-Anything-3-ori/da3_streaming/weights/dino_salad.ckpt",
            )
        ),
        help="Path to dino_salad checkpoint.",
    )
    parser.add_argument("--salad_image_size", type=int, default=322)
    parser.add_argument("--salad_num_trainable_blocks", type=int, default=4)
    parser.add_argument(
        "--descriptor_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for descriptor extraction.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "runs" / "relocal_eval_tum")
    parser.add_argument("--cache_dir", type=Path, default=PROJECT_ROOT / "runs" / "relocal_eval_tum" / "cache")

    parser.add_argument("--loger_repo", type=Path, default=PROJECT_ROOT / "LoGeR")
    parser.add_argument(
        "--loger_weights",
        type=Path,
        default=PROJECT_ROOT / "LoGeR" / "ckpts" / "LoGeR_star" / "latest.pt",
    )
    parser.add_argument(
        "--loger_config",
        type=Path,
        default=PROJECT_ROOT / "LoGeR" / "ckpts" / "LoGeR_star" / "original_config.yaml",
    )
    parser.add_argument("--loger_window_size", type=int, default=32)
    parser.add_argument("--loger_overlap_size", type=int, default=3)
    parser.add_argument("--loger_reset_every", type=int, default=0)
    parser.add_argument("--loger_num_iterations", type=int, default=1)
    parser.add_argument(
        "--loger_ttt",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Enable/disable LoGeR test-time tuning.",
    )
    parser.add_argument("--loger_use_sim3", type=int, default=1, choices=[0, 1], help="Forward sim3 flag to LoGeR.")
    parser.add_argument("--loger_use_se3", type=int, default=0, choices=[0, 1], help="Forward se3 flag to LoGeR.")
    parser.add_argument(
        "--loger_device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for LoGeR verification inference.",
    )
    parser.add_argument(
        "--loger_patch_multiple",
        type=int,
        default=14,
        help="Resize LoGeR input images so H/W are multiples of this value.",
    )

    parser.add_argument("--w1", type=float, default=1.0, help="Weight for reprojection-like 3D consistency error.")
    parser.add_argument("--w2", type=float, default=1.0, help="Weight for depth consistency error.")
    parser.add_argument("--w3", type=float, default=1.0, help="Weight for scale variance.")

    parser.add_argument("--translation_success_m", type=float, default=0.10)
    parser.add_argument("--rotation_success_deg", type=float, default=5.0)
    parser.add_argument("--max_queries_per_sequence", type=int, default=0, help="0 means no cap.")
    parser.add_argument(
        "--max_frames_per_sequence",
        type=int,
        default=0,
        help="Optional cap on frames per sequence (uniformly subsampled). 0 means full sequence.",
    )
    parser.add_argument("--run_baselines", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_example_pairs", type=int, default=12)
    return parser.parse_args()


def parse_tum_file(path: Path) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            ts = float(parts[0])
        except Exception:
            continue
        out.append((ts, parts[1]))
    out.sort(key=lambda x: x[0])
    return out


def parse_groundtruth(path: Path) -> List[Tuple[float, np.ndarray]]:
    out: List[Tuple[float, np.ndarray]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        try:
            ts = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
        except Exception:
            continue
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = quat_xyzw_to_rot(qx, qy, qz, qw)
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        out.append((ts, T))
    out.sort(key=lambda x: x[0])
    return out


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = q / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def find_nearest_pose(ts: float, gt: Sequence[Tuple[float, np.ndarray]], tol: float) -> Optional[np.ndarray]:
    if not gt:
        return None
    times = [x[0] for x in gt]
    i = int(np.searchsorted(times, ts))
    cand: List[Tuple[float, np.ndarray]] = []
    if 0 <= i < len(gt):
        cand.append(gt[i])
    if i - 1 >= 0:
        cand.append(gt[i - 1])
    if not cand:
        return None
    best = min(cand, key=lambda x: abs(x[0] - ts))
    if abs(best[0] - ts) > tol:
        return None
    return best[1]


def load_tum_sequence(tum_root: Path, sequence_short_name: str, assoc_tol: float) -> List[TumFrame]:
    seq_dir = tum_root / f"rgbd_dataset_{sequence_short_name}"
    rgb_txt = seq_dir / "rgb.txt"
    gt_txt = seq_dir / "groundtruth.txt"
    rgb_rows = parse_tum_file(rgb_txt)
    gt_rows = parse_groundtruth(gt_txt)
    frames: List[TumFrame] = []
    for i, (ts, rel_rgb) in enumerate(rgb_rows):
        pose = find_nearest_pose(ts, gt_rows, assoc_tol)
        if pose is None:
            continue
        rgb_path = (seq_dir / rel_rgb).resolve()
        if not rgb_path.exists():
            continue
        frames.append(
            TumFrame(
                seq_name=sequence_short_name,
                frame_idx=i,
                timestamp=ts,
                rgb_path=str(rgb_path),
                pose_w_c=pose.tolist(),
            )
        )
    return frames


class DescriptorExtractor:
    def __init__(
        self,
        model_name: str = "resnet18",
        device: str = "cpu",
        salad_root: Optional[Path] = None,
        salad_ckpt_path: Optional[Path] = None,
        salad_image_size: int = 322,
        salad_num_trainable_blocks: int = 4,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.salad_root = Path(salad_root) if salad_root is not None else None
        self.salad_ckpt_path = Path(salad_ckpt_path) if salad_ckpt_path is not None else None
        self.salad_image_size = int(salad_image_size)
        self.salad_num_trainable_blocks = int(salad_num_trainable_blocks)
        self.desc_dim = 512
        self._init_model()

    def _init_model(self) -> None:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.model_name == "salad":
            path_added = False
            salad_root_s = ""
            try:
                salad_root = self.salad_root
                if salad_root is None:
                    raise ValueError("salad_root is required when descriptor_model=salad")
                if not salad_root.exists():
                    raise FileNotFoundError(f"SALAD root not found: {salad_root}")
                salad_root_s = str(salad_root.resolve())
                if salad_root_s not in sys.path:
                    sys.path.insert(0, salad_root_s)
                    path_added = True
                from models import helper as salad_helper

                class _SaladVPRModel(torch.nn.Module):
                    def __init__(self, num_trainable_blocks: int = 4) -> None:
                        super().__init__()
                        self.backbone = salad_helper.get_backbone(
                            "dinov2_vitb14",
                            {
                                "num_trainable_blocks": int(num_trainable_blocks),
                                "return_token": True,
                                "norm_layer": True,
                            },
                        )
                        self.aggregator = salad_helper.get_aggregator(
                            "SALAD",
                            {
                                "num_channels": 768,
                                "num_clusters": 64,
                                "cluster_dim": 128,
                                "token_dim": 256,
                            },
                        )

                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        return self.aggregator(self.backbone(x))

                model = _SaladVPRModel(num_trainable_blocks=self.salad_num_trainable_blocks)
                ckpt = self.salad_ckpt_path
                if ckpt is None:
                    raise ValueError("salad_ckpt_path is required when descriptor_model=salad")
                if not ckpt.exists():
                    raise FileNotFoundError(f"SALAD checkpoint not found: {ckpt}")
                state = torch.load(str(ckpt), map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                    raw_state = state["state_dict"]
                else:
                    raw_state = state
                cleaned = {}
                for k, v in raw_state.items():
                    if k.startswith("module."):
                        cleaned[k[7:]] = v
                    else:
                        cleaned[k] = v
                model.load_state_dict(cleaned, strict=True)
                self.model = model.to(self.device).eval()
                self.preprocess = transforms.Compose(
                    [
                        transforms.Resize((self.salad_image_size, self.salad_image_size)),
                        transforms.ToTensor(),
                        norm,
                    ]
                )
                self._mode = "salad"
                self.desc_dim = 8448
                print(f"[info] Loaded SALAD descriptor model from {ckpt}")
                if path_added and salad_root_s in sys.path:
                    sys.path.remove(salad_root_s)
                return
            except Exception as e:
                if path_added and salad_root_s in sys.path:
                    sys.path.remove(salad_root_s)
                print(f"[warn] Failed to load SALAD ({e}). Falling back to DINOv2/ResNet.")

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                norm,
            ]
        )
        if self.model_name == "dinov2_vits14":
            try:
                model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
                self.model = model.to(self.device).eval()
                self._mode = "dinov2"
                self.desc_dim = 384
                return
            except Exception as e:
                print(f"[warn] Failed to load DINOv2 ({e}). Falling back to resnet18.")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        self.model = model.to(self.device).eval()
        self._mode = "resnet18"
        self.desc_dim = 512

    @torch.no_grad()
    def encode_paths(self, image_paths: Sequence[str], batch_size: int = 32) -> np.ndarray:
        descs: List[np.ndarray] = []
        for st in range(0, len(image_paths), batch_size):
            chunk = image_paths[st : st + batch_size]
            imgs = []
            for p in chunk:
                im = Image.open(p).convert("RGB")
                imgs.append(self.preprocess(im))
            x = torch.stack(imgs, dim=0).to(self.device, non_blocking=True)
            feat = self.model(x)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = feat.float()
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            descs.append(feat.detach().cpu().numpy())
        return np.concatenate(descs, axis=0) if descs else np.zeros((0, int(self.desc_dim)), dtype=np.float32)


def load_or_compute_descriptors(
    frames: Sequence[TumFrame],
    extractor: DescriptorExtractor,
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    image_paths = [f.rgb_path for f in frames]
    cache_ok = False
    if cache_path.exists():
        try:
            payload = np.load(cache_path, allow_pickle=True)
            cached_paths = payload["rgb_paths"].tolist()
            if cached_paths == image_paths:
                desc = payload["descriptors"].astype(np.float32)
                cache_ok = True
                print(f"[cache] loaded {cache_path} ({desc.shape[0]} frames)")
                return desc
        except Exception:
            cache_ok = False
    if not cache_ok:
        desc = extractor.encode_paths(image_paths, batch_size=batch_size).astype(np.float32)
        np.savez_compressed(cache_path, descriptors=desc, rgb_paths=np.array(image_paths, dtype=object))
        print(f"[cache] wrote {cache_path} ({desc.shape[0]} frames)")
        return desc
    return np.zeros((0, int(getattr(extractor, "desc_dim", 512))), dtype=np.float32)


def compute_pose_jumps(frames: Sequence[TumFrame]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(frames)
    trans = np.zeros((n,), dtype=np.float64)
    rot = np.zeros((n,), dtype=np.float64)
    if n <= 1:
        return trans, rot
    I = np.eye(3, dtype=np.float64)
    for i in range(1, n):
        T_prev = np.array(frames[i - 1].pose_w_c, dtype=np.float64)
        T_cur = np.array(frames[i].pose_w_c, dtype=np.float64)
        T_rel = invert_T(T_prev) @ T_cur
        trans[i] = float(np.linalg.norm(T_rel[:3, 3]))
        rot[i] = float(rotation_error_deg(T_rel[:3, :3], I))
    return trans, rot


def build_valid_query_indices(total: int, map_exclude_radius: int) -> List[int]:
    valid: List[int] = []
    for qi in range(total):
        lo = max(0, qi - map_exclude_radius)
        hi = min(total - 1, qi + map_exclude_radius)
        map_size = total - (hi - lo + 1)
        if map_size > 0:
            valid.append(qi)
    return valid


def sample_query_indices(
    frames: Sequence[TumFrame],
    n_queries: int,
    radius: int,
    map_exclude_radius: int,
    seed: int,
    mode: str,
    kidnap_min_translation_m: float,
    kidnap_min_rotation_deg: float,
) -> List[int]:
    total = len(frames)
    _ = radius  # reserved for compatibility; sampling itself is map-window based.
    candidates = build_valid_query_indices(total, map_exclude_radius=max(0, int(map_exclude_radius)))
    if not candidates:
        return []
    rnd = random.Random(seed)
    if mode == "random":
        rnd.shuffle(candidates)
        return sorted(candidates[: min(n_queries, len(candidates))])

    jump_t, jump_r = compute_pose_jumps(frames)
    preferred: List[Tuple[float, int]] = []
    fallback: List[Tuple[float, int]] = []
    for qi in candidates:
        score = float(jump_t[qi] + (jump_r[qi] / 45.0))
        if jump_t[qi] >= kidnap_min_translation_m or jump_r[qi] >= kidnap_min_rotation_deg:
            preferred.append((score, qi))
        else:
            fallback.append((score, qi))

    rnd.shuffle(preferred)
    rnd.shuffle(fallback)
    preferred.sort(key=lambda x: x[0], reverse=True)
    fallback.sort(key=lambda x: x[0], reverse=True)
    picked = [qi for _, qi in preferred[: min(n_queries, len(preferred))]]
    if len(picked) < n_queries:
        rest = [qi for _, qi in fallback]
        picked.extend(rest[: max(0, n_queries - len(picked))])
    return sorted(set(picked))


def subsample_frames_uniform(frames: Sequence[TumFrame], max_frames: int) -> List[TumFrame]:
    n = len(frames)
    if max_frames <= 0 or n <= max_frames:
        return list(frames)
    idxs = np.linspace(0, n - 1, num=max_frames, dtype=np.int64)
    uniq = sorted(set(int(i) for i in idxs.tolist()))
    return [frames[i] for i in uniq]


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    d = R_est.T @ R_gt
    tr = float(np.trace(d))
    c = (tr - 1.0) * 0.5
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def compose_T(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B


def resolve_device(requested: str) -> str:
    req = str(requested).lower().strip()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    if req not in {"cpu", "cuda"}:
        return "cpu"
    return req


def _resize_to_multiple(img: Image.Image, multiple: int) -> Image.Image:
    if multiple <= 1:
        return img
    w, h = img.size
    nw = max(multiple, (w // multiple) * multiple)
    nh = max(multiple, (h // multiple) * multiple)
    if nw == w and nh == h:
        return img
    return img.resize((nw, nh), Image.Resampling.BICUBIC)


def _load_loger_image(rgb_path: str, patch_multiple: int) -> Tuple[torch.Tensor, np.ndarray]:
    img = Image.open(rgb_path).convert("RGB")
    img = _resize_to_multiple(img, int(patch_multiple))
    arr_u8 = np.asarray(img, dtype=np.uint8)
    arr_f = arr_u8.astype(np.float32) / 255.0
    t = torch.from_numpy(arr_f).permute(2, 0, 1).contiguous()
    return t, arr_u8


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    if src.shape[0] < 3 or dst.shape[0] < 3:
        return None
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst
    cov = (Y.T @ X) / float(src.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    var_src = float((X * X).sum() / src.shape[0])
    if var_src < 1e-12:
        return None
    scale = float(np.trace(np.diag(S) @ D) / var_src)
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def orb_match_pairs(img_a_rgb: np.ndarray, img_b_rgb: np.ndarray, nfeatures: int = 1200) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    if img_a_rgb is None or img_b_rgb is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    if img_a_rgb.ndim == 3:
        img_a = cv2.cvtColor(img_a_rgb, cv2.COLOR_RGB2GRAY)
    else:
        img_a = img_a_rgb
    if img_b_rgb.ndim == 3:
        img_b = cv2.cvtColor(img_b_rgb, cv2.COLOR_RGB2GRAY)
    else:
        img_b = img_b_rgb
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kpa, desa = orb.detectAndCompute(img_a, None)
    kpb, desb = orb.detectAndCompute(img_b, None)
    if desa is None or desb is None or len(kpa) < 8 or len(kpb) < 8:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desa, desb)
    if not matches:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[: min(300, len(matches))]
    pa = np.array([kpa[m.queryIdx].pt for m in matches], dtype=np.float32)
    pb = np.array([kpb[m.trainIdx].pt for m in matches], dtype=np.float32)
    return pa, pb


class LoGeRVerifier:
    def __init__(
        self,
        loger_repo: Path,
        weights_path: Path,
        config_path: Path,
        window_size: int,
        overlap_size: int,
        reset_every: int,
        num_iterations: int,
        patch_multiple: int,
        turn_off_ttt: bool,
        use_sim3: bool,
        use_se3: bool,
        device: str = "cpu",
    ) -> None:
        _ensure_loger_importable(loger_repo)
        from eval.pi3_adapter import load_pi3_model, merge_forward_kwargs, run_pi3_inference_on_views

        self._run_pi3_inference_on_views = run_pi3_inference_on_views
        self.device = torch.device(device)
        self.model, base_fw = load_pi3_model(
            weights_path=str(weights_path),
            config_path=str(config_path),
            device=self.device,
            strict=False,
        )
        self.forward_kwargs = merge_forward_kwargs(
            base_fw,
            {
                "window_size": int(window_size),
                "overlap_size": int(overlap_size),
                "reset_every": int(reset_every),
                "num_iterations": int(num_iterations),
                "turn_off_ttt": bool(turn_off_ttt),
                "sim3": bool(use_sim3),
                "se3": bool(use_se3),
            },
        )
        self.patch_multiple = int(max(1, patch_multiple))

    def evaluate_candidate(
        self,
        frames: Sequence[TumFrame],
        query_idx: int,
        cand_idx: int,
        radius: int,
        w1: float,
        w2: float,
        w3: float,
    ) -> Dict[str, object]:
        chunk_ids = list(range(max(0, cand_idx - radius), min(len(frames), cand_idx + radius + 1)))
        if query_idx not in chunk_ids:
            chunk_ids.append(query_idx)
        chunk_ids = sorted(set(chunk_ids))
        if len(chunk_ids) < 2:
            return {"valid": False, "reason": "chunk_too_small"}

        local_pos = {idx: i for i, idx in enumerate(chunk_ids)}
        i_c = local_pos[cand_idx]
        i_q = local_pos[query_idx]

        preloaded: Dict[int, Tuple[torch.Tensor, np.ndarray]] = {
            idx: _load_loger_image(frames[idx].rgb_path, self.patch_multiple) for idx in chunk_ids
        }
        views = [{"img": preloaded[idx][0]} for idx in chunk_ids]
        _, seq_out = self._run_pi3_inference_on_views(
            self.model,
            views,
            forward_kwargs=self.forward_kwargs,
            device=self.device,
            output_device=torch.device("cpu"),
        )

        poses = seq_out.camera_poses.detach().cpu().numpy().astype(np.float64)  # (T,4,4)
        lp = seq_out.local_points.detach().cpu().numpy().astype(np.float64)  # (T,H,W,3)

        T_wc = poses[i_c]
        T_wq = poses[i_q]
        T_cq = invert_T(T_wc) @ T_wq
        R_cq = T_cq[:3, :3]
        t_cq = T_cq[:3, 3]

        def fallback(reason: str, match_count: int = 0) -> Dict[str, object]:
            # Keep candidate usable with a large penalty so ranking still works,
            # and downstream can still evaluate pose from LoGeR relative SE3.
            return {
                "valid": True,
                "fallback": True,
                "reason": reason,
                "score": 1e6 + float(max(0, match_count)),
                "match_count": int(match_count),
                "reprojection_error": 1e6,
                "depth_error": 1e6,
                "scale_variance": 1e6,
                "scale": 1.0,
                "T_cq_se3": T_cq,
                "sim3": None,
                "candidate_local_index": int(i_c),
                "query_local_index": int(i_q),
            }

        kp_c, kp_q = orb_match_pairs(preloaded[cand_idx][1], preloaded[query_idx][1])
        if kp_c.shape[0] < 8:
            return fallback("few_matches", int(kp_c.shape[0]))

        Hc, Wc = lp[i_c].shape[:2]
        Hq, Wq = lp[i_q].shape[:2]

        pts_c = []
        pts_q = []
        for (uc, vc), (uq, vq) in zip(kp_c, kp_q):
            xc = int(round(float(uc)))
            yc = int(round(float(vc)))
            xq = int(round(float(uq)))
            yq = int(round(float(vq)))
            if not (0 <= xc < Wc and 0 <= yc < Hc and 0 <= xq < Wq and 0 <= yq < Hq):
                continue
            pc = lp[i_c, yc, xc]
            pq = lp[i_q, yq, xq]
            if not (np.isfinite(pc).all() and np.isfinite(pq).all()):
                continue
            if pc[2] <= 1e-4 or pq[2] <= 1e-4:
                continue
            pts_c.append(pc)
            pts_q.append(pq)

        if len(pts_c) < 6:
            return fallback("few_valid_3d", int(len(pts_c)))

        P = np.asarray(pts_c, dtype=np.float64)
        Q = np.asarray(pts_q, dtype=np.float64)
        P_pred = (R_cq @ P.T).T + t_cq[None, :]

        err_3d = np.linalg.norm(P_pred - Q, axis=1)
        reproj_like = float(np.mean(err_3d))
        depth_err = float(np.mean(np.abs(P_pred[:, 2] - Q[:, 2])))

        denom = np.linalg.norm(P_pred, axis=1) + 1e-8
        scale_ratio = np.linalg.norm(Q, axis=1) / denom
        scale_ratio = scale_ratio[np.isfinite(scale_ratio)]
        scale_var = float(np.var(scale_ratio)) if scale_ratio.size > 0 else 1e6

        sim3 = umeyama_sim3(P, Q)
        if sim3 is None:
            return fallback("sim3_failed", int(P.shape[0]))
        s, R_sim3, t_sim3 = sim3

        score = float(w1 * reproj_like + w2 * depth_err + w3 * math.sqrt(max(0.0, scale_var)))
        return {
            "valid": True,
            "score": score,
            "match_count": int(P.shape[0]),
            "reprojection_error": reproj_like,
            "depth_error": depth_err,
            "scale_variance": scale_var,
            "scale": float(s),
            "T_cq_se3": T_cq,
            "sim3": (float(s), R_sim3, t_sim3),
            "candidate_local_index": int(i_c),
            "query_local_index": int(i_q),
        }


def retrieve_topk(query_desc: np.ndarray, map_descs: np.ndarray, map_indices: Sequence[int], k: int) -> List[int]:
    if len(map_indices) == 0:
        return []
    sims = np.dot(map_descs, query_desc.reshape(-1))
    order = np.argsort(-sims)
    out = []
    for j in order[: min(k, len(order))]:
        out.append(int(map_indices[int(j)]))
    return out


def make_se3_from_sim3(sim3_tuple: Tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    s, R, t = sim3_tuple
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    # Convert Sim3 translation into SE3-ish translation for camera composition.
    # If s ~= 1 this is equivalent; when s drifts, dividing by s keeps magnitude bounded.
    s_safe = float(s if abs(s) > 1e-8 else 1.0)
    T[:3, 3] = t / s_safe
    return T


def summarize_metrics(records: List[EvalRecord], prefix: str = "") -> Dict[str, float]:
    total = len(records)
    valid = [r for r in records if r.translation_error_m is not None and r.rotation_error_deg is not None]
    if total == 0:
        return {
            f"{prefix}count": 0.0,
            f"{prefix}success_rate": 0.0,
            f"{prefix}mean_translation_error_m": float("nan"),
            f"{prefix}mean_rotation_error_deg": float("nan"),
            f"{prefix}ate_rmse_success_m": float("nan"),
            f"{prefix}recall_success": 0.0,
        }
    if not valid:
        return {
            f"{prefix}count": float(total),
            f"{prefix}success_rate": 0.0,
            f"{prefix}mean_translation_error_m": float("nan"),
            f"{prefix}mean_rotation_error_deg": float("nan"),
            f"{prefix}ate_rmse_success_m": float("nan"),
            f"{prefix}recall_success": 0.0,
        }
    trans = np.array([r.translation_error_m for r in valid], dtype=np.float64)
    rot = np.array([r.rotation_error_deg for r in valid], dtype=np.float64)
    succ = np.array([bool(r.success) for r in records], dtype=np.bool_)
    succ_valid = np.array([bool(r.success) for r in valid], dtype=np.bool_)
    succ_trans = trans[succ_valid]
    ate_rmse = float(np.sqrt(np.mean(succ_trans * succ_trans))) if succ_trans.size > 0 else float("nan")
    success_rate = float(np.mean(succ.astype(np.float64))) if succ.size > 0 else 0.0
    return {
        f"{prefix}count": float(total),
        f"{prefix}success_rate": success_rate,
        f"{prefix}mean_translation_error_m": float(np.mean(trans)),
        f"{prefix}mean_rotation_error_deg": float(np.mean(rot)),
        f"{prefix}ate_rmse_success_m": ate_rmse,
        f"{prefix}recall_success": success_rate,
    }


def save_plots(records: List[EvalRecord], out_dir: Path, max_examples: int = 12) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in records if r.translation_error_m is not None and r.rotation_error_deg is not None]
    if valid:
        trans = np.array([r.translation_error_m for r in valid], dtype=np.float64)
        rot = np.array([r.rotation_error_deg for r in valid], dtype=np.float64)
        succ = np.array([bool(r.success) for r in valid], dtype=np.bool_)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(trans, bins=30, color="#1f77b4", alpha=0.9)
        plt.xlabel("translation error (m)")
        plt.ylabel("count")
        plt.title("Translation Error Histogram")
        plt.subplot(1, 2, 2)
        plt.hist(rot, bins=30, color="#ff7f0e", alpha=0.9)
        plt.xlabel("rotation error (deg)")
        plt.ylabel("count")
        plt.title("Rotation Error Histogram")
        plt.tight_layout()
        plt.savefig(out_dir / "error_histograms.png", dpi=160)
        plt.close()

        success_examples = [r for r in valid if r.success][: max_examples // 2]
        fail_examples = [r for r in valid if not r.success][: max_examples // 2]
        ex = success_examples + fail_examples
        if ex:
            cols = 2
            rows = len(ex)
            plt.figure(figsize=(9, 3.2 * rows))
            for i, rec in enumerate(ex):
                q = np.asarray(Image.open(rec.query_rgb).convert("RGB"))
                c = None
                if rec.best_candidate_rgb:
                    c = np.asarray(Image.open(rec.best_candidate_rgb).convert("RGB"))
                plt.subplot(rows, cols, 2 * i + 1)
                plt.imshow(q)
                plt.axis("off")
                plt.title(f"Q {rec.sequence}:{rec.query_idx}")
                plt.subplot(rows, cols, 2 * i + 2)
                if c is not None:
                    plt.imshow(c)
                plt.axis("off")
                tag = "S" if rec.success else "F"
                plt.title(
                    f"{tag} cand={rec.best_candidate_idx} "
                    f"te={rec.translation_error_m:.3f}m re={rec.rotation_error_deg:.2f}deg"
                )
            plt.tight_layout()
            plt.savefig(out_dir / "success_failure_examples.png", dpi=160)
            plt.close()


def _preview_relpath(src_path: str) -> str:
    h = hashlib.sha1(src_path.encode("utf-8")).hexdigest()[:12]
    ext = Path(src_path).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    return f"previews/{h}{ext}"


def _materialize_previews(records: List[EvalRecord], out_dir: Path, max_side: int = 320) -> Dict[str, str]:
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    path_map: Dict[str, str] = {}
    unique_paths = set()
    for r in records:
        if r.query_rgb:
            unique_paths.add(r.query_rgb)
        if r.best_candidate_rgb:
            unique_paths.add(r.best_candidate_rgb)
    for src in sorted(unique_paths):
        src_p = Path(src)
        if not src_p.exists():
            continue
        rel = _preview_relpath(src)
        dst = out_dir / rel
        if not dst.exists():
            try:
                img = Image.open(src_p).convert("RGB")
                w, h = img.size
                scale = min(1.0, float(max_side) / float(max(1, max(w, h))))
                if scale < 1.0:
                    nw = max(1, int(round(w * scale)))
                    nh = max(1, int(round(h * scale)))
                    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
                img.save(dst, quality=90)
            except Exception:
                continue
        path_map[src] = rel
    return path_map


def save_qualitative_html(records: List[EvalRecord], out_dir: Path, max_rows: int = 500) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path_map = _materialize_previews(records, out_dir)
    rows = records[: max(0, int(max_rows))]

    def img_cell(src_abs: Optional[str]) -> str:
        if not src_abs:
            return "-"
        rel = path_map.get(src_abs, "")
        if not rel:
            return "-"
        return f'<a href="{escape(rel)}"><img src="{escape(rel)}" width="220"></a>'

    tr_html = []
    for r in rows:
        trans = "-" if r.translation_error_m is None else f"{r.translation_error_m:.4f}"
        rot = "-" if r.rotation_error_deg is None else f"{r.rotation_error_deg:.3f}"
        score = "-" if r.score is None else f"{r.score:.5f}"
        mc = "-" if r.match_count is None else str(int(r.match_count))
        cand = "-" if r.best_candidate_idx is None else str(int(r.best_candidate_idx))
        map_sz = "-" if r.map_size is None else str(int(r.map_size))
        excl = (
            "-"
            if r.map_exclude_start is None or r.map_exclude_end is None
            else f"[{int(r.map_exclude_start)}, {int(r.map_exclude_end)}]"
        )
        topk = "-" if not r.topk_candidates else ",".join(str(int(x)) for x in r.topk_candidates)
        jump_t = "-" if r.query_jump_translation_m is None else f"{r.query_jump_translation_m:.3f}"
        jump_r = "-" if r.query_jump_rotation_deg is None else f"{r.query_jump_rotation_deg:.2f}"
        tr_html.append(
            "<tr>"
            f"<td>{escape(r.sequence)}</td>"
            f"<td>{int(r.query_idx)}</td>"
            f"<td>{escape(r.status)}</td>"
            f"<td>{'yes' if r.success else 'no'}</td>"
            f"<td>{excl}</td>"
            f"<td>{map_sz}</td>"
            f"<td>{jump_t}</td>"
            f"<td>{jump_r}</td>"
            f"<td>{trans}</td>"
            f"<td>{rot}</td>"
            f"<td>{cand}</td>"
            f"<td>{score}</td>"
            f"<td>{mc}</td>"
            f"<td>{topk}</td>"
            f"<td>{img_cell(r.query_rgb)}</td>"
            f"<td>{img_cell(r.best_candidate_rgb)}</td>"
            "</tr>"
        )

    body = "\n".join(tr_html) if tr_html else "<tr><td colspan='16'>No records</td></tr>"
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Cold-start Relocalization Qualitative Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 14px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
    th {{ background: #f4f4f4; position: sticky; top: 0; }}
    img {{ border: 1px solid #bbb; border-radius: 4px; }}
    .meta {{ margin-bottom: 10px; }}
  </style>
</head>
<body>
  <div class="meta">
    <h3>Cold-start Relocalization Qualitative Results</h3>
    <p>Total rows: {len(rows)} / {len(records)}</p>
  </div>
  <table>
    <thead>
      <tr>
        <th>Seq</th><th>QueryIdx</th><th>Status</th><th>Success</th>
        <th>Excluded</th><th>MapSize</th><th>Jump_t(m)</th><th>Jump_r(deg)</th>
        <th>t_err(m)</th><th>r_err(deg)</th><th>BestCand</th><th>Score</th><th>Matches</th><th>TopK</th>
        <th>Query</th><th>Best Candidate</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""
    out_path = out_dir / "qualitative_viewer.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def evaluate_sequence(
    seq_name: str,
    frames: List[TumFrame],
    descriptors: np.ndarray,
    verifier: LoGeRVerifier,
    args: argparse.Namespace,
) -> List[EvalRecord]:
    out: List[EvalRecord] = []
    if len(frames) < 8:
        print(f"[warn] {seq_name}: too few frames ({len(frames)}), skipping.")
        return out

    jump_t, jump_r = compute_pose_jumps(frames)
    q_idx = sample_query_indices(
        frames=frames,
        n_queries=int(args.n_queries),
        radius=int(args.candidate_radius),
        map_exclude_radius=int(args.map_exclude_radius),
        seed=int(args.seed),
        mode=str(args.query_mode),
        kidnap_min_translation_m=float(args.kidnap_min_translation_m),
        kidnap_min_rotation_deg=float(args.kidnap_min_rotation_deg),
    )
    if args.max_queries_per_sequence > 0:
        q_idx = q_idx[: int(args.max_queries_per_sequence)]
    print(
        f"[eval] {seq_name}: {len(q_idx)} queries over {len(frames)} frames "
        f"(query_mode={args.query_mode}, map_exclude_radius={int(args.map_exclude_radius)})"
    )

    all_indices = list(range(len(frames)))
    rng = random.Random(args.seed + 1234)

    for qi in q_idx:
        query_desc = descriptors[qi]
        excl_lo = max(0, int(qi) - int(args.map_exclude_radius))
        excl_hi = min(len(frames) - 1, int(qi) + int(args.map_exclude_radius))
        map_indices = [i for i in all_indices if i < excl_lo or i > excl_hi]
        if not map_indices:
            out.append(
                EvalRecord(
                    sequence=seq_name,
                    query_idx=qi,
                    query_timestamp=frames[qi].timestamp,
                    query_rgb=frames[qi].rgb_path,
                    status="empty_map",
                    best_candidate_idx=None,
                    best_candidate_rgb=None,
                    score=None,
                    match_count=None,
                    translation_error_m=None,
                    rotation_error_deg=None,
                    success=False,
                    map_size=0,
                    map_exclude_start=excl_lo,
                    map_exclude_end=excl_hi,
                    topk_candidates=[],
                    query_jump_translation_m=float(jump_t[qi]),
                    query_jump_rotation_deg=float(jump_r[qi]),
                )
            )
            continue
        map_desc = descriptors[np.array(map_indices, dtype=np.int64)]
        topk = retrieve_topk(query_desc, map_desc, map_indices, args.top_k)

        best = None
        for cand_idx in topk:
            try:
                cand_eval = verifier.evaluate_candidate(
                    frames=frames,
                    query_idx=qi,
                    cand_idx=cand_idx,
                    radius=args.candidate_radius,
                    w1=args.w1,
                    w2=args.w2,
                    w3=args.w3,
                )
            except Exception as e:
                cand_eval = {"valid": False, "reason": f"exception:{e}"}
            if not cand_eval.get("valid", False):
                continue
            if best is None or float(cand_eval["score"]) < float(best["score"]):
                best = {"cand_idx": int(cand_idx), **cand_eval}

        if best is None:
            out.append(
                EvalRecord(
                    sequence=seq_name,
                    query_idx=qi,
                    query_timestamp=frames[qi].timestamp,
                    query_rgb=frames[qi].rgb_path,
                    status="no_valid_candidate",
                    best_candidate_idx=None,
                    best_candidate_rgb=None,
                    score=None,
                    match_count=None,
                    translation_error_m=None,
                    rotation_error_deg=None,
                    success=False,
                    map_size=int(len(map_indices)),
                    map_exclude_start=excl_lo,
                    map_exclude_end=excl_hi,
                    topk_candidates=[int(x) for x in topk],
                    query_jump_translation_m=float(jump_t[qi]),
                    query_jump_rotation_deg=float(jump_r[qi]),
                )
            )
            continue

        cand_idx = int(best["cand_idx"])
        T_w_c = np.array(frames[cand_idx].pose_w_c, dtype=np.float64)
        if "sim3" in best and best["sim3"] is not None:
            T_c_q = make_se3_from_sim3(best["sim3"])
        else:
            T_c_q = np.array(best["T_cq_se3"], dtype=np.float64)
        T_w_q_est = compose_T(T_w_c, T_c_q)
        T_w_q_gt = np.array(frames[qi].pose_w_c, dtype=np.float64)

        t_err = float(np.linalg.norm(T_w_q_est[:3, 3] - T_w_q_gt[:3, 3]))
        r_err = float(rotation_error_deg(T_w_q_est[:3, :3], T_w_q_gt[:3, :3]))
        success = bool(t_err < args.translation_success_m and r_err < args.rotation_success_deg)

        rec = EvalRecord(
            sequence=seq_name,
            query_idx=qi,
            query_timestamp=frames[qi].timestamp,
            query_rgb=frames[qi].rgb_path,
            status="ok",
            best_candidate_idx=cand_idx,
            best_candidate_rgb=frames[cand_idx].rgb_path,
            score=float(best["score"]),
            match_count=int(best.get("match_count", 0)),
            translation_error_m=t_err,
            rotation_error_deg=r_err,
            success=success,
            map_size=int(len(map_indices)),
            map_exclude_start=excl_lo,
            map_exclude_end=excl_hi,
            topk_candidates=[int(x) for x in topk],
            query_jump_translation_m=float(jump_t[qi]),
            query_jump_rotation_deg=float(jump_r[qi]),
        )

        if int(args.run_baselines) == 1:
            # Baseline 1: VPR-only (top-1 candidate pose as query pose)
            if topk:
                b_idx = int(topk[0])
                T_w_q_b = np.array(frames[b_idx].pose_w_c, dtype=np.float64)
                rec.baseline_vpr_translation_error_m = float(np.linalg.norm(T_w_q_b[:3, 3] - T_w_q_gt[:3, 3]))
                rec.baseline_vpr_rotation_error_deg = float(rotation_error_deg(T_w_q_b[:3, :3], T_w_q_gt[:3, :3]))
            # Baseline 2: random candidate pose
            rb = int(rng.choice(map_indices))
            T_w_q_r = np.array(frames[rb].pose_w_c, dtype=np.float64)
            rec.baseline_random_translation_error_m = float(np.linalg.norm(T_w_q_r[:3, 3] - T_w_q_gt[:3, 3]))
            rec.baseline_random_rotation_error_deg = float(rotation_error_deg(T_w_q_r[:3, :3], T_w_q_gt[:3, :3]))

        out.append(rec)
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    seq_names = [s.strip() for s in args.sequences.split(",") if s.strip()]
    print(f"[cfg] sequences={seq_names}")
    print(f"[cfg] tum_root={args.tum_root}")
    print(
        f"[cfg] query_mode={args.query_mode}, map_exclude_radius={int(args.map_exclude_radius)}, "
        f"kidnap_min_t={float(args.kidnap_min_translation_m):.3f}, "
        f"kidnap_min_r={float(args.kidnap_min_rotation_deg):.2f}"
    )
    descriptor_device = resolve_device(args.descriptor_device)
    loger_device = resolve_device(args.loger_device)
    print(
        f"[cfg] descriptor={args.descriptor_model}, "
        f"descriptor_device={descriptor_device}, loger_device={loger_device}, "
        f"loger_ttt={args.loger_ttt}, sim3={int(args.loger_use_sim3)}, se3={int(args.loger_use_se3)}"
    )

    extractor = DescriptorExtractor(
        model_name=args.descriptor_model,
        device=descriptor_device,
        salad_root=args.salad_root,
        salad_ckpt_path=args.salad_ckpt_path,
        salad_image_size=args.salad_image_size,
        salad_num_trainable_blocks=args.salad_num_trainable_blocks,
    )
    verifier = LoGeRVerifier(
        loger_repo=args.loger_repo,
        weights_path=args.loger_weights,
        config_path=args.loger_config,
        window_size=args.loger_window_size,
        overlap_size=args.loger_overlap_size,
        reset_every=args.loger_reset_every,
        num_iterations=args.loger_num_iterations,
        patch_multiple=args.loger_patch_multiple,
        turn_off_ttt=(str(args.loger_ttt).lower() == "off"),
        use_sim3=bool(int(args.loger_use_sim3)),
        use_se3=bool(int(args.loger_use_se3)),
        device=loger_device,
    )

    all_records: List[EvalRecord] = []
    seq_summary: Dict[str, Dict[str, float]] = {}

    for seq_name in seq_names:
        frames = load_tum_sequence(args.tum_root, seq_name, assoc_tol=float(args.assoc_tolerance_sec))
        if not frames:
            print(f"[warn] sequence missing or empty: {seq_name}")
            continue
        if int(args.max_frames_per_sequence) > 0:
            before_n = len(frames)
            frames = subsample_frames_uniform(frames, int(args.max_frames_per_sequence))
            print(f"[info] {seq_name}: subsampled frames {before_n} -> {len(frames)}")
        cache_path = args.cache_dir / f"{seq_name}_{args.descriptor_model}.npz"
        desc = load_or_compute_descriptors(
            frames=frames,
            extractor=extractor,
            cache_path=cache_path,
            batch_size=int(args.descriptor_batch_size),
        )
        records = evaluate_sequence(
            seq_name=seq_name,
            frames=frames,
            descriptors=desc,
            verifier=verifier,
            args=args,
        )
        all_records.extend(records)
        seq_summary[seq_name] = summarize_metrics(records, prefix="")
        print(
            f"[summary][{seq_name}] success={seq_summary[seq_name]['success_rate']:.3f} "
            f"mean_t={seq_summary[seq_name]['mean_translation_error_m']:.4f} "
            f"mean_r={seq_summary[seq_name]['mean_rotation_error_deg']:.3f}"
        )

    overall = summarize_metrics(all_records, prefix="")
    print("\n=== Overall ===")
    print(f"queries: {int(overall['count'])}")
    print(f"success_rate: {overall['success_rate'] * 100.0:.2f}%")
    print(f"mean_translation_error_m: {overall['mean_translation_error_m']:.6f}")
    print(f"mean_rotation_error_deg: {overall['mean_rotation_error_deg']:.6f}")
    print(f"ATE_RMSE_success_m: {overall['ate_rmse_success_m']:.6f}")
    print(f"Recall@success: {overall['recall_success'] * 100.0:.2f}%")

    out_json = args.output_dir / "records.json"
    out_metrics = args.output_dir / "metrics.json"
    out_json.write_text(json.dumps([asdict(r) for r in all_records], indent=2), encoding="utf-8")
    payload = {"overall": overall, "per_sequence": seq_summary}
    if int(args.run_baselines) == 1:
        valid = [r for r in all_records if r.baseline_vpr_translation_error_m is not None]
        if valid:
            vpr_te = np.array([r.baseline_vpr_translation_error_m for r in valid], dtype=np.float64)
            vpr_re = np.array([r.baseline_vpr_rotation_error_deg for r in valid], dtype=np.float64)
            rnd_te = np.array([r.baseline_random_translation_error_m for r in valid], dtype=np.float64)
            rnd_re = np.array([r.baseline_random_rotation_error_deg for r in valid], dtype=np.float64)
            payload["baseline_vpr_only"] = {
                "count": int(vpr_te.size),
                "mean_translation_error_m": float(np.mean(vpr_te)),
                "mean_rotation_error_deg": float(np.mean(vpr_re)),
            }
            payload["baseline_random"] = {
                "count": int(rnd_te.size),
                "mean_translation_error_m": float(np.mean(rnd_te)),
                "mean_rotation_error_deg": float(np.mean(rnd_re)),
            }
    out_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_plots(all_records, args.output_dir, max_examples=int(args.num_example_pairs))
    out_html = save_qualitative_html(all_records, args.output_dir, max_rows=1000)
    print(f"\nSaved records: {out_json}")
    print(f"Saved metrics: {out_metrics}")
    print(f"Saved qualitative viewer: {out_html}")
    print(f"Saved plots: {args.output_dir / 'error_histograms.png'}, {args.output_dir / 'success_failure_examples.png'}")


if __name__ == "__main__":
    main()
