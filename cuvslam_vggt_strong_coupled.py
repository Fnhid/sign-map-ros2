#!/usr/bin/env python3
"""
Strong-coupled VGGT prior + cuVSLAM (RGB mono) runner.

Concept:
- Read VGGT pose priors from text file (frame_id tx ty tz qx qy qz qw).
- Inject/fuse prior pose into cuVSLAM SLAM state with set_slam_pose().
- Run cuVSLAM tracking per frame (RGB mono, CPU by default).
- Export web viewer with trajectories and per-frame side view.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import cuvslam


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VGGT prior + cuVSLAM strong-coupled runner (RGB mono).")
    p.add_argument("--image_folder", type=Path, required=True)
    p.add_argument("--vggt_pose_txt", type=Path, required=True, help="Text file: frame_id tx ty tz qx qy qz qw")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--max_frames", type=int, default=0, help="0 = all")
    p.add_argument("--use_gpu", type=int, default=0, choices=[0, 1], help="cuVSLAM internal GPU usage")
    p.add_argument("--inject_mode", type=str, default="both", choices=["pre", "post", "both", "off"])
    p.add_argument("--prior_weight", type=float, default=0.75, help="0.0 keeps raw SLAM, 1.0 fully trusts VGGT prior")
    p.add_argument("--thumb_w", type=int, default=480)
    p.add_argument("--thumb_h", type=int, default=360)
    p.add_argument("--cam_fx", type=float, default=0.0, help="0 means auto-estimate from image width")
    p.add_argument("--cam_fy", type=float, default=0.0, help="0 means auto-estimate from image width")
    p.add_argument("--cam_cx", type=float, default=0.0, help="0 means center principal point")
    p.add_argument("--cam_cy", type=float, default=0.0, help="0 means center principal point")
    p.add_argument("--max_cloud_points", type=int, default=50000, help="Max points exported to pointcloud.json (0=all)")
    p.add_argument(
        "--odometry_mode",
        type=str,
        default="mono",
        choices=["mono", "rgbd_vggt"],
        help="mono: cuVSLAM mono mode, rgbd_vggt: feed VGGT-derived depth map to cuVSLAM RGBD mode",
    )
    p.add_argument("--vggt_depth_log_dir", type=Path, default=None, help="Folder with VGGT *.npz logs (pointcloud, mask)")
    p.add_argument("--depth_metric", type=str, default="z", choices=["z", "norm"], help="Depth from pointcloud: z-axis or xyz norm")
    p.add_argument("--depth_scale_factor", type=float, default=1000.0, help="Raw depth scale for cuVSLAM (depth_m = raw / scale)")
    p.add_argument("--depth_min_m", type=float, default=0.15, help="Minimum valid depth in meters")
    p.add_argument("--depth_max_m", type=float, default=20.0, help="Maximum valid depth in meters")
    p.add_argument("--depth_nearest_max_gap", type=int, default=2, help="Max frame gap for nearest VGGT depth fallback")
    return p.parse_args()


def normalize_q(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_q(q0)
    q1 = normalize_q(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return normalize_q(q)
    theta_0 = math.acos(dot)
    sin_0 = math.sin(theta_0)
    theta_t = theta_0 * t
    s0 = math.sin(theta_0 - theta_t) / sin_0
    s1 = math.sin(theta_t) / sin_0
    return s0 * q0 + s1 * q1


def blend_pose(
    base_t: np.ndarray, base_q: np.ndarray, prior_t: np.ndarray, prior_q: np.ndarray, w: float
) -> Tuple[np.ndarray, np.ndarray]:
    w = float(max(0.0, min(1.0, w)))
    t = (1.0 - w) * base_t + w * prior_t
    q = slerp(base_q, prior_q, w)
    return t, q


def pose_to_obj(t: np.ndarray, q: np.ndarray) -> cuvslam.Pose:
    return cuvslam.Pose(rotation=[float(x) for x in q], translation=[float(x) for x in t])


def load_vggt_priors(path: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    priors: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 8:
            continue
        fid = int(round(float(parts[0]))) - 1
        t = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        q = normalize_q(np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64))
        priors[fid] = (t, q)
    return priors


def image_list(folder: Path) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    out: List[Path] = []
    for e in exts:
        out.extend(folder.glob(e))
    return sorted(out)


def landmarks_to_points(raw_landmarks) -> np.ndarray:
    if raw_landmarks is None:
        return np.zeros((0, 3), dtype=np.float32)
    if isinstance(raw_landmarks, dict):
        src = raw_landmarks.values()
    else:
        src = raw_landmarks

    points: List[List[float]] = []
    for item in src:
        coords = item.coords if hasattr(item, "coords") else item
        arr = np.asarray(coords, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            continue
        xyz = arr[:3]
        if not np.all(np.isfinite(xyz)):
            continue
        points.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def try_get_final_points(tracker, max_points: int) -> np.ndarray:
    points = np.zeros((0, 3), dtype=np.float32)
    try:
        points = landmarks_to_points(tracker.get_final_landmarks())
    except Exception:
        points = np.zeros((0, 3), dtype=np.float32)

    if points.shape[0] == 0:
        try:
            slam_data = tracker.get_slam_landmarks(tracker.SlamDataLayer.Map)
            if hasattr(slam_data, "landmarks"):
                points = landmarks_to_points(slam_data.landmarks)
        except Exception:
            points = np.zeros((0, 3), dtype=np.float32)

    if max_points > 0 and points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[idx]
    return points


def pack_cloud_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    if max_points > 0 and points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
        points = points[idx]
    return points


def load_vggt_log_points(log_dir: Path, max_points: int) -> np.ndarray:
    if not log_dir.exists():
        return np.zeros((0, 3), dtype=np.float32)

    files = sorted(log_dir.glob("*.npz"))
    if not files:
        return np.zeros((0, 3), dtype=np.float32)

    # Keep memory bounded by sampling each file first, then global sampling.
    per_file_cap = max(1000, max_points // max(1, len(files)) * 3) if max_points > 0 else 5000
    all_pts: List[np.ndarray] = []
    for f in files:
        try:
            data = np.load(f)
            if "pointcloud" not in data:
                continue
            pc = np.asarray(data["pointcloud"])
            if pc.ndim != 3 or pc.shape[-1] < 3:
                continue
            if "mask" in data:
                mk = np.asarray(data["mask"]).astype(bool)
                if mk.shape == pc.shape[:2]:
                    pts = pc[mk]
                else:
                    pts = pc.reshape(-1, pc.shape[-1])[:, :3]
            else:
                pts = pc.reshape(-1, pc.shape[-1])[:, :3]
            if pts.shape[0] == 0:
                continue
            pts = pts[:, :3]
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            if pts.shape[0] == 0:
                continue
            if per_file_cap > 0 and pts.shape[0] > per_file_cap:
                idx = np.linspace(0, pts.shape[0] - 1, per_file_cap, dtype=np.int64)
                pts = pts[idx]
            all_pts.append(pts.astype(np.float32))
        except Exception:
            continue

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32)

    merged = np.concatenate(all_pts, axis=0)
    return pack_cloud_points(merged, max_points)


def build_vggt_depth_index(log_dir: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    if not log_dir.exists():
        return out
    for f in sorted(log_dir.glob("*.npz")):
        stem = f.stem
        try:
            frame_num = int(round(float(stem)))
        except Exception:
            continue
        out[frame_num] = f
    return out


def infer_intrinsics_from_vggt_pointcloud(npz_path: Path) -> Optional[Dict[str, float]]:
    try:
        data = np.load(npz_path)
        if "pointcloud" not in data:
            return None
        pc = np.asarray(data["pointcloud"], dtype=np.float64)
        if pc.ndim != 3 or pc.shape[-1] < 3:
            return None
        h, w = pc.shape[:2]
        mask = np.asarray(data["mask"]).astype(bool) if "mask" in data and np.asarray(data["mask"]).shape == (h, w) else np.ones((h, w), dtype=bool)
        z = pc[..., 2]
        valid = mask & np.isfinite(pc[..., 0]) & np.isfinite(pc[..., 1]) & np.isfinite(z) & (np.abs(z) > 1e-8)
        if int(valid.sum()) < 200:
            return None
        u = np.broadcast_to(np.arange(w, dtype=np.float64)[None, :], (h, w))[valid]
        v = np.broadcast_to(np.arange(h, dtype=np.float64)[:, None], (h, w))[valid]
        xz = (pc[..., 0] / z)[valid]
        yz = (pc[..., 1] / z)[valid]

        ax = np.vstack([u, np.ones_like(u)]).T
        ay = np.vstack([v, np.ones_like(v)]).T
        kx, bx = np.linalg.lstsq(ax, xz, rcond=None)[0]
        ky, by = np.linalg.lstsq(ay, yz, rcond=None)[0]
        if abs(kx) < 1e-9 or abs(ky) < 1e-9:
            return None
        fx = 1.0 / kx
        fy = 1.0 / ky
        cx = -bx / kx
        cy = -by / ky
        if not all(np.isfinite(vv) for vv in [fx, fy, cx, cy]):
            return None
        if fx <= 1.0 or fy <= 1.0:
            return None
        pred_xz = ax @ np.array([kx, bx])
        pred_yz = ay @ np.array([ky, by])
        r2x = 1.0 - float(np.mean((xz - pred_xz) ** 2) / (np.var(xz) + 1e-12))
        r2y = 1.0 - float(np.mean((yz - pred_yz) ** 2) / (np.var(yz) + 1e-12))
        return {
            "w": float(w),
            "h": float(h),
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "r2x": float(r2x),
            "r2y": float(r2y),
            "source_npz": str(npz_path),
        }
    except Exception:
        return None


def _resize_depth_and_mask(depth_m: np.ndarray, valid: np.ndarray, out_w: int, out_h: int) -> Tuple[np.ndarray, np.ndarray]:
    depth_img = Image.fromarray(depth_m.astype(np.float32), mode="F")
    depth_rs = np.asarray(depth_img.resize((out_w, out_h), Image.BILINEAR), dtype=np.float32)
    mask_img = Image.fromarray((valid.astype(np.uint8) * 255), mode="L")
    mask_rs = np.asarray(mask_img.resize((out_w, out_h), Image.NEAREST), dtype=np.uint8) > 127
    return depth_rs, mask_rs


def load_vggt_depth_u16(
    frame_idx0: int,
    depth_index: Dict[int, Path],
    out_w: int,
    out_h: int,
    metric: str,
    min_m: float,
    max_m: float,
    raw_scale: float,
    nearest_max_gap: int,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    target = frame_idx0 + 1
    chosen = depth_index.get(target)
    used_nearest = False
    if chosen is None and depth_index:
        keys = np.fromiter(depth_index.keys(), dtype=np.int32)
        nearest_idx = int(np.argmin(np.abs(keys - target)))
        nearest_key = int(keys[nearest_idx])
        if abs(nearest_key - target) <= int(max(0, nearest_max_gap)):
            chosen = depth_index[nearest_key]
            used_nearest = True

    if chosen is None:
        return None, {"has_depth": False, "used_nearest": False, "path": None, "valid_ratio": 0.0}

    try:
        data = np.load(chosen)
        if "pointcloud" not in data:
            return None, {"has_depth": False, "used_nearest": used_nearest, "path": str(chosen), "valid_ratio": 0.0}
        pc = np.asarray(data["pointcloud"], dtype=np.float32)
        if pc.ndim != 3 or pc.shape[-1] < 3:
            return None, {"has_depth": False, "used_nearest": used_nearest, "path": str(chosen), "valid_ratio": 0.0}
        if metric == "norm":
            depth_m = np.linalg.norm(pc[..., :3], axis=-1)
        else:
            depth_m = pc[..., 2]
        if "mask" in data:
            mask = np.asarray(data["mask"]).astype(bool)
            if mask.shape != depth_m.shape:
                mask = np.ones_like(depth_m, dtype=bool)
        else:
            mask = np.ones_like(depth_m, dtype=bool)
        valid = mask & np.isfinite(depth_m) & (depth_m > 0.0)
        depth_m = np.where(valid, depth_m, 0.0)
        depth_rs, valid_rs = _resize_depth_and_mask(depth_m, valid, out_w, out_h)
        valid_rs = valid_rs & np.isfinite(depth_rs) & (depth_rs >= float(min_m)) & (depth_rs <= float(max_m))
        depth_rs = np.where(valid_rs, depth_rs, 0.0)
        raw = np.clip(np.round(depth_rs * float(raw_scale)), 0, 65535).astype(np.uint16)
        return raw, {
            "has_depth": True,
            "used_nearest": bool(used_nearest),
            "path": str(chosen),
            "valid_ratio": float(valid_rs.mean()) if valid_rs.size else 0.0,
        }
    except Exception:
        return None, {"has_depth": False, "used_nearest": used_nearest, "path": str(chosen), "valid_ratio": 0.0}


def normalize_label(raw: object) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    s = re.sub(r"[^\dA-Za-z가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    nums = re.findall(r"\d+", s)
    if nums:
        return " ".join(nums)
    return s


def load_category_points(run_dir: Path, max_points_per_cat: int = 5000) -> Dict[str, object]:
    out: Dict[str, List[List[float]]] = {"sign": [], "door": []}
    sign_labels: List[str] = []

    sign_path = run_dir / "sign_3d_detections.json"
    if sign_path.exists():
        try:
            data = json.loads(sign_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for d in data:
                    if not isinstance(d, dict):
                        continue
                    c = d.get("centroid_xyz")
                    if not (isinstance(c, list) and len(c) >= 3):
                        continue
                    xyz = np.asarray(c[:3], dtype=np.float64)
                    if np.all(np.isfinite(xyz)):
                        out["sign"].append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                        label_raw = (
                            d.get("final_label")
                            or d.get("label")
                            or d.get("best_label")
                            or d.get("ocr_text")
                            or ""
                        )
                        sign_labels.append(normalize_label(label_raw))
        except Exception:
            pass

    open_set_path = run_dir / "open_set_3d.json"
    if open_set_path.exists():
        try:
            data = json.loads(open_set_path.read_text(encoding="utf-8"))
            objs = data.get("objects", []) if isinstance(data, dict) else []
            if isinstance(objs, list):
                for d in objs:
                    if not isinstance(d, dict):
                        continue
                    q = str(d.get("query", "")).strip().lower()
                    if q not in {"door", "sign"}:
                        continue
                    c = d.get("center_xyz")
                    if not (isinstance(c, list) and len(c) >= 3):
                        c = d.get("centroid_xyz")
                    if not (isinstance(c, list) and len(c) >= 3):
                        continue
                    xyz = np.asarray(c[:3], dtype=np.float64)
                    if np.all(np.isfinite(xyz)):
                        out[q].append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
                        if q == "sign":
                            label_raw = d.get("label") or d.get("ocr_text") or ""
                            sign_labels.append(normalize_label(label_raw))
        except Exception:
            pass

    packed: Dict[str, np.ndarray] = {}
    for k in ["sign", "door"]:
        pts = np.asarray(out[k], dtype=np.float32) if out[k] else np.zeros((0, 3), dtype=np.float32)
        if max_points_per_cat > 0 and pts.shape[0] > max_points_per_cat:
            idx = np.linspace(0, pts.shape[0] - 1, max_points_per_cat, dtype=np.int64)
            pts = pts[idx]
            if k == "sign" and sign_labels:
                sign_labels = [sign_labels[int(i)] for i in idx]
        packed[k] = pts
    if len(sign_labels) < int(packed["sign"].shape[0]):
        sign_labels.extend([""] * (int(packed["sign"].shape[0]) - len(sign_labels)))
    elif len(sign_labels) > int(packed["sign"].shape[0]):
        sign_labels = sign_labels[: int(packed["sign"].shape[0])]
    return {"sign": packed["sign"], "door": packed["door"], "sign_labels": sign_labels}


def load_detection_cards(run_dir: Path, max_items: int = 80) -> List[dict]:
    path = run_dir / "sign_3d_detections.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    source_name = run_dir.name
    items: List[dict] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        frame_name = str(d.get("frame_name") or "")
        frame_idx = -1
        m = re.search(r"(\d+)", frame_name)
        if m:
            frame_idx = int(m.group(1))
        label = normalize_label(d.get("final_label") or d.get("ocr_text") or "")
        item = {
            "frame_name": frame_name,
            "frame_idx": frame_idx,
            "score": float(d.get("score", 0.0)),
            "bbox_xyxy": d.get("bbox_xyxy"),
            "text": label,
            "raw_text": str(d.get("ocr_text") or ""),
            "mapped_3d": bool(d.get("mapped_3d", False)),
            "point_count": int(d.get("point_count", 0)),
            "annotated_rel": "",
            "crop_rel": "",
        }
        af = d.get("annotated_frame_path")
        cp = d.get("crop_path")
        if isinstance(af, str) and af.strip():
            item["annotated_rel"] = f"../{source_name}/{af.strip()}"
        if isinstance(cp, str) and cp.strip():
            item["crop_rel"] = f"../{source_name}/{cp.strip()}"
        items.append(item)

    items.sort(key=lambda x: (0 if x["text"] else 1, -x["score"], -x["point_count"]))
    return items[:max_items]


def make_viewer(out_dir: Path) -> None:
    html = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VGGT + cuVSLAM Strong-Coupled Viewer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#f6f8fb; color:#1f2937; }
    .wrap { display:grid; grid-template-columns: 1.45fr 1fr; gap:10px; height:100vh; padding:10px; box-sizing:border-box; }
    .panel { background:#fff; border:1px solid #dbe2ea; border-radius:10px; overflow:hidden; }
    #traj { width:100%; height:100%; }
    .right { display:grid; grid-template-rows:auto 1fr auto 260px; }
    .meta { padding:10px 12px; border-bottom:1px solid #e5e7eb; font-size:13px; line-height:1.45; }
    .imgbox { display:flex; align-items:center; justify-content:center; background:#111827; }
    .imgbox img { max-width:100%; max-height:100%; object-fit:contain; }
    .ctrl { padding:10px 12px; border-top:1px solid #e5e7eb; }
    .detlist { border-top:1px solid #e5e7eb; overflow:auto; padding:8px; background:#f8fafc; }
    .detcard { display:grid; grid-template-columns:84px 1fr; gap:8px; border:1px solid #dbe2ea; border-radius:8px; background:#fff; margin-bottom:8px; padding:6px; }
    .detcard img { width:84px; height:58px; object-fit:cover; border-radius:6px; border:1px solid #e5e7eb; background:#111827; }
    .detmeta { font-size:12px; line-height:1.35; }
    .detmeta .t { font-weight:700; color:#0f172a; }
    .detmeta .s { color:#475569; }
    .mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }
    input[type=range] { width:100%; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel"><div id="traj"></div></div>
    <div class="panel right">
      <div class="meta" id="meta">loading...</div>
      <div class="imgbox"><img id="frameImg" alt="frame" /></div>
      <div class="ctrl">
        <div><b>Frame</b> <span id="frameIdx">0</span></div>
        <input id="slider" type="range" min="0" max="0" value="0" step="1" />
      </div>
      <div class="detlist" id="detList">loading detections...</div>
    </div>
  </div>
<script>
(async function () {
  const [summary, poses, cloud, semantic, detections] = await Promise.all([
    fetch('./summary.json').then(r => r.json()),
    fetch('./poses.json').then(r => r.json()),
    fetch('./pointcloud.json').then(r => r.json()).catch(() => ({points: []})),
    fetch('./semantic_points.json').then(r => r.json()).catch(() => ({sign: [], door: []})),
    fetch('./detections_view.json').then(r => r.json()).catch(() => ([])),
  ]);
  const slider = document.getElementById('slider');
  const frameIdx = document.getElementById('frameIdx');
  const meta = document.getElementById('meta');
  const img = document.getElementById('frameImg');
  const detList = document.getElementById('detList');

  function pickAny(p, keys) {
    for (const k of keys) {
      const v = p[k];
      if (Array.isArray(v) && v.length === 3) return v;
    }
    return [0,0,0];
  }

  const tX = poses.map(p => pickAny(p, ['fused_t', 'slam_t', 'odom_t'])[0]);
  const tY = poses.map(p => pickAny(p, ['fused_t', 'slam_t', 'odom_t'])[1]);
  const tZ = poses.map(p => pickAny(p, ['fused_t', 'slam_t', 'odom_t'])[2]);

  const cX = (cloud.points || []).map(p => p[0]);
  const cY = (cloud.points || []).map(p => p[1]);
  const cZ = (cloud.points || []).map(p => p[2]);
  const sX = (semantic.sign || []).map(p => p[0]);
  const sY = (semantic.sign || []).map(p => p[1]);
  const sZ = (semantic.sign || []).map(p => p[2]);
  const signLabels = (semantic.sign_labels || []).map(v => String(v || '').trim());
  const dX = (semantic.door || []).map(p => p[0]);
  const dY = (semantic.door || []).map(p => p[1]);
  const dZ = (semantic.door || []).map(p => p[2]);

  const traces = [];
  if (cX.length > 0) {
    traces.push({
      type:'scatter3d',
      mode:'markers',
      x:cX, y:cY, z:cZ,
      marker:{size:1.9,color:'rgba(70,70,70,0.88)'},
      name:'Point Cloud (all)'
    });
  }
  if (sX.length > 0) {
    traces.push({
      type:'scatter3d', mode:'markers', x:sX, y:sY, z:sZ,
      marker:{size:6,color:'#f97316',symbol:'diamond'},
      name:'sign points'
    });
    const byLabel = new Map();
    for (let i = 0; i < Math.min(sX.length, signLabels.length); i++) {
      const t = signLabels[i];
      if (!t) continue;
      if (!byLabel.has(t)) byLabel.set(t, {sx:0, sy:0, sz:0, n:0});
      const g = byLabel.get(t);
      g.sx += sX[i]; g.sy += sY[i]; g.sz += sZ[i]; g.n += 1;
    }
    const lsX = [], lsY = [], lsZ = [], lText = [];
    for (const [k, g] of byLabel.entries()) {
      lsX.push(g.sx / g.n);
      lsY.push(g.sy / g.n);
      lsZ.push((g.sz / g.n) + 0.06);
      lText.push(k);
    }
    if (lText.length > 0) {
      traces.push({
        type:'scatter3d',
        mode:'markers+text',
        x:lsX, y:lsY, z:lsZ,
        text:lText,
        textposition:'top center',
        textfont:{size:22,color:'#f8fafc'},
        marker:{size:11,color:'rgba(17,24,39,0.92)',symbol:'circle'},
        name:'sign labels'
      });
    }
  }
  if (dX.length > 0) {
    traces.push({
      type:'scatter3d', mode:'markers', x:dX, y:dY, z:dZ,
      marker:{size:7,color:'#22c55e',symbol:'circle'},
      name:'door points'
    });
  }
  traces.push({type:'scatter3d', mode:'lines', x:tX, y:tY, z:tZ, line:{width:5,color:'#16a34a'}, name:'Final Trajectory'});
  const currentTraceIndex = traces.length;
  traces.push({type:'scatter3d', mode:'markers', x:[tX[0]], y:[tY[0]], z:[tZ[0]], marker:{size:6,color:'#ef4444'}, name:'Current'});

  Plotly.newPlot('traj', traces, {
    margin:{l:0,r:0,b:0,t:30},
    title:`VGGT + cuVSLAM (odom=${summary.odometry_mode || 'mono'}, inject=${summary.inject_mode}, w=${summary.prior_weight})`,
    scene:{xaxis:{title:'x'}, yaxis:{title:'y'}, zaxis:{title:'z'}, aspectmode:'data'}
  }, {displaylogo:false, responsive:true});

  slider.max = String(Math.max(0, poses.length - 1));
  function fmt(a) { return (a || [0,0,0]).map(v => Number(v).toFixed(4)).join(', '); }
  function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }

  function renderDetections() {
    if (!Array.isArray(detections) || detections.length === 0) {
      detList.innerHTML = '<div class="detmeta s">No detection cards.</div>';
      return;
    }
    const cards = detections.map((d, idx) => {
      const bbox = Array.isArray(d.bbox_xyxy) ? `[${d.bbox_xyxy.join(', ')}]` : '-';
      const txt = d.text ? d.text : (d.raw_text || '(empty)');
      const src = d.annotated_rel || d.crop_rel || '';
      const score = Number(d.score || 0).toFixed(3);
      const frame = Number.isFinite(d.frame_idx) && d.frame_idx >= 0 ? d.frame_idx : idx;
      return `
        <div class="detcard" data-frame="${frame}">
          <img src="${esc(src)}" alt="det" />
          <div class="detmeta">
            <div class="t">text: ${esc(txt)}</div>
            <div class="s">frame: ${esc(d.frame_name || '-')} | score: ${score}</div>
            <div class="s">bbox: ${esc(bbox)} | points: ${esc(d.point_count || 0)}</div>
          </div>
        </div>
      `;
    }).join('');
    detList.innerHTML = cards;
    detList.querySelectorAll('.detcard').forEach(el => {
      el.addEventListener('click', () => {
        const i = Number(el.getAttribute('data-frame') || 0);
        const clamped = Math.max(0, Math.min(i, poses.length - 1));
        slider.value = String(clamped);
        render(clamped);
      });
    });
  }

  function render(i) {
    const p = poses[i];
    frameIdx.textContent = String(i);
    img.src = `./${p.image_rel}`;
    const dRaw = Number.isFinite(p.err_raw_vs_vggt_m) ? p.err_raw_vs_vggt_m.toFixed(4) : '-';
    const dFuse = Number.isFinite(p.err_fused_vs_vggt_m) ? p.err_fused_vs_vggt_m.toFixed(4) : '-';
    meta.innerHTML = [
      `<div><b>Frames</b>: ${summary.frames} | <b>Runtime</b>: ${summary.runtime_sec}s | <b>FPS</b>: ${summary.fps}</div>`,
      `<div><b>odometry</b>: ${summary.odometry_mode || 'mono'} | <b>depth exact/near/miss</b>: ${(summary.depth_frames_exact||0)}/${(summary.depth_frames_nearest||0)}/${(summary.depth_frames_missing||0)} | <b>depth valid mean</b>: ${(summary.depth_valid_ratio_mean||0).toFixed(3)}</div>`,
      `<div><b>3D points</b>: ${summary.pointcloud_points || 0} | <b>source</b>: ${summary.pointcloud_source || 'none'}</div>`,
      `<div><b>sign points</b>: ${(summary.sign_points || 0)} | <b>door points</b>: ${(summary.door_points || 0)}</div>`,
      `<div><b>sign labels</b>: ${(summary.sign_labels && summary.sign_labels.length) ? summary.sign_labels.join(', ') : '(none)'}</div>`,
      `<div><b>timestamp_sec</b>: ${p.timestamp_sec.toFixed(6)} | <b>prior</b>: ${p.has_vggt_prior ? 'yes':'no'}</div>`,
      `<div><b>err raw-prior (m)</b>: ${dRaw} | <b>err fused-prior (m)</b>: ${dFuse}</div>`,
      `<div class="mono"><b>VGGT t</b>: [${fmt(p.vggt_t)}]</div>`,
      `<div class="mono"><b>SLAM t</b>: [${fmt(p.slam_t || p.odom_t)}]</div>`,
      `<div class="mono"><b>Fused t</b>: [${fmt(p.fused_t || p.slam_t || p.odom_t)}]</div>`
    ].join('');
    Plotly.restyle('traj', {x:[[tX[i]]], y:[[tY[i]]], z:[[tZ[i]]]}, [currentTraceIndex]);
  }
  slider.addEventListener('input', (e) => render(Number(e.target.value)));
  renderDetections();
  render(0);
})();
</script>
</body>
</html>
"""
    (out_dir / "viewer.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = args.output_dir / "frames_small"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    priors = load_vggt_priors(args.vggt_pose_txt)
    images = image_list(args.image_folder)
    if not images:
        raise RuntimeError(f"No images found in {args.image_folder}")
    if args.max_frames > 0:
        images = images[: args.max_frames]

    depth_log_dir = args.vggt_depth_log_dir if args.vggt_depth_log_dir is not None else (args.vggt_pose_txt.parent / "vggt_poses_logs")
    depth_index: Dict[int, Path] = {}
    intrinsics_from_vggt: Optional[Dict[str, float]] = None
    if args.odometry_mode == "rgbd_vggt":
        depth_index = build_vggt_depth_index(depth_log_dir)
        if not depth_index:
            raise RuntimeError(f"RGBD mode requested but no VGGT depth logs found in {depth_log_dir}")
        # Infer camera intrinsics directly from VGGT pointcloud geometry if user didn't provide intrinsics.
        if args.cam_fx <= 0 and args.cam_fy <= 0 and args.cam_cx <= 0 and args.cam_cy <= 0:
            first_key = sorted(depth_index.keys())[0]
            intrinsics_from_vggt = infer_intrinsics_from_vggt_pointcloud(depth_index[first_key])

    first_img = Image.open(images[0])
    width, height = first_img.size
    if intrinsics_from_vggt is not None:
        sx = float(width) / float(intrinsics_from_vggt["w"])
        sy = float(height) / float(intrinsics_from_vggt["h"])
        fx = float(intrinsics_from_vggt["fx"] * sx)
        fy = float(intrinsics_from_vggt["fy"] * sy)
        cx = float(intrinsics_from_vggt["cx"] * sx)
        cy = float(intrinsics_from_vggt["cy"] * sy)
    else:
        fx = float(args.cam_fx if args.cam_fx > 0 else 0.9 * width)
        fy = float(args.cam_fy if args.cam_fy > 0 else 0.9 * width)
        cx = float(args.cam_cx if args.cam_cx > 0 else (width - 1) * 0.5)
        cy = float(args.cam_cy if args.cam_cy > 0 else (height - 1) * 0.5)

    cam = cuvslam.Camera()
    cam.size = (int(width), int(height))
    cam.principal = (cx, cy)
    cam.focal = (fx, fy)

    odom_cfg = cuvslam.Tracker.OdometryConfig()
    if args.odometry_mode == "rgbd_vggt":
        rgbd_settings = cuvslam.Tracker.OdometryRGBDSettings()
        rgbd_settings.depth_scale_factor = float(args.depth_scale_factor)
        rgbd_settings.depth_camera_id = 0
        rgbd_settings.enable_depth_stereo_tracking = False
        odom_cfg.odometry_mode = cuvslam.Tracker.OdometryMode.RGBD
        odom_cfg.rgbd_settings = rgbd_settings
    else:
        odom_cfg.odometry_mode = cuvslam.Tracker.OdometryMode.Mono
    odom_cfg.use_gpu = bool(args.use_gpu)
    odom_cfg.enable_observations_export = True
    odom_cfg.enable_landmarks_export = True
    if hasattr(odom_cfg, "enable_final_landmarks_export"):
        odom_cfg.enable_final_landmarks_export = True

    slam_cfg = cuvslam.Tracker.SlamConfig()
    slam_cfg.use_gpu = bool(args.use_gpu)
    slam_cfg.enable_reading_internals = True
    slam_cfg.sync_mode = True

    tracker = cuvslam.Tracker(cuvslam.Rig([cam]), odom_cfg, slam_cfg)

    recs: List[dict] = []
    streamed_points: List[np.ndarray] = []
    last_fused_t: Optional[np.ndarray] = None
    last_fused_q: Optional[np.ndarray] = None
    depth_exact_count = 0
    depth_nearest_count = 0
    depth_missing_count = 0
    depth_valid_ratios: List[float] = []
    t0 = time.time()

    for i, p in enumerate(images):
        ts_ns = int((i + 1) * 1e9)
        if args.odometry_mode == "rgbd_vggt":
            rgb = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
            frame = np.ascontiguousarray(rgb[:, :, ::-1])  # BGR
        else:
            frame = np.array(Image.open(p).convert("L"), dtype=np.uint8)
        depth_u16 = None
        depth_meta = {"has_depth": False, "used_nearest": False, "path": None, "valid_ratio": 0.0}
        if args.odometry_mode == "rgbd_vggt":
            depth_u16, depth_meta = load_vggt_depth_u16(
                frame_idx0=i,
                depth_index=depth_index,
                out_w=width,
                out_h=height,
                metric=args.depth_metric,
                min_m=float(args.depth_min_m),
                max_m=float(args.depth_max_m),
                raw_scale=float(args.depth_scale_factor),
                nearest_max_gap=int(args.depth_nearest_max_gap),
            )
            if depth_u16 is None:
                depth_u16 = np.zeros((height, width), dtype=np.uint16)
                depth_missing_count += 1
            else:
                if depth_meta.get("used_nearest", False):
                    depth_nearest_count += 1
                else:
                    depth_exact_count += 1
                depth_valid_ratios.append(float(depth_meta.get("valid_ratio", 0.0)))
        prior = priors.get(i)
        has_prior = prior is not None
        prior_t = prior[0] if has_prior else None
        prior_q = prior[1] if has_prior else None

        if has_prior and args.inject_mode in {"pre", "both"}:
            if last_fused_t is not None and last_fused_q is not None:
                inject_t, inject_q = blend_pose(last_fused_t, last_fused_q, prior_t, prior_q, float(args.prior_weight))
            else:
                inject_t, inject_q = prior_t, prior_q
            tracker.set_slam_pose(pose_to_obj(inject_t, inject_q))

        if args.odometry_mode == "rgbd_vggt":
            pose_est, slam_pose = tracker.track(ts_ns, images=[frame], depths=[depth_u16])
        else:
            pose_est, slam_pose = tracker.track(ts_ns, images=[frame])

        odom_t = [0.0, 0.0, 0.0]
        odom_q = [0.0, 0.0, 0.0, 1.0]
        if pose_est.world_from_rig is not None:
            odom_t = [float(x) for x in pose_est.world_from_rig.pose.translation]
            odom_q = [float(x) for x in pose_est.world_from_rig.pose.rotation]

        slam_t = None
        slam_q = None
        if slam_pose is not None:
            slam_t = [float(x) for x in slam_pose.translation]
            slam_q = [float(x) for x in slam_pose.rotation]

        fused_t = None
        fused_q = None
        if has_prior and args.inject_mode in {"post", "both"} and slam_t is not None and slam_q is not None:
            ft, fq = blend_pose(np.array(slam_t), np.array(slam_q), prior_t, prior_q, float(args.prior_weight))
            tracker.set_slam_pose(pose_to_obj(ft, fq))
            fused_t = [float(x) for x in ft]
            fused_q = [float(x) for x in fq]
            last_fused_t, last_fused_q = ft, fq
        elif slam_t is not None and slam_q is not None:
            last_fused_t, last_fused_q = np.array(slam_t), np.array(slam_q)

        try:
            last_landmarks = tracker.get_last_landmarks()
            pts = landmarks_to_points(last_landmarks)
            if pts.shape[0] > 0:
                streamed_points.append(pts)
        except Exception:
            pass

        thumb_name = f"frame_{i:06d}.jpg"
        Image.open(p).convert("RGB").resize((args.thumb_w, args.thumb_h)).save(thumb_dir / thumb_name, quality=85)

        err_raw = None
        err_fused = None
        if has_prior and slam_t is not None:
            err_raw = float(np.linalg.norm(np.array(slam_t) - prior_t))
        if has_prior and fused_t is not None:
            err_fused = float(np.linalg.norm(np.array(fused_t) - prior_t))

        recs.append(
            {
                "frame_idx": i,
                "timestamp_sec": float(ts_ns / 1e9),
                "image_rel": f"frames_small/{thumb_name}",
                "has_vggt_prior": bool(has_prior),
                "vggt_t": [float(x) for x in prior_t] if has_prior else None,
                "vggt_q": [float(x) for x in prior_q] if has_prior else None,
                "odom_t": odom_t,
                "odom_q": odom_q,
                "slam_t": slam_t,
                "slam_q": slam_q,
                "fused_t": fused_t,
                "fused_q": fused_q,
                "err_raw_vs_vggt_m": err_raw,
                "err_fused_vs_vggt_m": err_fused,
                "depth_has": bool(depth_meta.get("has_depth", False)) if args.odometry_mode == "rgbd_vggt" else None,
                "depth_used_nearest": bool(depth_meta.get("used_nearest", False)) if args.odometry_mode == "rgbd_vggt" else None,
                "depth_valid_ratio": float(depth_meta.get("valid_ratio", 0.0)) if args.odometry_mode == "rgbd_vggt" else None,
            }
        )

    elapsed = time.time() - t0
    cloud_pts = try_get_final_points(tracker, args.max_cloud_points)
    cloud_source = "cuvslam_final"
    if cloud_pts.shape[0] == 0 and streamed_points:
        cloud_pts = np.concatenate(streamed_points, axis=0)
        cloud_pts = pack_cloud_points(cloud_pts, args.max_cloud_points)
        cloud_source = "cuvslam_stream"
    if cloud_pts.shape[0] == 0:
        vggt_log_dir = args.vggt_pose_txt.parent / "vggt_poses_logs"
        cloud_pts = load_vggt_log_points(vggt_log_dir, args.max_cloud_points)
        if cloud_pts.shape[0] > 0:
            cloud_source = "vggt_logs"
    raw_errs = [r["err_raw_vs_vggt_m"] for r in recs if r["err_raw_vs_vggt_m"] is not None]
    fused_errs = [r["err_fused_vs_vggt_m"] for r in recs if r["err_fused_vs_vggt_m"] is not None]
    semantic = load_category_points(args.image_folder.parent)
    sign_labels_nonempty = sorted({s for s in semantic["sign_labels"] if isinstance(s, str) and s.strip()})
    detections = load_detection_cards(args.image_folder.parent)
    summary = {
        "frames": len(recs),
        "runtime_sec": round(elapsed, 3),
        "fps": round(len(recs) / elapsed, 3) if elapsed > 0 else 0.0,
        "odometry_mode": args.odometry_mode,
        "inject_mode": args.inject_mode,
        "prior_weight": float(args.prior_weight),
        "use_gpu": bool(args.use_gpu),
        "prior_frames": int(sum(1 for r in recs if r["has_vggt_prior"])),
        "mean_err_raw_vs_vggt_m": float(np.mean(raw_errs)) if raw_errs else None,
        "mean_err_fused_vs_vggt_m": float(np.mean(fused_errs)) if fused_errs else None,
        "image_folder": str(args.image_folder),
        "vggt_pose_txt": str(args.vggt_pose_txt),
        "pointcloud_points": int(cloud_pts.shape[0]),
        "pointcloud_source": cloud_source if cloud_pts.shape[0] > 0 else "none",
        "sign_points": int(semantic["sign"].shape[0]),
        "door_points": int(semantic["door"].shape[0]),
        "sign_labels": sign_labels_nonempty,
        "detection_cards": int(len(detections)),
        "depth_log_dir": str(depth_log_dir),
        "depth_frames_exact": int(depth_exact_count),
        "depth_frames_nearest": int(depth_nearest_count),
        "depth_frames_missing": int(depth_missing_count),
        "depth_valid_ratio_mean": float(np.mean(depth_valid_ratios)) if depth_valid_ratios else 0.0,
        "camera_fx": float(fx),
        "camera_fy": float(fy),
        "camera_cx": float(cx),
        "camera_cy": float(cy),
        "intrinsics_source": "vggt_pointcloud_fit" if intrinsics_from_vggt is not None else "manual_or_default",
    }
    if intrinsics_from_vggt is not None:
        summary["vggt_fit_r2x"] = float(intrinsics_from_vggt.get("r2x", 0.0))
        summary["vggt_fit_r2y"] = float(intrinsics_from_vggt.get("r2y", 0.0))
        summary["vggt_fit_npz"] = str(intrinsics_from_vggt.get("source_npz", ""))

    (args.output_dir / "poses.json").write_text(json.dumps(recs, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cloud_obj = {"points": cloud_pts.tolist()}
    (args.output_dir / "pointcloud.json").write_text(json.dumps(cloud_obj), encoding="utf-8")
    semantic_obj = {
        "sign": semantic["sign"].tolist(),
        "door": semantic["door"].tolist(),
        "sign_labels": list(semantic["sign_labels"]),
    }
    (args.output_dir / "semantic_points.json").write_text(json.dumps(semantic_obj), encoding="utf-8")
    (args.output_dir / "detections_view.json").write_text(json.dumps(detections, ensure_ascii=False), encoding="utf-8")
    make_viewer(args.output_dir)
    print(f"[done] {args.output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
