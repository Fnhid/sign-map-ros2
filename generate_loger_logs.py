#!/usr/bin/env python3
import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LoGeR inference and convert outputs to sign-map-ros2 framewise logs."
    )
    parser.add_argument("--image_folder", type=Path, required=True)
    parser.add_argument("--log_path", type=Path, required=True, help="Output pose txt path (e.g., .../vggt_poses.txt)")
    parser.add_argument("--loger_repo", type=Path, default=PROJECT_ROOT / "LoGeR")
    parser.add_argument("--model_name", type=str, default="ckpts/LoGeR_star/latest.pt")
    parser.add_argument("--config_path", type=str, default="ckpts/LoGeR_star/original_config.yaml")
    parser.add_argument("--output_folder", type=Path, default=None, help="Optional raw LoGeR output folder")
    parser.add_argument("--seq_name", type=str, default="signmap_loger")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--overlap_size", type=int, default=3)
    parser.add_argument("--conf_thres", type=float, default=0.05)
    return parser.parse_args()


def parse_frame_id(name: str):
    m = re.search(r"\d+(?:\.\d+)?", name)
    if not m:
        return None
    return float(m.group())


def sort_by_number(paths):
    def key_fn(p: Path):
        fid = parse_frame_id(p.stem)
        return (float("inf") if fid is None else fid, p.name)

    return sorted(paths, key=key_fn)


def mat_to_quat_xyzw(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_seq(x):
    arr = to_numpy(x)
    while arr.ndim > 0 and arr.shape[0] == 1 and arr.ndim >= 2:
        arr = arr[0]
    return arr


def resolve_path(repo: Path, maybe_rel: str):
    p = Path(maybe_rel)
    if p.is_absolute():
        return p
    return (repo / p).resolve()


def ensure_pose_4x4(pose):
    pose = np.asarray(pose)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = np.eye(4, dtype=pose.dtype)
        out[:3, :4] = pose
        return out
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


def main():
    args = parse_args()
    if not args.image_folder.exists():
        raise FileNotFoundError(f"image_folder not found: {args.image_folder}")

    image_paths = [p for p in args.image_folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    image_paths = sort_by_number(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_folder}")

    loger_repo = args.loger_repo.resolve()
    if not loger_repo.exists():
        raise FileNotFoundError(f"LoGeR repo not found: {loger_repo}")

    model_path = resolve_path(loger_repo, args.model_name)
    config_path = resolve_path(loger_repo, args.config_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"LoGeR checkpoint not found: {model_path}\n"
            "Download example:\n"
            "wget -O ckpts/LoGeR_star/latest.pt \"https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR_star/latest.pt?download=true\""
        )
    if not config_path.exists():
        raise FileNotFoundError(f"LoGeR config not found: {config_path}")

    raw_output = args.output_folder or (args.log_path.parent / "loger_raw")
    raw_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "demo_viser.py",
        "--input",
        str(args.image_folder.resolve()),
        "--start_frame",
        "0",
        "--end_frame",
        "-1",
        "--stride",
        "1",
        "--skip_viser",
        "--output_folder",
        str(raw_output.resolve()),
        "--seq_name",
        args.seq_name,
        "--window_size",
        str(args.window_size),
        "--overlap_size",
        str(args.overlap_size),
        "--model_name",
        str(model_path),
        "--config",
        str(config_path),
    ]
    print("[LoGeR] running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(loger_repo), check=True)

    expected_pt = raw_output / f"{args.seq_name}_0_-1_1.pt"
    if expected_pt.exists():
        pred_path = expected_pt
    else:
        pts = sorted(raw_output.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not pts:
            raise FileNotFoundError(f"No prediction .pt found in {raw_output}")
        pred_path = pts[0]
    print(f"[LoGeR] reading prediction: {pred_path}")

    preds = torch.load(pred_path, map_location="cpu", weights_only=False)
    if not isinstance(preds, dict):
        raise RuntimeError("Unexpected LoGeR prediction format: expected dict.")

    points = None
    for k in ("points", "world_points", "world_points_from_depth"):
        if k in preds:
            points = squeeze_seq(preds[k])
            break
    if points is None:
        raise RuntimeError(f"No 3D point tensor key found in prediction keys: {sorted(preds.keys())}")

    poses = preds.get("camera_poses", None)
    if poses is None:
        raise RuntimeError("LoGeR prediction does not contain camera_poses.")
    poses = squeeze_seq(poses)

    conf = None
    for k in ("conf", "world_points_conf", "depth_conf"):
        if k in preds:
            conf = squeeze_seq(preds[k])
            break
    if conf is None:
        conf = np.ones(points.shape[:3], dtype=np.float32)

    points = np.asarray(points)
    poses = np.asarray(poses)
    conf = np.asarray(conf)

    if points.ndim != 4 or points.shape[-1] != 3:
        raise RuntimeError(f"Unexpected points shape: {points.shape}, expected (N,H,W,3)")
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf.ndim != 3:
        raise RuntimeError(f"Unexpected conf shape: {conf.shape}, expected (N,H,W)")
    if poses.ndim != 3:
        raise RuntimeError(f"Unexpected camera_poses shape: {poses.shape}, expected (N,4,4) or (N,3,4)")

    n = min(len(image_paths), points.shape[0], conf.shape[0], poses.shape[0])
    if n <= 0:
        raise RuntimeError("No valid frames available after LoGeR conversion.")
    if n < len(image_paths):
        print(f"[LoGeR] warning: using first {n}/{len(image_paths)} frames due to shape mismatch.")

    log_dir = Path(str(args.log_path).replace(".txt", "_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.log_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            fid = parse_frame_id(image_paths[i].stem)
            if fid is None:
                continue

            pc = points[i]
            c = conf[i]
            if c.shape != pc.shape[:2]:
                c = np.full(pc.shape[:2], float(np.nanmean(c)), dtype=np.float32)
            mask = np.logical_and(np.isfinite(pc).all(axis=-1), np.isfinite(c) & (c > float(args.conf_thres)))
            np.savez(log_dir / f"{fid}.npz", pointcloud=pc.astype(np.float32), mask=mask.astype(bool))

            pose44 = ensure_pose_4x4(poses[i])
            R = pose44[:3, :3]
            t = pose44[:3, 3]
            qx, qy, qz, qw = mat_to_quat_xyzw(R)
            f.write(f"{fid:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n")
            written += 1

    print(f"[LoGeR] wrote pose file: {args.log_path}")
    print(f"[LoGeR] wrote framewise logs: {log_dir} ({written} frames)")


if __name__ == "__main__":
    main()
