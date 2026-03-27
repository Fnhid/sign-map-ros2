#!/usr/bin/env python3
import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate framewise pointcloud logs from Pi3/Pi3X for sign-map-ros2 pipeline."
    )
    parser.add_argument("--image_folder", type=Path, required=True)
    parser.add_argument("--log_path", type=Path, required=True, help="Output pose txt path (e.g., .../vggt_poses.txt)")
    parser.add_argument("--pi3_repo", type=Path, default=WORKSPACE_ROOT / "Pi3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conf_thres", type=float, default=0.05)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=6)
    parser.add_argument("--inject_condition", type=str, default="", help="Comma list, e.g. pose,depth")
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
    # Minimal robust conversion without scipy dependency.
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


def main():
    args = parse_args()
    image_paths = [p for p in args.image_folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    image_paths = sort_by_number(image_paths)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_folder}")

    # Lazy import PI3 after path injection.
    import sys

    sys.path.insert(0, str(args.pi3_repo))
    from pi3.models.pi3x import Pi3X
    from pi3.pipe.pi3x_vo import Pi3XVO
    from pi3.utils.basic import load_images_as_tensor

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        print("[PI3] warning: running on CPU. This may be very slow.")

    print(f"[PI3] loading model on {device}")
    model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
    pipe = Pi3XVO(model)

    print(f"[PI3] loading images from {args.image_folder}")
    imgs = load_images_as_tensor(str(args.image_folder), interval=1).to(device)  # (N,3,H,W)
    imgs_batched = imgs[None]  # (1,N,3,H,W)
    n_frames = imgs.shape[0]
    print(f"[PI3] frames: {n_frames}, tensor shape: {tuple(imgs_batched.shape)}")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    inject = [x.strip() for x in args.inject_condition.split(",") if x.strip()]

    with torch.no_grad():
        pred = pipe(
            imgs=imgs_batched,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            conf_thre=args.conf_thres,
            inject_condition=inject,
            dtype=dtype,
        )

    points = pred["points"][0].float().cpu().numpy()  # (N,H,W,3)
    conf = pred["conf"][0].float().cpu().numpy()      # (N,H,W)
    poses = pred["camera_poses"][0].float().cpu().numpy()  # (N,4,4)

    log_dir = Path(str(args.log_path).replace(".txt", "_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    if len(image_paths) != points.shape[0]:
        raise RuntimeError(
            f"Frame count mismatch: image_paths={len(image_paths)} vs points={points.shape[0]}"
        )

    written = 0
    with args.log_path.open("w", encoding="utf-8") as f:
        for i, img_path in enumerate(image_paths):
            fid = parse_frame_id(img_path.stem)
            if fid is None:
                continue
            pc = points[i]
            c = conf[i]
            mask = np.logical_and(np.isfinite(pc).all(axis=-1), c > args.conf_thres)
            np.savez(log_dir / f"{fid}.npz", pointcloud=pc.astype(np.float32), mask=mask.astype(bool))

            R = poses[i, :3, :3]
            t = poses[i, :3, 3]
            qx, qy, qz, qw = mat_to_quat_xyzw(R)
            line = f"{fid:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}\n"
            f.write(line)
            written += 1

    print(f"[PI3] wrote pose file: {args.log_path}")
    print(f"[PI3] wrote framewise logs: {log_dir} ({written} frames)")


if __name__ == "__main__":
    main()
