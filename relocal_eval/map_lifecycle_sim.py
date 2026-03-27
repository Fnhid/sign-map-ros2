#!/usr/bin/env python3
"""
Map lifecycle simulation on TUM RGB-D streams.

Goal:
- Repeatedly replay one sequence (or "infinite" stream semantics)
- Simulate multi-robot shared map growth
- Apply lifecycle management (score decay, tiering, prune, merge)
- Produce quantitative traces and an HTML report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Reuse tested utilities from relocal_eval pipeline.
from cold_start_relocal_eval import (
    DescriptorExtractor,
    TumFrame,
    load_or_compute_descriptors,
    load_tum_sequence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map lifecycle simulation with repeated TUM streams")
    parser.add_argument(
        "--tum_root",
        type=Path,
        default=Path(os.environ.get("TUM_ROOT", "/workspace/VGGT-SLAM-mixed-20260306_133724/datasets/tum")),
    )
    parser.add_argument("--sequence", type=str, default="freiburg1_room")
    parser.add_argument("--assoc_tolerance_sec", type=float, default=0.03)

    parser.add_argument("--descriptor_model", type=str, default="resnet18", choices=["resnet18", "dinov2_vits14", "salad"])
    parser.add_argument("--descriptor_batch_size", type=int, default=32)
    parser.add_argument("--descriptor_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lifecycle_sim" / "cache",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "lifecycle_sim" / "latest",
    )

    parser.add_argument("--num_robots", type=int, default=3)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_cycles", type=int, default=8, help="0 means no cycle cap (use --max_steps).")
    parser.add_argument("--max_steps", type=int, default=0, help="Global observation-step cap. 0 means derived from cycles.")
    parser.add_argument("--seed", type=int, default=7)

    # Keyframe and duplicate gating
    parser.add_argument("--kf_min_trans_m", type=float, default=0.12)
    parser.add_argument("--kf_min_rot_deg", type=float, default=5.0)
    parser.add_argument("--dup_desc_sim", type=float, default=0.985)
    parser.add_argument("--dup_trans_m", type=float, default=0.18)
    parser.add_argument("--dup_rot_deg", type=float, default=8.0)

    # Retrieval + geometric verification
    parser.add_argument("--retrieval_topk", type=int, default=8)
    parser.add_argument("--retrieval_min_gap_steps", type=int, default=50)
    parser.add_argument("--verify_trans_m", type=float, default=0.8)
    parser.add_argument("--verify_rot_deg", type=float, default=20.0)

    # Score / maintenance
    parser.add_argument("--maintenance_every", type=int, default=200)
    parser.add_argument("--score_init", type=float, default=1.0)
    parser.add_argument("--score_obs_gain", type=float, default=0.25)
    parser.add_argument("--score_loop_gain", type=float, default=1.2)
    parser.add_argument("--score_loop_penalty", type=float, default=0.4)
    parser.add_argument("--score_decay_per_step", type=float, default=0.0015)

    # Tiering / pruning
    parser.add_argument("--tier_hot_age_steps", type=int, default=900)
    parser.add_argument("--tier_hot_score", type=float, default=2.0)
    parser.add_argument("--tier_warm_score", type=float, default=0.8)
    parser.add_argument("--prune_score_th", type=float, default=0.45)
    parser.add_argument("--prune_min_age_steps", type=int, default=500)
    parser.add_argument("--max_nodes", type=int, default=2500)

    # Merge
    parser.add_argument("--merge_enable", type=int, default=1, choices=[0, 1])
    parser.add_argument("--merge_trans_m", type=float, default=0.25)
    parser.add_argument("--merge_rot_deg", type=float, default=12.0)
    parser.add_argument("--merge_desc_sim", type=float, default=0.985)

    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return name


def rotation_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = np.clip((np.trace(Ra.T @ Rb) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def estimate_blur_quality(path: str) -> float:
    # Lightweight no-opencv quality score: higher is sharper.
    img = Image.open(path).convert("L").resize((160, 120))
    a = np.asarray(img, dtype=np.float32) / 255.0
    dx = np.diff(a, axis=1)
    dy = np.diff(a, axis=0)
    # Gradient energy as a proxy for focus/sharpness.
    return float(np.mean(dx * dx) + np.mean(dy * dy))


def load_or_compute_quality_cache(frames: Sequence[TumFrame], cache_path: Path) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    paths = [f.rgb_path for f in frames]
    if cache_path.exists():
        try:
            payload = np.load(cache_path, allow_pickle=True)
            if payload["rgb_paths"].tolist() == paths:
                return payload["quality"].astype(np.float32)
        except Exception:
            pass
    q = np.array([estimate_blur_quality(p) for p in paths], dtype=np.float32)
    np.savez_compressed(cache_path, rgb_paths=np.array(paths, dtype=object), quality=q)
    return q


@dataclass
class RobotState:
    rid: int
    start_offset: int
    last_insert_t: Optional[np.ndarray] = None
    last_insert_R: Optional[np.ndarray] = None
    last_insert_desc: Optional[np.ndarray] = None


@dataclass
class MapNode:
    nid: int
    t: np.ndarray
    R: np.ndarray
    desc: np.ndarray
    created_step: int
    last_seen_step: int
    score: float
    observations: int = 1
    loop_verified: int = 0
    loop_rejected: int = 0
    tier: str = "hot"
    retrieval_active: bool = True
    robots_seen: set = field(default_factory=set)


class LifecycleMap:
    def __init__(self, args: argparse.Namespace, desc_dim: int) -> None:
        self.args = args
        self.desc_dim = int(desc_dim)
        self.nodes: Dict[int, MapNode] = {}
        self.next_nid = 0

        self.inserted = 0
        self.duplicate_updates = 0
        self.pruned = 0
        self.merged = 0
        self.loop_verified = 0
        self.loop_rejected = 0

        self._retrieval_ids: List[int] = []
        self._retrieval_desc: np.ndarray = np.zeros((0, self.desc_dim), dtype=np.float32)
        self._index_dirty = True

        self.last_maintenance_step = 0
        self.metrics: List[Dict[str, float]] = []

    def _rebuild_index(self) -> None:
        ids = [nid for nid, n in self.nodes.items() if n.retrieval_active]
        ids.sort()
        if not ids:
            self._retrieval_ids = []
            self._retrieval_desc = np.zeros((0, self.desc_dim), dtype=np.float32)
            self._index_dirty = False
            return
        self._retrieval_ids = ids
        self._retrieval_desc = np.stack([self.nodes[nid].desc for nid in ids], axis=0).astype(np.float32)
        self._index_dirty = False

    def _ensure_index(self) -> None:
        if self._index_dirty:
            self._rebuild_index()

    def _add_node(self, t: np.ndarray, R: np.ndarray, desc: np.ndarray, step: int, rid: int) -> int:
        nid = self.next_nid
        self.next_nid += 1
        node = MapNode(
            nid=nid,
            t=t.copy(),
            R=R.copy(),
            desc=desc.copy(),
            created_step=int(step),
            last_seen_step=int(step),
            score=float(self.args.score_init),
            robots_seen={int(rid)},
        )
        self.nodes[nid] = node
        self.inserted += 1
        self._index_dirty = True
        return nid

    def _update_duplicate(self, nid: int, step: int, rid: int) -> None:
        n = self.nodes[nid]
        n.last_seen_step = int(step)
        n.observations += 1
        n.score += float(self.args.score_obs_gain)
        n.robots_seen.add(int(rid))
        self.duplicate_updates += 1

    def process_observation(
        self,
        rid: int,
        step: int,
        t: np.ndarray,
        R: np.ndarray,
        desc: np.ndarray,
        quality: float,
        robot_state: RobotState,
    ) -> None:
        # Low-quality observation still ages map but does not update it.
        if quality <= 1e-8:
            return

        self._ensure_index()
        current_nid: Optional[int] = None

        # 1) Duplicate suppression vs existing retrieval-active nodes.
        if self._retrieval_desc.shape[0] > 0:
            sims = self._retrieval_desc @ desc
            j = int(np.argmax(sims))
            best_sim = float(sims[j])
            cand_id = self._retrieval_ids[j]
            cand = self.nodes[cand_id]
            dt = float(np.linalg.norm(t - cand.t))
            dr = rotation_error_deg(R, cand.R)
            if (
                best_sim >= float(self.args.dup_desc_sim)
                and dt <= float(self.args.dup_trans_m)
                and dr <= float(self.args.dup_rot_deg)
            ):
                self._update_duplicate(cand_id, step=step, rid=rid)
                current_nid = cand_id

        # 2) If not duplicate, apply keyframe gate and insert.
        if current_nid is None:
            do_insert = False
            if robot_state.last_insert_t is None:
                do_insert = True
            else:
                dt = float(np.linalg.norm(t - robot_state.last_insert_t))
                dr = rotation_error_deg(R, robot_state.last_insert_R)
                sim_prev = float(np.dot(desc, robot_state.last_insert_desc))
                if (
                    dt >= float(self.args.kf_min_trans_m)
                    or dr >= float(self.args.kf_min_rot_deg)
                    or sim_prev <= float(self.args.dup_desc_sim)
                ):
                    do_insert = True
            if do_insert:
                current_nid = self._add_node(t=t, R=R, desc=desc, step=step, rid=rid)
                robot_state.last_insert_t = t.copy()
                robot_state.last_insert_R = R.copy()
                robot_state.last_insert_desc = desc.copy()

        # 3) Loop verification scoring (if we have a current node and retrieval index).
        if current_nid is not None:
            self._ensure_index()
            if self._retrieval_desc.shape[0] > 1:
                sims = self._retrieval_desc @ desc
                order = np.argsort(-sims)
                k = int(self.args.retrieval_topk)
                checked = 0
                found_verified = False
                for jj in order.tolist():
                    cand_id = self._retrieval_ids[jj]
                    if cand_id == current_nid:
                        continue
                    cand = self.nodes[cand_id]
                    # Temporal gate in simulation step-space.
                    if abs(step - cand.last_seen_step) < int(self.args.retrieval_min_gap_steps):
                        continue
                    checked += 1
                    dt = float(np.linalg.norm(t - cand.t))
                    dr = rotation_error_deg(R, cand.R)
                    if dt <= float(self.args.verify_trans_m) and dr <= float(self.args.verify_rot_deg):
                        cur = self.nodes[current_nid]
                        cur.loop_verified += 1
                        cur.score += float(self.args.score_loop_gain)
                        cand.loop_verified += 1
                        cand.score += float(self.args.score_loop_gain) * 0.5
                        self.loop_verified += 1
                        found_verified = True
                        break
                    else:
                        # Rejected candidate causes a mild penalty.
                        cur = self.nodes[current_nid]
                        cur.loop_rejected += 1
                        cur.score -= float(self.args.score_loop_penalty) * 0.25
                        self.loop_rejected += 1
                    if checked >= k:
                        break
                if not found_verified:
                    self.nodes[current_nid].score -= float(self.args.score_loop_penalty) * 0.1

    def _assign_tiers(self, step: int) -> None:
        for n in self.nodes.values():
            age = step - n.last_seen_step
            if age <= int(self.args.tier_hot_age_steps) and n.score >= float(self.args.tier_hot_score):
                n.tier = "hot"
                n.retrieval_active = True
            elif n.score >= float(self.args.tier_warm_score):
                n.tier = "warm"
                n.retrieval_active = True
            else:
                n.tier = "cold"
                n.retrieval_active = False

    def _apply_decay(self, step: int) -> None:
        since = max(1, step - self.last_maintenance_step)
        decay = float(self.args.score_decay_per_step) * float(since)
        for n in self.nodes.values():
            unseen = max(0, step - n.last_seen_step)
            n.score -= decay * (1.0 + 0.25 * min(unseen, 1000) / 1000.0)
            if n.score < 0.0:
                n.score = 0.0

    def _prune_nodes(self, step: int) -> None:
        kill = []
        for nid, n in self.nodes.items():
            age = step - n.last_seen_step
            if n.score < float(self.args.prune_score_th) and age >= int(self.args.prune_min_age_steps):
                kill.append(nid)
        for nid in kill:
            self.nodes.pop(nid, None)
        self.pruned += len(kill)

        # Hard cap with score-first eviction (prefer removing cold).
        cap = int(self.args.max_nodes)
        if len(self.nodes) > cap:
            ids = list(self.nodes.keys())
            ids.sort(key=lambda x: (self.nodes[x].tier != "cold", self.nodes[x].score, self.nodes[x].observations))
            overflow = len(self.nodes) - cap
            drop = ids[:overflow]
            for nid in drop:
                self.nodes.pop(nid, None)
            self.pruned += len(drop)

    def _merge_nodes(self) -> None:
        if int(self.args.merge_enable) == 0:
            return
        ids = [nid for nid, n in self.nodes.items() if n.retrieval_active]
        if len(ids) < 2:
            return
        ids.sort(key=lambda x: self.nodes[x].score, reverse=True)
        used = set()
        removed = set()
        for i, nid in enumerate(ids):
            if nid in used or nid in removed or nid not in self.nodes:
                continue
            base = self.nodes[nid]
            group = [nid]
            for j in range(i + 1, len(ids)):
                mid = ids[j]
                if mid in used or mid in removed or mid not in self.nodes:
                    continue
                cand = self.nodes[mid]
                dt = float(np.linalg.norm(base.t - cand.t))
                if dt > float(self.args.merge_trans_m):
                    continue
                dr = rotation_error_deg(base.R, cand.R)
                if dr > float(self.args.merge_rot_deg):
                    continue
                sim = float(np.dot(base.desc, cand.desc))
                if sim < float(self.args.merge_desc_sim):
                    continue
                group.append(mid)
                used.add(mid)
            if len(group) <= 1:
                continue
            # Merge into base node.
            weights = np.array([max(self.nodes[g].score, 1e-3) for g in group], dtype=np.float64)
            weights = weights / np.sum(weights)
            t_stack = np.stack([self.nodes[g].t for g in group], axis=0)
            d_stack = np.stack([self.nodes[g].desc for g in group], axis=0)
            base.t = np.sum(t_stack * weights[:, None], axis=0)
            new_desc = np.sum(d_stack * weights[:, None], axis=0)
            nrm = float(np.linalg.norm(new_desc))
            if nrm > 1e-12:
                new_desc = new_desc / nrm
            base.desc = new_desc.astype(np.float32)
            base.observations = int(sum(self.nodes[g].observations for g in group))
            base.loop_verified = int(sum(self.nodes[g].loop_verified for g in group))
            base.loop_rejected = int(sum(self.nodes[g].loop_rejected for g in group))
            base.last_seen_step = int(max(self.nodes[g].last_seen_step for g in group))
            base.score = float(sum(self.nodes[g].score for g in group) * 0.85)
            rs = set()
            for g in group:
                rs |= self.nodes[g].robots_seen
            base.robots_seen = rs
            # Keep base orientation (higher score anchor behavior).
            for g in group[1:]:
                removed.add(g)
        for nid in removed:
            self.nodes.pop(nid, None)
        self.merged += len(removed)

    def maintenance(self, step: int) -> None:
        self._apply_decay(step=step)
        self._assign_tiers(step=step)
        self._prune_nodes(step=step)
        self._merge_nodes()
        self._assign_tiers(step=step)
        self._index_dirty = True
        self.last_maintenance_step = int(step)
        self.snapshot_metrics(step=step)

    def snapshot_metrics(self, step: int) -> None:
        if self.nodes:
            scores = np.array([n.score for n in self.nodes.values()], dtype=np.float64)
            mean_score = float(np.mean(scores))
            p50_score = float(np.percentile(scores, 50))
            p90_score = float(np.percentile(scores, 90))
        else:
            mean_score = p50_score = p90_score = 0.0
        hot = sum(1 for n in self.nodes.values() if n.tier == "hot")
        warm = sum(1 for n in self.nodes.values() if n.tier == "warm")
        cold = sum(1 for n in self.nodes.values() if n.tier == "cold")
        retrieval = sum(1 for n in self.nodes.values() if n.retrieval_active)
        mem_mb = float((len(self.nodes) * (self.desc_dim * 4 + 256)) / (1024.0 * 1024.0))
        self.metrics.append(
            {
                "step": float(step),
                "nodes_total": float(len(self.nodes)),
                "nodes_hot": float(hot),
                "nodes_warm": float(warm),
                "nodes_cold": float(cold),
                "nodes_retrieval": float(retrieval),
                "score_mean": mean_score,
                "score_p50": p50_score,
                "score_p90": p90_score,
                "inserted_total": float(self.inserted),
                "duplicate_updates_total": float(self.duplicate_updates),
                "pruned_total": float(self.pruned),
                "merged_total": float(self.merged),
                "loop_verified_total": float(self.loop_verified),
                "loop_rejected_total": float(self.loop_rejected),
                "memory_est_mb": mem_mb,
            }
        )


def write_metrics_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_plots(rows: Sequence[Dict[str, float]], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return {}
    x = np.array([r["step"] for r in rows], dtype=np.float64)
    y_nodes = np.array([r["nodes_total"] for r in rows], dtype=np.float64)
    y_hot = np.array([r["nodes_hot"] for r in rows], dtype=np.float64)
    y_warm = np.array([r["nodes_warm"] for r in rows], dtype=np.float64)
    y_cold = np.array([r["nodes_cold"] for r in rows], dtype=np.float64)
    y_score_p50 = np.array([r["score_p50"] for r in rows], dtype=np.float64)
    y_score_p90 = np.array([r["score_p90"] for r in rows], dtype=np.float64)
    y_insert = np.array([r["inserted_total"] for r in rows], dtype=np.float64)
    y_pruned = np.array([r["pruned_total"] for r in rows], dtype=np.float64)
    y_merged = np.array([r["merged_total"] for r in rows], dtype=np.float64)
    y_verified = np.array([r["loop_verified_total"] for r in rows], dtype=np.float64)

    files = {}

    fig = plt.figure(figsize=(9, 4))
    plt.plot(x, y_nodes, label="total")
    plt.plot(x, y_hot, label="hot")
    plt.plot(x, y_warm, label="warm")
    plt.plot(x, y_cold, label="cold")
    plt.xlabel("Step")
    plt.ylabel("Node Count")
    plt.title("Node Count Over Time")
    plt.legend()
    plt.tight_layout()
    p = out_dir / "nodes_over_time.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files["nodes_over_time"] = p.name

    fig = plt.figure(figsize=(9, 4))
    plt.plot(x, y_score_p50, label="score p50")
    plt.plot(x, y_score_p90, label="score p90")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Score Distribution Over Time")
    plt.legend()
    plt.tight_layout()
    p = out_dir / "scores_over_time.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files["scores_over_time"] = p.name

    fig = plt.figure(figsize=(9, 4))
    plt.plot(x, y_insert, label="inserted")
    plt.plot(x, y_pruned, label="pruned")
    plt.plot(x, y_merged, label="merged")
    plt.plot(x, y_verified, label="loop verified")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Count")
    plt.title("Lifecycle Event Counters")
    plt.legend()
    plt.tight_layout()
    p = out_dir / "events_over_time.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    files["events_over_time"] = p.name

    return files


def build_html_report(
    out_path: Path,
    summary: Dict[str, object],
    plot_files: Dict[str, str],
    metrics_csv_name: str,
) -> None:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Map Lifecycle Simulation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
    th {{ background: #f3f5f8; }}
    img {{ max-width: 100%; border: 1px solid #d8dde6; border-radius: 6px; }}
    .card {{ margin-bottom: 16px; padding: 10px; border: 1px solid #d8dde6; border-radius: 8px; }}
    .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <h2>Map Lifecycle Simulation (TUM)</h2>
  <div class="card">
    <div><b>Sequence</b>: {summary["sequence"]}</div>
    <div><b>Robots</b>: {summary["num_robots"]}</div>
    <div><b>Total Steps</b>: {summary["total_steps"]}</div>
    <div><b>Frames per Cycle</b>: {summary["frames_per_cycle"]}</div>
    <div><b>Final Nodes</b>: {summary["final_nodes"]}</div>
    <div><b>Final Retrieval Nodes</b>: {summary["final_retrieval_nodes"]}</div>
    <div><b>Inserted</b>: {summary["inserted_total"]} | <b>Pruned</b>: {summary["pruned_total"]} | <b>Merged</b>: {summary["merged_total"]}</div>
    <div><b>Loop Verified</b>: {summary["loop_verified_total"]} | <b>Loop Rejected</b>: {summary["loop_rejected_total"]}</div>
    <div class="mono">files: <a href="./summary.json">summary.json</a> | <a href="./{metrics_csv_name}">{metrics_csv_name}</a></div>
  </div>
  <div class="card">
    <h3>Node Counts</h3>
    <img src="./{plot_files.get("nodes_over_time", "")}" alt="nodes">
  </div>
  <div class="card">
    <h3>Scores</h3>
    <img src="./{plot_files.get("scores_over_time", "")}" alt="scores">
  </div>
  <div class="card">
    <h3>Lifecycle Events</h3>
    <img src="./{plot_files.get("events_over_time", "")}" alt="events">
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    descriptor_device = resolve_device(args.descriptor_device)
    print(f"[cfg] sequence={args.sequence}, descriptor={args.descriptor_model}, device={descriptor_device}")

    frames = load_tum_sequence(args.tum_root, args.sequence, assoc_tol=float(args.assoc_tolerance_sec))
    if not frames:
        raise RuntimeError(f"No frames loaded for sequence={args.sequence}")
    n_frames = len(frames)
    print(f"[info] loaded frames={n_frames}")

    extractor = DescriptorExtractor(
        model_name=args.descriptor_model,
        device=descriptor_device,
        salad_root=Path("/workspace/Depth-Anything-3-ori/da3_streaming/loop_utils/salad"),
        salad_ckpt_path=Path("/workspace/Depth-Anything-3-ori/da3_streaming/weights/dino_salad.ckpt"),
    )
    desc_cache = args.cache_dir / f"{args.sequence}_{args.descriptor_model}.npz"
    desc = load_or_compute_descriptors(frames, extractor, cache_path=desc_cache, batch_size=int(args.descriptor_batch_size))
    if desc.shape[0] != n_frames:
        raise RuntimeError("Descriptor count mismatch")
    # Ensure normalized.
    desc = desc.astype(np.float32)
    dnorm = np.linalg.norm(desc, axis=1, keepdims=True)
    dnorm = np.where(dnorm < 1e-12, 1.0, dnorm)
    desc = desc / dnorm

    quality_cache = args.cache_dir / f"{args.sequence}_quality.npz"
    quality = load_or_compute_quality_cache(frames, quality_cache)

    poses_t = []
    poses_R = []
    for f in frames:
        T = np.array(f.pose_w_c, dtype=np.float64)
        poses_t.append(T[:3, 3].copy())
        poses_R.append(T[:3, :3].copy())
    poses_t = np.stack(poses_t, axis=0)

    if int(args.num_robots) <= 0:
        raise ValueError("--num_robots must be >= 1")
    robot_offsets = [int(i * n_frames / int(args.num_robots)) % n_frames for i in range(int(args.num_robots))]
    robots = [RobotState(rid=i, start_offset=robot_offsets[i]) for i in range(int(args.num_robots))]

    if int(args.max_steps) > 0:
        total_steps = int(args.max_steps)
    else:
        if int(args.max_cycles) <= 0:
            raise ValueError("Use --max_steps when --max_cycles=0.")
        total_steps = int(args.max_cycles) * n_frames
    print(f"[cfg] total_steps={total_steps}, robots={len(robots)}")

    sim = LifecycleMap(args=args, desc_dim=int(desc.shape[1]))

    for step in range(total_steps):
        rid = step % len(robots)
        robot_tick = step // len(robots)
        rs = robots[rid]
        frame_idx = (rs.start_offset + robot_tick * int(args.frame_stride)) % n_frames
        sim.process_observation(
            rid=rid,
            step=step,
            t=poses_t[frame_idx],
            R=poses_R[frame_idx],
            desc=desc[frame_idx],
            quality=float(quality[frame_idx]),
            robot_state=rs,
        )
        if (step + 1) % int(args.maintenance_every) == 0:
            sim.maintenance(step=step + 1)

    # Final maintenance + snapshot.
    sim.maintenance(step=total_steps)

    metrics_csv = args.output_dir / "metrics.csv"
    write_metrics_csv(metrics_csv, sim.metrics)

    plots_dir = args.output_dir / "plots"
    plot_files = make_plots(sim.metrics, out_dir=plots_dir)

    final = sim.metrics[-1] if sim.metrics else {}
    args_json = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_json[k] = str(v)
        else:
            args_json[k] = v

    summary = {
        "sequence": args.sequence,
        "num_robots": int(args.num_robots),
        "frames_per_cycle": int(n_frames),
        "total_steps": int(total_steps),
        "descriptor_model": args.descriptor_model,
        "descriptor_device": descriptor_device,
        "final_nodes": int(final.get("nodes_total", 0)),
        "final_retrieval_nodes": int(final.get("nodes_retrieval", 0)),
        "final_hot_nodes": int(final.get("nodes_hot", 0)),
        "final_warm_nodes": int(final.get("nodes_warm", 0)),
        "final_cold_nodes": int(final.get("nodes_cold", 0)),
        "inserted_total": int(sim.inserted),
        "duplicate_updates_total": int(sim.duplicate_updates),
        "pruned_total": int(sim.pruned),
        "merged_total": int(sim.merged),
        "loop_verified_total": int(sim.loop_verified),
        "loop_rejected_total": int(sim.loop_rejected),
        "final_score_mean": float(final.get("score_mean", 0.0)),
        "final_score_p50": float(final.get("score_p50", 0.0)),
        "final_score_p90": float(final.get("score_p90", 0.0)),
        "args": args_json,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    build_html_report(
        out_path=args.output_dir / "viewer.html",
        summary=summary,
        plot_files=plot_files,
        metrics_csv_name=metrics_csv.name,
    )
    print(f"[done] output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
