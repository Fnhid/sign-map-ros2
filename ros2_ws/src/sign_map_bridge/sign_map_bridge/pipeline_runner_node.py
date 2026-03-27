#!/usr/bin/env python3
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PipelineRunnerNode(Node):
    def __init__(self) -> None:
        super().__init__("pipeline_runner_node")
        project_root = Path(__file__).resolve().parents[3]

        self.declare_parameter("frames_dir", "/tmp/sign_map_live/frames")
        self.declare_parameter("run_root", str(project_root / "runs" / "ros2_live"))
        self.declare_parameter("run_script", str(project_root / "run_full_pipeline.sh"))
        self.declare_parameter("min_frames_to_run", 40)
        self.declare_parameter("run_interval_sec", 10.0)
        self.declare_parameter("video_fps", 2)
        self.declare_parameter("backend", "vggt")

        self.frames_dir = Path(self.get_parameter("frames_dir").get_parameter_value().string_value).resolve()
        self.run_root = Path(self.get_parameter("run_root").get_parameter_value().string_value).resolve()
        self.run_script = Path(self.get_parameter("run_script").get_parameter_value().string_value).resolve()
        self.min_frames_to_run = int(self.get_parameter("min_frames_to_run").get_parameter_value().integer_value)
        self.video_fps = int(self.get_parameter("video_fps").get_parameter_value().integer_value)
        self.backend = self.get_parameter("backend").get_parameter_value().string_value

        self.run_root.mkdir(parents=True, exist_ok=True)

        self.last_processed_count = 0
        self.proc = None
        self.current_run_dir = None

        self.new_run_pub = self.create_publisher(String, "/sign_map/new_run", 10)
        interval = float(self.get_parameter("run_interval_sec").get_parameter_value().double_value)
        self.timer = self.create_timer(max(1.0, interval), self.on_timer)
        self.get_logger().info("Pipeline runner started (CPU mode, B+digits OCR policy).")

    def _frame_count(self) -> int:
        if not self.frames_dir.exists():
            return 0
        return len([p for p in self.frames_dir.glob("*.jpg")])

    def on_timer(self) -> None:
        if self.proc is not None:
            ret = self.proc.poll()
            if ret is None:
                return
            if ret == 0 and self.current_run_dir is not None:
                msg = String()
                msg.data = str(self.current_run_dir)
                self.new_run_pub.publish(msg)
                self.get_logger().info(f"new run published: {self.current_run_dir}")
            else:
                self.get_logger().warning(f"pipeline exited with code {ret}")
            self.proc = None
            return

        n = self._frame_count()
        if n < self.min_frames_to_run:
            return
        if n == self.last_processed_count:
            return

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.run_root / f"live_{ts}"
        env = os.environ.copy()
        env.update(
            {
                "USE_CPU": "1",
                "BACKEND": self.backend,
                "DETECTOR_BACKEND": "hybrid",
                "OCR_NUMERIC_MODE": "1",
                "OCR_NUMERIC_ONLY": "1",
                "OCR_NUMERIC_CHARS": "B0123456789 ",
                "OCR_TEXT_PATTERN": "[^B0-9\\s]+",
                "OCR_NUMERIC_MIN_DIGITS": "2",
                "STAGE2_MAX_FRAMES": "0",
            }
        )

        cmd = [str(self.run_script), str(self.frames_dir), str(run_dir), str(self.video_fps)]
        self.get_logger().info("starting pipeline: " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, env=env, cwd=str(self.run_script.parent))
        self.current_run_dir = run_dir
        self.last_processed_count = n


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PipelineRunnerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
