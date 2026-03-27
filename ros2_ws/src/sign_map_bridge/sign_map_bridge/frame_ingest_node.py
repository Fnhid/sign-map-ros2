#!/usr/bin/env python3
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


class FrameIngestNode(Node):
    def __init__(self) -> None:
        super().__init__("frame_ingest_node")
        self.declare_parameter("rgb_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "")
        self.declare_parameter("camera_info_topic", "")
        self.declare_parameter("session_root", "/tmp/sign_map_live")

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.session_root = Path(self.get_parameter("session_root").get_parameter_value().string_value).resolve()

        self.frames_dir = self.session_root / "frames"
        self.depth_dir = self.session_root / "depth"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.session_root / "frame_manifest.jsonl"

        self.bridge = CvBridge()
        self.frame_idx = 0
        self.latest_depth_path = ""
        self.camera_info_json = self.session_root / "camera_info.json"

        self.event_pub = self.create_publisher(String, "/sign_map/frame_event", 10)
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.on_rgb, 10)
        self.depth_sub = None
        self.info_sub = None
        if depth_topic:
            self.depth_sub = self.create_subscription(Image, depth_topic, self.on_depth, 10)
        if camera_info_topic:
            self.info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.on_camera_info, 10)

        self.get_logger().info(f"Frame ingest started. root={self.session_root}")

    def on_camera_info(self, msg: CameraInfo) -> None:
        payload = {
            "header": {
                "stamp": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
                "frame_id": msg.header.frame_id,
            },
            "height": int(msg.height),
            "width": int(msg.width),
            "k": list(msg.k),
            "d": list(msg.d),
            "r": list(msg.r),
            "p": list(msg.p),
        }
        self.camera_info_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def on_depth(self, msg: Image) -> None:
        try:
            arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().warning(f"depth conversion failed: {exc}")
            return
        depth_path = self.depth_dir / f"depth_{self.frame_idx:06d}.npy"
        np.save(depth_path, arr)
        self.latest_depth_path = str(depth_path)

    def on_rgb(self, msg: Image) -> None:
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"rgb conversion failed: {exc}")
            return

        frame_path = self.frames_dir / f"frame_{self.frame_idx:06d}.jpg"
        ok = cv2.imwrite(str(frame_path), img)
        if not ok:
            self.get_logger().warning(f"failed to save frame: {frame_path}")
            return

        ts = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        rec = {
            "frame_idx": int(self.frame_idx),
            "timestamp": ts,
            "frame_id": msg.header.frame_id,
            "rgb_path": str(frame_path),
            "depth_path": self.latest_depth_path,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        evt = String()
        evt.data = json.dumps(rec, ensure_ascii=False)
        self.event_pub.publish(evt)

        self.frame_idx += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrameIngestNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
