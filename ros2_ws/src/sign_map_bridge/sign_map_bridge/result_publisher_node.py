#!/usr/bin/env python3
import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class ResultPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("result_publisher_node")
        self.declare_parameter("map_frame", "map")
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value

        self.path_pub = self.create_publisher(NavPath, "/sign_map/trajectory", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/sign_map/object_markers", 10)
        self.detail_pub = self.create_publisher(String, "/sign_map/object_details", 10)
        self.run_sub = self.create_subscription(String, "/sign_map/new_run", self.on_new_run, 10)

        self.get_logger().info("Result publisher ready.")

    def on_new_run(self, msg: String) -> None:
        run_dir = Path(msg.data).resolve()
        self.get_logger().info(f"loading run outputs: {run_dir}")
        self.publish_trajectory(run_dir / "vggt_poses.txt")
        self.publish_objects(run_dir / "sign_3d_detections.json")

    def publish_trajectory(self, pose_txt: Path) -> None:
        if not pose_txt.exists():
            self.get_logger().warning(f"pose file missing: {pose_txt}")
            return

        nav = NavPath()
        nav.header.frame_id = self.map_frame
        nav.header.stamp = self.get_clock().now().to_msg()

        for line in pose_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = parts[:8]

            p = PoseStamped()
            p.header.frame_id = self.map_frame
            p.header.stamp = self.get_clock().now().to_msg()
            p.pose.position.x = float(tx)
            p.pose.position.y = float(ty)
            p.pose.position.z = float(tz)
            p.pose.orientation.x = float(qx)
            p.pose.orientation.y = float(qy)
            p.pose.orientation.z = float(qz)
            p.pose.orientation.w = float(qw)
            nav.poses.append(p)

        self.path_pub.publish(nav)
        self.get_logger().info(f"published trajectory poses: {len(nav.poses)}")

    def publish_objects(self, det_json: Path) -> None:
        if not det_json.exists():
            self.get_logger().warning(f"detections file missing: {det_json}")
            return

        try:
            detections = json.loads(det_json.read_text(encoding="utf-8"))
        except Exception as exc:
            self.get_logger().warning(f"failed to parse detections: {exc}")
            return

        markers = MarkerArray()
        details = []
        marker_id = 0

        for d in detections:
            c = d.get("centroid_xyz")
            if not c or len(c) < 3:
                continue
            category = str(d.get("object_class") or "sign")
            label = str(d.get("track_text_final") or d.get("ocr_text") or "")

            mk = Marker()
            mk.header.frame_id = self.map_frame
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.ns = category
            mk.id = marker_id
            marker_id += 1
            mk.type = Marker.SPHERE
            mk.action = Marker.ADD
            mk.pose.position.x = float(c[0])
            mk.pose.position.y = float(c[1])
            mk.pose.position.z = float(c[2])
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.18
            mk.scale.y = 0.18
            mk.scale.z = 0.18
            if category == "door":
                mk.color.r = 0.1
                mk.color.g = 0.45
                mk.color.b = 1.0
            else:
                mk.color.r = 0.1
                mk.color.g = 1.0
                mk.color.b = 0.2
            mk.color.a = 0.95
            markers.markers.append(mk)

            details.append(
                {
                    "category": category,
                    "label": label,
                    "centroid_xyz": [float(c[0]), float(c[1]), float(c[2])],
                    "frame_idx": int(d.get("frame_idx", -1)),
                }
            )

        self.marker_pub.publish(markers)
        msg = String()
        msg.data = json.dumps(details, ensure_ascii=False)
        self.detail_pub.publish(msg)
        self.get_logger().info(f"published objects: {len(details)}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ResultPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
