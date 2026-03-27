#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="sign_map_bridge",
                executable="frame_ingest_node",
                name="frame_ingest_node",
                output="screen",
            ),
            Node(
                package="sign_map_bridge",
                executable="pipeline_runner_node",
                name="pipeline_runner_node",
                output="screen",
            ),
            Node(
                package="sign_map_bridge",
                executable="result_publisher_node",
                name="result_publisher_node",
                output="screen",
            ),
        ]
    )
