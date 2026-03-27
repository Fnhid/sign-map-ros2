from setuptools import setup

package_name = "sign_map_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/sign_map_bridge"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/sign_map_stack.launch.py"]),
        (f"share/{package_name}/config", ["config/sign_map_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sign-map",
    maintainer_email="dev@example.com",
    description="ROS2 bridge for sign/object + point-cloud mapping pipeline.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "frame_ingest_node = sign_map_bridge.frame_ingest_node:main",
            "pipeline_runner_node = sign_map_bridge.pipeline_runner_node:main",
            "result_publisher_node = sign_map_bridge.result_publisher_node:main",
        ],
    },
)
