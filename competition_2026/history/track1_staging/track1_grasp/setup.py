from setuptools import find_packages, setup

package_name = "track1_grasp"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            [
                "config/track1_mock.yaml",
                "config/track1_real_template.yaml",
                "config/camera_to_base.example.yaml",
                "config/target_order.example.yaml",
                "config/task_b_place_poses.example.yaml",
                "config/ms42dc_gripper.example.yaml",
            ],
        ),
        (
            "share/" + package_name + "/scenarios",
            [
                "scenarios/mock_track1_scene.yaml",
                "scenarios/task_a_ground_truth.example.yaml",
                "scenarios/mock_task_a_scene.yaml",
            ],
        ),
        (
            "share/" + package_name + "/launch",
            [
                "launch/track1_mock.launch.py",
                "launch/track1_real.launch.py",
            ],
        ),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="sunshine",
    maintainer_email="sunshine@example.com",
    description="Track 1 (A/B/C/D/E) camera-arm-gripper orchestration framework",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mock_demo = track1_grasp.nodes.mock_demo:main",
            "track1_node = track1_grasp.nodes.track1_node:main",
            "pick_node = track1_grasp.nodes.pick_node:main",
            "annotate_node = track1_grasp.nodes.annotate_node:main",
            "gripper_bench = track1_grasp.tools.gripper_bench:main",
        ],
    },
)
