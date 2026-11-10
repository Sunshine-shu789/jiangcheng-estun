from glob import glob

from setuptools import find_packages, setup

package_name = "track1_grasp"

config_files = sorted(glob("config/*.yaml"))
scenario_files = sorted(glob("scenarios/*.yaml"))
launch_files = sorted(glob("launch/*.launch.py"))

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", config_files),
        ("share/" + package_name + "/scenarios", scenario_files),
        ("share/" + package_name + "/launch", launch_files),
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
            "search_approach_node = track1_grasp.nodes.search_approach_node:main",
            "arm_motion_check = track1_grasp.nodes.arm_motion_check_node:main",
            "gripper_bench = track1_grasp.tools.gripper_bench:main",
            "score_grasp_quality = track1_grasp.tools.score_grasp_quality:main",
        ],
    },
)
