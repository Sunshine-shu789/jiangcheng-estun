from setuptools import find_packages, setup

package_name = "task_cde_grasp"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            [
                "config/cde_mock.yaml",
                "config/cde_real_template.yaml",
                "config/camera_to_base.example.yaml",
                "config/target_order.example.yaml",
            ],
        ),
        (
            "share/" + package_name + "/scenarios",
            ["scenarios/mock_cde_scene.yaml"],
        ),
        (
            "share/" + package_name + "/launch",
            ["launch/cde_mock.launch.py", "launch/cde_real.launch.py"],
        ),
    ],
    install_requires=["setuptools", "PyYAML"],
    zip_safe=True,
    maintainer="sunshine",
    maintainer_email="sunshine@example.com",
    description="Task C/D/E camera-arm-gripper orchestration framework",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mock_demo = task_cde_grasp.nodes.mock_demo:main",
            "pick_node = task_cde_grasp.nodes.pick_node:main",
        ],
    },
)
