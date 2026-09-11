#!/usr/bin/env python3
"""Load the Track 1 objects into the MoveIt planning scene as collision objects.

The objects are only bounding volumes (see config/track1_items.yaml) - enough
to plan approaches and check clearances, not to simulate grasping.

    ros2 run s3_60_eco_2024_moveit_config load_track1_items.py            # subset 1
    ros2 run s3_60_eco_2024_moveit_config load_track1_items.py --subset 3
    ros2 run s3_60_eco_2024_moveit_config load_track1_items.py --all
    ros2 run s3_60_eco_2024_moveit_config load_track1_items.py --clear    # 只删除

``--all`` 会把 4 个 Subset 摆成 2x2 的展台（每个 Subset 一块桌面），只用于一次
看全 40 件模型；后排离底座 1 m 以上，机器人够不到。逐件可达性检查请用
``check_reachability.py --all``，它是每个 Subset 单独摆、单独检查。
"""

import argparse
import sys
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

import os


def parse_args(argv):
    default_yaml = os.path.join(
        get_package_share_directory("s3_60_eco_2024_moveit_config"),
        "config",
        "track1_items.yaml",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", default=default_yaml, help="物品尺寸表")
    parser.add_argument("--subset", type=int, default=1, help="要加载的 Subset（1-4）")
    parser.add_argument("--all", action="store_true", help="加载全部 40 件")
    parser.add_argument("--no-table", action="store_true", help="不加载桌面")
    parser.add_argument("--clear", action="store_true", help="只清空，不加载")
    return parser.parse_args(argv)


def make_pose(x, y, z):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.w = 1.0
    return pose


def make_primitive(item):
    primitive = SolidPrimitive()
    size = item["size"]
    if item["shape"] == "box":
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(v) for v in size]
    elif item["shape"] == "cylinder":
        primitive.type = SolidPrimitive.CYLINDER
        # SolidPrimitive.CYLINDER 的 dimensions = [高度, 半径]
        primitive.dimensions = [float(size[1]), float(size[0]) / 2.0]
    elif item["shape"] == "sphere":
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [float(size[0]) / 2.0]
    else:
        raise ValueError(f"未知形状 {item['shape']!r}（id={item['id']}）")
    return primitive


def half_height(item):
    size = item["size"]
    if item["shape"] == "box":
        return float(size[2]) / 2.0
    if item["shape"] == "cylinder":
        return float(size[1]) / 2.0
    return float(size[0]) / 2.0


def build_scene(config, items, with_table, layout_offset=(0.0, 0.0)):
    """Return (collision_objects, table_id) for the requested items.

    ``layout_offset`` 把桌面和物品整体平移 (dx, dy)：单个 Subset 用默认的
    (0, 0)；一次摆多个 Subset（``--all``）时每个 Subset 给一个不同的偏移，
    摆成互不重叠的"展台"。
    """
    objects = []
    frame = config["frame_id"]
    table = config["table"]
    table_id = "track1_table"
    offset_x, offset_y = layout_offset

    if with_table:
        obj = CollisionObject()
        obj.header.frame_id = frame
        obj.id = table_id
        obj.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(v) for v in table["size"]]
        obj.primitives = [primitive]
        obj.primitive_poses = [
            make_pose(
                table["center_xy"][0] + offset_x,
                table["center_xy"][1] + offset_y,
                float(table["top_z"]) - float(table["size"][2]) / 2.0,
            )
        ]
        objects.append(obj)

    layout = config["layout"]
    for index, item in enumerate(items):
        column = index % int(layout["columns"])
        row = index // int(layout["columns"])
        x = float(layout["x_start"]) + column * float(layout["x_step"]) + offset_x
        y = float(layout["y_start"]) + row * float(layout["y_step"]) + offset_y
        z = float(table["top_z"]) + half_height(item)

        obj = CollisionObject()
        obj.header.frame_id = frame
        obj.id = f"track1_{item['id']:02d}_{item['name_en'].lower().replace(' ', '_')}"
        obj.operation = CollisionObject.ADD
        obj.primitives = [make_primitive(item)]
        obj.primitive_poses = [make_pose(x, y, z)]
        objects.append(obj)
    return objects, table_id


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    config = load_config(args.yaml)
    items = [
        item
        for item in config["items"]
        if args.all or item["subset"] == args.subset
    ]

    rclpy.init()
    node = Node("track1_items_loader")
    publisher = node.create_publisher(PlanningScene, "planning_scene", 10)

    def publish(scene):
        # move_group needs a moment to match the publisher; a few retries are
        # cheaper than a discovery handshake dance.
        for _ in range(5):
            publisher.publish(scene)
            rclpy.spin_once(node, timeout_sec=0.2)
        time.sleep(0.5)

    clearing = PlanningScene()
    clearing.is_diff = True
    for obj in build_scene(config, config["items"] if args.all else items, True)[0]:
        removal = CollisionObject()
        removal.header.frame_id = config["frame_id"]
        removal.id = obj.id
        removal.operation = CollisionObject.REMOVE
        clearing.world.collision_objects.append(removal)
    publish(clearing)

    if args.clear:
        node.get_logger().info("已清空规划场景里的 track1 物体")
        node.destroy_node()
        rclpy.shutdown()
        return 0

    # 40 件物品的布局是"每 10 件摆一块桌面"，一次全摆会同位置重叠，
    # 所以 --all 时给 4 个 Subset 各一个偏移，摆成 2x2 的展台
    # （后排离底座 1 m 以上，机器人够不到，只用于一次看全模型）。
    all_offsets = {1: (0.0, 0.0), 2: (1.3, 0.0), 3: (0.0, 0.9), 4: (1.3, 0.9)}
    if args.all:
        objects = []
        for subset, offset in all_offsets.items():
            subset_items = [item for item in config["items"] if item["subset"] == subset]
            subset_objects, table_id = build_scene(
                config, subset_items, not args.no_table, offset
            )
            objects.extend(subset_objects)
    else:
        objects, table_id = build_scene(config, items, not args.no_table)
    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects = objects
    publish(scene)

    if args.all:
        node.get_logger().info(
            f"已加载全部 {len(items)} 件物品"
            + (f"（{len(all_offsets)} 块桌面，2x2 展台）" if not args.no_table else "（没有桌面）")
        )
        for subset, (offset_x, offset_y) in all_offsets.items():
            count = len([item for item in items if item["subset"] == subset])
            node.get_logger().info(
                f"  Subset {subset}：{count} 件 @ 桌面偏移 ({offset_x:+.2f}, {offset_y:+.2f})"
            )
        node.get_logger().info("提示：逐件可达性检查用 check_reachability.py --all，它每个 Subset 单独摆。")
    else:
        node.get_logger().info(
            f"已加载 Subset {args.subset}：{len(items)} 件物品"
            + (f" + 桌面（{table_id}）" if not args.no_table else "")
        )
        for index, item in enumerate(items):
            layout = config["layout"]
            x = float(layout["x_start"]) + (index % int(layout["columns"])) * float(layout["x_step"])
            y = float(layout["y_start"]) + (index // int(layout["columns"])) * float(layout["y_step"])
            node.get_logger().info(
                f"  {item['id']:2d} {item['name_cn']:<10s} {item['shape']:<8s} "
                f"size={item['size']} @ ({x:+.2f}, {y:+.2f}) [{item['source']}]"
            )

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
