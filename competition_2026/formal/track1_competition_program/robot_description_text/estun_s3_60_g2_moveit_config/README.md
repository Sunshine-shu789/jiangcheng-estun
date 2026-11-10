# Estun S3-60-G2 offline MoveIt configuration

This package is for offline FK, IK, collision checking, and OMPL planning only.
It deliberately does not start a robot driver or a trajectory controller, and
`allow_trajectory_execution` is hard-coded to `false`.

## Model scope

- Arm kinematics: `track1_grasp/kinematics.py::CONTROLLER_DH`
- Arm geometry: official `S3-60-G2.STEP`
- Gripper geometry: official `FAE2MR86M-D.STEP`
- Camera geometry: official Intel RealSense D405 xacro box and upstream STL
- Arm collision/visual geometry: voxel-decimated STEP component meshes
- Gripper collision/visual geometry: 2.0 mm / 0.8 mm voxel-decimated STEP meshes
- Hand-eye transform: calibrated and validated D405-to-flange transform
- Planning groups: `arm` (flange), `gripper` (two finger joints), and
  `arm_with_tool` (base to `grasp_tcp`)

The nominal TCP is 141.751924 mm from `flange_link` along the gripper approach
axis: 91.751924 mm from flange to gripper rear datum plus 50 mm to the TCP.

## Provisional gripper model and required checks

The CAD-derived gripper integration is suitable for offline development, not
for immediate physical execution. The following are still unmeasured or
approximated:

- Installed mounting clocking among the four possible 0/90/180/270 degree
  variants; the model uses canonical 0 degrees only.
- Installed shim stack and its effect on the flange-to-body transform.
- Flexible four-bar linkage, silicone-pad deformation, backlash, and
  motor-command-to-jaw-width behavior.
- The two independent prismatic joints are a rigid-CAD collision
  approximation. Full rigid closure is diagnostic only because the real fingers
  rotate and deform rather than translating through their roots.
- Tool payload, center of mass, and the controller's final tool-frame
  convention.
- D405 USB plug, cable, custom mounting bracket, and manufacturing tolerance.

Grasp candidate filtering must therefore use the measured JawProfile lookup,
not only a maximum opening scalar. Left/right finger collision, gripper-to-camera
collision, and arm-to-gripper collision remain enabled in the SRDF.

`d405_camera_link` is deliberately the calibrated
`camera_color_optical_frame`, not the camera's mechanical center.  The official
D405 case geometry is offset inside that link using the transforms published in
the upstream Intel RealSense xacro.  Its source commit, mesh SHA-256, license,
and rebased visual/collision origins are recorded in
`estun_s3_60_g2/config/geometry_estimates.yaml` and
`estun_s3_60_g2/mesh/camera/README.md`.  Do not recenter the camera mesh or
collision box on the optical frame to make a collision disappear.

The arm STEP's all-zero controller pose is retained as `zero` for reference
only. The folded shoulder/base geometry overlaps at that pose, so it is not
accepted as a planning seed.

## Build and run

```bash
/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/bin/ensure_runtime
source /opt/ros/humble/setup.bash
source /media/sunshine/ESTUN_RUNTIME/ros2_ws/install/setup.bash
colcon build --symlink-install --packages-select   estun_s3_60_g2_description estun_s3_60_g2_moveit_config
```

Start the offline MoveIt service:

```bash
ros2 launch estun_s3_60_g2_moveit_config move_group.launch.py
# Optional visualization:
ros2 launch estun_s3_60_g2_moveit_config move_group.launch.py rviz:=true
```

## Offline smoke test

With the launch file running in another terminal:

```bash
ros2 run estun_s3_60_g2_moveit_config moveit_offline_smoke.py   --output /media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序/robot_description/estun_s3_60_g2_moveit_config/validation/moveit_offline_smoke.json
```

The smoke test validates flange and TCP FK, the fixed tool-chain length,
`arm_with_tool` IK and pose round-trip error, full-state collision checking with
open and rigid-close-envelope fingers, and `arm_with_tool` OMPL RRTConnect
planning. It starts no controller and sends no robot or gripper command.

A passing report is geometry/software evidence only. It does not authorize
physical motion. Before live use, verify installed gripper clocking and shims,
measure jaw width/force behavior, validate tool payload/limits, and perform
explicitly supervised on-site tests.
