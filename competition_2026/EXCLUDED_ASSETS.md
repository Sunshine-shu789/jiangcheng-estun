# 未上传内容及原因

本仓库的目标是公开“除官方提供之外的自研内容”。以下内容经过核对后**排除**，不代表丢失；其中多数仍保存在数据盘正式目录，只有明确属于官方或第三方资产的条目才不随仓库发布。

## 1. 官方比赛材料

- 埃斯顿酷卓机器人初级/高级培训资料
- 第三届“埃斯顿”大学生机器人大赛比赛任务书、PDF、表格和物品清单
- `S3-60 Eco 2024` 官方 ROS 包、控制器资料和原始导出文件
- `ROS2通信协议_测试_解密`、`v1.6.6.19T`、`fae爪子` 等厂家资料
- 官方手册、xlsx、培训文档和固件

## 2. 厂家 CAD、网格和运行时

- `S3-60-G2.STEP`（约 113 MB）
- 厂家 `STL/STEP/DAE/OBJ` 网格和 `source/` 目录
- `ESTUN_ROS_RUNTIME.ext4` 及 ROS 2 运行镜像
- 厂家提供的 `ms42dc_vendor_ws/src/step_motor` 和 `codroid_bridge` 包

仓库仅保留为集成而编写的文本配置、xacro/URDF 生成脚本、验证脚本和最小补丁。

## 3. 第三方依赖与预训练权重

- GraspNet Baseline 完整源码和 checkpoint（学术/非商业条款）
- `ViT-B-32.pt` CLIP 权重（约 354 MB）
- 基础 YOLO 权重、YOLO-World、MobileSAM、YCB 物体模型包
- `tooling/` 下的 Python 虚拟环境、编译缓存和依赖二进制
- `subset1_rand.zip` 等第三方数据压缩包

`patches/` 只保存本项目对第三方代码的最小兼容改动、上游仓库地址和基础 commit，便于合法复现。

## 4. 可再生构建和缓存

- `__pycache__`、`.pytest_cache`、`*.pyc`
- ROS 2 的 `build/`、`install/`、`log/` 构建目录
- 本地 lock/pid 文件和临时编辑器缓存
- 指向主目录或数据盘绝对路径的不可移植符号链接

## 5. 本地检查状态

- `/home/sunshine/object-detector` 在当前工作区不存在，因此没有可上传副本。
- `/home/sunshine/codex/easton/codroid_bridge` 经文件时间和包元数据核对为厂家示例/桥接包，未直接上传；自研使用方式已记录在历史文档和日志中。
- 主目录只进行了只读盘点与复制，没有删除、移动或生成新的比赛程序。删除主目录备份仍需用户明确确认。
