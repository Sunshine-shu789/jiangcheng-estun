#!/usr/bin/env python3
"""相机安装计算器：给定相机参数和工作台尺寸，算出该装多高、ROI 裁哪一块。

依据（2026-09-20 实验结论）：
  - 仿真训练时，物体长边在网络输入里是 80–190 px（中位 138 px）
  - 尺度偏离的代价不对称：0.7× 只掉 1.2 个点，1.5× 掉 6 个点，2× 直接崩（39.4% → 4.1%）
  - 因此目标是让物体落在 0.7×–1.2× 训练尺度，宁可偏小不可偏大

用法示例（Gemini 335L，1280×800，水平视场约 91°）：
    python plan_camera_mount.py --width 1280 --height 800 --hfov 91 \
        --min-object-cm 8 --max-object-cm 30 --workspace-cm 120,80

输出：不同安装高度下物体的网络输入像素、ROI 裁剪框、是否落在安全区间。
"""
from __future__ import annotations

import argparse
import math

TRAIN_MEDIAN_PX = 138.0   # 仿真训练时物体长边的中位像素（网络输入 640）
SAFE_LO, SAFE_HI = 0.7, 1.2
NET = 640  # 网络输入边长


def f_px(width: int, hfov_deg: float) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True, help="相机彩色图宽度（px）")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--hfov", type=float, required=True, help="水平视场角（度）")
    parser.add_argument("--min-object-cm", type=float, default=8.0, help="最小物体的长边（cm）")
    parser.add_argument("--max-object-cm", type=float, default=30.0, help="最大物体的长边（cm）")
    parser.add_argument("--workspace-cm", default="120,80", help="需要覆盖的工作台 宽,深（cm）")
    parser.add_argument("--mount-cm", default="50,60,70,80,90,100,120,150", help="候选安装高度（cm）")
    parser.add_argument("--roi-native", action="store_true", help="按原生分辨率裁 ROI（推荐）")
    args = parser.parse_args()

    f = f_px(args.width, args.hfov)
    ws_w, ws_d = (float(x) for x in args.workspace_cm.split(","))
    print(f"相机：{args.width}×{args.height}，水平视场 {args.hfov}°，等效焦距 {f:.0f} px")
    print(f"训练尺度基准：物体长边中位 {TRAIN_MEDIAN_PX:.0f} px（安全区 {SAFE_LO}×–{SAFE_HI}×）")
    print(f"需要覆盖工作台 {ws_w:.0f}×{ws_d:.0f} cm\n")

    mid_cm = 15.0   # 以 15 cm 的"中位物体"为设计基准（10 类物体的大致中位尺寸）
    print(f"{'安装高度':>7} {'覆盖宽度':>8} {'物体原生':>8} {'建议ROI':>8} {'覆盖需要':>8} {'判定':>12}")
    print("-" * 62)
    for h_cm in [float(x) for x in args.mount_cm.split(",")]:
        z = h_cm / 100
        cover_w = 2 * z * math.tan(math.radians(args.hfov) / 2) * 100      # cm，全画幅覆盖宽度
        native_px = f * (mid_cm / 100) / z                                  # 物体在原生图里的像素
        # 为了把物体送回训练尺度(138px)，ROI 缩到 640 时允许的缩放比 → ROI 宽度
        roi_w = NET * native_px / TRAIN_MEDIAN_PX
        # 要覆盖整个工作台，ROI 至少要多宽（按比例换算回像素）
        need_w = args.width * min(1.0, ws_w / cover_w)
        if roi_w >= need_w:
            verdict = "✔ 可行"
        elif roi_w >= need_w * 0.85:
            verdict = "~ 勉强"
        else:
            verdict = "✘ 冲突"
        print(
            f"{h_cm:>6.0f}cm {cover_w:>7.0f}cm {native_px:>7.0f}px "
            f"{roi_w:>6.0f}px {need_w:>6.0f}px {verdict:>12}"
        )
    print(
        "\n读法：\n"
        "  · 「物体原生」= 15 cm 物体在相机原图里的长边像素；「建议ROI」= 裁多大一块再缩到 640，"
        "能把物体送回训练尺度 138 px。\n"
        "  · 「覆盖需要」= 要装下整个工作台至少得裁多少像素的宽度。\n"
        "  · 建议ROI ≥ 覆盖需要 → 可行（既装得下、尺度也对）；明显更小 → 尺度和覆盖不可兼得，"
        "要么抬高相机、要么换更广的镜头、要么接受偏小的物体（0.5× 仍有 30.7% 通过率，可接受但会掉点）。\n"
        "  · 小物体（8 cm）在最优点位约为表中数值的 "
        f"{(args.min_object_cm / mid_cm):.2f} 倍；大物体（{args.max_object_cm:.0f} cm）约为 "
        f"{(args.max_object_cm / mid_cm):.2f} 倍。\n"
        "  · INTEL D405 这类短距相机不适合固定位（最小工作距离只有 7 cm），留给腕部。"
    )

    print(
        "\n提示：\n"
        "  1) 覆盖宽度 < 工作台宽度时，相机装得太低，会看不到边缘的物体。\n"
        "  2) 「尺度」是以 15 cm 物体估的，落在 0.7–1.2× 最安全；偏小可以接受（0.5× 仍有 30.7% 通过率），偏大要避免。\n"
        "  3) ROI 用原生分辨率裁（不缩放）时，物体像素=表中数值；整帧缩放会再小一半。\n"
        "  4) 三台相机里把「物体像素最大」的那台放俯视位，因为它对 IoU≥0.75 影响最大。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
