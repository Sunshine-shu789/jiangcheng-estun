#!/usr/bin/env python3
"""Standalone live camera preview for Task A object detection.

The process only reads the camera and runs YOLO inference. It does not connect
to or control the robot.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion_postprocess import fuse_views, load_calibration  # 多视角融合/仲裁（默认不启用）

from depth_plane_segment import (
    adjusted_intrinsics,
    clean_mask,
    component_summary,
    fit_dominant_plane,
    overlay_result,
    unproject,
)
from scene_annotation import (
    add_detection_suggestions,
    render_scene_preview,
    save_scene,
)


DEFAULT_CAMERA = (
    "/dev/v4l/by-id/"
    "usb-Intel_R__RealSense_TM__Depth_Camera_405_"
    "Intel_R__RealSense_TM__Depth_Camera_405_260423072083-"
    "video-index4"
)
DEFAULT_WEIGHTS = (
    "/media/sunshine/新加卷/taskA/runs/"
    "yolo11s_real_ft_v12b_deploy_20261012/weights/best.pt"
)
DEFAULT_FALLBACK_WEIGHTS = (
    "/media/sunshine/新加卷/taskA/runs/"
    "yolo11s_real_ft_v5_grouped_20260926/weights/best.pt"
)
DEFAULT_PRIMARY_CLASSES = "0,1,2,3,4,5,6,7,8,9"
DEFAULT_CAPTURE_ROOT = "/media/sunshine/新加卷/taskA/real_raw"


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Task A 相机实时识别</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: system-ui, sans-serif;
      background: #0d1216;
      color: #edf2f7;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: #0d1216; }
    main { width: min(100%, 1120px); margin: 0 auto; padding: 18px; }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }
    h1 { margin: 0; font-size: 20px; font-weight: 650; }
    #status {
      color: #aebbc7;
      font-variant-numeric: tabular-nums;
      font-size: 14px;
      text-align: right;
    }
    .view {
      position: relative;
      overflow: hidden;
      border: 1px solid #313b45;
      border-radius: 8px;
      background: #050708;
      aspect-ratio: 4 / 3;
    }
    img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }
    .capture {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 10px;
      padding: 10px;
      border: 1px solid #313b45;
      border-radius: 8px;
      background: #131a20;
    }
    select, button {
      height: 36px;
      border: 1px solid #45515d;
      border-radius: 6px;
      background: #1d2630;
      color: #edf2f7;
      font: inherit;
    }
    select {
      min-width: 210px;
      padding: 0 10px;
    }
    button {
      min-width: 76px;
      padding: 0 13px;
      cursor: pointer;
    }
    button:hover { background: #293744; }
    button:disabled { cursor: wait; opacity: 0.6; }
    #capture-status {
      margin-left: auto;
      color: #aebbc7;
      font-size: 13px;
      text-align: right;
    }
    .scene {
      margin-top: 10px;
      padding: 10px;
      border: 1px solid #313b45;
      border-radius: 8px;
      background: #131a20;
    }
    .scene-actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    #scene-status {
      margin-left: auto;
      color: #aebbc7;
      font-size: 13px;
      text-align: right;
    }
    #scene-preview-wrap {
      display: none;
      margin-top: 10px;
    }
    #scene-preview-wrap.visible { display: block; }
    #scene-canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      touch-action: none;
    }
    #scene-preview-wrap.drawing #scene-canvas {
      pointer-events: auto;
      cursor: crosshair;
    }
    .assignments {
      display: grid;
      gap: 1px;
      margin-top: 10px;
      border: 1px solid #313b45;
      border-radius: 6px;
      overflow: hidden;
      background: #313b45;
    }
    .assignment {
      display: grid;
      grid-template-columns: minmax(120px, 1fr) minmax(210px, 2fr);
      align-items: center;
      gap: 10px;
      padding: 8px 10px;
      background: #131a20;
    }
    .assignment span {
      color: #c8d2dc;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    .assignment-controls {
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto;
      gap: 8px;
    }
    @media (max-width: 620px) {
      main { padding: 10px; }
      header { align-items: flex-start; flex-direction: column; gap: 4px; }
      #status { text-align: left; }
      .capture, .scene-actions { align-items: stretch; flex-wrap: wrap; }
      select { flex: 1 1 100%; }
      #capture-status, #scene-status {
        flex: 1 1 100%;
        margin-left: 0;
        text-align: left;
      }
      .assignment { grid-template-columns: 1fr; gap: 4px; }
      .assignment-controls { grid-template-columns: minmax(0, 1fr) auto; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Task A 相机实时识别</h1>
      <div id="status">正在连接相机...</div>
    </header>
    <div class="view">
      <img id="stream" src="/stream.mjpg" alt="D405 实时画面">
    </div>
    <div class="capture">
      <select id="class-select" aria-label="物品类别"></select>
      <button id="capture-one" type="button">采图</button>
      <button id="capture-burst" type="button">连拍 20 张</button>
      <div id="capture-status">等待模型加载...</div>
    </div>
    <div class="scene">
      <div class="scene-actions">
        <button id="scene-freeze" type="button">冻结并分割</button>
        <button id="scene-draw" type="button" disabled>手动框选</button>
        <button id="scene-save" type="button" disabled>保存标注</button>
        <button id="scene-discard" type="button" disabled>放弃</button>
        <div id="scene-status">放置物品后冻结当前画面</div>
      </div>
      <div class="view" id="scene-preview-wrap">
        <img id="scene-preview" alt="待标注的 RGB-D 场景">
        <canvas id="scene-canvas" width="640" height="480"></canvas>
      </div>
      <div class="assignments" id="assignments" hidden></div>
    </div>
  </main>
  <script>
    const status = document.getElementById("status");
    const stream = document.getElementById("stream");
    const classSelect = document.getElementById("class-select");
    const captureOne = document.getElementById("capture-one");
    const captureBurst = document.getElementById("capture-burst");
    const captureStatus = document.getElementById("capture-status");
    const sceneFreeze = document.getElementById("scene-freeze");
    const sceneDraw = document.getElementById("scene-draw");
    const sceneSave = document.getElementById("scene-save");
    const sceneDiscard = document.getElementById("scene-discard");
    const sceneStatus = document.getElementById("scene-status");
    const scenePreviewWrap = document.getElementById("scene-preview-wrap");
    const scenePreview = document.getElementById("scene-preview");
    const sceneCanvas = document.getElementById("scene-canvas");
    const sceneContext = sceneCanvas.getContext("2d");
    const assignments = document.getElementById("assignments");
    let classesReady = false;
    let captures = 0;
    let reconnectTimer = 0;
    let classNames = [];
    let pendingCandidates = [];
    let manualBoxes = [];
    let sceneFrozen = false;
    let drawMode = false;
    let dragStart = null;
    let dragBox = null;

    function reconnectStream() {
      if (reconnectTimer) {
        return;
      }
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = 0;
        stream.src = `/stream.mjpg?reconnect=${Date.now()}`;
      }, 1000);
    }

    stream.addEventListener("error", reconnectStream);

    function setCaptureBusy(busy) {
      captureOne.disabled = busy;
      captureBurst.disabled = busy;
      classSelect.disabled = busy;
    }

    function setSceneBusy(busy) {
      sceneFreeze.disabled = busy;
      sceneDraw.disabled = busy || !sceneFrozen;
      sceneSave.disabled =
        busy || !sceneFrozen ||
        (pendingCandidates.length === 0 && manualBoxes.length === 0);
      sceneDiscard.disabled = busy || !sceneFrozen;
    }

    async function captureFrame() {
      const isBackground = classSelect.value === "background";
      const response = await fetch("/api/capture", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          class_id: isBackground ? null : Number(classSelect.value)
        })
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return response.json();
    }

    captureOne.addEventListener("click", async () => {
      setCaptureBusy(true);
      try {
        const value = await captureFrame();
        captures += 1;
        captureStatus.textContent =
          `已采 ${captures} 张 · ${value.class_name}`;
      } catch (error) {
        captureStatus.textContent = `采图失败：${error.message}`;
      } finally {
        setCaptureBusy(false);
      }
    });

    captureBurst.addEventListener("click", async () => {
      setCaptureBusy(true);
      const started = captures;
      try {
        for (let i = 0; i < 20; i += 1) {
          await captureFrame();
          captures += 1;
          captureStatus.textContent = `连拍中 ${i + 1}/20 ...`;
          await new Promise(resolve => setTimeout(resolve, 300));
        }
        captureStatus.textContent =
          `连拍完成，本次新增 ${captures - started} 张`;
      } catch (error) {
        captureStatus.textContent = `连拍中断：${error.message}`;
      } finally {
        setCaptureBusy(false);
      }
    });

    function classOptions(selectedValue) {
      return classNames.map((name, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = name;
        if (String(index) === String(selectedValue)) {
          option.selected = true;
        }
        return option;
      });
    }

    function clearScene() {
      pendingCandidates = [];
      manualBoxes = [];
      sceneFrozen = false;
      drawMode = false;
      dragStart = null;
      dragBox = null;
      sceneCanvas.width = 640;
      sceneCanvas.height = 480;
      sceneContext.clearRect(0, 0, sceneCanvas.width, sceneCanvas.height);
      assignments.replaceChildren();
      assignments.hidden = true;
      scenePreviewWrap.classList.remove("visible");
      scenePreviewWrap.classList.remove("drawing");
      sceneDraw.textContent = "手动框选";
      scenePreview.removeAttribute("src");
      setSceneBusy(false);
    }

    function renderAssignments(candidates) {
      const candidateRows = candidates.map((candidate, index) => {
        const row = document.createElement("label");
        row.className = "assignment";
        const [x, y, width, height] = candidate.bbox_xywh;
        const summary = document.createElement("span");
        summary.textContent =
          `#${index + 1}  ${width}×${height}px  ` +
          `${candidate.area_pixels}px²`;
        const select = document.createElement("select");
        select.dataset.candidateId = String(index + 1);
        const suggestedIndex = classNames.indexOf(
          candidate.suggested_class || ""
        );
        select.append(
          ...classOptions(suggestedIndex >= 0 ? suggestedIndex : ""),
          Object.assign(document.createElement("option"), {
            value: "ignore",
            textContent: "忽略此候选"
          })
        );
        if (suggestedIndex < 0) {
          select.value = "ignore";
        }
        row.append(summary, select);
        return row;
      });
      const manualRows = manualBoxes.map((box, index) => {
        const row = document.createElement("div");
        row.className = "assignment";
        const [x1, y1, x2, y2] = box.bbox_xyxy;
        const summary = document.createElement("span");
        summary.textContent =
          `手动 M${index + 1}  ${Math.round(x2 - x1)}×` +
          `${Math.round(y2 - y1)}px`;
        const controls = document.createElement("div");
        controls.className = "assignment-controls";
        const select = document.createElement("select");
        select.dataset.manualIndex = String(index);
        select.append(...classOptions(box.classId));
        select.addEventListener("change", () => {
          manualBoxes[index].classId = Number(select.value);
          drawSceneCanvas();
        });
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "删除";
        remove.addEventListener("click", () => {
          manualBoxes.splice(index, 1);
          renderAssignments(pendingCandidates);
          drawSceneCanvas();
          setSceneBusy(false);
        });
        controls.append(select, remove);
        row.append(summary, controls);
        return row;
      });
      assignments.replaceChildren(...candidateRows, ...manualRows);
      assignments.hidden = false;
    }

    function drawSceneCanvas() {
      const width = sceneCanvas.width;
      const height = sceneCanvas.height;
      sceneContext.clearRect(0, 0, width, height);
      sceneContext.lineWidth = 2;
      manualBoxes.forEach((box, index) => {
        const [x1, y1, x2, y2] = box.bbox_xyxy;
        sceneContext.strokeStyle = "#f06edc";
        sceneContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
        sceneContext.fillStyle = "rgba(240, 110, 220, 0.88)";
        sceneContext.fillRect(x1, Math.max(0, y1 - 20), 34, 20);
        sceneContext.fillStyle = "#10161c";
        sceneContext.font = "13px sans-serif";
        sceneContext.fillText(`M${index + 1}`, x1 + 4, Math.max(14, y1 - 5));
      });
      if (dragBox !== null) {
        const [x1, y1, x2, y2] = dragBox;
        sceneContext.setLineDash([6, 4]);
        sceneContext.strokeStyle = "#ffd24d";
        sceneContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
        sceneContext.setLineDash([]);
      }
    }

    function canvasPoint(event) {
      const bounds = sceneCanvas.getBoundingClientRect();
      return {
        x: Math.max(
          0,
          Math.min(
            sceneCanvas.width,
            (event.clientX - bounds.left) * sceneCanvas.width / bounds.width
          )
        ),
        y: Math.max(
          0,
          Math.min(
            sceneCanvas.height,
            (event.clientY - bounds.top) * sceneCanvas.height / bounds.height
          )
        )
      };
    }

    sceneDraw.addEventListener("click", () => {
      if (!drawMode && classSelect.value === "background") {
        sceneStatus.textContent = "手动框选前请先在上方类别中选择物品";
        return;
      }
      drawMode = !drawMode;
      scenePreviewWrap.classList.toggle("drawing", drawMode);
      sceneDraw.textContent = drawMode ? "结束框选" : "手动框选";
      sceneStatus.textContent = drawMode
        ? `拖动鼠标框住物体；当前类别：${classNames[Number(classSelect.value)]}`
        : "框选已结束，可调整类别或保存";
    });

    sceneCanvas.addEventListener("pointerdown", event => {
      if (!drawMode) {
        return;
      }
      event.preventDefault();
      dragStart = canvasPoint(event);
      dragBox = [dragStart.x, dragStart.y, dragStart.x, dragStart.y];
      sceneCanvas.setPointerCapture(event.pointerId);
      drawSceneCanvas();
    });

    sceneCanvas.addEventListener("pointermove", event => {
      if (dragStart === null) {
        return;
      }
      const point = canvasPoint(event);
      dragBox = [
        Math.min(dragStart.x, point.x),
        Math.min(dragStart.y, point.y),
        Math.max(dragStart.x, point.x),
        Math.max(dragStart.y, point.y)
      ];
      drawSceneCanvas();
    });

    sceneCanvas.addEventListener("pointerup", event => {
      if (dragStart === null || dragBox === null) {
        return;
      }
      sceneCanvas.releasePointerCapture(event.pointerId);
      const [x1, y1, x2, y2] = dragBox;
      dragStart = null;
      dragBox = null;
      if (x2 - x1 < 8 || y2 - y1 < 8) {
        drawSceneCanvas();
        return;
      }
      if (classSelect.value === "background") {
        sceneStatus.textContent = "请先在上方类别中选择物品再拖框";
        drawSceneCanvas();
        return;
      }
      manualBoxes.push({
        bbox_xyxy: [x1, y1, x2, y2],
        classId: Number(classSelect.value)
      });
      renderAssignments(pendingCandidates);
      drawSceneCanvas();
      sceneStatus.textContent =
        `已添加 M${manualBoxes.length}，可继续框选或保存`;
    });

    sceneCanvas.addEventListener("pointercancel", () => {
      dragStart = null;
      dragBox = null;
      drawSceneCanvas();
    });

    scenePreview.addEventListener("load", () => {
      sceneCanvas.width = scenePreview.naturalWidth || 640;
      sceneCanvas.height = scenePreview.naturalHeight || 480;
      drawSceneCanvas();
    });

    sceneFreeze.addEventListener("click", async () => {
      sceneFreeze.disabled = true;
      sceneFrozen = false;
      sceneStatus.textContent = "正在分割深度候选...";
      try {
        const response = await fetch("/api/scene/freeze", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: "{}"
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const value = await response.json();
        pendingCandidates = value.candidates;
        manualBoxes = [];
        sceneFrozen = true;
        renderAssignments(pendingCandidates);
        scenePreview.src = `/api/scene/preview.jpg?t=${Date.now()}`;
        scenePreviewWrap.classList.add("visible");
        drawSceneCanvas();
        sceneStatus.textContent =
          `已找到 ${pendingCandidates.length} 个候选，请逐个选择类别`;
      } catch (error) {
        clearScene();
        sceneStatus.textContent = `分割失败：${error.message}`;
      } finally {
        setSceneBusy(false);
      }
    });

    sceneSave.addEventListener("click", async () => {
      sceneSave.disabled = true;
      const selected = {};
      assignments
        .querySelectorAll("select[data-candidate-id]")
        .forEach(select => {
          if (select.value !== "ignore") {
            selected[select.dataset.candidateId] = Number(select.value);
          }
        });
      try {
        const response = await fetch("/api/scene/save", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            assignments: selected,
            manual_boxes: manualBoxes.map(box => ({
              class_id: box.classId,
              bbox_xyxy: box.bbox_xyxy
            }))
          })
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const value = await response.json();
        sceneStatus.textContent =
          `已保存 ${value.objects} 个标注 · ${value.scene}`;
        clearScene();
      } catch (error) {
        sceneStatus.textContent = `保存失败：${error.message}`;
        sceneSave.disabled = false;
      }
    });

    sceneDiscard.addEventListener("click", async () => {
      sceneDiscard.disabled = true;
      try {
        await fetch("/api/scene/discard", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: "{}"
        });
      } finally {
        clearScene();
        sceneStatus.textContent = "已放弃当前场景";
      }
    });

    async function refresh() {
      try {
        const response = await fetch("/status.json", {cache: "no-store"});
        const value = await response.json();
        if (!value.ready) {
          status.textContent = "正在加载识别模型...";
          return;
        }
        if (!classesReady && value.classNames.length) {
          classNames = value.classNames;
          classSelect.replaceChildren(
            ...classNames.map((name, index) => {
              const option = document.createElement("option");
              option.value = String(index);
              option.textContent = name;
              return option;
            }),
            Object.assign(document.createElement("option"), {
              value: "background",
              textContent: "空背景（不摆放物品）"
            })
          );
          classSelect.value = "3";
          classesReady = true;
          captureStatus.textContent = "可开始采集";
        }
        const labels = value.detections.map(item =>
          `${item.class} ${((item.stableConfidence ?? item.confidence) * 100).toFixed(1)}%`
        );
        const parts = [
          `${value.fps.toFixed(1)} FPS`,
          `${value.width}x${value.height}`,
          `尺度 ${Math.round(value.detectionScale * 100)}%`,
        ];
        if (value.depth) {
          const tablePercent = (value.depth.tableFraction * 100).toFixed(1);
          parts.push(`桌面 ${tablePercent}%`);
          parts.push(`深度候选 ${value.depth.objectComponents}`);
        }
        parts.push(labels.length ? labels.join(" | ") : "未识别到物品");
        status.textContent = parts.join("  ·  ");
      } catch {
        status.textContent = "预览服务暂时不可用";
        reconnectStream();
      }
    }
    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""


def parse_camera(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task A live camera preview. "
            "The process never connects to the robot."
        )
    )
    parser.add_argument("--camera", type=parse_camera, default=DEFAULT_CAMERA)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--fallback-weights",
        default=DEFAULT_FALLBACK_WEIGHTS,
    )
    parser.add_argument(
        "--primary-classes",
        default=DEFAULT_PRIMARY_CLASSES,
        help="只从主权重读取的类别编号，逗号分隔",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--inference-conf", type=float, default=0.05)
    parser.add_argument("--fallback-conf", type=float, default=0.20)
    parser.add_argument(
        "--fallback-suppress-iou",
        type=float,
        default=0.0,
        help="回退类别命中时，抑制与其 IoU≥该值的主类别框；0=关闭",
    )
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--confirm-hits", type=int, default=3)
    parser.add_argument("--track-misses", type=int, default=3)
    parser.add_argument("--track-iou", type=float, default=0.20)
    parser.add_argument(
        "--stable-confidence-alpha",
        type=float,
        default=0.10,
        help="界面稳定置信度的 EMA 系数；1.0 表示不平滑",
    )
    parser.add_argument("--duplicate-iou", type=float, default=0.55)
    parser.add_argument("--cross-class-iou", type=float, default=0.40)
    parser.add_argument(
        "--cross-class-containment",
        type=float,
        default=0.60,
        help="跨类别重叠框需要达到的较小框包含比例",
    )
    parser.add_argument(
        "--cross-class-conf-ratio",
        type=float,
        default=1.50,
        help="跨类别去重要求高置信框至少为低置信框的倍数",
    )
    parser.add_argument("--containment-ratio", type=float, default=0.90)
    parser.add_argument("--max-per-class", type=int, default=1)
    parser.add_argument(
        "--class-switch-hits",
        type=int,
        default=3,
        help="轨迹切换类别前，新类别需要连续出现的次数",
    )
    parser.add_argument("--edge-margin-px", type=int, default=8)
    parser.add_argument("--edge-suppress-conf", type=float, default=0.55)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument("--rgbd", action="store_true")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--depth-process-interval", type=float, default=0.20)
    parser.add_argument("--plane-threshold-m", type=float, default=0.006)
    parser.add_argument("--object-min-height-m", type=float, default=0.010)
    parser.add_argument("--object-max-height-m", type=float, default=0.40)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--depth-sample-step", type=int, default=10)
    parser.add_argument("--depth-ransac-iterations", type=int, default=300)
    parser.add_argument("--depth-min-component-pixels", type=int, default=150)
    parser.add_argument("--depth-morph-kernel", type=int, default=5)
    parser.add_argument("--keep-border-components", action="store_true")
    parser.add_argument(
        "--detection-scale",
        default="1.0",
        help="检测缩放：单值或逗号分隔多尺度（如 0.65,0.75），多尺度结果自动合并",
    )
    parser.add_argument(
        "--fusion",
        choices=("legacy", "vote3"),
        default="legacy",
        help="后处理：legacy=旧的多尺度取最高分合并；vote3=跨视角投票+WBF+类别仲裁（离线验收 91.1%%，错类帧 29/225）",
    )
    parser.add_argument("--fusion-min-views", type=int, default=3,
                        help="vote3 下，一个框至少要在多少个视角里出现才保留")
    parser.add_argument("--fusion-tta", action="store_true",
                        help="vote3 下，每个尺度额外跑一次 TTA（更稳但更慢）")
    parser.add_argument("--fusion-arbitration", action="store_true",
                        help="vote3 下启用吸引子仲裁（压制与具体类重叠的 meat_can 框）")
    parser.add_argument("--fusion-calibration-dir", default="",
                        help="框校准目录（含 base.json/tta.json），仅对 1.00 尺度生效")
    parser.add_argument("--detect-interval", type=float, default=0.05)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--capture-root", default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument(
        "--scene-root",
        default="/media/sunshine/新加卷/taskA/random_real_scenes_auto",
    )
    parser.add_argument("--scene-box-padding-px", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--open-browser", action="store_true")
    return parser


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def parse_class_ids(
    value: str,
    class_count: int | None = None,
) -> set[int]:
    class_ids: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            class_id = int(item)
        except ValueError as error:
            raise ValueError(f"类别编号无效：{item!r}") from error
        if class_id < 0 or (
            class_count is not None and class_id >= class_count
        ):
            upper_bound = (
                f"0-{class_count - 1}"
                if class_count is not None
                else "非负整数"
            )
            raise ValueError(
                f"类别编号 {class_id} 超出范围 {upper_bound}"
            )
        class_ids.add(class_id)
    if not class_ids:
        raise ValueError("primary-classes 至少需要一个类别编号")
    return class_ids


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def box_containment(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min(box_area(first), box_area(second))
    return intersection / smaller if smaller > 0.0 else 0.0


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


@dataclass
class DetectionTrack:
    track_id: int
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float
    stable_confidence: float
    hits: int = 1
    missed: int = 0
    class_votes: dict[int, float] = field(default_factory=dict)
    confirm_after: int = 2
    challenger_class_id: int | None = None
    challenger_hits: int = 0

    @property
    def confirmed(self) -> bool:
        return self.hits >= self.confirm_after


class DetectionStabilizer:
    """Suppress duplicate boxes and keep detections stable across frames."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.confirm_hits = args.confirm_hits
        self.max_misses = args.track_misses
        self.match_iou = args.track_iou
        self.stable_confidence_alpha = args.stable_confidence_alpha
        self.duplicate_iou = args.duplicate_iou
        self.cross_class_iou = args.cross_class_iou
        self.cross_class_containment = args.cross_class_containment
        self.cross_class_conf_ratio = args.cross_class_conf_ratio
        self.containment_ratio = args.containment_ratio
        self.max_per_class = args.max_per_class
        self.class_switch_hits = args.class_switch_hits
        self.edge_margin_px = args.edge_margin_px
        self.edge_suppress_conf = args.edge_suppress_conf
        self.new_track_conf = args.conf
        self.keep_track_conf = max(
            args.inference_conf,
            args.conf * 0.60,
        )
        self.tracks: dict[int, DetectionTrack] = {}
        self.next_track_id = 1

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
        class_names: list[str],
    ) -> list[dict[str, Any]]:
        candidates = self._suppress_duplicates(
            detections,
            frame_width,
            frame_height,
        )
        existing_track_ids = set(self.tracks)
        diagonal = max(
            float(np.hypot(frame_width, frame_height)),
            1.0,
        )
        pairs: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(candidates):
            bbox = tuple(float(value) for value in detection["bbox"])
            for track_id, track in self.tracks.items():
                score = self._match_score(
                    bbox,
                    detection,
                    track,
                    diagonal,
                )
                if score is not None:
                    pairs.append((score, detection_index, track_id))

        assigned_detections: set[int] = set()
        assigned_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(pairs, reverse=True):
            if (
                detection_index in assigned_detections
                or track_id in assigned_tracks
            ):
                continue
            assigned_detections.add(detection_index)
            assigned_tracks.add(track_id)
            self._update_track(
                self.tracks[track_id],
                candidates[detection_index],
                class_names,
            )

        for detection_index, detection in enumerate(candidates):
            if detection_index in assigned_detections:
                continue
            confidence = float(detection["confidence"])
            if confidence < self.new_track_conf:
                continue
            self._create_track(detection, class_names)

        for track_id in existing_track_ids:
            if track_id not in self.tracks:
                continue
            if track_id in assigned_tracks:
                continue
            track = self.tracks[track_id]
            track.missed += 1
            track.confidence *= 0.82
            track.stable_confidence *= 0.90
            if (
                (not track.confirmed and track.missed >= 1)
                or track.missed > self.max_misses
            ):
                del self.tracks[track_id]

        visible = self._visible_tracks()
        return sorted(
            (self._to_detection(track) for track in visible),
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )

    def _suppress_duplicates(
        self,
        detections: list[dict[str, Any]],
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            (
                {
                    **detection,
                    "bbox": tuple(
                        float(value) for value in detection["bbox"]
                    ),
                }
                for detection in detections
                if float(detection["confidence"]) >= self.keep_track_conf
            ),
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
        kept: list[dict[str, Any]] = []
        class_counts: dict[int, int] = {}
        for candidate in ordered:
            candidate_class = int(candidate["class_id"])
            candidate_box = candidate["bbox"]
            if self._is_edge_false_positive(
                candidate_box,
                float(candidate["confidence"]),
                frame_width,
                frame_height,
            ):
                continue
            if any(
                self._boxes_are_duplicates(
                    candidate_box,
                    candidate_class,
                    float(candidate["confidence"]),
                    accepted["bbox"],
                    int(accepted["class_id"]),
                    float(accepted["confidence"]),
                )
                for accepted in kept
            ):
                continue
            if (
                self.max_per_class > 0
                and class_counts.get(candidate_class, 0)
                >= self.max_per_class
            ):
                continue
            kept.append(candidate)
            class_counts[candidate_class] = (
                class_counts.get(candidate_class, 0) + 1
            )
        return kept

    def _boxes_are_duplicates(
        self,
        first_box: tuple[float, float, float, float],
        first_class: int,
        first_confidence: float,
        second_box: tuple[float, float, float, float],
        second_class: int,
        second_confidence: float,
    ) -> bool:
        overlap = box_iou(first_box, second_box)
        containment = box_containment(first_box, second_box)
        if first_class == second_class:
            return (
                overlap >= self.duplicate_iou
                or containment >= self.containment_ratio
            )
        if not (
            overlap >= self.cross_class_iou
            or containment >= self.cross_class_containment
        ):
            return False
        stronger_confidence = max(first_confidence, second_confidence)
        weaker_confidence = min(first_confidence, second_confidence)
        if weaker_confidence <= 0.0:
            return True
        return (
            stronger_confidence
            >= weaker_confidence * self.cross_class_conf_ratio
        )

    def _is_edge_false_positive(
        self,
        bbox: tuple[float, float, float, float],
        confidence: float,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        if confidence >= self.edge_suppress_conf:
            return False
        margin = float(self.edge_margin_px)
        edge_count = sum(
            (
                bbox[0] <= margin,
                bbox[1] <= margin,
                bbox[2] >= frame_width - 1 - margin,
                bbox[3] >= frame_height - 1 - margin,
            )
        )
        return edge_count >= 2

    def _visible_tracks(self) -> list[DetectionTrack]:
        visible = [
            track
            for track in self.tracks.values()
            if (
                track.confirmed
                and track.missed <= self.max_misses
                and track.confidence >= self.new_track_conf
            )
        ]
        visible.sort(key=self._track_rank, reverse=True)
        deduplicated: list[DetectionTrack] = []
        for track in visible:
            if any(
                self._boxes_are_duplicates(
                    track.bbox,
                    track.class_id,
                    track.confidence,
                    accepted.bbox,
                    accepted.class_id,
                    accepted.confidence,
                )
                for accepted in deduplicated
            ):
                continue
            deduplicated.append(track)

        if self.max_per_class <= 0:
            return deduplicated

        by_class: dict[int, list[DetectionTrack]] = {}
        for track in deduplicated:
            by_class.setdefault(track.class_id, []).append(track)
        return [
            max(
                tracks,
                key=self._track_rank,
            )
            for tracks in by_class.values()
        ]

    @staticmethod
    def _track_rank(track: DetectionTrack) -> tuple[float, int, float]:
        return (
            track.confidence
            + min(track.hits, 30) * 0.002
            - track.missed * 0.12,
            track.hits,
            track.confidence,
        )

    def _match_score(
        self,
        bbox: tuple[float, float, float, float],
        detection: dict[str, Any],
        track: DetectionTrack,
        diagonal: float,
    ) -> float | None:
        overlap = box_iou(bbox, track.bbox)
        containment = box_containment(bbox, track.bbox)
        class_id = int(detection["class_id"])
        if class_id == track.class_id:
            if overlap >= self.match_iou:
                return 1.0 + overlap
            first_center = box_center(bbox)
            second_center = box_center(track.bbox)
            center_distance = float(
                np.hypot(
                    first_center[0] - second_center[0],
                    first_center[1] - second_center[1],
                )
            )
            first_area = max(box_area(bbox), 1.0)
            second_area = max(box_area(track.bbox), 1.0)
            area_ratio = first_area / second_area
            if (
                center_distance / diagonal <= 0.08
                and 0.35 <= area_ratio <= 2.85
            ):
                return 0.5 + max(0.0, 1.0 - center_distance / diagonal)
            return None

        if (
            overlap >= max(0.45, self.match_iou)
            or containment >= 0.75
        ):
            return 0.25 + max(overlap, containment)
        return None

    def _update_track(
        self,
        track: DetectionTrack,
        detection: dict[str, Any],
        class_names: list[str],
    ) -> None:
        bbox = tuple(float(value) for value in detection["bbox"])
        confidence = float(detection["confidence"])
        class_id = int(detection["class_id"])

        if class_id == track.class_id:
            track.challenger_class_id = None
            track.challenger_hits = 0
        elif class_id == track.challenger_class_id:
            track.challenger_hits += 1
        else:
            track.challenger_class_id = class_id
            track.challenger_hits = 1

        for known_class in list(track.class_votes):
            track.class_votes[known_class] *= 0.65
            if track.class_votes[known_class] < 0.01:
                del track.class_votes[known_class]
        track.class_votes[class_id] = (
            track.class_votes.get(class_id, 0.0) + confidence
        )
        best_class, best_score = max(
            track.class_votes.items(),
            key=lambda item: item[1],
        )
        current_score = track.class_votes.get(track.class_id, 0.0)
        if (
            best_class != track.class_id
            and best_class == track.challenger_class_id
            and track.challenger_hits >= self.class_switch_hits
            and best_score >= current_score * 1.35
            and confidence >= self.new_track_conf
        ):
            track.class_id = best_class
            track.class_name = self._class_name(best_class, class_names)
            track.class_votes = {best_class: best_score}
            track.stable_confidence = confidence
            track.challenger_class_id = None
            track.challenger_hits = 0

        center_distance = float(
            np.hypot(
                box_center(bbox)[0] - box_center(track.bbox)[0],
                box_center(bbox)[1] - box_center(track.bbox)[1],
            )
        )
        alpha = 0.85 if center_distance >= 0.10 * max(
            np.hypot(
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
            ),
            1.0,
        ) else 0.55
        track.bbox = tuple(
            old_value * (1.0 - alpha) + new_value * alpha
            for old_value, new_value in zip(track.bbox, bbox)
        )
        track.confidence = (
            track.confidence * (1.0 - alpha) + confidence * alpha
        )
        track.stable_confidence = (
            track.stable_confidence * (1.0 - self.stable_confidence_alpha)
            + confidence * self.stable_confidence_alpha
        )
        track.hits += 1
        track.missed = 0

    def _create_track(
        self,
        detection: dict[str, Any],
        class_names: list[str],
    ) -> None:
        class_id = int(detection["class_id"])
        confidence = float(detection["confidence"])
        track = DetectionTrack(
            track_id=self.next_track_id,
            class_id=class_id,
            class_name=self._class_name(class_id, class_names),
            bbox=tuple(float(value) for value in detection["bbox"]),
            confidence=confidence,
            stable_confidence=confidence,
            class_votes={class_id: confidence},
            confirm_after=self.confirm_hits,
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1

    @staticmethod
    def _class_name(class_id: int, class_names: list[str]) -> str:
        if 0 <= class_id < len(class_names):
            return class_names[class_id]
        return f"class_{class_id}"

    @staticmethod
    def _to_detection(track: DetectionTrack) -> dict[str, Any]:
        return {
            "trackId": track.track_id,
            "classId": track.class_id,
            "class": track.class_name,
            "confidence": round(track.confidence, 4),
            "stableConfidence": round(track.stable_confidence, 4),
            "bbox": [round(value, 1) for value in track.bbox],
            "hits": track.hits,
            "missed": track.missed,
        }


class RealSenseCamera:
    def __init__(self, args: argparse.Namespace) -> None:
        config = rs.config()
        if args.serial:
            config.enable_device(args.serial)
        config.enable_stream(
            rs.stream.color,
            args.camera_width,
            args.camera_height,
            rs.format.bgr8,
            args.camera_fps,
        )
        config.enable_stream(
            rs.stream.depth,
            args.camera_width,
            args.camera_height,
            rs.format.z16,
            args.camera_fps,
        )
        self.args = args
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(config)
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        self.intrinsics = adjusted_intrinsics(
            {
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "ppx": float(intrinsics.ppx),
                "ppy": float(intrinsics.ppy),
            },
            args.rotate,
        )
        for _ in range(args.warmup_frames):
            self.pipeline.wait_for_frames(timeout_ms=5000)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        frames = self.align.process(
            self.pipeline.wait_for_frames(timeout_ms=5000)
        )
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("无法读取对齐后的 RGB-D 帧")
        color = rotate_frame(
            np.asanyarray(color_frame.get_data()),
            self.args.rotate,
        )
        depth = rotate_frame(
            np.asanyarray(depth_frame.get_data()),
            self.args.rotate,
        )
        depth_m = depth.astype(np.float32) * self.depth_scale
        return color, depth_m

    def release(self) -> None:
        self.pipeline.stop()


def segment_depth_frame(
    depth_m: np.ndarray,
    intrinsics: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= args.min_depth_m)
        & (depth_m <= args.max_depth_m)
    )
    ys, xs = np.nonzero(valid)
    if len(ys) < 3:
        raise RuntimeError("有效深度点不足，无法拟合平面")
    points = unproject(depth_m, valid, intrinsics)
    plane, _ = fit_dominant_plane(
        points[:: args.depth_sample_step],
        args.plane_threshold_m,
        args.depth_ransac_iterations,
        seed=0,
    )
    distances = points @ plane.normal + plane.offset
    height_map = np.full(valid.shape, np.nan, dtype=np.float32)
    height_map[ys, xs] = distances
    table_mask = np.zeros(valid.shape, dtype=bool)
    table_mask[ys, xs] = np.abs(distances) <= args.plane_threshold_m
    raw_object_mask = np.zeros(valid.shape, dtype=bool)
    raw_object_mask[ys, xs] = (
        (distances >= args.object_min_height_m)
        & (distances <= args.object_max_height_m)
    )
    object_mask = clean_mask(
        raw_object_mask,
        args.depth_morph_kernel,
        args.depth_min_component_pixels,
        args.keep_border_components,
    )
    component_count, object_pixels, _, component_details = component_summary(
        object_mask,
        height_map,
    )
    for index, detail in enumerate(component_details, start=1):
        detail["candidate_id"] = index
    stats = {
        "tableFraction": float(table_mask.sum() / max(valid.sum(), 1)),
        "objectComponents": component_count,
        "objectPixels": object_pixels,
        "components": component_details,
    }
    return table_mask, object_mask, stats


def prepare_inference_frame(
    frame: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, int, int]:
    if scale == 1.0:
        return frame, 0, 0

    height, width = frame.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    resized = cv2.resize(
        frame,
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    x_offset = (width - scaled_width) // 2
    y_offset = (height - scaled_height) // 2
    canvas[
        y_offset : y_offset + scaled_height,
        x_offset : x_offset + scaled_width,
    ] = resized
    return canvas, x_offset, y_offset


def parse_detection_scales(value: str | float) -> tuple[float, ...]:
    """解析 --detection-scale：支持 "0.65,0.75" 这类多尺度写法。"""
    if isinstance(value, (int, float)):
        return (float(value),)
    scales: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        scales.append(float(part))
    if not scales:
        raise ValueError("detection-scale 不能为空")
    return tuple(scales)


def format_fusion_view(scale: float, use_tta: bool) -> str:
    """Stable view key shared by the detector and the offline fusion pipeline."""
    return f"{float(scale):.4f}|{'tta' if use_tta else 'base'}"


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_scale_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """多尺度推理：同一目标重复框只保留置信度最高的一个。"""
    kept: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: -float(item["confidence"])):
        duplicated = False
        for other in kept:
            if other["class_id"] != det["class_id"]:
                continue
            if box_iou(other["bbox"], det["bbox"]) >= iou_threshold:
                duplicated = True
                break
        if not duplicated:
            kept.append(det)
    return kept


def suppress_overlapping_classes(
    detections: list[dict[str, Any]],
    primary_classes: set[int],
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """回退模型命中时，抑制与之重叠的主模型旧类框。

    双模型部署下同一物体会被两个模型各报一次（例如泡沫砖被主模型认成
    10-meat_can），此处以回退类别为准，丢弃与其重叠的主类别框。
    """
    if iou_threshold <= 0.0:
        return detections
    fallback = [d for d in detections if int(d["class_id"]) not in primary_classes]
    if not fallback:
        return detections
    kept: list[dict[str, Any]] = []
    for det in detections:
        if int(det["class_id"]) in primary_classes and any(
            box_iou(det["bbox"], other["bbox"]) >= iou_threshold for other in fallback
        ):
            continue
        kept.append(det)
    return kept


class PreviewState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.sequence = 0
        self.jpeg: bytes | None = None
        self.raw_frame: np.ndarray | None = None
        self.raw_depth_m: np.ndarray | None = None
        self.depth_scale_m_per_unit = 0.0
        self.depth_intrinsics: Any | None = None
        self.depth_args: argparse.Namespace | None = None
        self.pending_scene: dict[str, Any] | None = None
        self.class_names: list[str] = []
        self.status: dict[str, Any] = {
            "ready": False,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "detectionScale": 1.0,
            "weights": None,
            "classNames": [],
            "detections": [],
        }

    def publish(
        self,
        jpeg: bytes,
        raw_frame: np.ndarray,
        raw_depth_m: np.ndarray | None,
        status: dict[str, Any],
    ) -> None:
        with self.condition:
            self.sequence += 1
            status = dict(status)
            status["frameSequence"] = self.sequence
            self.jpeg = jpeg
            self.raw_frame = raw_frame.copy()
            self.raw_depth_m = (
                raw_depth_m.copy() if raw_depth_m is not None else None
            )
            self.status = status
            self.condition.notify_all()

    def wait_for_frame(
        self,
        after_sequence: int,
        timeout: float = 5.0,
    ) -> tuple[int, bytes] | None:
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > after_sequence,
                timeout=timeout,
            )
            if self.sequence <= after_sequence or self.jpeg is None:
                return None
            return self.sequence, self.jpeg


class PreviewWorker(threading.Thread):
    def __init__(
        self,
        args: argparse.Namespace,
        state: PreviewState,
    ) -> None:
        super().__init__(name="task-a-live-preview", daemon=True)
        self.args = args
        self.state = state
        self.stop_event = threading.Event()
        self.weights = str(Path(args.weights).resolve())
        self.fallback_weights = str(Path(args.fallback_weights).resolve())
        self.model = YOLO(self.weights)
        self.fallback_model = YOLO(self.fallback_weights)
        self.state.class_names = [
            str(self.model.names[index])
            for index in sorted(self.model.names)
        ]
        fallback_class_names = [
            str(self.fallback_model.names[index])
            for index in sorted(self.fallback_model.names)
        ]
        if fallback_class_names != self.state.class_names:
            raise ValueError("主权重和回退权重的类别定义不一致")
        self.primary_classes = parse_class_ids(
            args.primary_classes,
            len(self.state.class_names),
        )
        self.fallback_classes = (
            set(range(len(self.state.class_names))) - self.primary_classes
        )
        # 主权重覆盖全部类别时 fallback_classes 为空，此时不使用回退模型。
        self.fusion = str(getattr(args, "fusion", "legacy"))
        self.fusion_min_views = int(getattr(args, "fusion_min_views", 3))
        self.fusion_tta = bool(getattr(args, "fusion_tta", False))
        self.fusion_arbitration = bool(getattr(args, "fusion_arbitration", False))
        cal_dir = str(getattr(args, "fusion_calibration_dir", "") or "")
        self.fusion_calibration = {}
        if self.fusion == "vote3" and cal_dir:
            root = Path(cal_dir)
            base_cal = load_calibration(root / "base.json")
            tta_cal = load_calibration(root / "tta.json")
            for scale in parse_detection_scales(args.detection_scale):
                key_b = format_fusion_view(scale, False)
                key_t = format_fusion_view(scale, True)
                if abs(float(scale) - 1.0) < 1e-9:
                    self.fusion_calibration[key_b] = base_cal
                    self.fusion_calibration[key_t] = tta_cal
                else:
                    self.fusion_calibration[key_b] = {}
                    self.fusion_calibration[key_t] = {}
        self.stabilizer = DetectionStabilizer(args)

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        args = self.args
        detection_scales = parse_detection_scales(args.detection_scale)
        if args.rgbd:
            camera: Any = RealSenseCamera(args)
            self.state.depth_scale_m_per_unit = camera.depth_scale
            self.state.depth_intrinsics = camera.intrinsics
            self.state.depth_args = args
        else:
            camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
            if not camera.isOpened():
                raise RuntimeError(f"无法打开相机：{args.camera!r}")

        try:
            if not args.rgbd:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.camera_width))
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.camera_height))
                for _ in range(args.warmup_frames):
                    camera.read()

            last_frame_at = time.monotonic()
            last_detection_at = 0.0
            last_depth_at = 0.0
            last_result = None
            last_depth_result = None
            depth_status: dict[str, Any] | None = None
            last_inference_ms = 0.0
            smoothed_fps = 0.0

            while not self.stop_event.is_set():
                depth_m = None
                if args.rgbd:
                    frame, depth_m = camera.read()
                else:
                    ok, frame = camera.read()
                    if not ok or frame is None:
                        time.sleep(0.03)
                        continue
                    frame = rotate_frame(frame, args.rotate)
                now = time.monotonic()
                if (
                    last_result is None
                    or now - last_detection_at >= args.detect_interval
                ):
                    started_at = time.monotonic()
                    raw_detections: list[dict[str, Any]] = []
                    detections_by_view: dict[str, list[dict[str, Any]]] = {}
                    tta_flags = (False, True) if (
                        self.fusion == "vote3" and self.fusion_tta
                    ) else (False,)
                    for detection_scale in detection_scales:
                        inference_frame, x_offset, y_offset = (
                            prepare_inference_frame(frame, detection_scale)
                        )
                        for use_tta in tta_flags:
                            view_key = format_fusion_view(detection_scale, use_tta)
                            view_dets: list[dict[str, Any]] = []
                            primary_result = self.model.predict(
                                source=inference_frame,
                                imgsz=args.imgsz,
                                conf=args.inference_conf,
                                iou=args.iou,
                                max_det=args.max_det,
                                device=args.device,
                                classes=sorted(self.primary_classes),
                                augment=use_tta,
                                verbose=False,
                            )[0]
                            view_dets.extend(
                                self.map_result(
                                    primary_result,
                                    x_offset,
                                    y_offset,
                                    detection_scale,
                                )["detections"]
                            )
                            if self.fallback_classes:
                                fallback_result = self.fallback_model.predict(
                                    source=inference_frame,
                                    imgsz=args.imgsz,
                                    conf=args.fallback_conf,
                                    iou=args.iou,
                                    max_det=args.max_det,
                                    device=args.device,
                                    classes=sorted(self.fallback_classes),
                                    augment=use_tta,
                                    verbose=False,
                                )[0]
                                view_dets.extend(
                                    self.map_result(
                                        fallback_result,
                                        x_offset,
                                        y_offset,
                                        detection_scale,
                                    )["detections"]
                                )
                            detections_by_view[view_key] = view_dets
                            raw_detections.extend(view_dets)
                    if (
                        args.fallback_suppress_iou > 0.0
                        and self.fallback_classes
                    ):
                        raw_detections = suppress_overlapping_classes(
                            raw_detections,
                            self.primary_classes,
                            args.fallback_suppress_iou,
                        )
                    last_inference_ms = (time.monotonic() - started_at) * 1000.0
                    if self.fusion == "vote3":
                        merged_detections = fuse_views(
                            detections_by_view,
                            min_views=self.fusion_min_views,
                            width=int(frame.shape[1]),
                            height=int(frame.shape[0]),
                            calibration_by_view=self.fusion_calibration,
                            class_names=self.state.class_names,
                            arbitration=self.fusion_arbitration,
                        )
                    else:
                        merged_detections = merge_scale_detections(raw_detections)
                    stabilized = self.stabilizer.update(
                        merged_detections,
                        int(frame.shape[1]),
                        int(frame.shape[0]),
                        self.state.class_names,
                    )
                    last_result = {"detections": stabilized}
                    last_detection_at = now

                if (
                    args.rgbd
                    and depth_m is not None
                    and (
                        last_depth_result is None
                        or now - last_depth_at >= args.depth_process_interval
                    )
                ):
                    try:
                        table_mask, object_mask, depth_status = (
                            segment_depth_frame(
                                depth_m,
                                camera.intrinsics,
                                args,
                            )
                        )
                        last_depth_result = (table_mask, object_mask)
                        last_depth_at = now
                    except RuntimeError as error:
                        last_depth_result = None
                        depth_status = {
                            "available": False,
                            "error": str(error),
                            "tableFraction": 0.0,
                            "objectComponents": 0,
                            "objectPixels": 0,
                        }

                if now > last_frame_at:
                    current_fps = 1.0 / (now - last_frame_at)
                    smoothed_fps = (
                        current_fps
                        if smoothed_fps == 0.0
                        else smoothed_fps * 0.85 + current_fps * 0.15
                    )
                last_frame_at = now

                annotated = frame
                if last_depth_result is not None:
                    table_mask, object_mask = last_depth_result
                    annotated = overlay_result(
                        annotated,
                        table_mask,
                        object_mask,
                    )
                annotated = self.annotate(annotated, last_result)
                overlay = (
                    f"Task A preview  {smoothed_fps:.1f} FPS  "
                    f"objects: {len(last_result['detections'])}"
                )
                if depth_status is not None:
                    overlay += (
                        f"  depth candidates: "
                        f"{depth_status['objectComponents']}"
                    )
                cv2.rectangle(annotated, (0, 0), (430, 29), (16, 20, 24), -1)
                cv2.putText(
                    annotated,
                    overlay,
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (240, 244, 248),
                    1,
                    cv2.LINE_AA,
                )

                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
                )
                if not encoded_ok:
                    continue

                status = {
                    "ready": True,
                    "fps": round(smoothed_fps, 2),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "detectionScale": args.detection_scale,
                    "detectionScales": list(detection_scales),
                    "fusion": self.fusion,
                    "fusionMinViews": self.fusion_min_views,
                    "fusionTta": self.fusion_tta,
                    "fusionArbitration": self.fusion_arbitration,
                    "inferenceMs": round(last_inference_ms, 1),
                    "imgsz": args.imgsz,
                    "stableConfidenceAlpha": args.stable_confidence_alpha,
                    "weights": self.weights,
                    "sceneRoot": str(PreviewRequestHandler.scene_root),
                    "fallbackWeights": self.fallback_weights,
                    "primaryClasses": sorted(self.primary_classes),
                    "fallbackClasses": sorted(self.fallback_classes),
                    "fallbackSuppressIou": float(args.fallback_suppress_iou),
                    "classNames": self.state.class_names,
                    "detections": last_result["detections"],
                    "depth": depth_status,
                }
                self.state.publish(
                    encoded.tobytes(),
                    frame,
                    depth_m,
                    status,
                )
        finally:
            camera.release()

    def map_result(
        self,
        result: Any,
        x_offset: int,
        y_offset: int,
        scale: float,
    ) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None:
            return {"detections": detections}

        for box, confidence, class_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy(),
        ):
            x1 = (float(box[0]) - x_offset) / scale
            y1 = (float(box[1]) - y_offset) / scale
            x2 = (float(box[2]) - x_offset) / scale
            y2 = (float(box[3]) - y_offset) / scale
            class_id = int(class_id)
            class_name = self.state.class_names[class_id]
            detections.append(
                {
                    "class_id": class_id,
                    "class": class_name,
                    "confidence": round(float(confidence), 4),
                    "bbox": [
                        round(x1, 1),
                        round(y1, 1),
                        round(x2, 1),
                        round(y2, 1),
                    ],
                }
            )
        return {"detections": detections}

    def annotate(
        self,
        frame: np.ndarray,
        result: dict[str, Any],
    ) -> np.ndarray:
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        for item in result["detections"]:
            x1, y1, x2, y2 = (int(round(value)) for value in item["bbox"])
            x1 = min(max(x1, 0), width - 1)
            y1 = min(max(y1, 0), height - 1)
            x2 = min(max(x2, 0), width - 1)
            y2 = min(max(y2, 0), height - 1)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (54, 211, 153), 2)
            label = (
                f"{item['class']}  "
                f"{item['confidence'] * 100:.1f}%"
            )
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                1,
            )
            label_top = y1 - text_height - baseline - 6
            if label_top < 31:
                label_top = min(y1 + 3, height - text_height - baseline - 6)
            label_bottom = label_top + text_height + baseline + 6
            label_right = min(x1 + text_width + 10, width - 1)
            cv2.rectangle(
                annotated,
                (x1, label_top),
                (label_right, label_bottom),
                (18, 116, 86),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 5, label_top + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated


class PreviewRequestHandler(BaseHTTPRequestHandler):
    state: PreviewState
    capture_root: Path
    scene_root: Path
    scene_box_padding_px: int
    jpeg_quality: int
    server_version = "TaskACameraPreview/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            self.send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if path == "/status.json":
            with self.state.condition:
                payload = json.dumps(
                    self.state.status,
                    ensure_ascii=False,
                )
            self.send_text(payload, "application/json; charset=utf-8")
            return
        if path == "/snapshot.jpg":
            with self.state.condition:
                jpeg = self.state.jpeg
            if jpeg is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self.send_bytes(jpeg, "image/jpeg")
            return
        if path == "/api/scene/preview.jpg":
            with self.state.condition:
                pending = self.state.pending_scene
                jpeg = (
                    pending.get("preview_jpeg")
                    if pending is not None
                    else None
                )
            if jpeg is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_bytes(jpeg, "image/jpeg")
            return
        if path == "/stream.mjpg":
            self.stream_frames()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/capture":
            self.handle_capture()
        elif path == "/api/scene/freeze":
            self.freeze_scene()
        elif path == "/api/scene/save":
            self.save_pending_scene()
        elif path == "/api/scene/discard":
            self.discard_pending_scene()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def handle_capture(self) -> None:
        try:
            payload = self.read_json()
            raw_class_id = payload.get("class_id")
            is_background = raw_class_id is None
            class_id = -1 if is_background else int(raw_class_id)
            with self.state.condition:
                if self.state.raw_frame is None:
                    raise RuntimeError("相机画面尚未就绪")
                if not is_background and not 0 <= class_id < len(self.state.class_names):
                    raise ValueError("类别编号无效")
                frame = self.state.raw_frame.copy()
                class_name = (
                    "empty_background"
                    if is_background
                    else self.state.class_names[class_id]
                )

            safe_name = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in class_name
            )
            target_dir = self.capture_root / safe_name
            target_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target = target_dir / f"{safe_name}_{timestamp}.jpg"
            if not cv2.imwrite(
                str(target),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            ):
                raise RuntimeError(f"无法写入图片：{target}")

            self.send_json(
                {
                    "ok": True,
                    "class_id": class_id,
                    "class_name": class_name,
                    "path": str(target),
                }
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )

    def freeze_scene(self) -> None:
        try:
            self.read_json()
            with self.state.condition:
                if (
                    self.state.raw_frame is None
                    or self.state.raw_depth_m is None
                ):
                    raise RuntimeError("RGB-D 画面尚未就绪")
                if (
                    self.state.depth_args is None
                    or self.state.depth_intrinsics is None
                ):
                    raise RuntimeError("深度分割参数尚未就绪")
                frame = self.state.raw_frame.copy()
                depth_m = self.state.raw_depth_m.copy()
                depth_args = self.state.depth_args
                intrinsics = self.state.depth_intrinsics
                depth_scale = self.state.depth_scale_m_per_unit
                detections = list(self.state.status.get("detections", []))

            table_mask, object_mask, depth_status = segment_depth_frame(
                depth_m,
                intrinsics,
                depth_args,
            )
            raw_candidates = list(depth_status.get("components", []))
            candidates = add_detection_suggestions(
                raw_candidates,
                detections,
            )

            preview = render_scene_preview(
                frame,
                table_mask,
                object_mask,
                candidates,
            )
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                preview,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not encoded_ok:
                raise RuntimeError("无法编码待标注画面")

            with self.state.condition:
                self.state.pending_scene = {
                    "frame": frame,
                    "depth_m": depth_m,
                    "depth_scale_m_per_unit": depth_scale,
                    "intrinsics": intrinsics,
                    "candidates": candidates,
                    "preview": preview,
                    "preview_jpeg": encoded.tobytes(),
                }
            self.send_json(
                {
                    "ok": True,
                    "candidates": candidates,
                    "tableFraction": depth_status["tableFraction"],
                }
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )

    def save_pending_scene(self) -> None:
        try:
            payload = self.read_json()
            assignments = payload.get("assignments")
            if not isinstance(assignments, dict):
                raise ValueError("assignments 必须是候选编号到类别编号的映射")
            manual_boxes = payload.get("manual_boxes", [])
            if not isinstance(manual_boxes, list):
                raise ValueError("manual_boxes 必须是手动框列表")
            with self.state.condition:
                pending = self.state.pending_scene
                if pending is None:
                    raise RuntimeError("没有待保存的场景，请先冻结画面")
                class_names = list(self.state.class_names)
            result = save_scene(
                scene_root=self.scene_root,
                frame=pending["frame"],
                depth_m=pending["depth_m"],
                depth_scale_m_per_unit=pending[
                    "depth_scale_m_per_unit"
                ],
                intrinsics=pending["intrinsics"],
                candidates=pending["candidates"],
                assignments=assignments,
                class_names=class_names,
                box_padding_px=self.scene_box_padding_px,
                manual_boxes=manual_boxes,
                preview=pending["preview"],
            )
            with self.state.condition:
                self.state.pending_scene = None
            self.send_json({"ok": True, **result})
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )

    def discard_pending_scene(self) -> None:
        try:
            self.read_json()
            with self.state.condition:
                self.state.pending_scene = None
            self.send_json({"ok": True})
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )

    def read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def send_text(self, body: str, content_type: str) -> None:
        self.send_bytes(body.encode("utf-8"), content_type)

    def send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def stream_frames(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        sequence = 0
        try:
            while True:
                item = self.state.wait_for_frame(sequence)
                if item is None:
                    continue
                sequence, jpeg = item
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format: str, *args: Any) -> None:
        return


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.weights).is_file():
        raise ValueError(f"权重文件不存在：{args.weights}")
    if not Path(args.fallback_weights).is_file():
        raise ValueError(f"回退权重文件不存在：{args.fallback_weights}")
    parse_class_ids(args.primary_classes)
    detection_scales = parse_detection_scales(args.detection_scale)
    if len(detection_scales) > 4:
        raise ValueError("detection-scale 最多支持 4 个尺度")
    if any(not 0.1 <= scale <= 1.0 for scale in detection_scales):
        raise ValueError("detection-scale 必须在 0.1 到 1.0 之间")
    if not 0.0 < args.inference_conf <= args.conf <= 1.0:
        raise ValueError("置信度必须满足 0 < inference-conf <= conf <= 1")
    if not 0.0 < args.fallback_conf <= 1.0:
        raise ValueError("fallback-conf 必须在 0 到 1 之间")
    if not 0.0 <= args.fallback_suppress_iou <= 1.0:
        raise ValueError("fallback-suppress-iou 必须在 0 到 1 之间")
    if not 0.0 < args.iou <= 1.0:
        raise ValueError("iou 必须在 0 到 1 之间")
    if args.max_det < 1:
        raise ValueError("max-det 必须为正整数")
    if args.imgsz < 1:
        raise ValueError("imgsz 必须为正整数")
    if args.confirm_hits < 1:
        raise ValueError("confirm-hits 必须为正整数")
    if args.track_misses < 0:
        raise ValueError("track-misses 不能为负数")
    if args.max_per_class < 0:
        raise ValueError("max-per-class 不能为负数")
    if args.class_switch_hits < 1:
        raise ValueError("class-switch-hits 必须为正整数")
    if args.edge_margin_px < 0:
        raise ValueError("edge-margin-px 不能为负数")
    if not 0.0 <= args.edge_suppress_conf <= 1.0:
        raise ValueError("edge-suppress-conf 必须在 0 到 1 之间")
    for name, value in (
        ("track-iou", args.track_iou),
        ("duplicate-iou", args.duplicate_iou),
        ("cross-class-iou", args.cross_class_iou),
        ("cross-class-containment", args.cross_class_containment),
        ("containment-ratio", args.containment_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} 必须在 0 到 1 之间")
    if args.cross_class_conf_ratio < 1.0:
        raise ValueError("cross-class-conf-ratio 不能小于 1")
    if args.detect_interval < 0:
        raise ValueError("detect-interval 不能为负数")
    if args.depth_process_interval <= 0:
        raise ValueError("depth-process-interval 必须大于 0")
    if args.plane_threshold_m <= 0:
        raise ValueError("plane-threshold-m 必须大于 0")
    if not 0 < args.object_min_height_m <= args.object_max_height_m:
        raise ValueError("物体高度范围无效")
    if not 0 < args.min_depth_m < args.max_depth_m:
        raise ValueError("深度范围无效")
    if args.depth_sample_step < 1:
        raise ValueError("depth-sample-step 必须为正整数")
    if args.depth_ransac_iterations < 1:
        raise ValueError("depth-ransac-iterations 必须为正整数")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality 必须在 1 到 100 之间")
    if not 0 <= args.scene_box_padding_px <= 20:
        raise ValueError("scene-box-padding-px 必须在 0 到 20 之间")
    if not 0 < args.port < 65536:
        raise ValueError("port 必须在 1 到 65535 之间")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        print(f"参数错误：{error}", file=sys.stderr)
        return 2

    state = PreviewState()
    PreviewRequestHandler.state = state
    PreviewRequestHandler.capture_root = Path(args.capture_root).resolve()
    PreviewRequestHandler.scene_root = Path(args.scene_root).resolve()
    PreviewRequestHandler.scene_box_padding_px = args.scene_box_padding_px
    PreviewRequestHandler.jpeg_quality = args.jpeg_quality
    server = ThreadingHTTPServer(
        (args.host, args.port),
        PreviewRequestHandler,
    )
    url = f"http://{args.host}:{args.port}/"

    try:
        worker = PreviewWorker(args, state)
    except Exception as error:
        server.server_close()
        print(f"预览初始化失败：{error}", file=sys.stderr)
        return 2

    worker.start()
    print(f"Task A camera preview: {url}", file=sys.stderr)
    print("robot commands: disabled", file=sys.stderr)
    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        server.shutdown()
        server.server_close()
        worker.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
