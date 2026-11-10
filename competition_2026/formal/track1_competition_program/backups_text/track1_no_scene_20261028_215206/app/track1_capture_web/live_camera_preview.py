#!/usr/bin/env python3
"""Track 1 RGB-D self-collection web service.

The service reads the D405, runs YOLO only as an auxiliary hint, and writes
labelled RGB-D frames. It never imports ROS or controls the robot/gripper.
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
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_store import (  # noqa: E402
    CaptureError,
    CaptureStore,
    ClassCatalog,
    load_class_catalog,
)
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


DEFAULT_CAMERA = "auto"
PROJECT_ROOT = Path(
    "/media/sunshine/新加卷/埃斯顿比赛_2026/赛道一比赛程序"
)
DEFAULT_CONFIG = str(PROJECT_ROOT / "config" / "track1_capture_web.yaml")
DEFAULT_WEIGHTS = str(PROJECT_ROOT / "models" / "best.pt")
DEFAULT_FALLBACK_WEIGHTS = ""
DEFAULT_PRIMARY_CLASSES = "0,1,2,3,4,5,6,7,8,9"
DEFAULT_CLASS_MAP = str(
    PROJECT_ROOT
    / "datasets"
    / "track1_selfcollected_v1"
    / "manifests"
    / "class_map.yaml"
)
DEFAULT_PHOTOS_ROOT = "/media/sunshine/新加卷/赛道一_C_D自采照片"
DEFAULT_SCENE_ROOT = str(
    PROJECT_ROOT / "datasets" / "track1_selfcollected_v1" / "annotations"
)


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>赛道一 RGB-D 自采数据</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, "Noto Sans SC", "Microsoft YaHei", system-ui, sans-serif;
      background: #0b1117;
      color: #eef4fa;
      --panel: #131c24;
      --panel-2: #18232d;
      --line: #2d3d4b;
      --muted: #9eb0c0;
      --accent: #31d0a5;
      --accent-2: #55a9ff;
      --danger: #ff6d7a;
      --warn: #ffd166;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; background: radial-gradient(circle at 15% -10%, #18334a 0, #0b1117 44%); }
    button, input, select, textarea { font: inherit; }
    button { cursor: pointer; }
    main { width: min(100%, 1480px); margin: 0 auto; padding: 18px; }
    header { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 13px; }
    h1 { margin: 0 0 5px; font-size: clamp(20px, 2vw, 28px); letter-spacing: .01em; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    .subhead { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.55; }
    .header-right { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }
    #status { color: #b8c7d5; font-size: 13px; line-height: 1.45; text-align: right; font-variant-numeric: tabular-nums; max-width: 480px; }
    .workspace { display: grid; grid-template-columns: minmax(0, 1.8fr) minmax(320px, .82fr); gap: 14px; align-items: start; }
    .card { border: 1px solid var(--line); border-radius: 12px; background: rgba(19, 28, 36, .94); box-shadow: 0 12px 35px rgba(0,0,0,.16); }
    .video-card { padding: 10px; }
    .view { position: relative; overflow: hidden; border: 1px solid #314252; border-radius: 8px; background: #030608; aspect-ratio: 4 / 3; }
    .view img { display: block; width: 100%; height: 100%; object-fit: contain; }
    .stream-foot { display: flex; justify-content: space-between; gap: 12px; padding: 9px 3px 1px; color: var(--muted); font-size: 13px; }
    #model-status { text-align: right; }
    .control-card { padding: 15px; }
    .field { display: grid; gap: 6px; margin-bottom: 11px; }
    .field > span, .field-label { color: #c8d4df; font-size: 12px; font-weight: 650; letter-spacing: .02em; }
    .field input, .field select, .field textarea {
      width: 100%; min-height: 38px; border: 1px solid #415261; border-radius: 7px;
      background: #0f1820; color: #eef4fa; padding: 8px 10px; outline: none;
    }
    .field textarea { min-height: 62px; resize: vertical; }
    .field input:focus, .field select:focus, .field textarea:focus { border-color: var(--accent-2); box-shadow: 0 0 0 2px rgba(85,169,255,.14); }
    .field input:disabled, .field select:disabled { opacity: .7; cursor: not-allowed; }
    .inline { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 7px; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 9px; }
    .hint { margin: -5px 0 11px; color: var(--muted); font-size: 12px; line-height: 1.5; overflow-wrap: anywhere; }
    .warning { padding: 9px 10px; border: 1px solid rgba(255,209,102,.35); border-radius: 8px; background: rgba(255,209,102,.08); color: #f4df9c; font-size: 12px; line-height: 1.5; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 13px; }
    button.primary, button.secondary, button.ghost, button.danger {
      min-height: 38px; border-radius: 7px; padding: 8px 12px; border: 1px solid transparent; color: #f5f9fc;
    }
    button.primary { background: linear-gradient(135deg, #1fb98f, #17876c); border-color: #31d0a5; font-weight: 700; }
    button.secondary { background: #2268a8; border-color: #438ed9; }
    button.ghost { background: #1a2630; border-color: #415261; }
    button.danger { background: #51242c; border-color: #a34857; color: #ffd8dd; }
    button:hover:not(:disabled) { filter: brightness(1.12); }
    button:disabled { cursor: not-allowed; opacity: .55; }
    #capture-status { margin-top: 11px; min-height: 39px; padding: 9px 10px; border-radius: 8px; background: #0e171f; border: 1px solid #2d3d4b; color: #b9c8d6; font-size: 12px; line-height: 1.55; }
    .progress { height: 7px; overflow: hidden; border-radius: 999px; background: #0a1117; border: 1px solid #2d3d4b; margin: 9px 0; display: none; }
    .progress.visible { display: block; }
    #progress-bar { width: 0; height: 100%; background: linear-gradient(90deg, #31d0a5, #55a9ff); transition: width .16s ease; }
    .output { margin-top: 11px; color: var(--muted); font-size: 11px; line-height: 1.55; overflow-wrap: anywhere; }
    .output strong { color: #c8d4df; font-size: 11px; }
    .recent-card { margin-top: 14px; padding: 14px; }
    .recent-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
    .recent-head h2 { margin: 0; }
    #session-total { color: var(--muted); font-size: 13px; }
    .counts { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 11px; }
    .count-chip { padding: 5px 8px; border-radius: 999px; border: 1px solid #344655; background: #111b23; color: #b9c8d6; font-size: 11px; }
    .recent-grid { display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 8px; }
    .recent-item { min-width: 0; color: var(--muted); font-size: 10px; text-decoration: none; }
    .recent-item img { display: block; width: 100%; aspect-ratio: 4/3; object-fit: cover; border: 1px solid #354756; border-radius: 6px; background: #060a0d; }
    .recent-item span { display: block; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; padding-top: 4px; }
    .empty { color: var(--muted); font-size: 12px; padding: 8px 0; }
    details.scene-card { margin-top: 14px; padding: 13px 14px; }
    details.scene-card summary { cursor: pointer; color: #cbd8e3; font-weight: 650; }
    .scene-copy { margin: 10px 0; color: var(--muted); font-size: 12px; line-height: 1.6; }
    .scene-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    #scene-status { margin-left: auto; color: var(--muted); font-size: 12px; text-align: right; }
    #scene-preview-wrap { display: none; margin-top: 10px; }
    #scene-preview-wrap.visible { display: block; }
    #scene-canvas { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; touch-action: none; }
    #scene-preview-wrap.drawing #scene-canvas { pointer-events: auto; cursor: crosshair; }
    .assignments { display: grid; gap: 1px; margin-top: 10px; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; background: var(--line); }
    .assignment { display: grid; grid-template-columns: minmax(140px, .8fr) minmax(220px, 1.2fr); align-items: center; gap: 9px; padding: 8px 10px; background: #111a22; }
    .assignment span { color: #c1cfda; font-size: 12px; }
    .assignment-controls { display: grid; grid-template-columns: minmax(180px, 1fr) auto; gap: 7px; }
    .banner { display: none; margin: 0 0 12px; padding: 10px 12px; border: 1px solid #a34857; border-radius: 9px; background: #3c1e24; color: #ffd9de; white-space: pre-wrap; }
    .banner.visible { display: block; }
    @media (max-width: 980px) {
      .workspace { grid-template-columns: 1fr; }
      .recent-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    }
    @media (max-width: 600px) {
      main { padding: 10px; }
      header { display: block; }
      .header-right { margin-top: 10px; justify-content: flex-start; }
      #status { text-align: left; }
      .two-col { grid-template-columns: 1fr; }
      .recent-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .assignment { grid-template-columns: 1fr; }
      .assignment-controls { grid-template-columns: 1fr auto; }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>赛道一 RGB-D 自采数据</h1>
      <p class="subhead">D405 实时预览 + 40 类手动标签采集。只采图、不控制机械臂和夹爪。</p>
    </div>
    <div class="header-right">
      <div id="status">正在加载采集服务...</div>
      <button id="stop-service" class="danger" type="button">停止服务</button>
    </div>
  </header>

  <div id="error-banner" class="banner" role="alert"></div>

  <div class="workspace">
    <section class="card video-card">
      <div class="view">
        <img id="stream" src="/stream.mjpg" alt="D405 实时画面">
      </div>
      <div class="stream-foot">
        <span id="stream-meta">正在连接 D405...</span>
        <span id="model-status">模型加载中</span>
      </div>
    </section>

    <aside class="card control-card">
      <h2>采集设置</h2>
      <label class="field">
        <span>采集会话 ID</span>
        <div class="inline">
          <input id="session-id" autocomplete="off" spellcheck="false" placeholder="20261027_session_001">
          <button id="new-session" class="ghost" type="button">新会话</button>
        </div>
      </label>
      <p class="hint" id="session-hint">同一批采集使用同一个会话；照片会统一进入 photos 根目录。</p>

      <label class="field">
        <span>物品名称（40 类 + 空背景）</span>
        <select id="class-select" aria-label="物品类别"></select>
      </label>
      <p class="hint" id="class-hint">正在加载类别表...</p>

      <div class="two-col">
        <label class="field">
          <span>采集场景</span>
          <select id="scene-type"></select>
        </label>
        <label class="field">
          <span>实例 ID（可选）</span>
          <input id="instance-id" autocomplete="off" spellcheck="false" placeholder="11-Padlock#01">
        </label>
      </div>
      <p class="hint" id="scene-hint"></p>

      <div class="two-col">
        <label class="field">
          <span>光照 / 场地 ID</span>
          <input id="lighting-id" value="site_A" autocomplete="off" spellcheck="false">
        </label>
        <label class="field">
          <span>备注（可选）</span>
          <input id="notes" autocomplete="off" placeholder="例如：透明箱壁反光">
        </label>
      </div>

      <div class="warning" id="model-warning">当前模型只识别前 10 类；第 11–40 类必须手动选择标签采集。</div>

      <div class="actions">
        <button id="capture-one" class="primary" type="button" disabled>采一张 RGB-D</button>
        <button id="capture-burst" class="secondary" type="button" disabled>连拍</button>
      </div>
      <div class="two-col" style="margin-top:8px">
        <label class="field">
          <span>连拍张数</span>
          <input id="burst-count" type="number" min="2" max="100" value="20">
        </label>
        <label class="field">
          <span>间隔（毫秒）</span>
          <input id="burst-interval" type="number" min="100" max="5000" step="50" value="350">
        </label>
      </div>
      <div class="progress" id="progress"><div id="progress-bar"></div></div>
      <div id="capture-status">等待类别表、相机和模型就绪...</div>
      <div class="output">
        <strong>统一照片目录</strong><br>
        <span id="photos-root">加载中...</span><br>
        <strong>当前会话目录</strong><br>
        <span id="session-path">加载中...</span>
      </div>
    </aside>
  </div>

  <section class="card recent-card">
    <div class="recent-head">
      <h2>最近采集</h2>
      <span id="session-total">本会话 0 张</span>
    </div>
    <div class="counts" id="counts"><span class="empty">本会话还没有采集记录。</span></div>
    <div class="recent-grid" id="recent-grid"></div>
  </section>

  <details class="card scene-card">
    <summary>高级：冻结当前 RGB-D 并做场景框选标注（可选）</summary>
    <p class="scene-copy">这一区域用于从当前帧做多物体 bbox 标注，保存到独立 annotations 目录；普通分类采集不需要使用。</p>
    <div class="scene-actions">
      <button id="scene-freeze" class="ghost" type="button">冻结并分割</button>
      <button id="scene-draw" class="ghost" type="button" disabled>手动框选</button>
      <button id="scene-save" class="ghost" type="button" disabled>保存标注</button>
      <button id="scene-discard" class="ghost" type="button" disabled>放弃</button>
      <div id="scene-status">放置物品后冻结当前画面</div>
    </div>
    <div class="view" id="scene-preview-wrap">
      <img id="scene-preview" alt="待标注的 RGB-D 场景">
      <canvas id="scene-canvas" width="640" height="480"></canvas>
    </div>
    <div class="assignments" id="assignments" hidden></div>
  </details>
</main>

<script>
  const statusEl = document.getElementById("status");
  const stream = document.getElementById("stream");
  const stopService = document.getElementById("stop-service");
  const errorBanner = document.getElementById("error-banner");
  const classSelect = document.getElementById("class-select");
  const classHint = document.getElementById("class-hint");
  const sessionInput = document.getElementById("session-id");
  const sessionHint = document.getElementById("session-hint");
  const newSessionButton = document.getElementById("new-session");
  const sceneSelect = document.getElementById("scene-type");
  const sceneHint = document.getElementById("scene-hint");
  const instanceInput = document.getElementById("instance-id");
  const lightingInput = document.getElementById("lighting-id");
  const notesInput = document.getElementById("notes");
  const modelWarning = document.getElementById("model-warning");
  const captureOne = document.getElementById("capture-one");
  const captureBurst = document.getElementById("capture-burst");
  const burstCount = document.getElementById("burst-count");
  const burstInterval = document.getElementById("burst-interval");
  const captureStatus = document.getElementById("capture-status");
  const progress = document.getElementById("progress");
  const progressBar = document.getElementById("progress-bar");
  const photosRoot = document.getElementById("photos-root");
  const sessionPath = document.getElementById("session-path");
  const sessionTotal = document.getElementById("session-total");
  const countsEl = document.getElementById("counts");
  const recentGrid = document.getElementById("recent-grid");
  const streamMeta = document.getElementById("stream-meta");
  const modelStatus = document.getElementById("model-status");

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

  let catalog = [];
  let backgroundEntry = null;
  let modelSupport = null;
  let sceneTypes = {};
  let catalogReady = false;
  let serviceReady = false;
  let captureBusy = false;
  let sessionLocked = false;
  let reconnectTimer = 0;
  let currentSession = "";
  let pendingCandidates = [];
  let manualBoxes = [];
  let sceneFrozen = false;
  let drawMode = false;
  let dragStart = null;
  let dragBox = null;

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.classList.add("visible");
  }

  function clearError() {
    errorBanner.textContent = "";
    errorBanner.classList.remove("visible");
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, {cache: "no-store", ...options});
    const text = await response.text();
    let value = null;
    try {
      value = text ? JSON.parse(text) : null;
    } catch {
      value = null;
    }
    if (!response.ok) {
      const message = value && value.error ? value.error : (text || `HTTP ${response.status}`);
      throw new Error(message);
    }
    return value;
  }

  function reconnectStream() {
    if (reconnectTimer) return;
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = 0;
      stream.src = `/stream.mjpg?reconnect=${Date.now()}`;
    }, 1200);
  }
  stream.addEventListener("error", reconnectStream);

  function selectedCatalogEntry() {
    const id = Number(classSelect.value);
    if (id === 0) return backgroundEntry;
    return catalog.find(item => item.class_id === id) || null;
  }

  function classOptionLabel(entry) {
    return `${String(entry.class_id).padStart(2, "0")} | ${entry.display_name} | ${entry.canonical_label}`;
  }

  function classOptions(selectedValue = "") {
    return [backgroundEntry, ...catalog].map(entry => {
      const option = document.createElement("option");
      option.value = String(entry.class_id);
      option.textContent = classOptionLabel(entry);
      if (String(entry.class_id) === String(selectedValue)) option.selected = true;
      return option;
    });
  }

  function classIndexByCanonicalLabel(label) {
    const index = catalog.findIndex(item =>
      item.canonical_label === label ||
      item.model_class_names.includes(label) ||
      item.english_name === label
    );
    return index >= 0 ? index : "";
  }

  function updateClassHint() {
    const entry = selectedCatalogEntry();
    if (!entry || !modelSupport) return;
    if (entry.is_background) {
      classHint.textContent = "空背景：画面中不摆放目标物品，用于负样本或桌面/箱底背景。";
      return;
    }
    const supported = modelSupport.supported_class_ids.includes(entry.class_id);
    classHint.textContent = supported
      ? `模型辅助类别：${entry.canonical_label}；采集标签仍以这里的选择为准。`
      : `当前模型不识别 ${entry.canonical_label}；请人工确认标签，模型框只作环境提示。`;
  }

  function updateSceneHint() {
    const value = sceneTypes[sceneSelect.value];
    if (!value) {
      sceneHint.textContent = "";
      return;
    }
    sceneHint.textContent = `${value.description} · 记录 box_type=${value.box_type}`;
  }

  async function loadCatalog() {
    const value = await requestJson("/api/classes");
    catalog = Array.isArray(value.classes) ? value.classes : [];
    backgroundEntry = value.background;
    modelSupport = value.model_support || {};
    sceneTypes = value.scene_types || {};
    if (catalog.length !== 40 || !backgroundEntry) {
      throw new Error("类别表不是 40 类，请检查 class_map.yaml");
    }
    classSelect.replaceChildren(...classOptions(11));
    sceneSelect.replaceChildren(...Object.entries(sceneTypes).map(([key, item]) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = item.label || key;
      return option;
    }));
    if (sceneSelect.options.length) sceneSelect.value = "tabletop";
    catalogReady = true;
    updateClassHint();
    updateSceneHint();
    const supported = modelSupport.supported_class_ids || [];
    modelWarning.textContent = supported.length
      ? `当前模型只识别第 ${supported[0]}–${supported[supported.length - 1]} 类；第 11–40 类必须手动选择标签采集。`
      : "当前没有可用模型；40 类全部手动选择标签采集。";
    updateCaptureButtons();
  }

  function updateCaptureButtons() {
    const canCapture = catalogReady && serviceReady && !captureBusy;
    captureOne.disabled = !canCapture;
    captureBurst.disabled = !canCapture;
    newSessionButton.disabled = captureBusy;
    sessionInput.disabled = captureBusy || sessionLocked;
    classSelect.disabled = captureBusy;
    sceneSelect.disabled = captureBusy;
  }

  function setCaptureBusy(busy) {
    captureBusy = busy;
    updateCaptureButtons();
  }

  function normalizeOverlayUrl(url) {
    if (!url) return "";
    return `${url}${url.includes("?") ? "&" : "?"}v=${Date.now()}`;
  }

  function renderCaptureSummary(value) {
    const session = value.session || {};
    const total = Number(session.total_frames || 0);
    sessionTotal.textContent = `本会话 ${total} 张`;
    const counts = session.counts_by_class || {};
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    if (!entries.length) {
      countsEl.innerHTML = '<span class="empty">本会话还没有采集记录。</span>';
    } else {
      countsEl.replaceChildren(...entries.map(([label, count]) => {
        const chip = document.createElement("span");
        chip.className = "count-chip";
        chip.textContent = `${label} × ${count}`;
        return chip;
      }));
    }
    const recent = Array.isArray(value.recent) ? value.recent : [];
    recentGrid.replaceChildren(...recent.map(item => {
      const anchor = document.createElement("a");
      anchor.className = "recent-item";
      anchor.target = "_blank";
      anchor.rel = "noopener";
      anchor.href = item.overlay_url || "#";
      if (item.overlay_url) {
        const img = document.createElement("img");
        img.loading = "lazy";
        img.src = normalizeOverlayUrl(item.overlay_url);
        img.alt = `${item.canonical_label || ""} ${item.frame_id || ""}`;
        anchor.append(img);
      }
      const caption = document.createElement("span");
      caption.textContent = `#${item.frame_id || "?"} ${item.display_name || item.canonical_label || ""}`;
      anchor.append(caption);
      return anchor;
    }));
    if (!recent.length) {
      const empty = document.createElement("span");
      empty.className = "empty";
      empty.textContent = "还没有可显示的缩略图。";
      recentGrid.append(empty);
    }
    photosRoot.textContent = value.photosRoot || "未知";
    sessionPath.textContent = value.sessionPath || "未知";
  }

  async function loadCaptureSummary() {
    const session = sessionInput.value.trim();
    const query = session ? `?session_id=${encodeURIComponent(session)}` : "";
    const value = await requestJson(`/api/capture/summary${query}`);
    currentSession = value.sessionId || session;
    if (!sessionInput.value.trim()) sessionInput.value = currentSession;
    renderCaptureSummary(value);
  }

  async function nextSession() {
    clearError();
    try {
      const value = await requestJson("/api/session/next");
      sessionInput.value = value.sessionId;
      sessionLocked = false;
      sessionHint.textContent = "已切换到新会话；采下第一张后会话 ID 会锁定。";
      updateCaptureButtons();
      await loadCaptureSummary();
      captureStatus.textContent = `新会话：${value.sessionId}`;
    } catch (error) {
      showError(`新建会话失败：${error.message}`);
    }
  }

  function capturePayload(mode, group = "") {
    return {
      session_id: sessionInput.value.trim(),
      class_id: Number(classSelect.value),
      scene_type: sceneSelect.value,
      instance_id: instanceInput.value.trim(),
      lighting_id: lightingInput.value.trim(),
      notes: notesInput.value.trim(),
      capture_mode: mode,
      capture_group: group
    };
  }

  async function captureFrame(mode = "single", group = "") {
    const value = await requestJson("/api/capture", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(capturePayload(mode, group))
    });
    sessionLocked = true;
    updateCaptureButtons();
    const summary = {
      photosRoot: photosRoot.textContent,
      sessionId: value.session_id,
      sessionPath: value.session_path,
      session: {
        total_frames: value.total_frames,
        counts_by_class: value.counts_by_class
      },
      recent: value.recent
    };
    renderCaptureSummary(summary);
    return value;
  }

  captureOne.addEventListener("click", async () => {
    clearError();
    setCaptureBusy(true);
    captureStatus.textContent = "正在读取同一对齐帧的 RGB 与深度并写入 PNG...";
    try {
      const value = await captureFrame("single");
      captureStatus.textContent = `已保存 #${value.frame_id} · ${value.class_name} · 本会话共 ${value.total_frames} 张`;
    } catch (error) {
      showError(`采图失败：${error.message}`);
      captureStatus.textContent = "采图失败，请查看上方错误提示。";
    } finally {
      setCaptureBusy(false);
    }
  });

  captureBurst.addEventListener("click", async () => {
    clearError();
    const count = Math.max(2, Math.min(100, Number(burstCount.value) || 20));
    const interval = Math.max(100, Math.min(5000, Number(burstInterval.value) || 350));
    const group = `burst-${Date.now()}`;
    setCaptureBusy(true);
    progress.classList.add("visible");
    progressBar.style.width = "0%";
    try {
      for (let index = 0; index < count; index += 1) {
        const value = await captureFrame("burst", group);
        const percent = ((index + 1) / count) * 100;
        progressBar.style.width = `${percent.toFixed(1)}%`;
        captureStatus.textContent = `连拍 ${index + 1}/${count} · #${value.frame_id} · ${value.class_name}`;
        if (index + 1 < count) await new Promise(resolve => setTimeout(resolve, interval));
      }
      captureStatus.textContent = `连拍完成：本次 ${count} 张。`;
    } catch (error) {
      showError(`连拍中断：${error.message}`);
      captureStatus.textContent = "连拍中断；已完成部分仍保留在会话中。";
    } finally {
      setCaptureBusy(false);
      setTimeout(() => progress.classList.remove("visible"), 700);
    }
  });

  classSelect.addEventListener("change", updateClassHint);
  sceneSelect.addEventListener("change", updateSceneHint);
  sessionInput.addEventListener("change", async () => {
    sessionLocked = false;
    updateCaptureButtons();
    try { await loadCaptureSummary(); } catch (error) { showError(`读取会话失败：${error.message}`); }
  });
  newSessionButton.addEventListener("click", nextSession);

  stopService.addEventListener("click", async () => {
    if (!window.confirm("确认停止视觉采集服务？已保存的数据不会删除。")) return;
    try {
      await requestJson("/api/stop", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: "{}"
      });
      serviceReady = false;
      updateCaptureButtons();
      statusEl.textContent = "采集服务正在停止...";
      captureStatus.textContent = "服务已请求停止，可以关闭网页。";
    } catch (error) {
      showError(`停止服务失败：${error.message}`);
    }
  });

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
    scenePreviewWrap.classList.remove("visible", "drawing");
    sceneDraw.textContent = "手动框选";
    scenePreview.removeAttribute("src");
    setSceneBusy(false);
  }

  function setSceneBusy(busy) {
    sceneFreeze.disabled = busy;
    sceneDraw.disabled = busy || !sceneFrozen;
    sceneSave.disabled = busy || !sceneFrozen || (pendingCandidates.length === 0 && manualBoxes.length === 0);
    sceneDiscard.disabled = busy || !sceneFrozen;
  }

  function renderAssignments(candidates) {
    const candidateRows = candidates.map((candidate, index) => {
      const row = document.createElement("label");
      row.className = "assignment";
      const [x, y, width, height] = candidate.bbox_xywh;
      const summary = document.createElement("span");
      summary.textContent = `#${index + 1}  ${width}×${height}px  ${candidate.area_pixels}px²`;
      const select = document.createElement("select");
      select.dataset.candidateId = String(index + 1);
      const suggestedIndex = candidate.suggested_class
        ? classIndexByCanonicalLabel(candidate.suggested_class)
        : "";
      select.append(
        ...classOptions(suggestedIndex),
        Object.assign(document.createElement("option"), {value: "ignore", textContent: "忽略此候选"})
      );
      if (suggestedIndex === "") select.value = "ignore";
      row.append(summary, select);
      return row;
    });
    const manualRows = manualBoxes.map((box, index) => {
      const row = document.createElement("div");
      row.className = "assignment";
      const [x1, y1, x2, y2] = box.bbox_xyxy;
      const summary = document.createElement("span");
      summary.textContent = `手动 M${index + 1}  ${Math.round(x2 - x1)}×${Math.round(y2 - y1)}px`;
      const controls = document.createElement("div");
      controls.className = "assignment-controls";
      const select = document.createElement("select");
      select.dataset.manualIndex = String(index);
      select.append(...classOptions(String(box.classId + 1)));
      select.value = String(box.classId + 1);
      select.addEventListener("change", () => {
        manualBoxes[index].classId = Number(select.value) - 1;
        drawSceneCanvas();
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "ghost";
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
    sceneContext.clearRect(0, 0, sceneCanvas.width, sceneCanvas.height);
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
      x: Math.max(0, Math.min(sceneCanvas.width, (event.clientX - bounds.left) * sceneCanvas.width / bounds.width)),
      y: Math.max(0, Math.min(sceneCanvas.height, (event.clientY - bounds.top) * sceneCanvas.height / bounds.height))
    };
  }

  sceneDraw.addEventListener("click", () => {
    if (!drawMode && selectedCatalogEntry()?.is_background) {
      sceneStatus.textContent = "手动框选前请先在上方类别中选择物品";
      return;
    }
    drawMode = !drawMode;
    scenePreviewWrap.classList.toggle("drawing", drawMode);
    sceneDraw.textContent = drawMode ? "结束框选" : "手动框选";
    sceneStatus.textContent = drawMode
      ? `拖动鼠标框住物体；当前类别：${selectedCatalogEntry()?.canonical_label || ""}`
      : "框选已结束，可调整类别或保存";
  });

  sceneCanvas.addEventListener("pointerdown", event => {
    if (!drawMode) return;
    event.preventDefault();
    dragStart = canvasPoint(event);
    dragBox = [dragStart.x, dragStart.y, dragStart.x, dragStart.y];
    sceneCanvas.setPointerCapture(event.pointerId);
    drawSceneCanvas();
  });

  sceneCanvas.addEventListener("pointermove", event => {
    if (dragStart === null) return;
    const point = canvasPoint(event);
    dragBox = [
      Math.min(dragStart.x, point.x), Math.min(dragStart.y, point.y),
      Math.max(dragStart.x, point.x), Math.max(dragStart.y, point.y)
    ];
    drawSceneCanvas();
  });

  sceneCanvas.addEventListener("pointerup", event => {
    if (dragStart === null || dragBox === null) return;
    sceneCanvas.releasePointerCapture(event.pointerId);
    const [x1, y1, x2, y2] = dragBox;
    dragStart = null;
    dragBox = null;
    if (x2 - x1 < 8 || y2 - y1 < 8) {
      drawSceneCanvas();
      return;
    }
    if (selectedCatalogEntry()?.is_background) {
      sceneStatus.textContent = "请先在上方类别中选择物品再拖框";
      drawSceneCanvas();
      return;
    }
    manualBoxes.push({bbox_xyxy: [x1, y1, x2, y2], classId: Number(classSelect.value) - 1});
    renderAssignments(pendingCandidates);
    drawSceneCanvas();
    sceneStatus.textContent = `已添加 M${manualBoxes.length}，可继续框选或保存`;
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
      const value = await requestJson("/api/scene/freeze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: "{}"
      });
      pendingCandidates = value.candidates || [];
      manualBoxes = [];
      sceneFrozen = true;
      renderAssignments(pendingCandidates);
      scenePreview.src = `/api/scene/preview.jpg?t=${Date.now()}`;
      scenePreviewWrap.classList.add("visible");
      drawSceneCanvas();
      sceneStatus.textContent = `已找到 ${pendingCandidates.length} 个候选，请逐个选择类别`;
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
    assignments.querySelectorAll("select[data-candidate-id]").forEach(select => {
      if (select.value !== "ignore") selected[select.dataset.candidateId] = Number(select.value) - 1;
    });
    try {
      const value = await requestJson("/api/scene/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          assignments: selected,
          manual_boxes: manualBoxes.map(box => ({class_id: box.classId, bbox_xyxy: box.bbox_xyxy}))
        })
      });
      sceneStatus.textContent = `已保存 ${value.objects} 个标注 · ${value.scene}`;
      clearScene();
    } catch (error) {
      sceneStatus.textContent = `保存失败：${error.message}`;
      sceneSave.disabled = false;
    }
  });

  sceneDiscard.addEventListener("click", async () => {
    sceneDiscard.disabled = true;
    try {
      await requestJson("/api/scene/discard", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: "{}"
      });
    } finally {
      clearScene();
      sceneStatus.textContent = "已放弃当前场景";
    }
  });

  async function refreshStatus() {
    try {
      const value = await requestJson("/status.json");
      if (!value.ready) {
        serviceReady = false;
        statusEl.textContent = value.error ? `识别服务错误：${value.error}` : "正在加载 D405、YOLO 和深度模块...";
        if (value.error) showError(value.error);
        updateCaptureButtons();
        return;
      }
      serviceReady = true;
      clearError();
      const parts = [
        `${Number(value.fps || 0).toFixed(1)} FPS`,
        `${value.width}x${value.height}`,
        `推理 ${Number(value.inferenceMs || 0).toFixed(0)} ms`
      ];
      if (value.depth) parts.push(`深度候选 ${value.depth.objectComponents ?? 0}`);
      statusEl.textContent = parts.join(" · ");
      streamMeta.textContent = `D405 实时画面 · ${value.width}x${value.height}`;
      const labels = (value.detections || []).map(item =>
        `${item.class} ${((item.stableConfidence ?? item.confidence) * 100).toFixed(1)}%`
      );
      modelStatus.textContent = labels.length ? `模型辅助：${labels.join(" | ")}` : "模型辅助：未识别到前 10 类物品";
      updateCaptureButtons();
    } catch {
      serviceReady = false;
      statusEl.textContent = "采集服务暂时不可用";
      updateCaptureButtons();
      reconnectStream();
    }
  }

  async function boot() {
    try {
      await loadCatalog();
      await loadCaptureSummary();
    } catch (error) {
      showError(`初始化失败：${error.message}`);
      captureStatus.textContent = "初始化失败，请检查类别表和采集目录。";
    }
    await refreshStatus();
    window.setInterval(refreshStatus, 500);
  }

  boot();
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
            "Track 1 RGB-D self-collection web service. "
            "The process never connects to the robot."
        )
    )
    parser.add_argument(
        "--camera",
        type=parse_camera,
        default=DEFAULT_CAMERA,
        help="RealSense 自动选择时为 auto；兼容旧参数",
    )
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
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=180)
    parser.add_argument(
        "--rgbd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 D405 RGB-D；采集必须保持启用",
    )
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
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="采集网页 YAML 配置；从 capture.scene_types 读取场景定义",
    )
    parser.add_argument("--class-map", default=DEFAULT_CLASS_MAP)
    parser.add_argument("--photos-root", default=DEFAULT_PHOTOS_ROOT)
    parser.add_argument("--scene-root", default=DEFAULT_SCENE_ROOT)
    parser.add_argument(
        "--scene-types-json",
        default="",
        help="场景类型 JSON；设置后覆盖 --config 中的场景定义",
    )
    parser.add_argument(
        "--default-lighting-id",
        default="site_A",
        help="网页未填写光照/场地 ID 时使用的默认值",
    )
    parser.add_argument(
        "--save-overlay-jpeg",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--scene-box-padding-px", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9110)
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
        try:
            self.serial = str(
                device.get_info(rs.camera_info.serial_number)
            ).strip()
        except RuntimeError as error:
            raise RuntimeError("无法读取 D405 相机序列号") from error
        if not self.serial:
            raise RuntimeError("D405 相机序列号为空")
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

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        depth_units = rotate_frame(
            np.asanyarray(depth_frame.get_data()),
            self.args.rotate,
        )
        if depth_units.dtype != np.uint16:
            raise RuntimeError(
                f"D405 深度帧格式异常：{depth_units.dtype}，预期 uint16"
            )
        depth_m = depth_units.astype(np.float32) * self.depth_scale
        return color, depth_units, depth_m

    def release(self) -> None:
        try:
            self.pipeline.stop()
        except RuntimeError:
            pass


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
        self.raw_depth_units: np.ndarray | None = None
        self.raw_depth_m: np.ndarray | None = None
        self.depth_scale_m_per_unit = 0.0
        self.depth_intrinsics: Any | None = None
        self.depth_args: argparse.Namespace | None = None
        self.camera_serial = ""
        self.camera_model = ""
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
        raw_depth_units: np.ndarray | None,
        raw_depth_m: np.ndarray | None,
        status: dict[str, Any],
    ) -> None:
        with self.condition:
            self.sequence += 1
            status = dict(status)
            status["frameSequence"] = self.sequence
            self.jpeg = jpeg
            self.raw_frame = raw_frame.copy()
            self.raw_depth_units = (
                raw_depth_units.copy()
                if raw_depth_units is not None
                else None
            )
            self.raw_depth_m = (
                raw_depth_m.copy() if raw_depth_m is not None else None
            )
            self.status = status
            self.condition.notify_all()

    def set_error(self, message: str) -> None:
        with self.condition:
            status = dict(self.status)
            status.update(
                {
                    "ready": False,
                    "error": str(message),
                    "frameSequence": self.sequence,
                }
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
        super().__init__(name="track1-capture-preview", daemon=True)
        self.args = args
        self.state = state
        self.stop_event = threading.Event()
        self.weights = str(Path(args.weights).resolve())
        self.fallback_weights = (
            str(Path(args.fallback_weights).resolve())
            if args.fallback_weights
            else ""
        )
        self.model = YOLO(self.weights)
        self.fallback_model = (
            YOLO(self.fallback_weights) if self.fallback_weights else None
        )
        self.state.class_names = [
            str(self.model.names[index])
            for index in sorted(self.model.names)
        ]
        if self.fallback_model is not None:
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
            if self.fallback_model is not None
            else set()
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
        try:
            self._run()
        except Exception as error:  # noqa: BLE001 - expose startup/runtime failure to web UI
            self.state.set_error(f"采集线程异常：{error}")

    def _run(self) -> None:
        args = self.args
        detection_scales = parse_detection_scales(args.detection_scale)
        if args.rgbd:
            camera: Any = RealSenseCamera(args)
            self.state.depth_scale_m_per_unit = camera.depth_scale
            self.state.depth_intrinsics = camera.intrinsics
            self.state.depth_args = args
            self.state.camera_serial = camera.serial
            self.state.camera_model = "Intel RealSense D405"
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
                depth_units = None
                depth_m = None
                if args.rgbd:
                    frame, depth_units, depth_m = camera.read()
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
                    f"Track 1 RGB-D capture  {smoothed_fps:.1f} FPS  "
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
                    depth_units,
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
    capture_store: CaptureStore
    catalog: ClassCatalog
    scene_root: Path
    scene_types: dict[str, dict[str, Any]]
    scene_box_padding_px: int
    jpeg_quality: int
    save_overlay_jpeg: bool = True
    server_version = "Track1CaptureWeb/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        path = parsed.path
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
        if path == "/api/classes":
            payload = self.catalog.to_api()
            payload["scene_types"] = self.scene_types
            payload["capture"] = {
                "photos_root": str(self.capture_store.photos_root),
                "save_overlay_jpeg": bool(self.save_overlay_jpeg),
            }
            self.send_json(payload)
            return
        if path == "/api/capture/summary":
            query = parse_qs(parsed.query)
            session_id = (query.get("session_id") or [""])[0].strip()
            try:
                if session_id:
                    session_id = self.capture_store.validate_session_id(
                        session_id
                    )
                payload = self.capture_store.summary(
                    session_id or None,
                    recent_limit=8,
                )
            except (CaptureError, OSError, ValueError) as error:
                self.send_json(
                    {"ok": False, "error": str(error)},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            self.send_json(payload)
            return
        if path == "/api/capture/overlay.jpg":
            query = parse_qs(parsed.query)
            session_id = (query.get("session_id") or [""])[0]
            frame_id = (query.get("frame_id") or [""])[0]
            try:
                overlay_path = self.capture_store.overlay_path(
                    session_id,
                    frame_id,
                )
                body = overlay_path.read_bytes()
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            except (CaptureError, OSError, ValueError) as error:
                self.send_json(
                    {"ok": False, "error": str(error)},
                    HTTPStatus.BAD_REQUEST,
                )
                return
            self.send_bytes(body, "image/jpeg")
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
        elif path == "/api/session/next":
            self.next_session()
        elif path == "/api/stop":
            self.stop_service()
        elif path == "/api/scene/freeze":
            self.freeze_scene()
        elif path == "/api/scene/save":
            self.save_pending_scene()
        elif path == "/api/scene/discard":
            self.discard_pending_scene()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def next_session(self) -> None:
        try:
            self.read_json()
            session_id = self.capture_store.next_session_id()
            self.send_json(
                {
                    "ok": True,
                    "sessionId": session_id,
                    "sessionPath": str(
                        self.capture_store.photos_root / session_id
                    ),
                    "photosRoot": str(self.capture_store.photos_root),
                }
            )
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )

    def stop_service(self) -> None:
        try:
            self.read_json()
            self.send_json({"ok": True, "stopping": True})
            threading.Thread(
                target=self.server.shutdown,
                name="track1-capture-stop",
                daemon=True,
            ).start()
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                HTTPStatus.BAD_REQUEST,
            )

    def handle_capture(self) -> None:
        try:
            payload = self.read_json()
            session_id = str(payload.get("session_id") or "").strip()
            if "class_id" not in payload:
                raise ValueError("缺少 class_id")
            class_id = int(payload["class_id"])
            self.catalog.by_id(class_id)
            scene_type = str(payload.get("scene_type") or "").strip()
            with self.state.condition:
                if self.state.raw_frame is None:
                    raise RuntimeError("相机画面尚未就绪，请等待实时预览")
                if self.state.raw_depth_units is None:
                    raise RuntimeError("深度帧尚未就绪，请检查 D405 RGB-D")
                if self.state.depth_intrinsics is None:
                    raise RuntimeError("相机内参尚未就绪")
                if not self.state.camera_serial:
                    raise RuntimeError("D405 相机序列号尚未就绪")
                frame = self.state.raw_frame.copy()
                depth_units = self.state.raw_depth_units.copy()
                depth_scale = float(self.state.depth_scale_m_per_unit)
                intrinsics = self.state.depth_intrinsics
                camera_serial = self.state.camera_serial
                detections = [dict(item) for item in self.state.status.get("detections", [])]
            result = self.capture_store.capture(
                session_id=session_id,
                class_id=class_id,
                frame=frame,
                depth_raw=depth_units,
                depth_scale_m_per_unit=depth_scale,
                intrinsics=intrinsics,
                camera_serial=camera_serial,
                rotate_180=bool(getattr(self.state.depth_args, "rotate", 0) == 180),
                scene_type=scene_type,
                instance_id=str(payload.get("instance_id") or ""),
                lighting_id=str(payload.get("lighting_id") or ""),
                notes=str(payload.get("notes") or ""),
                capture_mode=str(payload.get("capture_mode") or "single"),
                capture_group=str(payload.get("capture_group") or ""),
                model_predictions=detections,
                save_overlay_jpeg=bool(self.save_overlay_jpeg),
            )
            self.send_json(result)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as error:
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
            raw_assignments = payload.get("assignments")
            if not isinstance(raw_assignments, dict):
                raise ValueError("assignments 必须是候选编号到类别编号的映射")
            assignments: dict[int, int] = {}
            for key, value in raw_assignments.items():
                class_index = int(value)
                if not 0 <= class_index < len(self.catalog.entries):
                    raise ValueError("候选标注类别索引必须在 0-39")
                assignments[int(key)] = class_index
            manual_boxes = payload.get("manual_boxes", [])
            if not isinstance(manual_boxes, list):
                raise ValueError("manual_boxes 必须是手动框列表")
            for box in manual_boxes:
                if not isinstance(box, dict):
                    raise ValueError("手动标注框必须是对象")
                class_index = int(box.get("class_id", -1))
                if not 0 <= class_index < len(self.catalog.entries):
                    raise ValueError("手动框类别索引必须在 0-39，不能使用空背景")
            with self.state.condition:
                pending = self.state.pending_scene
                if pending is None:
                    raise RuntimeError("没有待保存的场景，请先冻结画面")
            class_names = [
                entry.canonical_label for entry in self.catalog.entries
            ]
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
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

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


def load_scene_types(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Load scene metadata from JSON or the formal YAML configuration."""
    source = ""
    payload: Any
    if str(getattr(args, "scene_types_json", "") or "").strip():
        source = "--scene-types-json"
        try:
            payload = json.loads(args.scene_types_json)
        except json.JSONDecodeError as error:
            raise ValueError(f"场景类型 JSON 无效：{error}") from error
    else:
        config_path = Path(getattr(args, "config", DEFAULT_CONFIG)).expanduser()
        source = str(config_path)
        if not config_path.is_file():
            raise ValueError(f"采集配置文件不存在：{config_path}")
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"无法读取采集配置：{error}") from error
        if not isinstance(config, dict):
            raise ValueError("采集配置根节点必须是对象")
        capture = config.get("capture")
        if not isinstance(capture, dict):
            raise ValueError("采集配置缺少 capture 对象")
        payload = capture.get("scene_types")

    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"没有有效的场景类型定义：{source}")
    scene_types: dict[str, dict[str, Any]] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key).strip()
        if not key or not isinstance(raw_value, dict):
            raise ValueError(f"场景类型定义无效：{raw_key!r}")
        label = str(raw_value.get("label") or "").strip()
        box_type = str(raw_value.get("box_type") or "").strip()
        if not label or not box_type:
            raise ValueError(f"场景 {key} 缺少 label 或 box_type")
        scene_types[key] = {
            "label": label,
            "box_type": box_type,
            "description": str(raw_value.get("description") or "").strip(),
        }
    required = {
        "tabletop",
        "box_11l",
        "box_11l_empty",
        "box_11l_clutter",
        "box_11l_near_wall",
        "box_11l_low_angle",
    }
    missing = sorted(required - set(scene_types))
    if missing:
        raise ValueError(f"场景类型缺少：{', '.join(missing)}")
    return scene_types


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.weights).is_file():
        raise ValueError(f"权重文件不存在：{args.weights}")
    if args.fallback_weights and not Path(args.fallback_weights).is_file():
        raise ValueError(f"回退权重文件不存在：{args.fallback_weights}")
    if not Path(args.class_map).is_file():
        raise ValueError(f"类别表不存在：{args.class_map}")
    if not args.rgbd:
        raise ValueError("正式采图必须启用 RGB-D，不能使用 --no-rgbd")
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
        catalog = load_class_catalog(args.class_map)
        scene_types = load_scene_types(args)
        capture_store = CaptureStore(
            args.photos_root,
            catalog,
            scene_types=scene_types,
            default_lighting_id=args.default_lighting_id,
        )
    except (ValueError, CaptureError, OSError) as error:
        print(f"采集配置错误：{error}", file=sys.stderr)
        return 2

    state = PreviewState()
    PreviewRequestHandler.state = state
    PreviewRequestHandler.catalog = catalog
    PreviewRequestHandler.capture_store = capture_store
    PreviewRequestHandler.scene_root = Path(args.scene_root).resolve()
    PreviewRequestHandler.scene_types = scene_types
    PreviewRequestHandler.scene_box_padding_px = args.scene_box_padding_px
    PreviewRequestHandler.jpeg_quality = args.jpeg_quality
    PreviewRequestHandler.save_overlay_jpeg = bool(args.save_overlay_jpeg)
    try:
        server = ThreadingHTTPServer(
            (args.host, args.port),
            PreviewRequestHandler,
        )
    except OSError as error:
        print(f"网页端口启动失败：{args.host}:{args.port}：{error}", file=sys.stderr)
        return 2
    url = f"http://{args.host}:{args.port}/"

    try:
        worker = PreviewWorker(args, state)
    except Exception as error:
        server.server_close()
        print(f"视觉模型初始化失败：{error}", file=sys.stderr)
        return 2

    worker.start()
    print(f"Track 1 RGB-D capture web: {url}", file=sys.stderr)
    print(f"class map: {args.class_map}", file=sys.stderr)
    print(f"photos root: {capture_store.photos_root}", file=sys.stderr)
    print("robot commands: disabled", file=sys.stderr)
    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        server.server_close()
        worker.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
