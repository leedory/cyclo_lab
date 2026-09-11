#!/usr/bin/env python3
"""Serve a read-only Task000458 HDF5 contract and episode viewer."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from PIL import Image


MAX_ATTRIBUTE_ITEMS = 200
MAX_PREVIEW_ELEMENTS = 10_000
MAX_IMAGE_PIXELS = 16_000_000
DEFAULT_CHILD_LIMIT = 500


DATASET_SEMANTICS = {
    "actions": {
        "title": "Current raw action command",
        "category": "Command",
        "meaning": "Absolute 19D joint-position command A[t] that is applied during the current environment step.",
        "timing": "pre-step, current row",
        "axes": "[time, action joint]",
        "order": "action",
    },
    "processed_actions": {
        "title": "Processed current action",
        "category": "Command",
        "meaning": "Action-term output after scaling and offset processing for A[t], recorded after the current step.",
        "timing": "post-step, current row",
        "axes": "[time, action joint]",
        "order": "action",
    },
    "obs/actions": {
        "title": "Previous action in the observation",
        "category": "Observation context",
        "meaning": "The last_action observation. For rows after the first, obs/actions[t] equals actions[t-1].",
        "timing": "pre-step observation, previous command",
        "axes": "[time, action joint]",
        "order": "action",
    },
    "obs/joint_pos": {
        "title": "Realized simulator joint position",
        "category": "Robot observation",
        "meaning": "Actual SG2 articulation joint positions read from the simulator before A[t] is applied.",
        "timing": "pre-step, current robot state",
        "axes": "[time, published joint]",
        "order": "observation",
    },
    "obs/joint_pos_target": {
        "title": "Previous simulator joint target",
        "category": "Robot observation",
        "meaning": "Joint target buffer before A[t]. For rows after the first, it equals actions[t-1] after name-based reordering.",
        "timing": "pre-step, previous target",
        "axes": "[time, published joint]",
        "order": "observation",
    },
    "obs/left_eef_pose": {
        "title": "Left end-effector pose",
        "category": "Robot observation",
        "meaning": "Realized left end-effector pose relative to the robot root frame.",
        "timing": "pre-step, current robot state",
        "axes": "[time, x, y, z, qw, qx, qy, qz]",
        "order": "eef_pose",
    },
    "obs/right_eef_pose": {
        "title": "Right end-effector pose",
        "category": "Robot observation",
        "meaning": "Realized right end-effector pose relative to the robot root frame.",
        "timing": "pre-step, current robot state",
        "axes": "[time, x, y, z, qw, qx, qy, qz]",
        "order": "eef_pose",
    },
    "obs/step_index": {
        "title": "Episode step index",
        "category": "Time",
        "meaning": "Zero-based control-step index recorded for this episode.",
        "timing": "pre-step",
        "axes": "[time, 1]",
        "order": None,
    },
    "obs/timestamp_s": {
        "title": "Episode timestamp",
        "category": "Time",
        "meaning": "Seconds since recording start, computed as step_index / control_hz.",
        "timing": "pre-step",
        "axes": "[time, 1], seconds",
        "order": None,
    },
}


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HDF5 구조 뷰어</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #111a2e;
      --panel-2: #16213a;
      --line: #283652;
      --text: #e9eefb;
      --muted: #93a3bf;
      --accent: #66d9c2;
      --accent-2: #8fb8ff;
      --danger: #ff9b9b;
      --shadow: 0 18px 50px rgb(0 0 0 / 28%);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 10% 0%, rgb(53 88 150 / 20%), transparent 34rem),
        var(--bg);
      color: var(--text);
      font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      gap: 16px;
      min-height: 72px;
      padding: 14px 24px;
      border-bottom: 1px solid var(--line);
      background: rgb(11 16 32 / 86%);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand {
      font-size: 18px;
      font-weight: 760;
      letter-spacing: -.02em;
      white-space: nowrap;
    }
    .brand span { color: var(--accent); }
    .file-pill {
      min-width: 0;
      padding: 7px 11px;
      color: var(--muted);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 9px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .read-only {
      margin-left: auto;
      padding: 5px 9px;
      color: #9df0df;
      border: 1px solid rgb(102 217 194 / 35%);
      background: rgb(102 217 194 / 9%);
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      min-height: calc(100vh - 73px);
    }
    aside {
      border-right: 1px solid var(--line);
      background: rgb(17 26 46 / 72%);
      overflow: auto;
      height: calc(100vh - 73px);
      position: sticky;
      top: 73px;
    }
    .side-head {
      padding: 18px 18px 12px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    #tree { padding: 0 10px 24px; }
    .tree-row {
      display: flex;
      align-items: center;
      min-width: 0;
      min-height: 34px;
      padding: 3px 7px;
      border-radius: 7px;
      cursor: pointer;
      user-select: none;
    }
    .tree-row:hover { background: rgb(143 184 255 / 8%); }
    .tree-row.selected { background: rgb(102 217 194 / 13%); color: #baf8ec; }
    .toggle {
      width: 24px;
      height: 24px;
      flex: 0 0 24px;
      padding: 0;
      color: var(--muted);
      background: transparent;
      border: 0;
      cursor: pointer;
      font-size: 11px;
    }
    .tree-icon { width: 22px; flex: 0 0 22px; }
    .tree-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tree-children { margin-left: 17px; border-left: 1px solid var(--line); padding-left: 2px; }
    main { min-width: 0; padding: 28px clamp(20px, 4vw, 54px) 56px; }
    .path {
      color: var(--accent-2);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    h1 { margin: 8px 0 4px; font-size: clamp(25px, 3vw, 38px); line-height: 1.15; letter-spacing: -.035em; }
    .subtitle { color: var(--muted); margin-bottom: 22px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin: 18px 0 24px;
    }
    .card { padding: 14px; background: var(--panel); border: 1px solid var(--line); border-radius: 11px; box-shadow: var(--shadow); }
    .card-label { color: var(--muted); font-size: 11px; font-weight: 750; letter-spacing: .07em; text-transform: uppercase; }
    .card-value { margin-top: 5px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
    section { margin-top: 28px; }
    h2 { margin: 0 0 12px; font-size: 17px; letter-spacing: -.01em; }
    .panel { padding: 16px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px 11px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { width: 190px; color: var(--muted); font-size: 12px; }
    tr:last-child th, tr:last-child td { border-bottom: 0; }
    code, pre, input { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    pre {
      margin: 0;
      max-height: 520px;
      overflow: auto;
      padding: 15px;
      color: #d9e5ff;
      background: #090e1a;
      border: 1px solid #202c45;
      border-radius: 9px;
      white-space: pre-wrap;
      word-break: break-word;
      tab-size: 2;
    }
    .controls { display: flex; align-items: end; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    input {
      min-width: min(430px, 72vw);
      padding: 9px 11px;
      color: var(--text);
      background: #0b1222;
      border: 1px solid var(--line);
      border-radius: 8px;
      outline: none;
    }
    input:focus { border-color: var(--accent-2); box-shadow: 0 0 0 3px rgb(143 184 255 / 10%); }
    input[type=number] { min-width: 100px; width: 120px; }
    button.action {
      padding: 9px 13px;
      color: #081713;
      background: var(--accent);
      border: 0;
      border-radius: 8px;
      font-weight: 760;
      cursor: pointer;
    }
    button.action:hover { filter: brightness(1.06); }
    .hint { margin: 8px 0 0; color: var(--muted); font-size: 12px; }
    .stats { display: flex; flex-wrap: wrap; gap: 7px; margin: 0 0 10px; }
    .stat { padding: 4px 8px; color: var(--muted); background: var(--panel-2); border-radius: 6px; font-size: 12px; }
    .image-wrap { display: grid; place-items: center; min-height: 160px; padding: 12px; background: #070b14; border: 1px solid var(--line); border-radius: 9px; overflow: auto; }
    .image-wrap img { max-width: 100%; max-height: 68vh; image-rendering: auto; }
    .callout {
      padding: 14px 16px;
      margin: 12px 0;
      color: #d8e6ff;
      background: rgb(143 184 255 / 9%);
      border: 1px solid rgb(143 184 255 / 26%);
      border-left: 4px solid var(--accent-2);
      border-radius: 9px;
    }
    .callout.good {
      color: #c5f8ee;
      background: rgb(102 217 194 / 8%);
      border-color: rgb(102 217 194 / 25%);
      border-left-color: var(--accent);
    }
    .flow {
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 9px;
      align-items: stretch;
    }
    .flow-step {
      position: relative;
      padding: 14px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 10px;
    }
    .flow-step:not(:last-child)::after {
      content: "→";
      position: absolute;
      right: -10px;
      top: calc(50% - 12px);
      z-index: 2;
      color: var(--accent);
      font-size: 18px;
      font-weight: 800;
    }
    .flow-title { font-weight: 760; color: var(--text); }
    .flow-copy { margin-top: 6px; color: var(--muted); font-size: 12px; }
    .category-grid, .semantic-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 10px;
    }
    .category, .semantic-item {
      padding: 14px;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 10px;
    }
    .category h3 { margin: 0 0 6px; font-size: 14px; }
    .category p { margin: 0 0 10px; color: var(--muted); font-size: 12px; }
    .path-list { display: grid; gap: 5px; }
    .path-chip {
      width: fit-content;
      max-width: 100%;
      padding: 3px 7px;
      color: #c7d8ff;
      background: #0b1222;
      border-radius: 5px;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11px;
    }
    .semantic-label { color: var(--muted); font-size: 11px; font-weight: 750; text-transform: uppercase; letter-spacing: .06em; }
    .semantic-value { margin-top: 5px; }
    .table-scroll { overflow: auto; border: 1px solid var(--line); border-radius: 9px; }
    .table-scroll table { min-width: 760px; }
    .table-scroll th { width: auto; white-space: nowrap; background: #111a2e; position: sticky; top: 0; z-index: 1; }
    .table-scroll td { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space: nowrap; }
    tr.mismatch { background: rgb(255 190 105 / 7%); }
    .badge {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      color: var(--muted);
      background: var(--panel-2);
      border: 1px solid var(--line);
      font-size: 11px;
      font-weight: 700;
    }
    .badge.warn { color: #ffd29a; border-color: rgb(255 190 105 / 35%); }
    .episode-link {
      padding: 3px 7px;
      color: var(--accent-2);
      background: transparent;
      border: 0;
      cursor: pointer;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-weight: 700;
    }
    .episode-link:hover { text-decoration: underline; }
    .frame-summary { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 14px; }
    .camera-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; margin-top: 14px; }
    .camera-card { padding: 10px; background: #070b14; border: 1px solid var(--line); border-radius: 9px; }
    .camera-card h3 { margin: 0 0 8px; font-size: 13px; }
    .camera-card img { display: block; width: 100%; height: auto; border-radius: 6px; }
    .positive { color: #9df0df; }
    .negative { color: #ffb6b6; }
    .section-copy { margin: -5px 0 13px; color: var(--muted); }
    .empty { padding: 22px; color: var(--muted); text-align: center; }
    #status {
      position: fixed;
      right: 18px;
      bottom: 18px;
      max-width: min(480px, calc(100vw - 36px));
      padding: 10px 13px;
      color: var(--text);
      background: #18233c;
      border: 1px solid var(--line);
      border-radius: 9px;
      box-shadow: var(--shadow);
      opacity: 0;
      pointer-events: none;
      transform: translateY(8px);
      transition: .18s ease;
    }
    #status.show { opacity: 1; transform: translateY(0); }
    #status.error { color: #ffd0d0; border-color: rgb(255 155 155 / 45%); }
    @media (max-width: 800px) {
      .layout { grid-template-columns: 1fr; }
      aside { position: relative; top: 0; height: auto; max-height: 42vh; border-right: 0; border-bottom: 1px solid var(--line); }
      main { padding-top: 22px; }
      .read-only { display: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">HDF5 <span>구조 뷰어</span></div>
    <div id="file-pill" class="file-pill">파일을 불러오는 중…</div>
    <div class="read-only">읽기 전용</div>
  </header>
  <div class="layout">
    <aside>
      <div class="side-head">HDF5 계층</div>
      <div id="tree"></div>
    </aside>
    <main id="detail">
      <div class="empty">HDF5 메타데이터를 불러오는 중…</div>
    </main>
  </div>
  <div id="status"></div>
  <script>
    const state = { selectedPath: null };

    function showStatus(message, isError = false) {
      const status = document.getElementById('status');
      status.textContent = message;
      status.className = `show${isError ? ' error' : ''}`;
      clearTimeout(showStatus.timer);
      showStatus.timer = setTimeout(() => { status.className = ''; }, 4200);
    }

    async function api(url) {
      const response = await fetch(url);
      if (!response.ok) {
        let message = `${response.status} ${response.statusText}`;
        try { message = (await response.json()).detail || message; } catch (_) {}
        throw new Error(message);
      }
      return response.json();
    }

    function el(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function formatBytes(bytes) {
      if (!Number.isFinite(bytes)) return String(bytes);
      const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
      let value = bytes;
      let index = 0;
      while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
      return `${value.toFixed(index ? 2 : 0)} ${units[index]}`;
    }

    function displayValue(value) {
      if (typeof value === 'string') return value;
      return JSON.stringify(value, null, 2);
    }

    function clearSelected() {
      document.querySelectorAll('.tree-row.selected').forEach(node => node.classList.remove('selected'));
    }

    function makeTreeItem(item, isRoot = false) {
      const wrapper = el('div', 'tree-item');
      const row = el('div', 'tree-row');
      row.dataset.path = item.path;
      const toggle = el('button', 'toggle', item.kind === 'group' ? '▶' : '');
      toggle.setAttribute('aria-label', item.kind === 'group' ? 'Expand group' : 'Dataset');
      const icon = el('span', 'tree-icon', item.kind === 'group' ? '▰' : '▤');
      const name = el('span', 'tree-name', isRoot ? '/' : item.name);
      name.title = item.path;
      row.append(toggle, icon, name);
      wrapper.append(row);

      row.addEventListener('click', async event => {
        if (event.target === toggle && item.kind === 'group') {
          await toggleGroup(wrapper, item.path, toggle);
          return;
        }
        clearSelected();
        row.classList.add('selected');
        await selectNode(item.path);
      });
      if (item.kind === 'group') {
        toggle.addEventListener('click', event => event.stopPropagation());
        toggle.addEventListener('click', () => toggleGroup(wrapper, item.path, toggle));
      }
      return wrapper;
    }

    async function toggleGroup(wrapper, path, toggle) {
      const existing = wrapper.querySelector(':scope > .tree-children');
      if (existing) {
        const hidden = existing.hidden = !existing.hidden;
        toggle.textContent = hidden ? '▶' : '▼';
        return;
      }
      toggle.textContent = '…';
      try {
        const result = await api(`/api/children?path=${encodeURIComponent(path)}`);
        const children = el('div', 'tree-children');
        for (const item of result.children) children.append(makeTreeItem(item));
        if (result.truncated) children.append(el('div', 'empty', `Showing first ${result.children.length} children`));
        wrapper.append(children);
        toggle.textContent = '▼';
      } catch (error) {
        toggle.textContent = '▶';
        showStatus(error.message, true);
      }
    }

    function addCard(container, label, value) {
      const card = el('div', 'card');
      card.append(el('div', 'card-label', label), el('div', 'card-value', String(value)));
      container.append(card);
    }

    function attributesSection(attributes) {
      const section = el('section');
      section.append(el('h2', '', `Attributes (${Object.keys(attributes).length})`));
      const panel = el('div', 'panel');
      if (!Object.keys(attributes).length) {
        panel.append(el('div', 'empty', 'No attributes on this object.'));
      } else {
        const table = document.createElement('table');
        for (const [key, value] of Object.entries(attributes)) {
          const row = document.createElement('tr');
          const heading = document.createElement('th');
          const cell = document.createElement('td');
          const code = document.createElement('code');
          heading.textContent = key;
          code.textContent = displayValue(value);
          cell.append(code);
          row.append(heading, cell);
          table.append(row);
        }
        panel.append(table);
      }
      section.append(panel);
      return section;
    }

    function datasetSection(node) {
      const section = el('section');
      section.append(el('h2', '', 'Data preview'));
      const panel = el('div', 'panel');
      const controls = el('div', 'controls');
      const label = document.createElement('label');
      label.append(document.createTextNode('Slice expression'));
      const input = document.createElement('input');
      input.id = 'slice-input';
      input.value = node.suggested_slice;
      input.placeholder = 'Example: 0:10, 0:7';
      label.append(input);
      const button = el('button', 'action', 'Load values');
      button.type = 'button';
      controls.append(label, button);
      const stats = el('div', 'stats');
      const output = document.createElement('pre');
      output.textContent = 'Loading preview…';
      panel.append(controls, stats, output);
      panel.append(el('p', 'hint', `At most ${node.preview_limit.toLocaleString()} selected elements are read. Use comma-separated Python-style indices and slices.`));
      section.append(panel);

      async function loadValues() {
        button.disabled = true;
        output.textContent = 'Loading preview…';
        stats.replaceChildren();
        try {
          const result = await api(`/api/values?path=${encodeURIComponent(node.path)}&slice=${encodeURIComponent(input.value)}`);
          stats.append(el('span', 'stat', `selection ${result.selection}`));
          stats.append(el('span', 'stat', `shape ${JSON.stringify(result.selected_shape)}`));
          stats.append(el('span', 'stat', `${result.element_count.toLocaleString()} elements`));
          for (const [key, value] of Object.entries(result.statistics || {})) {
            stats.append(el('span', 'stat', `${key} ${value}`));
          }
          output.textContent = JSON.stringify(result.data, null, 2);
        } catch (error) {
          output.textContent = error.message;
          showStatus(error.message, true);
        } finally {
          button.disabled = false;
        }
      }
      button.addEventListener('click', loadValues);
      input.addEventListener('keydown', event => { if (event.key === 'Enter') loadValues(); });
      queueMicrotask(loadValues);
      return section;
    }

    function imageSection(node) {
      const section = el('section');
      section.append(el('h2', '', 'Image preview'));
      const panel = el('div', 'panel');
      const controls = el('div', 'controls');
      const label = document.createElement('label');
      label.append(document.createTextNode('Frame index'));
      const input = document.createElement('input');
      input.type = 'number';
      input.min = '0';
      input.max = String(Math.max(0, node.image_frame_count - 1));
      input.value = '0';
      input.disabled = node.image_frame_count <= 1;
      label.append(input);
      const button = el('button', 'action', 'Show image');
      const wrap = el('div', 'image-wrap');
      const image = document.createElement('img');
      image.alt = `Preview of ${node.path}`;
      wrap.append(image);
      controls.append(label, button);
      panel.append(controls, wrap);
      panel.append(el('p', 'hint', 'Floating-point images are normalized to 0–255 for display only. The HDF5 data is never modified.'));
      section.append(panel);

      function loadImage() {
        const frame = Math.max(0, Number.parseInt(input.value || '0', 10));
        image.src = `/api/image?path=${encodeURIComponent(node.path)}&index=${frame}&t=${Date.now()}`;
      }
      image.addEventListener('error', () => showStatus('Could not render this dataset as an image.', true));
      button.addEventListener('click', loadImage);
      queueMicrotask(loadImage);
      return section;
    }

    function formatNumber(value, digits = 6) {
      if (value === null || value === undefined) return '—';
      if (!Number.isFinite(value)) return String(value);
      if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 1e-4)) {
        return value.toExponential(4);
      }
      return value.toFixed(digits).replace(/\.?0+$/, '');
    }

    function structuredTable(headers, rows, rowClass) {
      const wrap = el('div', 'table-scroll');
      const table = document.createElement('table');
      const head = document.createElement('thead');
      const headerRow = document.createElement('tr');
      for (const header of headers) headerRow.append(el('th', '', header.label));
      head.append(headerRow);
      const body = document.createElement('tbody');
      rows.forEach((item, index) => {
        const row = document.createElement('tr');
        if (rowClass) row.className = rowClass(item, index) || '';
        for (const header of headers) {
          const cell = document.createElement('td');
          const value = typeof header.value === 'function' ? header.value(item, index) : item[header.value];
          if (value instanceof Node) cell.append(value);
          else cell.textContent = value === null || value === undefined ? '—' : String(value);
          row.append(cell);
        }
        body.append(row);
      });
      table.append(head, body);
      wrap.append(table);
      return wrap;
    }

    function semanticSection(node) {
      if (!node.semantic) return null;
      const section = el('section');
      section.append(el('h2', '', '이 데이터의 의미'));
      const panel = el('div', 'panel');
      const grid = el('div', 'semantic-grid');
      const entries = [
        ['분류', node.semantic.category],
        ['설명', node.semantic.meaning],
        ['기록 시점', node.semantic.timing],
        ['축 구조', node.semantic.axes],
      ];
      for (const [label, value] of entries) {
        const item = el('div', 'semantic-item');
        item.append(el('div', 'semantic-label', label), el('div', 'semantic-value', value));
        grid.append(item);
      }
      panel.append(el('h3', '', node.semantic.title), grid);
      section.append(panel);
      return section;
    }

    function vectorSchemaSection(node) {
      if (!node.vector_schema?.length) return null;
      const section = el('section');
      section.append(el('h2', '', '마지막 축의 요소 (' + node.vector_schema.length + ')'));
      section.append(el('p', 'section-copy', 'shape의 마지막 축을 인덱스, 이름, 단위로 해석한 표입니다.'));
      section.append(structuredTable(
        [
          { label: '인덱스', value: 'index' },
          { label: '요소 이름', value: 'name' },
          { label: '단위', value: 'unit' },
        ],
        node.vector_schema,
      ));
      return section;
    }

    function overviewSection(info) {
      const container = el('div', 'structured-overview');

      const contract = el('section');
      contract.append(el('h2', '', '데이터셋 계약'));
      const cards = el('div', 'cards');
      addCard(cards, 'Schema', info.contract.schema_version);
      addCard(cards, 'Robot contract', info.contract.robot_contract_id);
      addCard(cards, 'Episodes', info.episode_count);
      addCard(cards, 'Frames', info.total_frames.toLocaleString());
      addCard(cards, 'Control rate', info.rates.control_hz + ' Hz');
      addCard(cards, 'Camera rate', info.rates.camera_hz + ' Hz');
      const contractPanel = el('div', 'panel');
      contractPanel.append(cards);
      const task = el('div', 'callout good');
      task.append(el('div', 'flow-title', 'Task ' + info.task.task_id + ': ' + info.task.target_object));
      task.append(el('div', 'flow-copy', info.task.instruction));
      task.append(el('div', 'flow-copy', '환경 ' + info.task.environment + ' · ' + info.task.target_side + ' gripper · success ' + info.task.success_criterion));
      contractPanel.append(task);
      contract.append(contractPanel);
      container.append(contract);

      const timeline = el('section');
      timeline.append(el('h2', '', '한 행 t의 시간 구조'));
      timeline.append(el('p', 'section-copy', '같은 HDF5 행에 저장되지만 모든 값이 같은 시점의 같은 종류는 아닙니다.'));
      const flow = el('div', 'flow');
      const flowItems = [
        ['obs/*[t]', 'A[t] 적용 전 상태. joint_pos는 현재 시뮬레이터 관절 위치이고 obs/actions는 이전 명령입니다.'],
        ['actions[t] = A[t]', '현재 physics step에 적용할 raw absolute joint command입니다.'],
        ['physics update', 'A[t]를 적용하고 simulator를 decimation 횟수만큼 진행합니다.'],
        ['states/*[t]', 'A[t] 적용 뒤의 full scene state이며 replay·진단용 privileged data입니다.'],
      ];
      for (const [title, copy] of flowItems) {
        const step = el('div', 'flow-step');
        step.append(el('div', 'flow-title', title), el('div', 'flow-copy', copy));
        flow.append(step);
      }
      const timelinePanel = el('div', 'panel');
      timelinePanel.append(flow);
      timelinePanel.append(el('div', 'callout', '정렬 규칙: obs/actions[t+1] = actions[t]. obs/joint_pos_target[t+1]도 action을 관절명 기준으로 재정렬하면 actions[t]와 같습니다. obs/joint_pos는 복사값이 아니라 physics가 만든 실제 관절 위치입니다.'));
      timeline.append(timelinePanel);
      container.append(timeline);

      const structure = el('section');
      structure.append(el('h2', '', '데이터 역할별 구조'));
      const categoryGrid = el('div', 'category-grid');
      for (const category of info.categories) {
        const card = el('div', 'category');
        card.append(el('h3', '', category.title), el('p', '', category.role));
        const paths = el('div', 'path-list');
        for (const item of category.paths) paths.append(el('span', 'path-chip', item));
        card.append(paths);
        categoryGrid.append(card);
      }
      const structurePanel = el('div', 'panel');
      structurePanel.append(categoryGrid);
      structure.append(structurePanel);
      container.append(structure);

      const order = el('section');
      order.append(el('h2', '', '19D 관절 순서 비교'));
      order.append(el('p', 'section-copy', 'index 16–18에서 action 순서와 observation 순서가 다릅니다. shape만 보고 연결하지 말고 이름으로 재정렬해야 합니다.'));
      order.append(structuredTable(
        [
          { label: '인덱스', value: 'index' },
          { label: 'actions', value: item => item.action_name + ' (' + item.action_unit + ')' },
          { label: 'obs/joint_pos', value: item => item.observation_name + ' (' + item.observation_unit + ')' },
          { label: '상태', value: item => {
            return el('span', 'badge' + (item.same ? '' : ' warn'), item.same ? '동일' : '순서 다름');
          }},
        ],
        info.joint_order,
        item => item.same ? '' : 'mismatch',
      ));
      container.append(order);

      const episodes = el('section');
      episodes.append(el('h2', '', '에피소드 (' + info.episode_count + ')'));
      episodes.append(el('p', 'section-copy', '에피소드를 선택하면 같은 프레임의 action, 이전 action, target, 실제 joint 위치와 카메라를 함께 볼 수 있습니다.'));
      episodes.append(structuredTable(
        [
          { label: '에피소드', value: item => {
            const button = el('button', 'episode-link', item.name);
            button.addEventListener('click', () => selectNode(item.path));
            return button;
          }},
          { label: '프레임', value: item => item.frames.toLocaleString() },
          { label: '시간', value: item => formatNumber(item.duration_s, 3) + ' s' },
          { label: '성공', value: item => item.success ? 'true' : 'false' },
          { label: 'Schema', value: 'schema_version' },
          { label: 'Target', value: 'target_object_name' },
        ],
        info.episodes,
      ));
      container.append(episodes);
      return container;
    }

    function episodeInspector(path) {
      const section = el('section');
      section.append(el('h2', '', '에피소드 프레임 검사기'));
      section.append(el('p', 'section-copy', '한 프레임만 읽어 모든 관절을 이름 기준으로 정렬합니다. 카메라도 같은 frame index를 표시합니다.'));
      const panel = el('div', 'panel');
      const controls = el('div', 'controls');
      const label = document.createElement('label');
      label.append(document.createTextNode('Frame index'));
      const input = document.createElement('input');
      input.type = 'number';
      input.min = '0';
      input.value = '0';
      label.append(input);
      const button = el('button', 'action', '프레임 불러오기');
      controls.append(label, button);
      const content = el('div');
      content.append(el('div', 'empty', '프레임을 불러오는 중…'));
      panel.append(controls, content);
      section.append(panel);

      async function loadFrame() {
        button.disabled = true;
        const frame = Math.max(0, Number.parseInt(input.value || '0', 10));
        try {
          const result = await api('/api/episode?path=' + encodeURIComponent(path) + '&index=' + frame);
          input.max = String(result.frame_count - 1);
          input.value = String(result.frame);
          content.replaceChildren();
          const summary = el('div', 'frame-summary');
          summary.append(el('span', 'stat', 'frame ' + result.frame + ' / ' + (result.frame_count - 1)));
          summary.append(el('span', 'stat', 'step ' + (result.step_index ?? '—')));
          summary.append(el('span', 'stat', 'time ' + formatNumber(result.timestamp_s, 6) + ' s'));
          content.append(summary);

          const alignment = result.alignment;
          const message = el('div', alignment.has_previous_recorded_action ? 'callout good' : 'callout');
          if (alignment.has_previous_recorded_action) {
            message.textContent = '이전 명령 정렬 검증: obs/actions 최대 오차 ' + formatNumber(alignment.obs_actions_vs_previous_max_abs) + ' · joint target 최대 오차 ' + formatNumber(alignment.target_vs_previous_max_abs) + '.';
          } else {
            message.textContent = 'frame 0의 obs/actions와 target은 녹화를 시작하기 전 마지막 simulator 명령을 나타냅니다.';
          }
          content.append(message);

          content.append(structuredTable(
            [
              { label: '관절', value: 'joint' },
              { label: '단위', value: 'unit' },
              { label: 'A index', value: 'action_index' },
              { label: 'Obs index', value: 'observation_index' },
              { label: '현재 action A[t]', value: item => formatNumber(item.action_t) },
              { label: 'obs 이전 action', value: item => formatNumber(item.previous_action_obs) },
              { label: 'joint target', value: item => formatNumber(item.joint_target) },
              { label: '실제 joint pos', value: item => formatNumber(item.joint_position) },
              { label: 'pos-target', value: item => {
                return el('span', Math.abs(item.tracking_error) < 1e-3 ? 'positive' : 'negative', formatNumber(item.tracking_error));
              }},
            ],
            result.joints,
            item => item.action_index === item.observation_index ? '' : 'mismatch',
          ));

          if (result.eef.left || result.eef.right) {
            const eefRows = result.eef.components.map((name, index) => ({
              component: name,
              left: result.eef.left?.[index],
              right: result.eef.right?.[index],
            }));
            const eefHeading = el('h3', '', 'End-effector pose');
            eefHeading.style.marginTop = '18px';
            content.append(eefHeading);
            content.append(structuredTable(
              [
                { label: '요소', value: 'component' },
                { label: 'Left EEF', value: item => formatNumber(item.left) },
                { label: 'Right EEF', value: item => formatNumber(item.right) },
              ],
              eefRows,
            ));
          }

          if (result.cameras.length) {
            const cameraHeading = el('h3', '', '같은 프레임의 카메라');
            cameraHeading.style.marginTop = '18px';
            const grid = el('div', 'camera-grid');
            for (const camera of result.cameras) {
              const card = el('div', 'camera-card');
              card.append(el('h3', '', camera.name + ' · ' + camera.shape.join('×')));
              const image = document.createElement('img');
              image.alt = camera.name + ' frame ' + result.frame;
              image.loading = 'lazy';
              image.src = '/api/image?path=' + encodeURIComponent(camera.path) + '&index=' + result.frame + '&t=' + Date.now();
              card.append(image);
              grid.append(card);
            }
            content.append(cameraHeading, grid);
          }
        } catch (error) {
          content.replaceChildren(el('div', 'empty', error.message));
          showStatus(error.message, true);
        } finally {
          button.disabled = false;
        }
      }

      button.addEventListener('click', loadFrame);
      input.addEventListener('keydown', event => { if (event.key === 'Enter') loadFrame(); });
      queueMicrotask(loadFrame);
      return section;
    }

    async function selectNode(path) {
      state.selectedPath = path;
      const detail = document.getElementById('detail');
      detail.replaceChildren(el('div', 'empty', '객체 메타데이터를 불러오는 중…'));
      try {
        const node = await api('/api/node?path=' + encodeURIComponent(path));
        if (state.selectedPath !== path) return;
        detail.replaceChildren();
        detail.append(el('div', 'path', node.path));
        detail.append(el('h1', '', node.path === '/' ? '파일 루트' : node.name));
        detail.append(el('div', 'subtitle', node.kind === 'group' ? 'HDF5 그룹' : 'HDF5 데이터셋'));
        const cards = el('div', 'cards');
        addCard(cards, 'Type', node.kind);
        if (node.kind === 'group') {
          addCard(cards, 'Children', node.child_count);
        } else {
          addCard(cards, 'Shape', JSON.stringify(node.shape));
          addCard(cards, 'Data type', node.dtype);
          addCard(cards, 'Elements', node.size.toLocaleString());
          addCard(cards, 'Storage', node.storage_size === null ? 'unknown' : formatBytes(node.storage_size));
          addCard(cards, 'Compression', node.compression || 'none');
          addCard(cards, 'Chunks', node.chunks ? JSON.stringify(node.chunks) : 'contiguous');
        }
        detail.append(cards);

        if (path === '/' || path === '/data') {
          const overview = await api('/api/overview');
          if (state.selectedPath !== path) return;
          detail.append(overviewSection(overview));
        }
        if (/^\/data\/demo_\d+$/.test(path)) {
          detail.append(episodeInspector(path));
        }
        if (node.semantic) {
          detail.append(semanticSection(node));
        }
        if (node.vector_schema?.length) {
          detail.append(vectorSchemaSection(node));
        }

        detail.append(attributesSection(node.attributes));
        if (node.kind === 'dataset') {
          detail.append(datasetSection(node));
          if (node.image_capable && (node.ndim >= 3 || node.semantic?.category === 'Vision')) {
            detail.append(imageSection(node));
          }
        }
      } catch (error) {
        detail.replaceChildren(el('div', 'empty', error.message));
        showStatus(error.message, true);
      }
    }

    async function initialize() {
      try {
        const info = await api('/api/file');
        const pill = document.getElementById('file-pill');
        pill.textContent = `${info.name} · ${formatBytes(info.size_bytes)}`;
        pill.title = info.path;
        const root = makeTreeItem({ path: '/', name: '/', kind: 'group' }, true);
        document.getElementById('tree').append(root);
        root.querySelector('.tree-row').classList.add('selected');
        await Promise.all([
          toggleGroup(root, '/', root.querySelector('.toggle')),
          selectNode('/'),
        ]);
      } catch (error) {
        document.getElementById('detail').replaceChildren(el('div', 'empty', error.message));
        showStatus(error.message, true);
      }
    }

    initialize();
  </script>
</body>
</html>
"""


def natural_sort_key(value: str) -> list[tuple[int, Any]]:
    """Sort names like demo_2 before demo_10."""
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
    ]


def normalize_path(path: str) -> str:
    """Normalize an HDF5 object path without interpreting it as a filesystem path."""
    if not path:
        return "/"
    return path if path.startswith("/") else f"/{path}"


def json_value(value: Any, *, max_items: int = MAX_ATTRIBUTE_ITEMS) -> Any:
    """Convert h5py/numpy values into bounded JSON-safe values."""
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return json_value(value.item(), max_items=max_items)
        if value.size > max_items:
            flattened = value.reshape(-1)
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "preview": [json_value(item, max_items=max_items) for item in flattened[:max_items]],
                "truncated": True,
            }
        return json_value(value.tolist(), max_items=max_items)
    if isinstance(value, np.void):
        if value.dtype.names:
            return {name: json_value(value[name], max_items=max_items) for name in value.dtype.names}
        return str(value)
    if isinstance(value, np.generic):
        return json_value(value.item(), max_items=max_items)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return {"bytes_hex": value.hex(), "length": len(value)}
    if isinstance(value, h5py.Reference):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): json_value(item, max_items=max_items) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > max_items:
            return {
                "preview": [json_value(item, max_items=max_items) for item in value[:max_items]],
                "length": len(value),
                "truncated": True,
            }
        return [json_value(item, max_items=max_items) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def attributes_for(obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    """Return JSON-safe HDF5 attributes."""
    return {str(key): json_value(obj.attrs[key]) for key in obj.attrs.keys()}


def suggested_slice(shape: tuple[int, ...]) -> str:
    """Choose a small, useful slice for the first value preview."""
    if not shape:
        return ""
    if len(shape) == 1:
        return f"0:{min(shape[0], 100)}"
    if len(shape) == 2:
        return f"0:{min(shape[0], 20)}, 0:{min(shape[1], 10)}"
    if shape[-1] <= 8:
        leading = ["0"] * max(0, len(shape) - 3)
        tail = [
            f"0:{min(shape[-3], 5)}",
            f"0:{min(shape[-2], 5)}",
            f"0:{shape[-1]}",
        ]
        return ", ".join(leading + tail)
    leading = ["0"] * (len(shape) - 2)
    return ", ".join(leading + [f"0:{min(shape[-2], 10)}", f"0:{min(shape[-1], 10)}"])


def parse_slice(expression: str, shape: tuple[int, ...]) -> tuple[tuple[Any, ...], str, int]:
    """Parse a bounded subset of numpy's integer/slice syntax."""
    if not shape:
        if expression.strip():
            raise ValueError("A scalar dataset does not accept a slice expression.")
        return (), "scalar", 1

    parts = [part.strip() for part in expression.split(",")] if expression.strip() else []
    if len(parts) > len(shape):
        raise ValueError(f"Expected at most {len(shape)} dimensions, got {len(parts)}.")
    parts.extend([":"] * (len(shape) - len(parts)))

    selection: list[int | slice] = []
    normalized_labels: list[str] = []
    element_count = 1
    for dimension, (part, length) in enumerate(zip(parts, shape, strict=True)):
        if ":" not in part:
            try:
                index = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid index {part!r} in dimension {dimension}.") from exc
            if index < 0:
                index += length
            if index < 0 or index >= length:
                raise ValueError(f"Index {part} is outside dimension {dimension} with length {length}.")
            selection.append(index)
            normalized_labels.append(str(index))
            continue

        fields = part.split(":")
        if len(fields) > 3:
            raise ValueError(f"Invalid slice {part!r} in dimension {dimension}.")
        try:
            values = [int(field) if field else None for field in fields]
        except ValueError as exc:
            raise ValueError(f"Invalid slice {part!r} in dimension {dimension}.") from exc
        values.extend([None] * (3 - len(values)))
        candidate = slice(values[0], values[1], values[2])
        start, stop, step = candidate.indices(length)
        if step <= 0:
            raise ValueError("Only positive slice steps are supported.")
        count = len(range(start, stop, step))
        element_count *= count
        selection.append(candidate)
        normalized_labels.append(f"{start}:{stop}:{step}")

    if element_count > MAX_PREVIEW_ELEMENTS:
        raise ValueError(
            f"The selection contains {element_count:,} elements; narrow it to "
            f"{MAX_PREVIEW_ELEMENTS:,} elements or fewer."
        )
    return tuple(selection), ", ".join(normalized_labels), element_count


def image_layout(dataset: h5py.Dataset) -> tuple[bool, int]:
    """Return whether a dataset can be rendered and its frame count."""
    shape = dataset.shape
    if not shape or not np.issubdtype(dataset.dtype, np.number) and dataset.dtype != np.bool_:
        return False, 0
    if len(shape) == 2:
        return shape[0] * shape[1] <= MAX_IMAGE_PIXELS, 1
    if len(shape) == 3:
        if shape[-1] in (1, 3, 4):
            return shape[0] * shape[1] <= MAX_IMAGE_PIXELS, 1
        return shape[1] * shape[2] <= MAX_IMAGE_PIXELS, shape[0]
    if len(shape) == 4 and shape[-1] in (1, 3, 4):
        return shape[1] * shape[2] <= MAX_IMAGE_PIXELS, shape[0]
    return False, 0


def image_array(dataset: h5py.Dataset, index: int) -> np.ndarray:
    """Read one image or image frame without loading an entire image sequence."""
    capable, frame_count = image_layout(dataset)
    if not capable:
        raise ValueError("This dataset does not have a supported image shape or dtype.")
    if index < 0 or index >= frame_count:
        raise ValueError(f"Frame index must be between 0 and {frame_count - 1}.")

    if dataset.ndim == 4 or (dataset.ndim == 3 and dataset.shape[-1] not in (1, 3, 4)):
        array = np.asarray(dataset[index])
    else:
        array = np.asarray(dataset[...])
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in (2, 3):
        raise ValueError("The selected frame is not a 2D grayscale or RGB(A) image.")

    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    numeric = array.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8)
    low = float(np.min(numeric[finite]))
    high = float(np.max(numeric[finite]))
    if high <= low:
        result = np.zeros(array.shape, dtype=np.uint8)
        result[finite] = np.clip(low, 0, 255).astype(np.uint8)
        return result
    normalized = (numeric - low) * (255.0 / (high - low))
    normalized[~finite] = 0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def attribute_list(attributes: h5py.AttributeManager, key: str) -> list[str]:
    """Parse a list attribute stored as JSON text or an HDF5 array."""
    value = attributes.get(key, [])
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return [str(item) for item in value]


def relative_episode_path(path: str) -> str | None:
    """Return a path relative to /data/demo_N, if this is an episode object."""
    match = re.match(r"^/data/demo_\d+/(.+)$", path)
    return match.group(1) if match else None


def semantics_for_path(path: str) -> dict[str, Any] | None:
    """Describe known Isaac Lab recorder datasets while keeping a generic fallback."""
    relative = relative_episode_path(path)
    if relative is None:
        return None
    if relative in DATASET_SEMANTICS:
        return dict(DATASET_SEMANTICS[relative])
    if relative.startswith("obs/cam_"):
        return {
            "title": "RGB camera observation",
            "category": "Vision",
            "meaning": "Rendered RGB observation available before the current action A[t] is applied.",
            "timing": "pre-step observation",
            "axes": "[time, height, width, RGB channel]",
            "order": None,
        }
    if relative.startswith("initial_state/"):
        if relative.endswith("/root_pose"):
            detail = "Reset-time pose relative to the environment origin: [x, y, z, qw, qx, qy, qz]."
        elif relative.endswith("/root_velocity"):
            detail = "Reset-time velocity: [linear x, y, z, angular x, y, z]."
        elif relative.endswith("/joint_position"):
            detail = "Reset-time full articulation joint positions in articulation order, not the 19D policy order."
        elif relative.endswith("/joint_velocity"):
            detail = "Reset-time full articulation joint velocities in articulation order."
        else:
            detail = "Reset-time simulator state used to reproduce the episode."
        return {
            "title": "Initial simulator state",
            "category": "Replay / privileged state",
            "meaning": detail,
            "timing": "post-reset, one sample per episode",
            "axes": "object-specific simulator state",
            "order": None,
        }
    if relative.startswith("states/"):
        if relative.endswith("/root_pose"):
            detail = "Post-step pose after A[t]: [x, y, z, qw, qx, qy, qz]."
        elif relative.endswith("/root_velocity"):
            detail = "Post-step velocity after A[t]: [linear x, y, z, angular x, y, z]."
        elif relative.endswith("/joint_position"):
            detail = "Post-step full articulation positions after A[t], using the 31D articulation order."
        elif relative.endswith("/joint_velocity"):
            detail = "Post-step full articulation velocities after A[t], using the 31D articulation order."
        else:
            detail = "Full post-step simulator state after the current action and physics update."
        return {
            "title": "Post-step simulator state",
            "category": "Replay / privileged state",
            "meaning": detail,
            "timing": "post-step, after A[t]",
            "axes": "object-specific simulator state",
            "order": None,
        }
    return None


def vector_schema_for(
    path: str,
    data_attributes: h5py.AttributeManager | None,
) -> list[dict[str, Any]]:
    """Return semantic labels for the final vector axis when the contract defines them."""
    semantic = semantics_for_path(path)
    if semantic is None:
        return []
    order = semantic.get("order")
    if order == "action" and data_attributes is not None:
        names = attribute_list(data_attributes, "action_names")
        units = attribute_list(data_attributes, "action_units")
    elif order == "observation" and data_attributes is not None:
        names = attribute_list(data_attributes, "observation_state_names")
        units = attribute_list(data_attributes, "observation_state_units")
    elif order == "eef_pose":
        names = ["x", "y", "z", "qw", "qx", "qy", "qz"]
        units = ["m", "m", "m", "1", "1", "1", "1"]
    else:
        return []
    return [
        {"index": index, "name": name, "unit": units[index] if index < len(units) else ""}
        for index, name in enumerate(names)
    ]


def create_app(hdf5_path: Path) -> FastAPI:
    """Create a viewer application bound to one validated HDF5 file."""
    file_path = hdf5_path.resolve(strict=True)
    app = FastAPI(title="HDF5 Viewer", docs_url=None, redoc_url=None)

    def open_file() -> h5py.File:
        try:
            return h5py.File(file_path, "r")
        except OSError as exc:
            raise HTTPException(status_code=409, detail=f"Could not open HDF5 file: {exc}") from exc

    def lookup(hdf5_file: h5py.File, object_path: str) -> h5py.Group | h5py.Dataset:
        normalized = normalize_path(object_path)
        try:
            return hdf5_file[normalized]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"HDF5 object not found: {normalized}") from exc

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/file")
    def file_info() -> dict[str, Any]:
        stat = file_path.stat()
        with open_file() as hdf5_file:
            root_attributes = attributes_for(hdf5_file)
        return {
            "name": file_path.name,
            "path": str(file_path),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
            "root_attributes": root_attributes,
            "mode": "read-only",
        }

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        with open_file() as hdf5_file:
            if "/data" not in hdf5_file:
                raise HTTPException(status_code=404, detail="This file does not contain /data.")
            data = hdf5_file["/data"]
            action_names = attribute_list(data.attrs, "action_names")
            action_units = attribute_list(data.attrs, "action_units")
            state_names = attribute_list(data.attrs, "observation_state_names")
            state_units = attribute_list(data.attrs, "observation_state_units")
            control_hz = float(data.attrs.get("control_hz", 0.0))
            episode_names = sorted(
                [
                    name
                    for name in data.keys()
                    if re.fullmatch(r"demo_\d+", name) and isinstance(data[name], h5py.Group)
                ],
                key=natural_sort_key,
            )
            episodes = []
            camera_names: list[str] = []
            total_frames = 0
            for episode_name in episode_names:
                episode = data[episode_name]
                if "actions" in episode:
                    frames = int(episode["actions"].shape[0])
                else:
                    frames = int(episode.attrs.get("num_samples", 0))
                total_frames += frames
                if not camera_names and "obs" in episode:
                    camera_names = sorted(
                        [
                            name
                            for name, child in episode["obs"].items()
                            if isinstance(child, h5py.Dataset)
                            and child.ndim == 4
                            and child.shape[-1] in (1, 3, 4)
                        ],
                        key=natural_sort_key,
                    )
                episodes.append(
                    {
                        "name": episode_name,
                        "path": f"/data/{episode_name}",
                        "frames": frames,
                        "duration_s": (frames - 1) / control_hz if frames and control_hz else None,
                        "success": bool(episode.attrs.get("success", False)),
                        "schema_version": str(episode.attrs.get("schema_version", "")),
                        "target_object_name": str(episode.attrs.get("target_object_name", "")),
                    }
                )

            width = max(len(action_names), len(state_names))
            joint_order = []
            for index in range(width):
                action_name = action_names[index] if index < len(action_names) else ""
                state_name = state_names[index] if index < len(state_names) else ""
                joint_order.append(
                    {
                        "index": index,
                        "action_name": action_name,
                        "action_unit": action_units[index] if index < len(action_units) else "",
                        "observation_name": state_name,
                        "observation_unit": state_units[index] if index < len(state_units) else "",
                        "same": action_name == state_name,
                    }
                )

            return {
                "contract": {
                    "schema_version": str(data.attrs.get("schema_version", "")),
                    "robot_contract_id": str(data.attrs.get("robot_contract_id", "")),
                    "dataset_origin": str(data.attrs.get("dataset_origin", "")),
                    "action_semantics": str(data.attrs.get("action_semantics", "")),
                    "observation_semantics": str(data.attrs.get("observation_semantics", "")),
                    "scene_state_semantics": str(data.attrs.get("scene_state_semantics", "")),
                },
                "task": {
                    "task_id": str(data.attrs.get("task_id", "")),
                    "instruction": str(data.attrs.get("task_instruction", "")),
                    "target_object": str(data.attrs.get("target_object_name", "")),
                    "target_side": str(data.attrs.get("target_side", "")),
                    "environment": str(data.attrs.get("task_env_name", "")),
                    "success_criterion": str(data.attrs.get("success_criterion_id", "")),
                },
                "rates": {"control_hz": control_hz, "camera_hz": float(data.attrs.get("camera_hz", 0.0))},
                "episode_count": len(episodes),
                "total_frames": total_frames,
                "stored_total": int(data.attrs.get("total", 0)),
                "episodes": episodes,
                "camera_names": camera_names,
                "joint_order": joint_order,
                "categories": [
                    {
                        "title": "Commands",
                        "role": "Values sent to the Isaac Lab action manager.",
                        "paths": ["actions", "processed_actions"],
                    },
                    {
                        "title": "Pre-step observations",
                        "role": "Policy-visible state before the current action is applied.",
                        "paths": [
                            "obs/actions",
                            "obs/joint_pos",
                            "obs/joint_pos_target",
                            "obs/left_eef_pose",
                            "obs/right_eef_pose",
                        ],
                    },
                    {
                        "title": "Vision and time",
                        "role": "Synchronized RGB observations and episode-relative time.",
                        "paths": [*(f"obs/{name}" for name in camera_names), "obs/step_index", "obs/timestamp_s"],
                    },
                    {
                        "title": "Replay / privileged state",
                        "role": "Full simulator snapshots for reset, replay and diagnostics; not policy input by default.",
                        "paths": ["initial_state/...", "states/articulation/...", "states/rigid_object/..."],
                    },
                ],
            }

    @app.get("/api/children")
    def children(
        path: str = Query(default="/"),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=DEFAULT_CHILD_LIMIT, ge=1, le=2_000),
    ) -> dict[str, Any]:
        normalized = normalize_path(path)
        with open_file() as hdf5_file:
            obj = lookup(hdf5_file, normalized)
            if not isinstance(obj, h5py.Group):
                raise HTTPException(status_code=400, detail=f"Not a group: {normalized}")
            names = sorted(obj.keys(), key=natural_sort_key)
            selected_names = names[offset : offset + limit]
            result = []
            for name in selected_names:
                child = obj[name]
                child_path = f"/{name}" if normalized == "/" else f"{normalized.rstrip('/')}/{name}"
                result.append(
                    {
                        "name": name,
                        "path": child_path,
                        "kind": "group" if isinstance(child, h5py.Group) else "dataset",
                    }
                )
        return {
            "path": normalized,
            "children": result,
            "total": len(names),
            "offset": offset,
            "limit": limit,
            "truncated": offset + len(result) < len(names),
        }

    @app.get("/api/node")
    def node(path: str = Query(default="/")) -> dict[str, Any]:
        normalized = normalize_path(path)
        with open_file() as hdf5_file:
            obj = lookup(hdf5_file, normalized)
            data_attributes = hdf5_file["/data"].attrs if "/data" in hdf5_file else None
            common = {
                "path": normalized,
                "name": "/" if normalized == "/" else normalized.rsplit("/", 1)[-1],
                "attributes": attributes_for(obj),
                "semantic": semantics_for_path(normalized),
            }
            if isinstance(obj, h5py.Group):
                return {**common, "kind": "group", "child_count": len(obj)}
            capable, frame_count = image_layout(obj)
            try:
                storage_size: int | None = int(obj.id.get_storage_size())
            except (RuntimeError, ValueError):
                storage_size = None
            return {
                **common,
                "kind": "dataset",
                "shape": list(obj.shape),
                "maxshape": list(obj.maxshape) if obj.maxshape is not None else None,
                "ndim": obj.ndim,
                "size": int(obj.size),
                "dtype": str(obj.dtype),
                "chunks": list(obj.chunks) if obj.chunks is not None else None,
                "compression": obj.compression,
                "compression_options": json_value(obj.compression_opts),
                "storage_size": storage_size,
                "suggested_slice": suggested_slice(obj.shape),
                "preview_limit": MAX_PREVIEW_ELEMENTS,
                "image_capable": capable,
                "image_frame_count": frame_count,
                "vector_schema": vector_schema_for(normalized, data_attributes),
            }

    @app.get("/api/episode")
    def episode_frame(path: str, index: int = Query(default=0, ge=0)) -> dict[str, Any]:
        normalized = normalize_path(path).rstrip("/")
        if not re.fullmatch(r"/data/demo_\d+", normalized):
            raise HTTPException(status_code=400, detail="Episode path must look like /data/demo_N.")
        with open_file() as hdf5_file:
            episode = lookup(hdf5_file, normalized)
            if not isinstance(episode, h5py.Group):
                raise HTTPException(status_code=400, detail=f"Not an episode group: {normalized}")
            required = (
                "actions",
                "processed_actions",
                "obs/actions",
                "obs/joint_pos",
                "obs/joint_pos_target",
            )
            missing = [name for name in required if name not in episode]
            if missing:
                raise HTTPException(status_code=400, detail=f"Episode is missing: {missing}")
            frames = int(episode["actions"].shape[0])
            if index >= frames:
                raise HTTPException(status_code=400, detail=f"Frame index must be between 0 and {frames - 1}.")

            data = hdf5_file["/data"]
            action_names = attribute_list(data.attrs, "action_names")
            action_units = attribute_list(data.attrs, "action_units")
            state_names = attribute_list(data.attrs, "observation_state_names")
            state_units = attribute_list(data.attrs, "observation_state_units")
            if len(action_names) != episode["actions"].shape[-1]:
                raise HTTPException(status_code=409, detail="action_names does not match the actions width.")
            if len(state_names) != episode["obs/joint_pos"].shape[-1]:
                raise HTTPException(status_code=409, detail="observation_state_names does not match joint_pos.")

            action = np.asarray(episode["actions"][index], dtype=np.float64)
            processed = np.asarray(episode["processed_actions"][index], dtype=np.float64)
            observed_action = np.asarray(episode["obs/actions"][index], dtype=np.float64)
            joint_position = np.asarray(episode["obs/joint_pos"][index], dtype=np.float64)
            joint_target = np.asarray(episode["obs/joint_pos_target"][index], dtype=np.float64)
            action_index = {name: position for position, name in enumerate(action_names)}
            state_index = {name: position for position, name in enumerate(state_names)}
            joints = []
            for observation_index, name in enumerate(state_names):
                current_action_index = action_index.get(name)
                unit = state_units[observation_index] if observation_index < len(state_units) else ""
                if not unit and current_action_index is not None and current_action_index < len(action_units):
                    unit = action_units[current_action_index]
                current_action = (
                    float(action[current_action_index]) if current_action_index is not None else None
                )
                previous_action = (
                    float(observed_action[current_action_index])
                    if current_action_index is not None
                    else None
                )
                processed_action = (
                    float(processed[current_action_index]) if current_action_index is not None else None
                )
                realized = float(joint_position[observation_index])
                target = float(joint_target[observation_index])
                joints.append(
                    {
                        "joint": name,
                        "unit": unit,
                        "action_index": current_action_index,
                        "observation_index": observation_index,
                        "action_t": current_action,
                        "processed_action_t": processed_action,
                        "previous_action_obs": previous_action,
                        "joint_target": target,
                        "joint_position": realized,
                        "tracking_error": realized - target,
                    }
                )

            alignment: dict[str, Any] = {
                "has_previous_recorded_action": index > 0,
                "obs_actions_vs_previous_max_abs": None,
                "target_vs_previous_max_abs": None,
            }
            if index > 0:
                previous = np.asarray(episode["actions"][index - 1], dtype=np.float64)
                previous_by_state = np.asarray([previous[action_index[name]] for name in state_names])
                alignment["obs_actions_vs_previous_max_abs"] = float(
                    np.max(np.abs(observed_action - previous))
                )
                alignment["target_vs_previous_max_abs"] = float(
                    np.max(np.abs(joint_target - previous_by_state))
                )

            def optional_vector(name: str) -> list[float] | None:
                if name not in episode:
                    return None
                return [float(value) for value in np.asarray(episode[name][index]).reshape(-1)]

            timestamp = optional_vector("obs/timestamp_s")
            step_index = optional_vector("obs/step_index")
            cameras = []
            if "obs" in episode:
                for name, child in episode["obs"].items():
                    if (
                        isinstance(child, h5py.Dataset)
                        and child.ndim == 4
                        and child.shape[-1] in (1, 3, 4)
                    ):
                        cameras.append(
                            {
                                "name": name,
                                "path": f"{normalized}/obs/{name}",
                                "shape": list(child.shape[1:]),
                            }
                        )
            cameras.sort(key=lambda item: natural_sort_key(item["name"]))
            return {
                "path": normalized,
                "frame": index,
                "frame_count": frames,
                "timestamp_s": timestamp[0] if timestamp else None,
                "step_index": int(step_index[0]) if step_index else None,
                "joints": joints,
                "alignment": alignment,
                "eef": {
                    "left": optional_vector("obs/left_eef_pose"),
                    "right": optional_vector("obs/right_eef_pose"),
                    "components": ["x", "y", "z", "qw", "qx", "qy", "qz"],
                },
                "cameras": cameras,
                "attributes": attributes_for(episode),
            }

    @app.get("/api/values")
    def values(path: str, slice: str = Query(default="")) -> dict[str, Any]:  # noqa: A002
        normalized = normalize_path(path)
        with open_file() as hdf5_file:
            obj = lookup(hdf5_file, normalized)
            if not isinstance(obj, h5py.Dataset):
                raise HTTPException(status_code=400, detail=f"Not a dataset: {normalized}")
            try:
                selection, label, element_count = parse_slice(slice, obj.shape)
                selected = obj[selection] if selection else obj[()]
            except (ValueError, TypeError, IndexError, OSError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        selected_array = np.asarray(selected)
        statistics: dict[str, Any] = {}
        if selected_array.size and (
            np.issubdtype(selected_array.dtype, np.number)
            and not np.issubdtype(selected_array.dtype, np.complexfloating)
        ):
            try:
                numeric = selected_array.astype(np.float64, copy=False)
                finite = numeric[np.isfinite(numeric)]
                if finite.size:
                    statistics = {
                        "min": json_value(float(np.min(finite))),
                        "max": json_value(float(np.max(finite))),
                        "mean": json_value(float(np.mean(finite))),
                    }
            except (TypeError, ValueError):
                statistics = {}
        return {
            "path": normalized,
            "selection": label,
            "selected_shape": list(selected_array.shape),
            "element_count": element_count,
            "dtype": str(selected_array.dtype),
            "statistics": statistics,
            "data": json_value(selected, max_items=MAX_PREVIEW_ELEMENTS),
        }

    @app.get("/api/image")
    def image(path: str, index: int = Query(default=0, ge=0)) -> Response:
        normalized = normalize_path(path)
        with open_file() as hdf5_file:
            obj = lookup(hdf5_file, normalized)
            if not isinstance(obj, h5py.Dataset):
                raise HTTPException(status_code=400, detail=f"Not a dataset: {normalized}")
            try:
                pixels = image_array(obj, index)
            except (ValueError, TypeError, IndexError, OSError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        output = io.BytesIO()
        Image.fromarray(pixels).save(output, format="PNG")
        return Response(content=output.getvalue(), media_type="image/png")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open one HDF5 file in a read-only, lazy-loading web viewer."
    )
    parser.add_argument("hdf5_file", type=Path, help="HDF5 file to inspect")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default: 8765)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.hdf5_file.is_file():
        raise SystemExit(f"HDF5 file not found: {args.hdf5_file}")
    try:
        with h5py.File(args.hdf5_file, "r"):
            pass
    except OSError as exc:
        raise SystemExit(f"Could not open HDF5 file: {exc}") from exc
    app = create_app(args.hdf5_file)
    print(f"Viewing {args.hdf5_file.resolve()} (read-only)")
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
