#!/usr/bin/env python3
"""
Browser mask painter for building a (possibly partial) shape model.

Intended workflow for orientation-rich templates:
  1. Switch to「框选分割」and drag a box over a distinctive PCB region
     (e.g. connector edge) — not the whole square board.
  2. Server segments green PCB (+ metal) inside the box only.
  3. Refine with brush / erase, then「保存并建模型」.

  .venv/bin/python paint_mask.py \
    --image ../captures/260322274901_20261010_112805/color/color_0000.jpg \
    --out models/color_0000_mask.png \
    --port 8765
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def bgr_to_data_url(bgr: np.ndarray, ext: str = ".jpg", quality: int = 90) -> str:
    if ext.lower() in {".jpg", ".jpeg"}:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        mime = "image/jpeg"
    else:
        ok, buf = cv2.imencode(".png", bgr)
        mime = "image/png"
    if not ok:
        raise RuntimeError("imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def mask_to_data_url(mask: np.ndarray) -> str:
    return bgr_to_data_url(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), ".png")


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        raise ValueError("box too small")
    return x1, y1, x2, y2


def segment_in_box(bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """
    Segment PCB pixels inside a user box (partial feature OK).

    Tray and PCB share similar teal hue in these captures, so HSV alone
    fails. Primary method: GrabCut with the user rectangle. Then lightly
    expand to bright metal connectors that touch the cut.
    Result is full-frame mask (zeros outside a small halo of the box).
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = _clip_box(*box, w, h)
    rw, rh = x2 - x1, y2 - y1
    rect = (x1, y1, rw, rh)

    gc = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        cut = np.where(
            (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except cv2.error:
        cut = np.zeros((h, w), np.uint8)

    # Restrict to the user box (GrabCut can leak slightly).
    boxed = np.zeros_like(cut)
    boxed[y1:y2, x1:x2] = cut[y1:y2, x1:x2]

    # Attach silver connectors that touch the GrabCut region (right-edge ports).
    roi = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    metal = ((gray > 135) & (gray < 250) & (hsv[:, :, 1] < 70)).astype(np.uint8) * 255
    touch = cv2.dilate(boxed[y1:y2, x1:x2], np.ones((7, 7), np.uint8), 1)
    metal = cv2.bitwise_and(metal, touch)
    boxed[y1:y2, x1:x2] = np.maximum(boxed[y1:y2, x1:x2], metal)

    # Keep largest component inside box.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        (boxed > 0).astype(np.uint8), 8
    )
    out = np.zeros_like(boxed)
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out[lab == i] = 255
    else:
        out = boxed

    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        filled = np.zeros_like(out)
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        out = filled
    return out


def seed_green_pcb_mask(bgr: np.ndarray) -> np.ndarray:
    """Optional full-board seed (usually skip for orientation templates)."""
    h, w = bgr.shape[:2]
    x_cut = int(w * 0.86)
    hsv = cv2.cvtColor(bgr[:, :x_cut], cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([38, 45, 35]), np.array([88, 255, 230]))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), 2)
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), 1)
    cnts, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    if not cnts:
        return mask
    best = max(cnts, key=cv2.contourArea)
    cv2.drawContours(mask, [best], -1, 255, -1)
    return mask


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>PCB Mask Painter</title>
<style>
  :root {
    --bg: #1a1d21; --panel: #252a31; --text: #e8eaed; --muted: #9aa0a6;
    --accent: #3ddc97; --danger: #ff6b6b; --warn: #ffe66d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg); color: var(--text); height: 100vh;
    display: flex; flex-direction: column;
  }
  header {
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
    padding: 10px 14px; background: var(--panel); border-bottom: 1px solid #333;
  }
  header h1 { font-size: 15px; margin: 0 8px 0 0; font-weight: 600; }
  .hint {
    width: 100%; font-size: 12px; color: var(--warn); margin: 0;
    order: 99;
  }
  .btn {
    border: 1px solid #444; background: #323842; color: var(--text);
    padding: 6px 10px; border-radius: 6px; cursor: pointer; font-size: 13px;
  }
  .btn:hover { background: #3c4450; }
  .btn.active { border-color: var(--accent); color: var(--accent); }
  .btn.primary { background: #1f6f4a; border-color: var(--accent); }
  .btn.danger { border-color: var(--danger); color: #ffb4b4; }
  label { font-size: 12px; color: var(--muted); display: flex; align-items: center; gap: 6px; }
  input[type=range] { width: 100px; }
  #stage-wrap {
    flex: 1; overflow: auto; display: flex; align-items: flex-start; justify-content: center;
    padding: 16px;
    background: radial-gradient(ellipse at 30% 20%, #243028 0%, transparent 50%),
      linear-gradient(160deg, #16191d, #1e2329 60%, #151820);
  }
  #stage { position: relative; box-shadow: 0 12px 40px rgba(0,0,0,.45); line-height: 0; }
  canvas { display: block; cursor: crosshair; max-width: 100%; height: auto; }
  #overlay { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
  #status { margin-left: auto; font-size: 12px; color: var(--muted); max-width: 460px; text-align: right; }
</style>
</head>
<body>
<header>
  <h1>PCB Mask Painter</h1>
  <button class="btn active" id="modeBox" type="button">框选分割 (R)</button>
  <button class="btn" id="modePaint" type="button">画笔 (B)</button>
  <button class="btn" id="modeErase" type="button">橡皮 (E)</button>
  <button class="btn" id="modePoly" type="button">多边形 (P)</button>
  <button class="btn" id="closePoly" type="button">闭合多边形</button>
  <label>笔刷 <input id="brush" type="range" min="2" max="80" value="14"/> <span id="brushVal">14</span></label>
  <label>透明度 <input id="alpha" type="range" min="10" max="90" value="45"/> <span id="alphaVal">45%</span></label>
  <label><input id="merge" type="checkbox" checked/> 叠加到已有 mask</label>
  <button class="btn" id="btnClear" type="button">清空</button>
  <button class="btn" id="btnUndo" type="button">撤销 (Z)</button>
  <button class="btn primary" id="btnSave" type="button">保存并建模型</button>
  <button class="btn danger" id="btnSaveOnly" type="button">仅保存 mask</button>
  <div id="status">拖框选特征区（如接口侧），会在框内自动抠 PCB</div>
  <p class="hint">提示：不必框整板。方板轮廓难定方向，请只框有特征的局部（接口/缺口等），用局部轮廓建模型。</p>
</header>
<div id="stage-wrap"><div id="stage">
  <canvas id="base"></canvas>
  <canvas id="overlay"></canvas>
</div></div>
<script>
const CFG = __CFG__;
const base = document.getElementById('base');
const overlay = document.getElementById('overlay');
const bctx = base.getContext('2d');
const octx = overlay.getContext('2d');
const statusEl = document.getElementById('status');

let W=0, H=0;
let mask = null;
let mode = 'box';
let drawing = false;
let brush = 14;
let alphaView = 0.45;
let undoStack = [];
let polyPts = [];
let boxStart = null;
let boxCur = null;
let busy = false;

function setStatus(t){ statusEl.textContent = t; }

function syncOverlaySize() {
  const rect = base.getBoundingClientRect();
  overlay.style.width = rect.width + 'px';
  overlay.style.height = rect.height + 'px';
}

function pushUndo() {
  undoStack.push(new Uint8ClampedArray(mask.data));
  if (undoStack.length > 30) undoStack.shift();
}
function undo() {
  if (!undoStack.length) return;
  mask.data.set(undoStack.pop());
  redraw();
}

function redraw() {
  octx.clearRect(0,0,W,H);
  const img = octx.createImageData(W,H);
  const d = mask.data;
  const a = Math.round(alphaView * 255);
  for (let i=0;i<d.length;i+=4) {
    if (d[i] > 127) {
      img.data[i]=40; img.data[i+1]=220; img.data[i+2]=120; img.data[i+3]=a;
    }
  }
  octx.putImageData(img,0,0);
  if (polyPts.length) {
    octx.strokeStyle = '#ffe66d'; octx.fillStyle = 'rgba(255,230,109,.25)';
    octx.lineWidth = 2; octx.beginPath();
    octx.moveTo(polyPts[0].x, polyPts[0].y);
    for (let i=1;i<polyPts.length;i++) octx.lineTo(polyPts[i].x, polyPts[i].y);
    octx.stroke();
    for (const p of polyPts) { octx.beginPath(); octx.arc(p.x,p.y,3,0,Math.PI*2); octx.fill(); }
  }
  if (boxStart && boxCur) {
    const x = Math.min(boxStart.x, boxCur.x);
    const y = Math.min(boxStart.y, boxCur.y);
    const w = Math.abs(boxCur.x - boxStart.x);
    const h = Math.abs(boxCur.y - boxStart.y);
    octx.strokeStyle = '#4cc9f0'; octx.lineWidth = 2;
    octx.setLineDash([6,4]);
    octx.strokeRect(x, y, w, h);
    octx.setLineDash([]);
    octx.fillStyle = 'rgba(76,201,240,.12)';
    octx.fillRect(x, y, w, h);
  }
}

function canvasPos(ev) {
  const r = overlay.getBoundingClientRect();
  return {
    x: (ev.clientX - r.left) * (W / r.width),
    y: (ev.clientY - r.top) * (H / r.height)
  };
}

function stamp(x, y, erase) {
  const r = brush, r2 = r*r, d = mask.data;
  const x0 = Math.max(0, Math.floor(x-r));
  const y0 = Math.max(0, Math.floor(y-r));
  const x1 = Math.min(W-1, Math.ceil(x+r));
  const y1 = Math.min(H-1, Math.ceil(y+r));
  const val = erase ? 0 : 255;
  for (let yy=y0; yy<=y1; yy++) for (let xx=x0; xx<=x1; xx++) {
    const dx=xx-x, dy=yy-y;
    if (dx*dx+dy*dy <= r2) {
      const i = (yy*W+xx)*4;
      d[i]=d[i+1]=d[i+2]=val; d[i+3]=255;
    }
  }
}

function fillPoly(pts, erase) {
  const c = document.createElement('canvas'); c.width=W; c.height=H;
  const ctx = c.getContext('2d');
  ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath(); ctx.fillStyle='#fff'; ctx.fill();
  const pdata = ctx.getImageData(0,0,W,H).data;
  const d = mask.data; const val = erase ? 0 : 255;
  for (let i=0;i<pdata.length;i+=4) if (pdata[i+3] > 0) {
    d[i]=d[i+1]=d[i+2]=val; d[i+3]=255;
  }
}

function applyMaskPngUrl(url, merge) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas'); c.width=W; c.height=H;
      const ctx = c.getContext('2d'); ctx.drawImage(img,0,0,W,H);
      const s = ctx.getImageData(0,0,W,H).data;
      const d = mask.data;
      for (let i=0;i<d.length;i+=4) {
        const v = s[i] > 127 ? 255 : 0;
        if (merge) {
          if (v) { d[i]=d[i+1]=d[i+2]=255; d[i+3]=255; }
        } else {
          d[i]=d[i+1]=d[i+2]=v; d[i+3]=255;
        }
      }
      redraw(); resolve();
    };
    img.onerror = reject;
    img.src = url;
  });
}

async function segmentBox(x1,y1,x2,y2) {
  if (busy) return;
  busy = true;
  setStatus('框内分割中...');
  try {
    const resp = await fetch('/segment', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({x1,y1,x2,y2})
    });
    const j = await resp.json();
    if (!resp.ok) throw new Error(j.error || resp.status);
    pushUndo();
    await applyMaskPngUrl(j.mask_data_url, document.getElementById('merge').checked);
    setStatus(`框内已抠出 ${j.fg_pixels} px · 可再框其他特征或用画笔修边`);
  } catch (e) {
    setStatus('分割失败: ' + e.message);
  } finally {
    busy = false;
  }
}

function setMode(m) {
  mode = m;
  ['Box','Paint','Erase','Poly'].forEach(k => {
    document.getElementById('mode'+k).classList.toggle('active',
      m === k.toLowerCase());
  });
}

async function loadImages() {
  const img = new Image();
  img.src = CFG.image_data_url;
  await img.decode();
  W = img.naturalWidth; H = img.naturalHeight;
  base.width = W; base.height = H;
  overlay.width = W; overlay.height = H;
  bctx.drawImage(img, 0, 0);
  mask = octx.createImageData(W,H);
  for (let i=0;i<mask.data.length;i+=4) mask.data[i+3]=255;
  if (CFG.seed_data_url) await applyMaskPngUrl(CFG.seed_data_url, false);
  syncOverlaySize();
  redraw();
  setStatus(`已加载 ${W}x${H} · 用「框选分割」框局部特征（如右侧接口）`);
}

function maskPngBlob() {
  const c = document.createElement('canvas'); c.width=W; c.height=H;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(W,H);
  const d = mask.data;
  for (let i=0;i<d.length;i+=4) {
    const v = d[i] > 127 ? 255 : 0;
    img.data[i]=img.data[i+1]=img.data[i+2]=v; img.data[i+3]=255;
  }
  ctx.putImageData(img,0,0);
  return new Promise(res => c.toBlob(res, 'image/png'));
}

function bytesToB64(buf) {
  const bytes = new Uint8Array(buf);
  let s = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk)
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  return btoa(s);
}

async function save(build) {
  const blob = await maskPngBlob();
  const b64 = bytesToB64(await blob.arrayBuffer());
  setStatus('保存中...');
  const resp = await fetch('/save', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ mask_png_b64: b64, build: !!build })
  });
  const j = await resp.json();
  if (!resp.ok) { setStatus('失败: ' + (j.error||resp.status)); return; }
  setStatus(j.message || JSON.stringify(j));
}

overlay.addEventListener('mousedown', (ev) => {
  const p = canvasPos(ev);
  if (mode === 'box') {
    if (ev.button !== 0) return;
    boxStart = p; boxCur = p; drawing = true; redraw();
    return;
  }
  if (mode === 'poly') {
    if (ev.button === 0) { polyPts.push(p); redraw(); }
    return;
  }
  pushUndo();
  drawing = true;
  stamp(p.x, p.y, mode==='erase' || ev.button===2);
  redraw();
  ev.preventDefault();
});
overlay.addEventListener('mousemove', (ev) => {
  if (!drawing) return;
  const p = canvasPos(ev);
  if (mode === 'box') { boxCur = p; redraw(); return; }
  if (mode === 'poly') return;
  stamp(p.x, p.y, mode==='erase' || ev.buttons===2);
  redraw();
});
window.addEventListener('mouseup', async (ev) => {
  if (mode === 'box' && drawing && boxStart && boxCur) {
    drawing = false;
    const x1 = boxStart.x, y1 = boxStart.y, x2 = boxCur.x, y2 = boxCur.y;
    boxStart = null; boxCur = null; redraw();
    if (Math.abs(x2-x1) > 8 && Math.abs(y2-y1) > 8)
      await segmentBox(x1,y1,x2,y2);
    return;
  }
  drawing = false;
});
overlay.addEventListener('contextmenu', e => e.preventDefault());

document.getElementById('closePoly').onclick = () => {
  if (polyPts.length < 3) return;
  pushUndo(); fillPoly(polyPts, mode==='erase'); polyPts=[]; redraw();
};
document.getElementById('modeBox').onclick = () => setMode('box');
document.getElementById('modePaint').onclick = () => setMode('paint');
document.getElementById('modeErase').onclick = () => setMode('erase');
document.getElementById('modePoly').onclick = () => setMode('poly');
document.getElementById('btnClear').onclick = () => {
  pushUndo();
  const d=mask.data; for(let i=0;i<d.length;i+=4){d[i]=d[i+1]=d[i+2]=0;d[i+3]=255;}
  polyPts=[]; redraw();
};
document.getElementById('btnUndo').onclick = undo;
document.getElementById('btnSave').onclick = () => save(true);
document.getElementById('btnSaveOnly').onclick = () => save(false);
document.getElementById('brush').oninput = (e) => {
  brush = +e.target.value; document.getElementById('brushVal').textContent = brush;
};
document.getElementById('alpha').oninput = (e) => {
  alphaView = (+e.target.value)/100;
  document.getElementById('alphaVal').textContent = e.target.value+'%'; redraw();
};
window.addEventListener('keydown', (e) => {
  if (e.key==='r'||e.key==='R') setMode('box');
  if (e.key==='b'||e.key==='B') setMode('paint');
  if (e.key==='e'||e.key==='E') setMode('erase');
  if (e.key==='p'||e.key==='P') setMode('poly');
  if (e.key==='z'||e.key==='Z') undo();
  if (e.key==='Enter' && mode==='poly') document.getElementById('closePoly').click();
});
window.addEventListener('resize', syncOverlaySize);
loadImages();
</script>
</body>
</html>
"""


class AppState:
    def __init__(self, image_path: Path, out_mask: Path, model_out: Path, seed: np.ndarray | None):
        self.image_path = image_path
        self.out_mask = out_mask
        self.model_out = model_out
        self.bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if self.bgr is None:
            raise FileNotFoundError(image_path)
        self.seed = seed
        self.last_result: dict | None = None


def build_model_from_saved_mask(
    mask_path: Path, model_out: Path, angle_step: float, max_points: int
) -> dict:
    from shape_matcher import build_model_from_mask, load_mask

    mask = load_mask(mask_path)
    model = build_model_from_mask(mask, angle_step_deg=angle_step, max_points=max_points)
    model.save(model_out)
    edges = np.zeros(mask.shape, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(edges, [max(contours, key=cv2.contourArea)], -1, 255, 1)
    preview = model_out.with_suffix(".edges.png")
    cv2.imwrite(str(preview), edges)
    return {
        "out": str(model_out),
        "n_points": int(len(model.points)),
        "edges_preview": str(preview),
        "angle_step_deg": model.angle_step_deg,
        "from_mask": True,
    }


def make_handler(state: AppState, angle_step: float, max_points: int):
    image_url = bgr_to_data_url(state.bgr, ".jpg")
    seed_url = mask_to_data_url(state.seed) if state.seed is not None else ""
    cfg = {
        "image_data_url": image_url,
        "seed_data_url": seed_url,
        "out_mask": str(state.out_mask),
        "model_out": str(state.model_out),
    }
    page = HTML.replace("__CFG__", json.dumps(cfg))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[paint_mask] {self.address_string()} {fmt % args}")

        def _json(self, code: int, obj: dict):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as e:
                self._json(400, {"ok": False, "error": f"bad json: {e}"})
                return

            if path == "/segment":
                try:
                    box = (
                        payload["x1"],
                        payload["y1"],
                        payload["x2"],
                        payload["y2"],
                    )
                    mask = segment_in_box(state.bgr, box)
                    fg = int((mask > 0).sum())
                    if fg < 30:
                        self._json(
                            400,
                            {
                                "ok": False,
                                "error": "框内几乎抠不到 PCB，请框住绿色板子/接口区域，或换更大一点的框",
                            },
                        )
                        return
                    self._json(
                        200,
                        {
                            "ok": True,
                            "fg_pixels": fg,
                            "mask_data_url": mask_to_data_url(mask),
                        },
                    )
                except Exception as e:
                    self._json(500, {"ok": False, "error": str(e)})
                return

            if path == "/save":
                try:
                    data = base64.b64decode(payload["mask_png_b64"])
                    arr = np.frombuffer(data, dtype=np.uint8)
                    mask_bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                    if mask_bgr is None:
                        raise ValueError("decode mask failed")
                    if mask_bgr.ndim == 3:
                        mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                    else:
                        mask = mask_bgr
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    state.out_mask.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(state.out_mask), mask)

                    cut = state.bgr.copy()
                    cut[mask == 0] = 0
                    cut_path = state.out_mask.with_name(state.out_mask.stem + "_cutout.jpg")
                    cv2.imwrite(str(cut_path), cut)

                    ov = state.bgr.copy()
                    tint = ov.copy()
                    tint[mask > 0] = (40, 220, 120)
                    ov = cv2.addWeighted(ov, 0.55, tint, 0.45, 0)
                    ov_path = state.out_mask.with_name(state.out_mask.stem + "_overlay.jpg")
                    cv2.imwrite(str(ov_path), ov)

                    result = {
                        "ok": True,
                        "mask": str(state.out_mask),
                        "cutout": str(cut_path),
                        "overlay": str(ov_path),
                        "fg_pixels": int((mask > 0).sum()),
                    }
                    msg = f"已保存 mask → {state.out_mask.name} ({result['fg_pixels']} px)"
                    if payload.get("build"):
                        built = build_model_from_saved_mask(
                            state.out_mask, state.model_out, angle_step, max_points
                        )
                        result["model"] = built
                        msg += f" · 模型 → {state.model_out.name} (n={built['n_points']})"
                    result["message"] = msg
                    state.last_result = result
                    self._json(200, result)
                except Exception as e:
                    self._json(500, {"ok": False, "error": str(e)})
                return

            self.send_error(404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="Manual / box-segment PCB mask painter")
    ap.add_argument(
        "--image",
        default=str(
            ROOT.parent
            / "captures/260322274901_20261010_112805/color/color_0000.jpg"
        ),
    )
    ap.add_argument("--out", default=str(ROOT / "models/color_0000_mask.png"))
    ap.add_argument("--model-out", default=str(ROOT / "models/d435_pcb.npz"))
    ap.add_argument("--seed-auto", action="store_true", help="Preload full-board auto seed")
    ap.add_argument("--seed", default="", help="Optional seed mask image")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--angle-step", type=float, default=2.0)
    ap.add_argument("--max-points", type=int, default=600)
    args = ap.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    out_mask = Path(args.out).expanduser().resolve()
    model_out = Path(args.model_out).expanduser().resolve()

    seed = None
    if args.seed:
        seed = cv2.imread(args.seed, cv2.IMREAD_GRAYSCALE)
        if seed is None:
            raise FileNotFoundError(args.seed)
        _, seed = cv2.threshold(seed, 127, 255, cv2.THRESH_BINARY)
    elif args.seed_auto:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(image_path)
        seed = seed_green_pcb_mask(bgr)
        print(f"seed fg={int((seed > 0).sum())}")

    state = AppState(image_path, out_mask, model_out, seed)
    handler = make_handler(state, args.angle_step, args.max_points)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Mask painter: {url}")
    print(f"image: {image_path}")
    print(f"mask out: {out_mask}")
    print(f"model out: {model_out}")
    print("Use「框选分割」on a distinctive partial region, then save. Ctrl+C to stop.")

    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        if state.last_result:
            print(json.dumps(state.last_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
