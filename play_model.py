"""Serve a small live browser window for the trained connectome controller."""

from __future__ import annotations

import argparse
import base64
import json
import struct
import threading
import time
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from env import ACTIONS, RICH_ACTIONS
from gym_env import (
    PartialPixelGymRunnerEnv,
    PixelGymRunnerEnv,
    RichPartialPixelGymRunnerEnv,
    RichPixelGymRunnerEnv,
)
from male_cns_neuron_graph import MaleCNSNeuronGraphExtractor  # noqa: F401
from train_fly_cns_retina import FlyCNSRetinaPolicy
from train_visual_teacher import VisualPolicy, near_field_view


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fly-Brain Runner</title>
  <link rel="icon" href="data:,">
  <style>
    :root {
      color-scheme: dark;
      font: 15px/1.4 ui-sans-serif, system-ui, -apple-system, sans-serif;
      --ink: #070b13;
      --panel: #0d1420;
      --panel-hi: #111c2c;
      --line: #263550;
      --line-hi: #3a4d70;
      --text: #edf3ff;
      --muted: #8d9bb5;
      --faint: #5f6d86;
      --mint: #64e6b0;
      --cyan: #72d8ff;
      --amber: #f6bf62;
      --coral: #ff7e75;
      --violet: #bd9cff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 0%, #14243a 0, transparent 34rem),
        radial-gradient(circle at 96% 90%, #19152b 0, transparent 28rem),
        var(--ink);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: .25;
      background-image: linear-gradient(#ffffff05 1px, transparent 1px),
        linear-gradient(90deg, #ffffff05 1px, transparent 1px);
      background-size: 32px 32px;
      mask-image: linear-gradient(to bottom, #000, transparent 75%);
    }
    .shell { width: min(1180px, calc(100vw - 40px)); margin: 0 auto; padding: 38px 0 28px; }
    .topbar { display: flex; justify-content: space-between; align-items: end; gap: 24px; margin-bottom: 26px; }
    .eyebrow, .section-label, .frame-spec, .model-name, kbd {
      font: 700 11px/1 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: .13em;
      text-transform: uppercase;
    }
    .eyebrow { color: var(--cyan); display: flex; align-items: center; gap: 9px; }
    .brand-mark { display: inline-grid; place-items: center; width: 25px; height: 25px;
      color: var(--ink); background: var(--mint); font-size: 10px; letter-spacing: -.06em; }
    h1 { margin: 11px 0 5px; font-size: clamp(1.8rem, 4vw, 2.55rem); line-height: 1; letter-spacing: -.045em; }
    .lede { margin: 0; color: var(--muted); max-width: 520px; }
    .session-state { display: grid; justify-items: end; gap: 7px; color: var(--muted); font-size: 12px; }
    .session-pill { display: inline-flex; align-items: center; gap: 8px; padding: 8px 11px;
      border: 1px solid var(--line); background: #0e1726; color: var(--text); }
    .status-dot, .event-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--mint);
      box-shadow: 0 0 0 4px #64e6b01c; }
    .workspace { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(300px, .8fr); gap: 16px; align-items: start; }
    .panel { border: 1px solid var(--line); background: linear-gradient(145deg, #101a2a, var(--panel)); box-shadow: 0 18px 50px #00000022; }
    .feed-panel { padding: 18px; }
    .panel-head { display: flex; justify-content: space-between; align-items: start; gap: 18px; margin-bottom: 14px; }
    .section-label { color: var(--faint); }
    h2 { margin: 7px 0 0; font-size: 1.12rem; letter-spacing: -.02em; }
    .frame-spec { color: var(--cyan); white-space: nowrap; padding-top: 3px; }
    .feed-stage { position: relative; overflow: hidden; aspect-ratio: 1; border: 1px solid var(--line-hi); background: #090f1a; }
    .feed-stage::after { content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .10;
      background: repeating-linear-gradient(0deg, transparent 0 5px, #a9c5e90d 6px); }
    #frame { display: block; width: 100%; height: 100%; image-rendering: pixelated; object-fit: contain; }
    .corner { position: absolute; z-index: 1; width: 18px; height: 18px; border-color: var(--mint); border-style: solid; opacity: .9; }
    .corner.tl { top: 12px; left: 12px; border-width: 2px 0 0 2px; }
    .corner.tr { top: 12px; right: 12px; border-width: 2px 2px 0 0; }
    .corner.bl { bottom: 12px; left: 12px; border-width: 0 0 2px 2px; }
    .corner.br { bottom: 12px; right: 12px; border-width: 0 2px 2px 0; }
    .feed-badge { position: absolute; z-index: 2; left: 14px; bottom: 13px; padding: 6px 8px;
      background: #07101ddd; border: 1px solid #3a4d70; color: var(--text);
      font: 700 10px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .09em; text-transform: uppercase; }
    .feed-meta { display: flex; justify-content: space-between; gap: 12px; padding: 13px 2px 16px; color: var(--muted); font-size: 12px; }
    .feed-meta strong { color: var(--text); font-weight: 600; }
    .scene-map { display: flex; align-items: center; flex-wrap: wrap; gap: 7px; padding: 11px 0 14px; border-top: 1px solid var(--line); }
    .scene-objects { display: flex; flex-wrap: wrap; gap: 6px; }
    .scene-empty { color: var(--faint); font-size: 11px; }
    .object-chip { padding: 4px 6px; border: 1px solid var(--line-hi); color: var(--text); background: #152033; font: 700 10px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .03em; }
    .object-chip.danger { border-color: #a94f58; color: #ffb0a9; }
    .object-chip.jump { border-color: #ad8746; color: #ffd68a; }
    .object-chip.roll { border-color: #4e83ae; color: #a8ddff; }
    .object-chip.train { border-color: #7f8794; color: #e1e5ea; }
    .object-chip.pickup { border-color: #9a7c3d; color: #ffe59c; }
    .object-chip.route { border-color: #4b9b80; color: #9ef0c6; }
    .legend { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding-top: 14px; border-top: 1px solid var(--line); }
    .legend-item { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 11px; }
    .swatch { width: 9px; height: 9px; flex: 0 0 auto; }
    .swatch.mint { background: var(--mint); } .swatch.amber { background: var(--amber); }
    .swatch.cyan { background: var(--cyan); } .swatch.coral { background: var(--coral); }
    .swatch.violet { background: var(--violet); } .swatch.train { background: #d4d9e0; }
    .swatch.gap { background: #111722; border: 1px solid #6c7890; } .swatch.pickup { background: #f7d342; }
    .side-rail { display: grid; gap: 16px; }
    .decision-card, .metrics-panel, .controls-panel { padding: 17px; }
    .decision-top, .metric-head, .model-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .decision-top { color: var(--muted); font-size: 12px; }
    #step { color: var(--cyan); font: 700 11px ui-monospace, SFMono-Regular, Menlo, monospace; letter-spacing: .1em; }
    .action-display { display: flex; align-items: center; gap: 15px; padding: 22px 0 18px; }
    .action-glyph { display: grid; place-items: center; width: 58px; height: 58px; color: var(--ink); background: var(--mint);
      font: 700 29px/1 ui-monospace, SFMono-Regular, Menlo, monospace; }
    .action-label { font-size: 1.65rem; font-weight: 750; letter-spacing: -.045em; }
    .action-detail { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .reward-readout { display: flex; align-items: center; justify-content: space-between; padding-top: 13px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }
    #reward { color: var(--mint); font: 700 1.05rem ui-monospace, SFMono-Regular, Menlo, monospace; }
    .metrics-panel { background: var(--panel); }
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; margin: 14px 0; border: 1px solid var(--line); background: var(--line); }
    .metric { min-width: 0; padding: 12px; background: var(--panel); }
    .metric span { display: block; color: var(--muted); font-size: 11px; }
    .metric strong { display: block; margin-top: 5px; color: var(--text); font: 700 1.05rem ui-monospace, SFMono-Regular, Menlo, monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .event-line { display: flex; align-items: center; gap: 9px; color: var(--muted); font-size: 12px; }
    .event-dot { background: var(--amber); box-shadow: 0 0 0 4px #f6bf621c; }
    .controls { display: grid; gap: 8px; margin-top: 13px; }
    button { display: flex; align-items: center; justify-content: space-between; width: 100%; padding: 11px 12px;
      border: 1px solid var(--line-hi); border-radius: 0; color: var(--text); background: #17243a; cursor: pointer; font: 600 13px ui-sans-serif, system-ui, sans-serif; text-align: left; }
    button:hover { border-color: var(--cyan); background: #1b2d49; }
    button:focus-visible { outline: 2px solid var(--mint); outline-offset: 2px; }
    button.primary { color: var(--ink); border-color: var(--mint); background: var(--mint); }
    button.primary:hover { background: #86f0c5; }
    kbd { color: currentColor; opacity: .65; letter-spacing: .06em; }
    .hint { margin: 12px 0 0; color: var(--faint); font-size: 11px; }
    .model-card { padding: 14px 17px; border: 1px solid var(--line); background: #0b111c; }
    .model-name { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); letter-spacing: .02em; text-transform: none; font-size: 11px; }
    .footer-note { margin-top: 17px; color: var(--faint); text-align: right; font-size: 11px; }
    @media (max-width: 820px) {
      .shell { width: min(100% - 24px, 620px); padding-top: 24px; }
      .topbar { align-items: start; flex-direction: column; gap: 16px; }
      .session-state { justify-items: start; }
      .workspace { grid-template-columns: 1fr; }
      .legend { grid-template-columns: repeat(2, 1fr); }
      .footer-note { text-align: left; }
    }
    @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; transition: none !important; } }
  </style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div>
      <div class="eyebrow"><span class="brand-mark">FB</span> FLY-BRAIN / RUN CONTROL</div>
      <h1>Live perception lab</h1>
      <p class="lede">Watch the policy read a frame, choose an action, and survive the next piece of track.</p>
    </div>
    <div class="session-state">
      <div class="session-pill"><span id="status-dot" class="status-dot"></span><strong id="status">CONNECTING</strong></div>
      <span>LOCAL SIMULATION · NO PHONE INPUT</span>
    </div>
  </header>

  <main class="workspace">
    <section class="panel feed-panel" aria-labelledby="feed-heading">
      <div class="panel-head">
        <div><div class="section-label">01 / PIXEL FEED</div><h2 id="feed-heading">What the fly sees</h2></div>
        <div class="frame-spec">84 × 84 RGB</div>
      </div>
      <div class="feed-stage">
        <img id="frame" alt="Current runner frame">
        <span class="corner tl"></span><span class="corner tr"></span><span class="corner bl"></span><span class="corner br"></span>
        <div id="feed-badge" class="feed-badge">WAITING FOR FRAME</div>
      </div>
      <div class="feed-meta"><span>event <strong id="feed-event">waiting for first decision</strong></span><span id="view">rich full pixels</span></div>
      <div class="scene-map"><span class="section-label">SCENE MAP</span><div id="scene-objects" class="scene-objects"><span class="scene-empty">waiting for frame</span></div></div>
      <div class="legend" aria-label="Color legend">
        <div class="legend-item"><span class="swatch mint"></span>player / safe route</div>
        <div class="legend-item"><span class="swatch amber"></span>jump barrier</div>
        <div class="legend-item"><span class="swatch cyan"></span>roll / board</div>
        <div class="legend-item"><span class="swatch train"></span>train</div>
        <div class="legend-item"><span class="swatch violet"></span>either action</div>
        <div class="legend-item"><span class="swatch gap"></span>track gap</div>
        <div class="legend-item"><span class="swatch pickup"></span>pickup</div>
        <div class="legend-item"><span class="swatch coral"></span>solid danger</div>
      </div>
    </section>

    <aside class="side-rail">
      <section class="panel decision-card" aria-live="polite">
        <div class="decision-top"><span>LATEST POLICY OUTPUT</span><span id="step">STEP 000</span></div>
        <div class="action-display"><span id="action-glyph" class="action-glyph">•</span><div><div id="action-label" class="action-label">NOOP</div><div id="action-detail" class="action-detail">Holding position</div></div></div>
        <div class="reward-readout"><span>immediate reward</span><strong id="reward">+0.00</strong></div>
      </section>

      <section class="panel metrics-panel">
        <div class="section-label">02 / RUN TELEMETRY</div>
        <div class="metrics">
          <div class="metric"><span>Score</span><strong id="score">0</strong></div>
          <div class="metric"><span>Lane</span><strong id="lane">2 / 3</strong></div>
          <div class="metric"><span>Speed</span><strong id="speed">1.0×</strong></div>
          <div class="metric"><span>Motion</span><strong id="motion">ground</strong></div>
        </div>
        <div class="event-line"><span class="event-dot"></span><span id="event-detail">Reset episode</span></div>
      </section>

      <section class="panel controls-panel">
        <div class="section-label">03 / CONTROL THE RUN</div>
        <div class="controls">
          <button id="toggle" class="primary" onclick="command('/toggle')"><span>Pause run</span><kbd>SPACE</kbd></button>
          <button onclick="command('/step')"><span>Step once</span><kbd>ENTER</kbd></button>
          <button onclick="command('/reset')"><span>Reset episode</span><kbd>R</kbd></button>
        </div>
        <p class="hint">Playback advances automatically. Pause it to inspect a decision frame by frame.</p>
      </section>

      <div class="model-card"><div class="model-row"><span class="section-label">MODEL</span><span id="model" class="model-name">loading checkpoint</span></div></div>
    </aside>
  </main>
  <div class="footer-note">__VIEW_DESCRIPTION__ · recurrent state resets at episode boundaries</div>
</div>
<script>
const interval = __INTERVAL__;
const actionMeta = {
  noop: ['•', 'NOOP', 'Holding position'],
  left: ['←', 'LEFT', 'Shift one lane left'],
  right: ['→', 'RIGHT', 'Shift one lane right'],
  jump: ['↑', 'JUMP', 'Clear a low obstacle'],
  roll: ['↓', 'ROLL', 'Slide under an overhead obstacle'],
  hoverboard: ['◇', 'BOARD', 'Spend one hoverboard charge']
};
const sceneMeta = {
  solid: ['solid blocker', 'danger'], jump: ['jump barrier', 'jump'], roll: ['roll bar', 'roll'],
  either: ['jump or roll', 'roll'], gap: ['track gap', 'danger'], roof_route: ['roof route', 'route'],
  block: ['solid block', 'danger'], barrier: ['jump barrier', 'jump'], roll_bar: ['roll bar', 'roll'],
  either_hurdle: ['jump or roll', 'roll'], train: ['train', 'train'], bus: ['bus', 'train'],
  tunnel: ['low tunnel', 'roll'], ramp: ['roof ramp', 'route'],
  coin: ['coin', 'pickup'], key: ['key', 'pickup'], jetpack: ['jetpack', 'pickup'],
  super_sneakers: ['super sneakers', 'pickup'], coin_magnet: ['coin magnet', 'pickup'],
  multiplier: ['multiplier', 'pickup'], pogo: ['pogo', 'pickup'], speed_pad: ['speed pad', 'pickup'],
  mystery_box: ['mystery box', 'pickup']
};
const $ = (id) => document.getElementById(id);
function set(id, value) { $(id).textContent = value; }
function renderScene(objects) {
  const target = $('scene-objects');
  target.replaceChildren();
  if (!objects || !objects.length) {
    const empty = document.createElement('span'); empty.className = 'scene-empty'; empty.textContent = 'clear track'; target.append(empty); return;
  }
  objects.slice(0, 8).forEach((object) => {
    const visual = object.appearance || object.kind;
    const meta = visual === 'train' && Number(object.speed_scale) > 1
      ? ['oncoming train', 'danger']
      : (sceneMeta[visual] || sceneMeta[object.kind] || [object.kind, 'danger']);
    const chip = document.createElement('span'); chip.className = 'object-chip ' + meta[1];
    const behavior = visual !== object.kind ? ` · ${object.kind}` : '';
    chip.title = `${object.kind} behavior · ${visual} appearance`;
    chip.textContent = `L${object.lane} · ${meta[0]}${behavior} · ${Number(object.distance).toFixed(1)}m`;
    target.append(chip);
  });
}
async function command(path) {
  await fetch(path, {method: 'POST', cache: 'no-store'});
  refresh(false);
}
async function refresh(advance) {
  try {
    const response = await fetch('/state?advance=' + (advance ? '1' : '0') + '&t=' + Date.now(), {cache: 'no-store'});
    const state = await response.json();
    const meta = actionMeta[state.action_name] || actionMeta.noop;
    $('frame').src = 'data:image/png;base64,' + state.frame;
    set('status', state.status.toUpperCase());
    set('step', 'STEP ' + String(state.step).padStart(3, '0'));
    set('action-glyph', meta[0]); set('action-label', meta[1]); set('action-detail', meta[2]);
    set('reward', (state.reward >= 0 ? '+' : '') + state.reward.toFixed(2));
    set('score', state.score.toFixed ? state.score.toFixed(1) : state.score);
    set('lane', (state.lane + 1) + ' / 3'); set('speed', state.speed.toFixed(1) + '×'); set('motion', state.motion);
    set('feed-event', state.event || 'survived'); set('event-detail', state.event || 'survived');
    set('view', state.view); set('model', state.model); set('feed-badge', state.status === 'finished' ? 'EPISODE FINISHED' : 'LIVE PIXEL FEED');
    renderScene(state.objects);
    $('status-dot').style.background = state.status === 'finished' ? 'var(--coral)' : state.status === 'paused' ? 'var(--amber)' : 'var(--mint)';
    $('toggle').querySelector('span').textContent = state.status === 'paused' ? 'Resume run' : 'Pause run';
  } catch (error) {
    set('status', 'DISCONNECTED'); set('feed-badge', 'VIEWER OFFLINE');
    $('status-dot').style.background = 'var(--coral)';
  }
  window.setTimeout(() => refresh(true), interval);
}
window.addEventListener('keydown', (event) => {
  if (event.target.tagName === 'BUTTON') return;
  if (event.code === 'Space') { event.preventDefault(); command('/toggle'); }
  if (event.code === 'Enter') { event.preventDefault(); command('/step'); }
  if (event.key.toLowerCase() === 'r') command('/reset');
});
refresh(false);
</script>
</body>
</html>"""


def encode_png(frame: np.ndarray) -> bytes:
    """Encode a small RGB frame without adding an image dependency."""

    pixels = np.asarray(frame, dtype=np.uint8)
    if pixels.shape != (84, 84, 3):
        raise ValueError("expected an 84x84 RGB frame")
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(84))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 84, 84, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(
        b"IDAT", zlib.compress(raw, level=1)
    ) + chunk(b"IEND", b"")


class Controller:
    def __init__(
        self,
        model_path: Path,
        seed: int,
        full_view: bool,
        rich: bool = False,
        checkpoint_dir: Path | None = None,
        visual_teacher: bool = False,
        near_field: bool = False,
        fly_cns_retina: bool = False,
    ) -> None:
        if visual_teacher and not rich:
            raise ValueError("visual_teacher requires rich=True")
        if fly_cns_retina and (not rich or not full_view):
            raise ValueError("fly_cns_retina requires rich=True and full_view=True")
        if fly_cns_retina and visual_teacher:
            raise ValueError("choose one visual model")
        if rich:
            env_class = RichPixelGymRunnerEnv if full_view else RichPartialPixelGymRunnerEnv
        else:
            env_class = PixelGymRunnerEnv if full_view else PartialPixelGymRunnerEnv
        self.env = env_class(max_steps=300, spawn_interval=8)
        self.visual_teacher = visual_teacher
        self.near_field = near_field
        self.fly_cns_retina = fly_cns_retina
        if visual_teacher:
            self.model = VisualPolicy(input_channels=6 if near_field else 3)
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            self.model.eval()
        elif fly_cns_retina:
            self.model = FlyCNSRetinaPolicy(self.env.observation_space)
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu", weights_only=True)
            )
            self.model.eval()
        else:
            self.model = RecurrentPPO.load(model_path, device="cpu")
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.action_map = RICH_ACTIONS if rich else ACTIONS
        self.seed = seed
        self.view = ("rich " if rich else "") + ("full pixels" if full_view else "partial pixels")
        if near_field:
            self.view += " + near-field"
        if fly_cns_retina:
            self.view += " + learned retina + MaleCNS graph"
        self.loaded_checkpoint = model_path.name
        self.lock = threading.Lock()
        self._reset_locked()

    def _reload_latest_locked(self) -> None:
        if self.visual_teacher or self.fly_cns_retina or self.checkpoint_dir is None:
            return
        checkpoints = sorted(self.checkpoint_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime)
        if not checkpoints:
            return
        latest = checkpoints[-1]
        if latest.name == self.loaded_checkpoint:
            return
        # Do not read a checkpoint while the trainer is still writing it.
        if time.time() - latest.stat().st_mtime < 0.5:
            return
        try:
            model = RecurrentPPO.load(latest, device="cpu")
        except (OSError, EOFError, ValueError, RuntimeError):
            return
        self.model = model
        self.model_path = latest
        self.loaded_checkpoint = latest.name
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.observation, _ = self.env.reset(seed=self.seed)
        self.lstm_state = None
        self.episode_start = np.ones((1,), dtype=bool)
        self.running = True
        self.finished = False
        self.step = 0
        self.score = 0
        self.action = 0
        self.reward = 0.0
        self.last_event = "reset"

    def reset(self) -> None:
        with self.lock:
            self._reset_locked()

    def toggle(self) -> None:
        with self.lock:
            if not self.finished:
                self.running = not self.running

    def advance(self) -> None:
        with self.lock:
            self._advance_locked()

    def _advance_locked(self) -> None:
        if self.finished or not self.running:
            return
        self._reload_latest_locked()
        if self.visual_teacher or self.fly_cns_retina:
            with torch.no_grad():
                observation = (
                    near_field_view(self.observation)
                    if self.fly_cns_retina
                    else (
                        near_field_view(self.observation)
                        if self.near_field
                        else self.observation
                    )
                )
                logits = self.model(
                    torch.from_numpy(observation)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                )
            self.action = int(logits.argmax(1).item())
        else:
            action, self.lstm_state = self.model.predict(
                self.observation[None, ...],
                state=self.lstm_state,
                episode_start=self.episode_start,
                deterministic=True,
            )
            self.action = int(np.asarray(action).reshape(-1)[0])
        self.observation, self.reward, terminated, truncated, info = self.env.step(
            self.action
        )
        self.step += 1
        self.score = float(info["score"])
        self.last_event = str(info.get("event", "survived"))
        self.episode_start[:] = False
        if terminated or truncated:
            self.finished = True
            self.running = False

    def state(self, advance: bool = False) -> dict[str, object]:
        with self.lock:
            if advance:
                self._advance_locked()
            if self.finished:
                status = "finished"
            else:
                status = "running" if self.running else "paused"
            core = getattr(self.env, "core", None)
            objects = []
            if core is not None:
                objects = [
                    {
                        "lane": item.lane + 1,
                        "kind": item.kind,
                        "distance": round(item.distance, 2),
                        **(
                            {
                                "appearance": item.appearance,
                                "speed_scale": round(item.speed_scale, 2),
                                "roofable": item.roofable,
                            }
                            if hasattr(item, "appearance")
                            else {}
                        ),
                    }
                    for item in [*core._obstacles, *core._pickups]
                    if 0 < item.distance <= core.spawn_distance + 1.0
                ]
                objects.sort(key=lambda item: item["distance"])
            return {
                "frame": base64.b64encode(encode_png(self.observation)).decode("ascii"),
                "status": status,
                "step": self.step,
                "score": self.score,
                "action": self.action,
                "action_name": self.action_map[self.action],
                "reward": float(self.reward),
                "event": self.last_event,
                "lane": int(getattr(core, "player_lane", 1)),
                "motion": str(getattr(core, "motion", "ground")),
                "speed": float(getattr(core, "speed", 1.0)),
                "objects": objects,
                "view": self.view,
                "model": self.loaded_checkpoint,
            }

    def close(self) -> None:
        self.env.close()


def make_handler(controller: Controller, interval_ms: int, view_description: str):
    page = HTML.replace("__INTERVAL__", str(interval_ms)).replace(
        "__VIEW_DESCRIPTION__", view_description
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def _send(self, content_type: str, body: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send("text/html; charset=utf-8", page.encode())
                return
            if path == "/state":
                advance = "advance=1" in self.path
                body = json.dumps(controller.state(advance)).encode()
                self._send("application/json; charset=utf-8", body)
                return
            self._send("text/plain; charset=utf-8", b"not found\n", 404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path == "/toggle":
                controller.toggle()
            elif path == "/step":
                controller.advance()
            elif path == "/reset":
                controller.reset()
            else:
                self._send("text/plain; charset=utf-8", b"not found\n", 404)
                return
            self._send("application/json; charset=utf-8", b"{\"ok\":true}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "results/male_cns_neuron_plastic_50k/"
            "male_cns_neuron_plastic_recurrent_ppo.zip"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--full-view", action="store_true")
    parser.add_argument("--rich", action="store_true", help="use the rich Subway-style environment")
    parser.add_argument(
        "--visual-teacher",
        action="store_true",
        help="load the bootstrapped pixel CNN instead of a recurrent PPO zip",
    )
    parser.add_argument(
        "--near-field",
        action="store_true",
        help="use the six-channel visual checkpoint with a near-track zoom",
    )
    parser.add_argument(
        "--fly-cns-retina",
        action="store_true",
        help="load the learned-retina MaleCNS graph action policy",
    )
    parser.add_argument(
        "--watch-checkpoint-dir",
        type=Path,
        help="reload the newest checkpoint from this directory while the viewer runs",
    )
    parser.add_argument("--open", action="store_true", help="open the viewer in the default browser")
    args = parser.parse_args()
    if args.fps <= 0 or not 1 <= args.port <= 65535:
        raise ValueError("fps must be positive and port must be between 1 and 65535")
    if args.visual_teacher and not args.rich:
        parser.error("--visual-teacher requires --rich")
    if args.near_field and not args.visual_teacher:
        parser.error("--near-field requires --visual-teacher")
    if args.fly_cns_retina and (not args.rich or not args.full_view):
        parser.error("--fly-cns-retina requires --rich --full-view")
    if args.fly_cns_retina and args.visual_teacher:
        parser.error("choose one of --visual-teacher or --fly-cns-retina")

    controller = Controller(
        args.model,
        args.seed,
        args.full_view,
        rich=args.rich,
        checkpoint_dir=args.watch_checkpoint_dir,
        visual_teacher=args.visual_teacher,
        near_field=args.near_field,
        fly_cns_retina=args.fly_cns_retina,
    )
    server = ThreadingHTTPServer(
        (
            args.host,
            args.port,
        ),
        make_handler(
            controller,
            round(1000 / args.fps),
            "Rich Subway-style mechanics environment" if args.rich else "Toy runner environment",
        ),
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"Fly-Brain Runner viewer: {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        controller.close()


if __name__ == "__main__":
    main()
