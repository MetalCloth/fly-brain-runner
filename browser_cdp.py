"""Small Chrome DevTools Protocol bridge for the Poki browser game."""

from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from collections.abc import Callable
from pathlib import Path
from typing import Any


DEFAULT_URL = "https://poki.com/en/g/subway-surfers"
KEY_ACTIONS = {"ArrowLeft": 1, "ArrowRight": 2, "ArrowUp": 3, "ArrowDown": 4}
ACTION_KEYS = {value: key for key, value in KEY_ACTIONS.items()}


class BrowserCdpError(RuntimeError):
    """Raised when the controlled browser cannot be reached."""


def parse_clip(value: str) -> dict[str, float]:
    """Parse a CDP page clip as ``x,y,width,height``."""

    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise ValueError("clip must be x,y,width,height") from error
    if len(parts) != 4 or parts[0] < 0 or parts[1] < 0 or parts[2] < 1 or parts[3] < 1:
        raise ValueError("clip must be x,y,width,height with a positive size")
    return {"x": parts[0], "y": parts[1], "width": parts[2], "height": parts[3]}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _targets(port: int) -> list[dict[str, Any]]:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/json/list", timeout=0.5
        ) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return []
    return [target for target in value if isinstance(target, dict)]


def _find_browser() -> str:
    for candidate in ("brave", "brave-browser", "chromium", "google-chrome"):
        path = shutil.which(candidate)
        if path:
            return path
    raise BrowserCdpError("could not find brave/chromium on PATH")


def _page_target(
    targets: list[dict[str, Any]], url: str
) -> dict[str, Any] | None:
    matching = next(
        (
            item
            for item in targets
            if item.get("type") == "page"
            and item.get("webSocketDebuggerUrl")
            and urlparse(str(item.get("url", ""))).netloc == urlparse(url).netloc
        ),
        None,
    )
    if matching is not None:
        return matching
    return next(
        (
            item
            for item in targets
            if item.get("type") == "page" and item.get("webSocketDebuggerUrl")
        ),
        None,
    )


ACTION_LISTENER_SOURCE = r"""
(() => {
  if (window.__flyActionListenerInstalled) return;
  window.__flyActionListenerInstalled = true;
  const keys = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Escape']);
  const publish = (type, event) => {
    if (!keys.has(event.key) || typeof window.flyAction !== 'function') return;
    try {
      window.flyAction(JSON.stringify({
        type,
        key: event.key,
        repeat: Boolean(event.repeat),
        at: performance.now()
      }));
    } catch (_) {}
  };
  window.addEventListener('keydown', event => publish('down', event), true);
  window.addEventListener('keyup', event => publish('up', event), true);
})();
"""


class BrowserPage:
    """Connect to one visible Chromium page through CDP."""

    def __init__(
        self,
        websocket_url: str,
        *,
        port: int,
        browser_process: subprocess.Popen[bytes] | None = None,
    ) -> None:
        try:
            import websocket
        except ImportError as error:
            raise BrowserCdpError(
                "browser control needs websocket-client; run "
                "./.venv/bin/python -m pip install -r requirements.txt"
            ) from error
        try:
            self._socket = websocket.create_connection(
                websocket_url,
                timeout=10,
                http_proxy_host=None,
            )
        except Exception as error:
            raise BrowserCdpError(f"could not connect to browser target: {error}") from error
        self.port = port
        self.browser_process = browser_process
        self._next_id = 1
        self._event_handler: Callable[[str, dict[str, Any]], None] | None = None

    @classmethod
    def connect_or_launch(
        cls,
        url: str = DEFAULT_URL,
        *,
        port: int = 9222,
        profile_dir: Path = Path("/tmp/fly-brain-runner-browser-profile"),
    ) -> "BrowserPage":
        selected_port = port
        existing = _targets(selected_port)
        process: subprocess.Popen[bytes] | None = None
        if existing and any(target.get("webSocketDebuggerUrl") for target in existing):
            target = _page_target(existing, url)
            if target is not None:
                try:
                    return cls(
                        str(target["webSocketDebuggerUrl"]),
                        port=selected_port,
                    )
                except BrowserCdpError as error:
                    if "Handshake status 403" not in str(error):
                        raise
                    # A previous browser may have been started without the
                    # origin flag; leave it alone and use a fresh port/profile.
                    selected_port = _free_port()
                    profile_dir = Path(
                        tempfile.mkdtemp(prefix="fly-brain-runner-browser-")
                    )
                    existing = []
            else:
                selected_port = _free_port()
                existing = []
        elif existing:
            selected_port = _free_port()
            existing = []

        if not existing:
            browser = _find_browser()
            profile_dir.mkdir(parents=True, exist_ok=True)
            process = subprocess.Popen(
                [
                    browser,
                    f"--remote-debugging-port={selected_port}",
                    f"--remote-allow-origins=http://127.0.0.1:{selected_port}",
                    f"--user-data-dir={profile_dir}",
                    "--new-window",
                    url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                existing = _targets(selected_port)
                if any(
                    item.get("type") == "page"
                    and item.get("webSocketDebuggerUrl")
                    and urlparse(str(item.get("url", ""))).netloc
                    == urlparse(url).netloc
                    for item in existing
                ):
                    break
                time.sleep(0.25)
            if not _page_target(existing, url):
                process.kill()
                raise BrowserCdpError(
                    f"browser did not expose CDP on 127.0.0.1:{selected_port}"
                )

        target = _page_target(existing, url)
        if target is None:
            raise BrowserCdpError("no debuggable browser page was found")
        return cls(
            str(target["webSocketDebuggerUrl"]),
            port=selected_port,
            browser_process=process,
        )

    def set_event_handler(
        self, handler: Callable[[str, dict[str, Any]], None] | None
    ) -> None:
        self._event_handler = handler

    def _read_message(self) -> dict[str, Any]:
        raw = self._socket.recv()
        if raw is None:
            raise BrowserCdpError("browser websocket closed")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise BrowserCdpError("browser sent invalid CDP JSON") from error
        if not isinstance(value, dict):
            raise BrowserCdpError("browser sent an invalid CDP message")
        return value

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        message_id = self._next_id
        self._next_id += 1
        self._socket.send(
            json.dumps({"id": message_id, "method": method, "params": params or {}})
        )
        while True:
            message = self._read_message()
            if "method" in message:
                if self._event_handler is not None:
                    self._event_handler(
                        str(message["method"]),
                        dict(message.get("params", {})),
                    )
                continue
            if message.get("id") != message_id:
                continue
            if "error" in message:
                raise BrowserCdpError(f"{method} failed: {message['error']}")
            return dict(message.get("result", {}))

    def evaluate(self, expression: str) -> Any:
        result = self.send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        if "exceptionDetails" in result:
            raise BrowserCdpError(f"browser evaluation failed: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def prepare(self, url: str = DEFAULT_URL) -> None:
        self.send("Runtime.enable")
        self.send("Page.enable")
        try:
            self.send("Runtime.addBinding", {"name": "flyAction"})
        except BrowserCdpError as error:
            if "already exists" not in str(error):
                raise
        self.send("Page.addScriptToEvaluateOnNewDocument", {"source": ACTION_LISTENER_SOURCE})
        self.send("Page.bringToFront")
        self.send("Page.navigate", {"url": url})
        time.sleep(2.0)
        try:
            self.evaluate(ACTION_LISTENER_SOURCE)
        except BrowserCdpError:
            pass

    def viewport(self) -> tuple[int, int]:
        value = self.evaluate(
            "JSON.stringify({width: window.innerWidth, height: window.innerHeight})"
        )
        if isinstance(value, str):
            parsed = json.loads(value)
            return max(1, int(parsed["width"])), max(1, int(parsed["height"]))
        raise BrowserCdpError("could not determine browser viewport")

    def find_game_clip(self, timeout: float = 15.0) -> dict[str, float]:
        expression = r"""
        (() => {
          const candidates = [...document.querySelectorAll('iframe, canvas, video')];
          return candidates.map(element => {
            const rect = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return {
              x: rect.left, y: rect.top, width: rect.width, height: rect.height,
              area: rect.width * rect.height, tag: element.tagName,
              visible: style.display !== 'none' && style.visibility !== 'hidden'
            };
          }).filter(item => item.visible && item.width >= 300 && item.height >= 180)
            .sort((a, b) => b.area - a.area)[0] || null;
        })()
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.evaluate(expression)
            if isinstance(value, dict):
                return {
                    key: float(value[key])
                    for key in ("x", "y", "width", "height")
                }
            time.sleep(0.5)
        width, height = self.viewport()
        return {"x": 0.0, "y": 0.0, "width": float(width), "height": float(height)}

    def capture_png(self, clip: dict[str, float] | None = None) -> bytes:
        params: dict[str, Any] = {"format": "png", "fromSurface": True}
        if clip is not None:
            params["clip"] = {**clip, "scale": 1.0}
        result = self.send("Page.captureScreenshot", params)
        try:
            return base64.b64decode(str(result["data"]))
        except (KeyError, ValueError) as error:
            raise BrowserCdpError("browser returned an invalid screenshot") from error

    def dispatch_key(self, key: str) -> None:
        if key not in ACTION_KEYS.values():
            raise ValueError(f"unsupported browser key: {key}")
        self.send(
            "Input.dispatchKeyEvent",
            {"type": "keyDown", "key": key, "code": key},
        )
        self.send(
            "Input.dispatchKeyEvent",
            {"type": "keyUp", "key": key, "code": key},
        )

    def close(self) -> None:
        self._socket.close()
