"""Small, explicit ADB bridge for screen capture and runner gestures."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence


ACTIONS = {0: "noop", 1: "left", 2: "right", 3: "jump", 4: "roll"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class AndroidBridgeError(RuntimeError):
    """Raised when ADB cannot perform the requested operation."""


class AndroidBridge:
    """Use ADB without hiding device or gesture assumptions."""

    def __init__(
        self,
        serial: str | None = None,
        adb: str = "adb",
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
        timeout: float = 10.0,
    ) -> None:
        self.serial = serial
        self.adb = adb
        self._runner = runner
        self.timeout = timeout

    def _command(self, *args: str) -> list[str]:
        command = [self.adb]
        if self.serial:
            command.extend(("-s", self.serial))
        command.extend(args)
        return command

    def _run(self, *args: str, input_bytes: bytes | None = None) -> bytes:
        result = self._runner(
            self._command(*args),
            input=input_bytes,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode:
            detail = result.stderr.decode(errors="replace").strip()
            raise AndroidBridgeError(
                f"ADB command failed ({result.returncode}): {' '.join(self._command(*args))}"
                + (f"\n{detail}" if detail else "")
            )
        return result.stdout

    def screen_size(self) -> tuple[int, int]:
        """Return the physical display size reported by Android."""

        output = self._run("shell", "wm", "size").decode(errors="replace")
        matches = re.findall(r"(\d+)x(\d+)", output)
        if not matches:
            raise AndroidBridgeError(f"could not parse display size from: {output!r}")
        width, height = (int(value) for value in matches[-1])
        if width < 1 or height < 1:
            raise AndroidBridgeError(f"invalid display size: {width}x{height}")
        return width, height

    def screenshot_png(self) -> bytes:
        """Capture one PNG frame from the device."""

        frame = self._run("exec-out", "screencap", "-p")
        if not frame.startswith(PNG_SIGNATURE):
            raise AndroidBridgeError("ADB returned a non-PNG screenshot")
        return frame

    def tap(self, x: int, y: int) -> None:
        """Tap one display coordinate."""

        if x < 0 or y < 0:
            raise ValueError("tap coordinates must be non-negative")
        self._run("shell", "input", "tap", str(x), str(y))

    def keyevent(self, key: str) -> None:
        """Send one Android key event, such as ``KEYCODE_BACK``."""

        if not key.strip():
            raise ValueError("key event must not be empty")
        self._run("shell", "input", "keyevent", key)

    def force_stop(self, package: str) -> None:
        """Stop an app without clearing its data."""

        if not package.strip():
            raise ValueError("package must not be empty")
        self._run("shell", "am", "force-stop", package)

    def launch(self, component: str) -> None:
        """Launch an explicit Android activity component."""

        if not component.strip() or "/" not in component:
            raise ValueError("component must be package/activity")
        self._run("shell", "am", "start", "-n", component)

    def send_action(
        self,
        action: int,
        *,
        width: int,
        height: int,
        horizontal_fraction: float = 0.22,
        vertical_fraction: float = 0.18,
        start_y_fraction: float = 0.72,
        duration_ms: int = 120,
    ) -> None:
        """Send one Subway-Surfers-style swipe; action 0 intentionally does nothing."""

        if action not in ACTIONS:
            raise ValueError(f"unknown action {action}; expected one of {sorted(ACTIONS)}")
        if width < 1 or height < 1:
            raise ValueError("screen dimensions must be positive")
        if not 0 < horizontal_fraction < 0.5:
            raise ValueError("horizontal_fraction must be between 0 and 0.5")
        if not 0 < vertical_fraction < 0.5:
            raise ValueError("vertical_fraction must be between 0 and 0.5")
        if not 0 < start_y_fraction < 1:
            raise ValueError("start_y_fraction must be between 0 and 1")
        if start_y_fraction - vertical_fraction < 0 or start_y_fraction + vertical_fraction > 1:
            raise ValueError("vertical swipe would leave the display")
        if duration_ms < 1:
            raise ValueError("duration_ms must be positive")
        if action == 0:
            return

        x = width // 2
        y = int(height * start_y_fraction)
        dx = int(width * horizontal_fraction)
        dy = int(height * vertical_fraction)
        deltas = {1: (-dx, 0), 2: (dx, 0), 3: (0, -dy), 4: (0, dy)}
        delta_x, delta_y = deltas[action]
        end_x, end_y = x + delta_x, y + delta_y
        self._run(
            "shell",
            "input",
            "swipe",
            str(x),
            str(y),
            str(end_x),
            str(end_y),
            str(duration_ms),
        )


def parse_crop(value: str) -> tuple[int, int, int, int]:
    """Parse ``left,top,width,height`` for the game viewport."""

    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 4 or parts[2] < 1 or parts[3] < 1 or parts[0] < 0 or parts[1] < 0:
        raise ValueError("crop must be left,top,width,height with non-negative origin")
    return parts


def command_text(command: Sequence[str]) -> str:
    """Expose a safe command string for diagnostics and tests."""

    return " ".join(command)
