"""Check the ADB bridge without requiring a connected phone."""

from __future__ import annotations

from subprocess import CompletedProcess

from android_bridge import AndroidBridge, PNG_SIGNATURE, parse_crop


def main() -> None:
    calls: list[list[str]] = []
    responses = [b"Physical size: 1080x1920\n", PNG_SIGNATURE + b"test"]

    def fake_runner(command, **_kwargs):
        calls.append(command)
        stdout = responses.pop(0) if responses else b""
        return CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    bridge = AndroidBridge(serial="demo", runner=fake_runner)
    assert bridge.screen_size() == (1080, 1920)
    assert bridge.screenshot_png().startswith(PNG_SIGNATURE)
    bridge.send_action(0, width=1080, height=1920)
    bridge.send_action(1, width=1080, height=1920)
    assert calls[-1] == [
        "adb",
        "-s",
        "demo",
        "shell",
        "input",
        "swipe",
        "540",
        "1382",
        "303",
        "1382",
        "120",
    ]
    assert parse_crop("10,20,800,1400") == (10, 20, 800, 1400)
    bridge.tap(10, 20)
    bridge.keyevent("KEYCODE_BACK")
    bridge.force_stop("com.kiloo.subwaysurf")
    bridge.launch("com.kiloo.subwaysurf/com.example.MainActivity")
    assert calls[-4:] == [
        ["adb", "-s", "demo", "shell", "input", "tap", "10", "20"],
        ["adb", "-s", "demo", "shell", "input", "keyevent", "KEYCODE_BACK"],
        ["adb", "-s", "demo", "shell", "am", "force-stop", "com.kiloo.subwaysurf"],
        [
            "adb",
            "-s",
            "demo",
            "shell",
            "am",
            "start",
            "-n",
            "com.kiloo.subwaysurf/com.example.MainActivity",
        ],
    ]
    print("Android bridge check passed")


if __name__ == "__main__":
    main()
