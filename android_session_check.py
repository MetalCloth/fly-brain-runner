"""Assert-based checks for the supervised Android session gate."""

from __future__ import annotations

import numpy as np

from run_android_session import ScreenState, classify_screen, screen_signature


def main() -> None:
    menu = np.full((84, 84, 3), (70, 110, 180), dtype=np.uint8)
    menu[::3, :, :] = (110, 150, 210)
    near_menu = menu.copy()
    near_menu[10:20, 10:20] = (80, 120, 190)
    active = np.full((84, 84, 3), (160, 100, 50), dtype=np.uint8)
    blocked = np.full((84, 84, 3), 35, dtype=np.uint8)

    menu_signature = screen_signature(menu)
    assert classify_screen(near_menu, menu_signature) is ScreenState.MENU
    animated_menu = menu.copy()
    animated_menu[:60, :, :] = 220
    assert classify_screen(animated_menu, menu_signature) is ScreenState.MENU
    assert classify_screen(active, menu_signature) is ScreenState.ACTIVE
    assert classify_screen(blocked, menu_signature) is ScreenState.BLOCKED
    result = np.full((84, 84, 3), 235, dtype=np.uint8)
    assert classify_screen(result, menu_signature) is ScreenState.BLOCKED
    print("Android session gate check passed")


if __name__ == "__main__":
    main()
