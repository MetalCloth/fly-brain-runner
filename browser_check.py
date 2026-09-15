"""Check the browser pipeline tensor and action contracts."""

import numpy as np
import torch

from browser_cdp import ACTION_KEYS, KEY_ACTIONS, parse_clip
from browser_model import BrowserPolicy
from browser_pipeline import ACTIONS, parse_region, resize_nearest


if __name__ == "__main__":
    assert parse_region("271,130,1031,581") == (271, 130, 1031, 581)
    assert parse_clip("10,20,300,180") == {
        "x": 10.0,
        "y": 20.0,
        "width": 300.0,
        "height": 180.0,
    }
    assert KEY_ACTIONS == {
        "ArrowLeft": 1,
        "ArrowRight": 2,
        "ArrowUp": 3,
        "ArrowDown": 4,
    }
    assert ACTION_KEYS[3] == "ArrowUp"
    image = np.zeros((581, 1031, 3), dtype=np.uint8)
    small = resize_nearest(image)
    assert small.shape == (84, 84, 3)
    for history in (1, 4):
        model = BrowserPolicy(history=history)
        frames = torch.zeros((2, history * 3, 84, 84), dtype=torch.float32)
        logits = model(frames)
        assert tuple(logits.shape) == (2, len(ACTIONS))
    print("browser pipeline check passed")
