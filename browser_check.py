"""Check the browser pipeline tensor and action contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from browser_cdp import ACTION_KEYS, KEY_ACTIONS, parse_clip
from browser_model import BrowserPolicy
from browser_pipeline import (
    ACTIONS,
    FRAME_SIZE,
    BrowserSample,
    parse_region,
    resize_frame,
)
from train_browser_bc import BrowserDataset


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
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    small = resize_frame(image)
    assert small.shape == (*FRAME_SIZE, 3)
    for history in (1, 4):
        model = BrowserPolicy(history=history)
        frames = torch.zeros((2, history * 3, *FRAME_SIZE), dtype=torch.float32)
        logits = model(frames)
        assert tuple(logits.shape) == (2, len(ACTIONS))
    with TemporaryDirectory() as temporary:
        frame_path = Path(temporary) / "frame.npy"
        frame = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
        frame[:, 0, :] = 255
        np.save(frame_path, frame, allow_pickle=False)
        sample = BrowserSample((frame_path,), 2, Path(temporary), 0, 0.0)
        dataset = BrowserDataset([sample], mirror=True)
        original, original_action = dataset[0]
        mirrored, mirrored_action = dataset[1]
        assert len(dataset) == 2
        assert original_action == 2 and mirrored_action == 1
        assert torch.equal(original[:, :, 0], mirrored[:, :, -1])
    print("browser pipeline check passed")
