"""Check the live viewer's frame encoder and one model step."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from play_model import Controller, encode_png


def main() -> None:
    frame = np.zeros((84, 84, 3), dtype=np.uint8)
    frame[20:40, 20:40] = (255, 80, 80)
    png = encode_png(frame)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        input=png,
        capture_output=True,
        check=False,
    )
    assert decoded.returncode == 0
    assert len(decoded.stdout) == 84 * 84 * 3

    controller = Controller(
        Path("results/male_cns_neuron_plastic_50k/male_cns_neuron_plastic_recurrent_ppo.zip"),
        seed=0,
        full_view=False,
    )
    try:
        before = controller.state()
        controller.advance()
        after = controller.state()
        assert before["step"] == 0
        assert after["step"] == 1
        assert after["action"] in range(5)
    finally:
        controller.close()
    visual_controller = Controller(
        Path("results/visual_nearfield_v1/nearfield_visual_teacher_best.pt"),
        seed=0,
        full_view=True,
        rich=True,
        visual_teacher=True,
        near_field=True,
    )
    try:
        visual_controller.advance()
        assert visual_controller.state()["action"] in range(6)
        assert "near-field" in visual_controller.state()["view"]
    finally:
        visual_controller.close()
    brain_controller = Controller(
        Path(
            "results/male_cns_neuron_rich_plastic_50k/"
            "male_cns_neuron_plastic_full_rich_recurrent_ppo.zip"
        ),
        seed=0,
        full_view=True,
        rich=True,
    )
    try:
        brain_controller.advance()
        brain_state = brain_controller.state()
        assert brain_state["action"] in range(6)
        assert brain_state["view"] == "rich full pixels"
    finally:
        brain_controller.close()
    retina_controller = Controller(
        Path("results/fly_cns_retina_v1/fly_cns_retina_best.pt"),
        seed=0,
        full_view=True,
        rich=True,
        fly_cns_retina=True,
    )
    try:
        retina_controller.advance()
        retina_state = retina_controller.state()
        assert retina_state["action"] in range(6)
        assert "learned retina" in retina_state["view"]
    finally:
        retina_controller.close()
    print("Live viewer check passed")


if __name__ == "__main__":
    main()
