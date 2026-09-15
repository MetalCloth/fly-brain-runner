# Android testing boundary

> **Current workflow:** screenshots and predictions only. This project snapshot
> does not send phone input. Do not use an `--execute` flag with a real phone.

The local controller now has a small ADB bridge, but the phone stage is
deliberately gated. The current checkpoint was trained on the 84x84 toy
renderer; it is not expected to understand Subway Surfers screenshots without
viewport calibration and visual adaptation.

## What the bridge does

`android_bridge.py` keeps device operations explicit:

- capture a PNG with `adb exec-out screencap -p`;
- translate the five local actions into no-op or swipe gestures;
- tap a coordinate, send a key event, force-stop an app, and launch an
  explicit activity for session setup and recovery.

Those input and lifecycle methods exist for future guarded work; the current
workflow uses only screenshot capture and watch-only inference.

`run_android_policy.py` decodes each screenshot with FFmpeg, crops and resizes
it to 84x84 RGB, runs the recurrent checkpoint, and prints the selected action.
It is a dry run unless `--execute` is present. No gesture is sent during the
dry run.

The selected fly checkpoint is a custom PyTorch .pt graph policy rather than
the recurrent PPO .zip used by the older runner. watch_android_fly.py is the
read-only bridge for the current model:

```bash
./.venv/bin/python watch_android_fly.py \
  results/fly_cns_retina_v1/fly_cns_retina_best.pt \
  --crop 0,0,1220,2712 --steps 10 \
  --log results/android_observation/fly_watch_only.jsonl
```

It has no input or execute mode. It captures a screenshot, creates the
six-channel near-field view expected by the fly model, prints the top action
and confidence, and appends JSONL telemetry. Main-menu predictions are only
pipeline checks, not evidence of real-game competence.

## Session wrapper (dry-run only in this snapshot)

`run_android_session.py` contains lifecycle and safety-gate code for future
bounded experiments. In this snapshot use it only to inspect the current
screen and write telemetry; it does not save PNGs unless `--save-frames` is
requested.

Dry-run against the current screen:

```bash
./.venv/bin/python run_android_session.py \
  results/male_cns_neuron_plastic_50k/male_cns_neuron_plastic_recurrent_ppo.zip \
  --crop 0,0,1220,2712 --steps 30
```

The code contains an explicit live-input path for future supervised work, but
that path is outside the current project boundary. Do not add `--execute` to
the command above. The wrapper is still only a test harness; it does not make
the toy-trained checkpoint understand real Subway Surfers.

The current PNG capture path is not yet fast enough for a full-speed
unattended run: five device captures measured about 9 seconds on the connected
phone. Treat the session wrapper as lifecycle and safety preparation until a
lower-latency screen stream or smaller device-side capture path is added.

For visual adaptation data, `capture_android_sequence.py` records full PNG
frames and a JSONL file containing the scripted action beside each frame. Use
it in dry-run mode in this snapshot; do not add `--execute` to a real-phone
command:

```bash
./.venv/bin/python capture_android_sequence.py \
  --output-dir results/android_sequence \
  --actions 0,0,1,0,3,0,2
```

The sequence tool does not claim that scripted actions are correct labels for
the game; it simply preserves the exact input/frame history for later
inspection or adaptation.

## Calibration sequence

1. Install Subway Surfers on the test phone and leave it at a known screen.
2. Enable USB debugging and connect the phone to the laptop.
3. Run `adb devices` and accept the debugging prompt on the phone.
4. Capture one frame:

   ```bash
   ./.venv/bin/python capture_android_frame.py --output results/android_frame.png
   ```

5. Inspect the frame and record the game viewport as
   `left,top,width,height`. Exclude status/navigation bars and any unrelated
   overlays.
6. Run the model in dry-run mode for a short sequence:

   ```bash
   ./.venv/bin/python run_android_policy.py \
     results/male_cns_neuron_plastic_50k/male_cns_neuron_plastic_recurrent_ppo.zip \
     --crop left,top,width,height --steps 10
   ```

7. Check the printed actions and crop metadata. These predictions are not
   evidence of real-game competence, and the current workflow sends no phone
   input.

## Interpretation rule

The first phone run is an interface/calibration test, not proof that a
connectome model plays the commercial game. A valid result needs a saved frame,
the crop and display dimensions, action timing, game/version context, and a
separate report of survival or failure. The local 300-step benchmark and live
game measurements must remain separate.

The bridge requires `adb` and `ffmpeg` on the host. The Python checkpoint and
its graph manifest remain local; no phone data is uploaded by these scripts.

## Observed device smoke test

On 2026-09-15 the connected device was a Motorola Edge 60 Pro at 1220x2712,
with `com.kiloo.subwaysurf` in the foreground. Read-only captures verified the
screen-size check, screenshot decoding, near-field packing, checkpoint loading,
and telemetry logging. Predictions on the menu and a manually started run are
not gameplay results because the checkpoint was trained on the toy renderer and
no model action was executed.
