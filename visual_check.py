"""Check the pixel policy's input/output contract without a checkpoint."""

import numpy as np
import torch
from pathlib import Path

from env import RICH_ACTIONS
from gym_env import RichPixelGymRunnerEnv
from train_visual_ppo import TeacherRewardWrapper, VisualBootstrapExtractor
from train_visual_teacher import (
    VisualPolicy,
    guarded_motion_action,
    initialize_near_field,
    near_field_view,
    timed_action,
)
from train_visual_temporal import (
    TemporalVisualPolicy,
    initialize_from_static,
    stack_history,
)


if __name__ == "__main__":
    action, locked, remaining = timed_action(4, 4, 2, 2)
    assert (action, locked, remaining) == (4, 4, 2)
    action, locked, remaining = timed_action(3, 4, 1, 2)
    assert (action, locked, remaining) == (4, 4, 0)
    action, locked, remaining = timed_action(1, 4, 1, 2)
    assert (action, locked, remaining) == (1, 4, 0)
    print("Action timing check passed")
    state = (None, 0, None, 0)
    outputs = []
    for proposed in (4, 4, 4, 4, 3, 3, 3):
        action, *state = guarded_motion_action(proposed, *state, 2)
        outputs.append(action)
    assert outputs == [4, 4, 4, 4, 4, 4, 3]
    action, *state = guarded_motion_action(3, 4, 2, None, 0, 2)
    assert (action, *state) == (3, 3, 1, None, 0)
    print("Targeted motion timing check passed")
    model = VisualPolicy()
    frame = torch.from_numpy(np.zeros((1, 3, 84, 84), dtype=np.float32))
    logits = model(frame)
    assert tuple(logits.shape) == (1, len(RICH_ACTIONS))
    print(f"Visual policy check passed shape={tuple(logits.shape)}")
    env = RichPixelGymRunnerEnv()
    extractor = VisualBootstrapExtractor(env.observation_space)
    features = extractor(torch.zeros((1, 3, 84, 84), dtype=torch.uint8))
    assert tuple(features.shape) == (1, 128)
    env.close()
    print(f"Visual PPO extractor check passed shape={tuple(features.shape)}")
    wrapped = TeacherRewardWrapper(RichPixelGymRunnerEnv(), weight=0.01, seed=7)
    wrapped.reset(seed=7)
    _, reward, _, _, info = wrapped.step(0)
    assert info["teacher_action"] in RICH_ACTIONS
    assert np.isfinite(reward)
    wrapped.close()
    print("Teacher reward wrapper check passed")
    temporal = TemporalVisualPolicy()
    sample = np.zeros((84, 84, 3), dtype=np.uint8)
    temporal_logits = temporal(torch.from_numpy(stack_history([sample] * 4)).unsqueeze(0))
    assert tuple(temporal_logits.shape) == (1, len(RICH_ACTIONS))
    print(f"Temporal visual policy check passed shape={tuple(temporal_logits.shape)}")
    static = VisualPolicy()
    checkpoint = Path("results/visual_dagger_dense_v2/visual_teacher_best.pt")
    static.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    transferred = TemporalVisualPolicy()
    initialize_from_static(transferred, checkpoint)
    current = np.random.default_rng(7).integers(0, 256, (84, 84, 3), dtype=np.uint8)
    history = [np.zeros_like(current), np.full_like(current, 255), current, current]
    with torch.no_grad():
        static_logits = static(torch.from_numpy(current).permute(2, 0, 1).unsqueeze(0).float())
        transferred_logits = transferred(torch.from_numpy(stack_history(history)).unsqueeze(0))
    assert torch.allclose(static_logits, transferred_logits, atol=1e-5)
    print("Temporal transfer preserves the static policy")
    near_view = near_field_view(current)
    assert near_view.shape == (84, 84, 6)
    near = VisualPolicy(input_channels=6)
    initialize_near_field(near, checkpoint)
    with torch.no_grad():
        near_logits = near(torch.from_numpy(near_view).permute(2, 0, 1).unsqueeze(0).float())
    assert torch.allclose(static_logits, near_logits, atol=1e-5)
    print("Near-field transfer preserves the static policy")
