"""Assert the hybrid policy preserves its base and exposes the fly path."""

from __future__ import annotations

import torch

from gym_env import RichPixelGymRunnerEnv
from train_fly_cns_hybrid import FlyCNSHybridPolicy
from train_fly_cns_retina import FlyCNSRetinaPolicy
from train_visual_teacher import near_field_view


def main() -> None:
    env = RichPixelGymRunnerEnv(seed=7, rich_layer=2, level_variant="mixed")
    try:
        observation, _ = env.reset(seed=7)
        model = FlyCNSHybridPolicy(env.observation_space)
        frame = torch.from_numpy(near_field_view(observation)).permute(2, 0, 1)
        frame = frame.unsqueeze(0).float()
        with torch.no_grad():
            latent = model.base.head[:2](model.base.features(frame / 255.0))
            base_logits = model.base.head[2](latent)
            hybrid_logits = model(frame)
        assert torch.allclose(base_logits, hybrid_logits, atol=1e-6)
        assert model.graph.neuron_count == 196
        assert model.graph.active_edges == 247
        assert torch.isfinite(model.graph.forward_retina(
            torch.sigmoid(model.retina_projection(latent))
        )).all()

        with torch.no_grad():
            model.residual.weight.fill_(0.01)
            changed_logits = model(frame)
        assert not torch.allclose(base_logits, changed_logits)

        brain = FlyCNSRetinaPolicy(env.observation_space)
        with torch.no_grad():
            brain_logits = brain(frame)
        assert brain_logits.shape == (1, 6)
        assert torch.isfinite(brain_logits).all()
        assert brain.graph.neuron_count == 196
        assert all(
            not parameter.requires_grad for parameter in brain.visual.parameters()
        )
    finally:
        env.close()
    print("FlyCNS hybrid check passed: safe base fallback and connected fly branch")


if __name__ == "__main__":
    main()
