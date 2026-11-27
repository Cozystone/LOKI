"""Prototype for ATANOR-side selection experiments. One instance is one body."""
import torch
from torch import nn


class EmbodiedChoice(nn.Module):
    def __init__(self, senses, channels, actions=90, path_width=30):
        super().__init__()
        inputs = senses + 1
        self.register_buffer("effect", torch.zeros(actions, inputs, path_width))
        self.register_buffer("precision", torch.eye(inputs).repeat(actions, 1, 1) * 100)
        self.register_buffer("relief_place", torch.zeros(channels, path_width))
        self.register_buffer("mass", torch.zeros(channels))

    def features(self, sense):
        return torch.cat((sense.new_ones(1), sense))

    @torch.no_grad()
    def experience(self, sense, path, deficits, action, next_path, next_deficits):
        """One aligned transition. Call only within the same body and lifetime."""
        x = self.features(sense)
        covariance = self.precision[action]
        direction = covariance @ x
        gain = direction / (1 + x @ direction)
        residual = next_path - path - x @ self.effect[action]
        self.effect[action].add_(torch.outer(gain, residual))
        covariance.sub_(torch.outer(gain, direction))
        rise = (deficits - next_deficits).clamp_min(0)
        self.mass.add_(rise).clamp_max_(1024)
        rate = rise / self.mass.clamp_min(1e-8)
        self.relief_place.lerp_(next_path.expand_as(self.relief_place), rate[:, None])

    @torch.no_grad()
    def choose(self, sense, path, deficits, probe=None):
        """Probe replaces desired sensory content only in the oracle ladder."""
        future = path + torch.einsum("i,aij->aj", self.features(sense), self.effect)
        targets = self.relief_place if probe is None else probe[None]
        weight = deficits.square() * (self.mass > 0) if probe is None else deficits.new_ones(1)
        distance = (future[:, None] - targets).square().mean(-1)
        return int((distance @ weight).argmin())
